#!/usr/bin/env python3
"""
NORMA -- Normality-Referenced Multimodal Anomaly grounding (single-file edition)
================================================================================

Open-ended, temporally grounded video-game glitch detection on VideoGlitchBench
(VGB), evaluated against GliDe (arXiv 2604.07818) under ONE harness in which
every system shares the same videos, frames, backbones, output schema, judge,
statistics and cost accounting.

Task (GliDe Sec. 3): given a gameplay video V, output the SET of glitch events
{(d_k, T_k)}: a free-text description d_k and one or more disjoint intervals
T_k (seconds).

--------------------------------------------------------------------------------
Systems implemented in this file (all share the same backbone adapter)
--------------------------------------------------------------------------------
  vanilla   off-the-shelf backbone, whole video, GliDe summariser JSON
  glide     GliDe reproduced from the paper's specification: 4 fps,
            non-overlapping 8-frame windows stitched into a 2x4 grid image,
            Scanner -> game-aware memory -> Planner/Executor/Reflector
            (Advocate/Skeptic/Judge, <= 5 rounds, stop at confidence >= 0.80)
            with vqa / zoom_in / object_tracking tools -> greedy event grounder
            -> Summariser; temperature 0.5 throughout, as in the paper.
            Documented deviation: SAM3 is replaced by normalised-cross-
            correlation template tracking feeding the SAME four physics rules.
  direct    the vanilla prompt after LoRA-SFT on the training folds
            (separates "fine-tuning" from "architecture")
  norma     this work (below)

--------------------------------------------------------------------------------
NORMA
--------------------------------------------------------------------------------
1. One early-exit prefill per 8 s chunk of the frozen VLM; per time unit the
   RMS-normalised mean/max of the visual-token states at two mid-depth decoder
   layers, Johnson-Lindenstrauss-projected to P dims. No decoding.
2. Self-Referential Normality: deviation of every unit from the video's own
   coordinate-wise median/MAD (50 % breakdown point), temporal surprise
   ||u_t - u_{t-1}||, and a leave-game-out kNN distance to a coreset of normal
   training units (PatchCore-style memory).
3. Temporal transformer heads: actionness, start/end boundaries, event count,
   and a unit-norm instance embedding that groups the separated intervals of
   one recurring bug into ONE multi-interval event.
4. Hybrid-HMM decoding with scaled likelihoods logit p - logit pi
   (Bourlard & Morgan 1994): the prior enters once, via the transitions.
5. LoRA-SFT describer on jitter-cropped events; P(NO_GLITCH) is a feature of
   a dev-fitted stacked calibrator (never a hard veto); MBR decoding.

--------------------------------------------------------------------------------
Fair-comparison contract (enforced by the code, recorded in every artefact)
--------------------------------------------------------------------------------
  data        all VGB videos; trained systems use game-disjoint K-fold CV, so
              each video is predicted by a model that never saw its game, and
              the union of test folds equals the set GliDe reports on.
  frames      one decoder, one 4 fps grid (GliDe's rate), one timestamp
              origin, one per-frame pixel budget for every system.
  backbone    identical checkpoint + revision for all systems of a run.
  schema      identical report schema and post-processing (clip to duration,
              merge overlaps) before scoring.
  judge       one named judge, one prompt, S/5 normalisation, cached by
              (judge, pred, gold); never seen by any system.
  sampling    stochastic systems (glide at T=0.5, norma MBR) are run with
              several seeds; mean +- sd over runs is reported.
  statistics  game-cluster bootstrap CIs; paired null-centred cluster
              bootstrap tests, Holm-corrected.
  cost        per-video model calls, input/output tokens, wall-clock and peak
              GPU memory for EVERY system, plus training GPU-seconds for the
              trained ones.

Commands:  preprocess | split | extract | train-head | train-desc | predict |
           baseline | evaluate | compare | cost | prepare-generic | selftest
Run `python norma_vgb.py selftest` (CPU, about a minute) before using a GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import hashlib
import importlib
import itertools
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("norma")

Interval = Tuple[float, float]

NO_GLITCH = "NO_GLITCH"
DEFAULT_FPS = 4.0                      # GliDe Sec. 5.1: frames sampled at 4 FPS
GLIDE_WINDOW = 8                       # GliDe: non-overlapping 8-frame windows
GLIDE_TEMPERATURE = 0.5                # GliDe: all agents at temperature 0.5
GLIDE_MAX_ROUNDS = 5                   # GliDe: at most 5 verification rounds
GLIDE_STOP_CONF = 0.80                 # GliDe: early stop at judge confidence 0.80
FRAME_CACHE_VERSION = 4
VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v", ".flv")
DEFAULT_HF_REPO = "SandyZheng33/VideoGlitchBench"
UTF8 = "utf-8"
VIDEOS_FILE = "videos.jsonl"
TRAIN_SUMMARY = "train_summary.json"


# =========================================================================== #
# 1. utilities
# =========================================================================== #


def setup_logging(verbosity: int = 1) -> None:
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    if level > logging.DEBUG:
        for noisy in ("urllib3", "filelock", "huggingface_hub", "datasets", "PIL", "httpx"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def set_seed(seed: int) -> None:
    """Seeds torch (the only global RNG used); numpy randomness always goes
    through explicitly seeded np.random.Generator objects."""
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def stable_hash(text: str, n: int = 16) -> str:
    """Content key for caches (SHA-256; not used for any security purpose)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def derived_seed(seed: int, *parts: Any) -> int:
    return int(stable_hash("|".join([str(seed), *map(str, parts)]), 8), 16)


# --------------------------------------------------------------------------- #
# path guard -- every file access in this script goes through safe_path()
# --------------------------------------------------------------------------- #

_ALLOWED_ROOTS: List[str] = []


def allow_root(path: Any) -> None:
    """Register a directory under which this process may read and write."""
    real = os.path.realpath(os.fspath(path))
    if real not in _ALLOWED_ROOTS:
        _ALLOWED_ROOTS.append(real)


def init_allowed_roots() -> None:
    """Allowed roots: the working directory plus every entry of the
    NORMA_DATA_ROOTS environment variable (os.pathsep-separated). CLI
    arguments can choose paths only INSIDE these roots, so a mistyped or
    model-generated argument cannot read or overwrite files elsewhere."""
    allow_root(os.getcwd())
    for entry in os.environ.get("NORMA_DATA_ROOTS", "").split(os.pathsep):
        if entry.strip():
            allow_root(entry.strip())


def safe_path(path: Any) -> Path:
    real = os.path.realpath(os.fspath(path))
    for root in _ALLOWED_ROOTS:
        if os.path.commonpath([real, root]) == root:
            return Path(real)
    raise PermissionError(
        f"path {real!r} is outside the allowed roots {_ALLOWED_ROOTS}; "
        "add its parent directory to NORMA_DATA_ROOTS"
    )


def read_jsonl(path: Any, required: bool = True) -> List[Dict[str, Any]]:
    target = safe_path(path)
    if not target.exists():
        if required:
            raise FileNotFoundError(f"required artefact not found: {target}")
        return []
    rows: List[Dict[str, Any]] = []
    with target.open("r", encoding=UTF8) as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{target}:{line_no} is not valid JSON") from exc
    return rows


def write_jsonl(path: Any, rows: Iterable[Dict[str, Any]]) -> int:
    target = safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding=UTF8) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_json(path: Any, required: bool = True) -> Any:
    target = safe_path(path)
    if not target.exists():
        if required:
            raise FileNotFoundError(f"required artefact not found: {target}")
        return None
    return json.loads(target.read_text(encoding=UTF8))


def write_json(path: Any, payload: Any) -> None:
    target = safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding=UTF8)


def make_rng(seed: int) -> np.random.Generator:
    """Seeded PCG64 generator. Reproducible experimental randomness (fold
    assignment, crops, bootstrap draws); never used for anything security
    related, where a seeded generator would be wrong by design."""
    return np.random.Generator(np.random.PCG64(seed))


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def harmonic(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


_JSON = json.JSONDecoder()


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """First JSON object embedded in model output (```json fences and
    surrounding prose tolerated)."""
    cleaned = re.sub(r"```(?:json)?", "", str(text))
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = _JSON.raw_decode(cleaned, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


# --------------------------------------------------------------------------- #
# interval algebra -- the single definition used by targets, decoding and tIoU
# --------------------------------------------------------------------------- #


def merge_intervals(intervals: Sequence[Interval], gap: float = 0.0) -> List[Interval]:
    ordered = sorted((float(a), float(b)) for a, b in intervals if float(b) > float(a))
    if not ordered:
        return []
    merged = [list(ordered[0])]
    for a, b in ordered[1:]:
        if a <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def normalise_intervals(raw: Any) -> List[Interval]:
    """Every [a, b] leaf below `raw`, depth-agnostic, ordered and merged."""
    out: List[Interval] = []

    def walk(node: Any) -> None:
        if node is None or not isinstance(node, (list, tuple)):
            return
        if len(node) == 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in node):
            a, b = sorted((float(node[0]), float(node[1])))
            if b > a:
                out.append((a, b))
            return
        for child in node:
            walk(child)

    walk(raw)
    return merge_intervals(out)


def clip_intervals(intervals: Sequence[Interval], duration: float) -> List[Interval]:
    if duration <= 0:
        return merge_intervals(intervals)
    return merge_intervals([(max(0.0, a), min(duration, b)) for a, b in intervals if a < duration])


def measure(intervals: Sequence[Interval]) -> float:
    return float(sum(b - a for a, b in merge_intervals(intervals)))


def intersect_measure(left: Sequence[Interval], right: Sequence[Interval]) -> float:
    a, b = merge_intervals(left), merge_intervals(right)
    total, i, j = 0.0, 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def interval_iou(left: Sequence[Interval], right: Sequence[Interval]) -> float:
    """tIoU between two interval SETS (multi-interval events), as in GliDe."""
    union = measure(list(left) + list(right))
    return intersect_measure(left, right) / union if union > 0.0 else 0.0


# --------------------------------------------------------------------------- #
# cost meter -- the same accounting wraps every system
# --------------------------------------------------------------------------- #


class CostMeter:
    """Counts model calls, input/output tokens and seconds per call kind."""

    def __init__(self) -> None:
        self.calls: Counter = Counter()
        self.in_tokens: Counter = Counter()
        self.out_tokens: Counter = Counter()
        self.seconds: Dict[str, float] = defaultdict(float)

    @contextlib.contextmanager
    def track(self, kind: str, in_tokens: int = 0) -> Iterator[Dict[str, int]]:
        box = {"out_tokens": 0, "in_tokens": int(in_tokens)}
        t0 = time.perf_counter()
        try:
            yield box
        finally:
            self.calls[kind] += 1
            self.in_tokens[kind] += box["in_tokens"]
            self.out_tokens[kind] += box["out_tokens"]
            self.seconds[kind] += time.perf_counter() - t0

    def snapshot(self) -> Dict[str, float]:
        return {"calls": float(sum(self.calls.values())), "in_tokens": float(sum(self.in_tokens.values())),
                "out_tokens": float(sum(self.out_tokens.values())), "seconds": float(sum(self.seconds.values()))}

    @staticmethod
    def delta(after: Dict[str, float], before: Dict[str, float]) -> Dict[str, float]:
        return {k: after[k] - before.get(k, 0.0) for k in after}

    def by_kind(self) -> Dict[str, Dict[str, float]]:
        return {k: {"calls": float(self.calls[k]), "in_tokens": float(self.in_tokens[k]),
                    "out_tokens": float(self.out_tokens[k]), "seconds": float(self.seconds[k])}
                for k in sorted(self.calls)}


def peak_gpu_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 2 ** 20)
    except ImportError:
        pass
    return 0.0


# =========================================================================== #
# 2. data: ingest, decoding, frame cache, targets, splits
# =========================================================================== #


def _as_python(value: Any) -> Any:
    """`datasets` may hand back numpy scalars / arrays / JSON-encoded lists."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_as_python(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_as_python(v) for v in value]
    if isinstance(value, str) and value.strip().startswith("[") and value.strip().endswith("]"):
        try:
            return _as_python(json.loads(value))
        except json.JSONDecodeError:
            return value
    return value


def parse_time_nodes(raw: Any, n_bugs: int, video_name: str) -> List[List[Interval]]:
    """
    `time_nodes` is a list OVER BUGS of [start, end] lists:
        [[[3, 11]]]                        1 bug, 1 interval
        [[[3, 11], [20, 24]], [[40, 45]]]  2 bugs, 3 intervals
    If the outer arity does not match len(bugs) the bug <-> span alignment is
    unrecoverable; we keep ONE unattributed group instead of silently pairing
    a description with the wrong span.
    """
    raw = _as_python(raw)
    if raw is None:
        return []
    groups = raw if isinstance(raw, list) else [raw]
    spans = [s for s in (normalise_intervals(g) for g in groups) if s]
    if n_bugs and len(spans) != n_bugs and spans:
        LOGGER.warning("time_nodes arity %d != len(bugs) %d for %s; alignment dropped",
                       len(spans), n_bugs, video_name)
        return [merge_intervals([iv for s in spans for iv in s])]
    return spans


def _first_present(row: Dict[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if row.get(name) is not None:
            return row[name]
    return default


def _video_basename(ref: Any) -> str:
    if ref is None:
        return ""
    if isinstance(ref, dict):
        return os.path.basename(str(ref.get("path", "")))
    for attr in ("path", "filename"):
        value = getattr(ref, attr, None)
        if isinstance(value, str) and value:
            return os.path.basename(value)
    return os.path.basename(str(ref))


def build_video_index(video_root: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """basename -> path and stem -> path. Duplicate basenames are an error:
    the metadata joins on basename and an ambiguous join pairs annotations
    with the wrong pixels."""
    root = safe_path(video_root)
    if not root.is_dir():
        raise NotADirectoryError(f"--video-root does not exist: {root}")
    by_name: Dict[str, Path] = {}
    duplicates: List[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            if path.name in by_name:
                duplicates.append(path.name)
            by_name[path.name] = path
    if duplicates:
        raise RuntimeError(f"ambiguous video basenames under {root}: {duplicates[:5]}")
    if not by_name:
        raise RuntimeError(f"no video files found under {root}")
    by_stem = {Path(name).stem: path for name, path in by_name.items()}
    return by_name, by_stem


def load_metadata_rows(hf_repo: str, split: str, metadata: Optional[str],
                       revision: Optional[str]) -> List[Dict[str, Any]]:
    """Annotation rows from a local JSON/JSONL file or the HF dataset. The
    requested split is used verbatim: no silent fallback to another split."""
    if metadata:
        if metadata.endswith(".jsonl"):
            return read_jsonl(metadata)
        payload = read_json(metadata)
        return payload if isinstance(payload, list) else list(payload.values())
    from datasets import load_dataset

    dset = load_dataset(hf_repo, split=split, revision=revision)
    LOGGER.info("loaded %d metadata rows (%s, split=%s)", len(dset), hf_repo, split)
    return [dict(row) for row in dset]


def record_from_row(row: Dict[str, Any], by_name: Dict[str, Path],
                    by_stem: Dict[str, Path]) -> Optional[Dict[str, Any]]:
    name = _video_basename(_first_present(row, ("video_name", "video", "file_name")))
    path = by_name.get(name) or by_stem.get(Path(name).stem)
    if not name or path is None:
        return None
    bugs_raw = _as_python(_first_present(row, ("bugs",), [])) or []
    bugs = [str(b).strip() for b in (bugs_raw if isinstance(bugs_raw, list) else [bugs_raw]) if str(b).strip()]
    spans = parse_time_nodes(_first_present(row, ("time_nodes",), []), len(bugs), name)
    return {
        "video_name": name, "video_path": str(path),
        "game_name": str(_as_python(_first_present(row, ("game_name", "game"), "unknown"))),
        "genre": str(_as_python(_first_present(row, ("genre",), ""))),
        "subgenre": str(_as_python(_first_present(row, ("subgenre",), ""))),
        "no_bugs": bool(_as_python(_first_present(row, ("no_bugs",), False))) or not spans,
        "bugs": bugs, "bug_spans": [[list(iv) for iv in s] for s in spans],
    }


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #


def letterbox(array: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Aspect-preserving resize + zero padding: angles and trajectories stay
    metric, which physics / clipping judgements depend on."""
    from PIL import Image

    h, w = array.shape[:2]
    if (h, w) == (out_h, out_w):
        return array
    scale = min(out_h / float(h), out_w / float(w))
    new_h, new_w = max(1, min(out_h, round(h * scale))), max(1, min(out_w, round(w * scale)))
    resized = np.asarray(Image.fromarray(array).resize((new_w, new_h), Image.BICUBIC), dtype=np.uint8)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    top, left = (out_h - new_h) // 2, (out_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def _target_size(h: int, w: int, short_side: int) -> Tuple[int, int]:
    if short_side <= 0:
        return h, w
    scale = short_side / float(min(h, w))
    return max(2, round(h * scale)), max(2, round(w * scale))


def _timed_frames(container: Any, stream: Any) -> Iterator[Tuple[float, Any]]:
    """(seconds since the FIRST decoded frame, frame). Annotation clocks start
    at the first displayed frame; raw pts * time_base does not whenever the
    stream has a non-zero start_time (edit lists, B-frame delay)."""
    base = float(stream.time_base) if stream.time_base else 0.0
    origin: Optional[float] = None
    for index, frame in enumerate(container.decode(video=0)):
        if frame.pts is None:
            continue
        t = float(frame.pts) * base if base else float(index)
        origin = t if origin is None else origin
        yield t - origin, frame


def decode_uniform_frames(video_path: Any, fps: float, short_side: int
                          ) -> Tuple[List[np.ndarray], List[float], float]:
    """For every grid point t_k = k / fps keep the first frame at or after
    t_k. One geometry per video (from the first frame)."""
    import av

    if fps <= 0:
        raise ValueError("fps must be positive")
    step = 1.0 / fps
    frames: List[np.ndarray] = []
    stamps: List[float] = []
    size: Optional[Tuple[int, int]] = None
    with av.open(str(safe_path(video_path))) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        next_t = 0.0
        for t, frame in _timed_frames(container, stream):
            if t + 1e-9 < next_t:
                continue
            array = frame.to_ndarray(format="rgb24")
            size = size or _target_size(array.shape[0], array.shape[1], short_side)
            frames.append(letterbox(array, *size))
            stamps.append(t)
            next_t = (math.floor(t / step) + 1) * step
    if not frames:
        raise RuntimeError(f"decoded zero frames from {video_path}")
    return frames, stamps, stamps[-1] + step


def cache_fingerprint(fps: float, short_side: int, quality: int) -> str:
    return stable_hash(json.dumps({"v": FRAME_CACHE_VERSION, "fps": round(fps, 6), "short": short_side,
                                   "q": quality}, sort_keys=True), 12)


def _valid_cache(marker: Path, fingerprint: str) -> Optional[Dict[str, Any]]:
    if not marker.exists():
        return None
    try:
        cached = json.loads(marker.read_text(encoding=UTF8))
    except json.JSONDecodeError:
        return None
    paths = cached.get("frame_paths") or []
    if cached.get("fingerprint") != fingerprint or not paths or not all(Path(p).exists() for p in paths):
        return None
    return cached


def _write_frames(out_dir: Path, frames: Sequence[np.ndarray], quality: int) -> List[str]:
    from PIL import Image

    for stale in out_dir.glob("*.jpg"):
        stale.unlink()
    paths = []
    for i, array in enumerate(frames):
        path = out_dir / f"{i:06d}.jpg"
        Image.fromarray(array).save(path, format="JPEG", quality=quality)
        paths.append(str(path))
    return paths


@dataclasses.dataclass
class CacheJob:
    frames_dir: str
    fps: float = DEFAULT_FPS
    short_side: int = 360
    quality: int = 92
    overwrite: bool = False


def cache_video(record: Dict[str, Any], job: CacheJob) -> Dict[str, Any]:
    """Decode once on the shared 4 fps grid, cache JPEGs under a fingerprint,
    return the videos.jsonl row (with an explicit pointer to the marker)."""
    out_dir = safe_path(Path(job.frames_dir) / re.sub(r"[^\w.-]+", "_", record["game_name"])
                        / Path(record["video_name"]).stem)
    marker = out_dir / "_frames.json"
    fingerprint = cache_fingerprint(job.fps, job.short_side, job.quality)
    cached = None if job.overwrite else _valid_cache(marker, fingerprint)
    if cached is None:
        frames, stamps, duration = decode_uniform_frames(record["video_path"], job.fps, job.short_side)
        out_dir.mkdir(parents=True, exist_ok=True)
        cached = {"fingerprint": fingerprint, "frame_paths": _write_frames(out_dir, frames, job.quality),
                  "timestamps": stamps, "duration": duration, "target_fps": job.fps}
        marker.write_text(json.dumps(cached), encoding=UTF8)
    return {**{k: v for k, v in record.items() if k != "video_path"},
            "duration": float(cached["duration"]), "n_frames": len(cached["frame_paths"]),
            "frames_marker": str(marker)}


def _cache_worker(record: Dict[str, Any], job: CacheJob, roots: List[str]) -> Dict[str, Any]:
    for root in roots:
        allow_root(root)
    return cache_video(record, job)


def cache_all(records: Sequence[Dict[str, Any]], job: CacheJob, workers: int
              ) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    if workers <= 1:
        for record in records:
            try:
                rows.append(cache_video(record, job))
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append({"video_name": record["video_name"], "error": repr(exc)})
        return rows, failures
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # spawn, not fork: FFmpeg thread state does not survive fork reliably.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = {pool.submit(_cache_worker, r, job, list(_ALLOWED_ROOTS)): r for r in records}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append({"video_name": futures[future]["video_name"], "error": repr(exc)})
    return rows, failures


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class VideoItem:
    name: str
    game: str
    genre: str
    subgenre: str
    duration: float
    fps: float
    frame_paths: List[str]
    timestamps: List[float]
    no_bugs: bool
    bugs: List[str]
    bug_spans: List[List[Interval]]

    @property
    def stem(self) -> str:
        return Path(self.name).stem

    def game_text(self) -> Dict[str, str]:
        return {"game": self.game, "genre": self.genre or "unknown", "subgenre": self.subgenre or "unknown"}


def _item_from_row(row: Dict[str, Any]) -> Optional[VideoItem]:
    marker = read_json(row["frames_marker"], required=False) if row.get("frames_marker") else None
    if not marker or not marker.get("frame_paths"):
        return None
    spans = [normalise_intervals(s) for s in (row.get("bug_spans") or [])]
    return VideoItem(
        name=row["video_name"], game=str(row.get("game_name") or "unknown"),
        genre=str(row.get("genre") or ""), subgenre=str(row.get("subgenre") or ""),
        duration=float(row.get("duration") or marker.get("duration") or 0.0),
        fps=float(marker.get("target_fps") or DEFAULT_FPS),
        frame_paths=list(marker["frame_paths"]), timestamps=[float(t) for t in marker["timestamps"]],
        no_bugs=bool(row.get("no_bugs")) or not any(spans),
        bugs=[str(b).strip() for b in (row.get("bugs") or []) if str(b).strip()], bug_spans=spans,
    )


def load_corpus(work_dir: Any) -> Dict[str, VideoItem]:
    items: Dict[str, VideoItem] = {}
    rows = read_jsonl(Path(work_dir) / VIDEOS_FILE)
    for row in rows:
        item = _item_from_row(row)
        if item is not None:
            items[item.name] = item
    if len(items) < len(rows):
        LOGGER.warning("%d/%d videos have no frame cache and were skipped", len(rows) - len(items), len(rows))
    if not items:
        raise RuntimeError(f"no usable videos under {work_dir}; run `preprocess` first")
    return items


def gold_events(item: VideoItem) -> List[Dict[str, Any]]:
    """One event per annotated bug with its own (possibly multi-interval)
    support; an unattributed group keeps all descriptions joined so no gold
    event leaves the denominator M."""
    if item.no_bugs:
        return []
    spans = [s for s in item.bug_spans if s]
    if spans and len(spans) == len(item.bugs):
        return [{"description": d, "intervals": merge_intervals(s)} for d, s in zip(item.bugs, spans, strict=True)]
    if spans:
        flat = merge_intervals([iv for s in spans for iv in s])
        return [{"description": " ".join(item.bugs).strip() or "glitch", "intervals": flat}]
    return []


# --------------------------------------------------------------------------- #
# per-unit targets
# --------------------------------------------------------------------------- #


def _interval_mask(centres: np.ndarray, a: float, b: float) -> np.ndarray:
    inside = (centres >= a) & (centres < b)
    if not inside.any():
        # A glitch shorter than one unit claims the unit that contains it.
        inside = np.zeros(len(centres), dtype=bool)
        inside[int(np.argmin(np.abs(centres - 0.5 * (a + b))))] = True
    return inside


def _heat(n: int, centre: float, sigma: float) -> np.ndarray:
    idx = np.arange(n, dtype=np.float32)
    return np.exp(-0.5 * ((idx - centre) / sigma) ** 2).astype(np.float32)


def unit_targets(unit_times: np.ndarray, events: Sequence[Dict[str, Any]], sigma_units: float = 1.0
                 ) -> Dict[str, np.ndarray]:
    """
    y[t]      1 if the centre of unit t lies inside a gold interval
    inst[t]   owning event index; -1 background; -2 two concurrent events
              (excluded from the embedding loss)
    start/end Gaussian boundary heat-maps (BMN-style soft targets)
    """
    n = int(unit_times.shape[0])
    centres = unit_times.mean(axis=1)
    y = np.zeros(n, dtype=np.float32)
    inst = np.full(n, -1, dtype=np.int64)
    start = np.zeros(n, dtype=np.float32)
    end = np.zeros(n, dtype=np.float32)
    for k, event in enumerate(events):
        for a, b in event["intervals"]:
            inside = _interval_mask(centres, a, b)
            y[inside] = 1.0
            clash = inside & (inst >= 0) & (inst != k)
            inst[inside & (inst == -1)] = k
            inst[clash] = -2
            start = np.maximum(start, _heat(n, float(np.argmin(np.abs(unit_times[:, 0] - a))), sigma_units))
            end = np.maximum(end, _heat(n, float(np.argmin(np.abs(unit_times[:, 1] - b))), sigma_units))
    return {"y": y, "inst": inst, "start": start, "end": end}


def select_frames(item: VideoItem, t0: float, t1: float, max_frames: int) -> Tuple[List[str], List[float], float]:
    """Uniform subsample of cached frames inside [t0, t1], even count (Qwen
    pairs frames into temporal patches); returns frames, times, effective fps."""
    stamps = np.asarray(item.timestamps, dtype=float)
    keep = np.nonzero((stamps >= t0 - 1e-6) & (stamps <= t1 + 1e-6))[0]
    if keep.size == 0:
        keep = np.array([int(np.argmin(np.abs(stamps - 0.5 * (t0 + t1))))])
    if keep.size > max_frames:
        keep = keep[np.round(np.linspace(0, keep.size - 1, num=max_frames)).astype(int)]
    keep_list = list(dict.fromkeys(int(k) for k in keep))
    if len(keep_list) % 2:
        keep_list.append(keep_list[-1])
    span = max(stamps[keep_list[-1]] - stamps[keep_list[0]], 1e-3)
    eff_fps = (len(set(keep_list)) - 1) / span if len(set(keep_list)) > 1 else item.fps
    return ([item.frame_paths[k] for k in keep_list], [float(stamps[k]) for k in keep_list],
            float(max(eff_fps, 0.25)))


# --------------------------------------------------------------------------- #
# game-disjoint, genre-stratified folds
# --------------------------------------------------------------------------- #


def _assign_games(games: Dict[str, List[str]], genre_of: Dict[str, str], k: int,
                  rng: np.random.Generator) -> List[List[str]]:
    """Largest-first greedy packing inside every genre: balances the genre
    mix and the video count of each fold."""
    fold_games: List[List[str]] = [[] for _ in range(k)]
    fold_size = [0] * k
    fold_genre: List[Counter] = [Counter() for _ in range(k)]
    tiebreak = {g: float(rng.random()) for g in sorted(games)}
    for genre in sorted(set(genre_of.values())):
        pool = sorted((g for g in games if genre_of[g] == genre), key=lambda g: (-len(games[g]), tiebreak[g]))
        def load(f: int, gen: str = genre) -> Tuple[int, int, int]:
            return fold_genre[f][gen], fold_size[f], f

        for game in pool:
            target = min(range(k), key=load)
            fold_games[target].append(game)
            fold_size[target] += len(games[game])
            fold_genre[target][genre] += len(games[game])
    return fold_games


def _carve_dev(rest: List[str], games: Dict[str, List[str]], dev_frac: float,
               rng: np.random.Generator) -> List[str]:
    order = [rest[i] for i in rng.permutation(len(rest))]
    target = dev_frac * sum(len(games[g]) for g in rest)
    dev, acc = [], 0
    for game in order:
        if acc >= target:
            break
        dev.append(game)
        acc += len(games[game])
    return dev


def _games_of(items: Dict[str, VideoItem]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    games: Dict[str, List[str]] = defaultdict(list)
    genre_of: Dict[str, str] = {}
    for name, item in items.items():
        games[item.game].append(name)
        genre_of.setdefault(item.game, item.genre or "unknown")
    return games, genre_of


def _fold_record(f: int, fold_games: List[List[str]], games: Dict[str, List[str]], dev_frac: float,
                 seed: int) -> Dict[str, Any]:
    rest = sorted(g for i, gs in enumerate(fold_games) if i != f for g in gs)
    dev_games = _carve_dev(rest, games, dev_frac, make_rng(derived_seed(seed, "dev", f)))
    fold: Dict[str, Any] = {"fold": f, "test_games": sorted(fold_games[f]), "dev_games": sorted(dev_games),
                            "train_games": sorted(set(rest) - set(dev_games))}
    for part in ("test", "dev", "train"):
        fold[part] = sorted(v for g in fold[f"{part}_games"] for v in games[g])
    LOGGER.info("fold %d: train %d / dev %d / test %d videos (%d / %d / %d games)", f, len(fold["train"]),
                len(fold["dev"]), len(fold["test"]), len(fold["train_games"]), len(dev_games), len(fold["test_games"]))
    return fold


def make_folds(items: Dict[str, VideoItem], k: int, dev_frac: float, seed: int) -> Dict[str, Any]:
    """
    VGB has no official split and GliDe reports on all videos. Out-of-fold
    predictions over K game-disjoint folds cover exactly that set, and
    game-disjointness turns the score into a cross-game generalisation
    estimate (videos of one game share engine, art style and often the bug).
    """
    games, genre_of = _games_of(items)
    if len(games) < k:
        raise ValueError(f"only {len(games)} games; cannot build {k} game-disjoint folds")
    fold_games = _assign_games(games, genre_of, k, make_rng(seed))
    folds = [_fold_record(f, fold_games, games, dev_frac, seed) for f in range(k)]
    assert_disjoint(folds)
    return {"k": k, "seed": seed, "dev_frac": dev_frac, "folds": folds}


def assert_disjoint(folds: Sequence[Dict[str, Any]]) -> None:
    covered: Counter = Counter()
    for fold in folds:
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
            if set(fold[a]) & set(fold[b]) or set(fold[f"{a}_games"]) & set(fold[f"{b}_games"]):
                raise RuntimeError(f"fold {fold['fold']}: {a} and {b} overlap")
        covered.update(fold["test"])
    if any(c != 1 for c in covered.values()):
        raise RuntimeError("every video must be in exactly one test fold")


# =========================================================================== #
# 3. geometry and backbone adapters
# =========================================================================== #

QWEN_END = "<|im_end|>"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EXTRACT_SYSTEM = (
    "You are a video game QA engineer. The following gameplay footage is from \"{game}\" "
    "({genre} / {subgenre}). Watch it for software glitches: physics violations, clipping through "
    "geometry, rendering, texture or lighting errors, broken animation, nonsensical agent behaviour, "
    "corrupted UI, or invalid game state. Unusual but legitimate gameplay is not a glitch."
)
EXTRACT_USER = "Is there a software glitch in this footage, and when does it happen?"


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> Tuple[int, int]:
    """Qwen2/2.5-VL resize rule: both sides multiples of `factor`, area in
    [min_pixels, max_pixels], aspect preserved within one factor step.
    Idempotent, so the processor's own resize is a no-op afterwards."""
    if height <= 0 or width <= 0:
        raise ValueError(f"degenerate frame size {height}x{width}")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = max(factor, math.ceil(height * beta / factor) * factor)
        w_bar = max(factor, math.ceil(width * beta / factor) * factor)
    return int(h_bar), int(w_bar)


def load_rgb(path: Any) -> np.ndarray:
    from PIL import Image

    with Image.open(safe_path(path)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def load_uniform_video(frame_paths: Sequence[str], factor: int, min_pixels: int, max_pixels: int,
                       temporal_patch: int = 2) -> np.ndarray:
    """(T, H, W, 3) with ONE geometry, chosen from the modal frame size
    (mixed-resolution lists are letterboxed, never stretched); T padded to a
    multiple of the temporal patch by repeating the last frame."""
    raw = [load_rgb(p) for p in frame_paths]
    if not raw:
        raise ValueError("empty frame list")
    (mode_h, mode_w), _ = Counter(a.shape[:2] for a in raw).most_common(1)[0]
    out_h, out_w = smart_resize(mode_h, mode_w, factor, min_pixels, max_pixels)
    frames = [letterbox(a, out_h, out_w) for a in raw]
    while len(frames) % max(1, temporal_patch):
        frames.append(frames[-1].copy())
    return np.stack(frames, axis=0)


def make_grid(frame_paths: Sequence[str], first_index: int, cols: int = 4, rows: int = 2) -> Any:
    """GliDe input format: 8 frames stitched into a 2x4 grid, the frame index
    written in each tile's top-left corner."""
    from PIL import Image, ImageDraw

    arrays = [load_rgb(p) for p in frame_paths][: cols * rows]
    h, w = arrays[0].shape[:2]
    canvas = Image.new("RGB", (w * cols, h * rows))
    draw = ImageDraw.Draw(canvas)
    for i, arr in enumerate(arrays):
        x, y = (i % cols) * w, (i // cols) * h
        canvas.paste(Image.fromarray(letterbox(arr, h, w)), (x, y))
        draw.rectangle([x, y, x + 34, y + 16], fill=(0, 0, 0))
        draw.text((x + 3, y + 2), str(first_index + i), fill=(255, 255, 0))
    return canvas


class EarlyExit(Exception):
    """Raised by the deepest layer tap to stop the forward pass."""


class LayerTap:
    """Forward hooks on chosen decoder layers; aborts the forward pass right
    after the deepest one, so deeper layers are never computed."""

    def __init__(self, layers: Any, indices: Sequence[int]) -> None:
        self.layers = layers
        self.indices = sorted({int(i) for i in indices})
        self.cache: Dict[int, Any] = {}
        self._handles: List[Any] = []

    def _hook(self, idx: int) -> Callable[..., None]:
        deepest = self.indices[-1]

        def hook(_module: Any, _inp: Any, out: Any) -> None:
            hidden = out[0] if isinstance(out, (tuple, list)) else out
            self.cache[idx] = hidden.detach()
            if idx == deepest:
                raise EarlyExit()

        return hook

    def __enter__(self) -> "LayerTap":
        self._handles = [self.layers[i].register_forward_hook(self._hook(i)) for i in self.indices]
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return exc_type is EarlyExit


def find_decoder_layers(root: Any) -> Any:
    """The longest ModuleList named '*layers' outside the vision tower."""
    import torch

    candidates = [(len(m), name, m) for name, m in root.named_modules()
                  if isinstance(m, torch.nn.ModuleList) and name.endswith("layers")
                  and not any(tag in name for tag in ("visual", "vision", "vit"))]
    if not candidates:
        raise RuntimeError("could not locate the language-model decoder layers")
    return max(candidates, key=lambda c: c[0])[2]


def resolve_layers(n_layers: int, fractions: Sequence[float]) -> List[int]:
    return sorted({min(n_layers - 1, max(0, round(f * (n_layers - 1)))) for f in fractions})


def pool_tokens(tokens: Any) -> Any:
    """[U, N, D] -> [U, 2, D]: mean and max over the N visual tokens of each
    unit after per-token RMS normalisation (massive-activation channels,
    Sun et al. 2024, would otherwise dominate max-pooling)."""
    import torch

    x = tokens.float()
    x = x / x.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    return torch.stack([x.mean(dim=1), x.amax(dim=1)], dim=1)


def rms_unit(vector: Any) -> np.ndarray:
    v = vector.float()
    return (v / v.pow(2).mean().add(1e-6).sqrt()).cpu().numpy().astype(np.float32)


def qwen_unit_pool(hidden: Any, input_ids: Any, grid_thw: Sequence[int], video_token_id: int, merge: int) -> Any:
    """Video-token states of one sequence folded into units. Qwen2.5-VL emits
    merged video tokens in (t, h/merge, w/merge) row-major order (window
    attention reordering is undone inside the vision tower)."""
    t, h, w = (int(v) for v in grid_thw)
    per = (h // merge) * (w // merge)
    tokens = hidden[input_ids == video_token_id]
    if tokens.shape[0] != t * per:
        raise RuntimeError(f"video token count {tokens.shape[0]} != t*h*w/merge^2 = {t * per}")
    return pool_tokens(tokens.view(t, per, -1))


def unit_time_table(timestamps: Sequence[float], n_units: int, unit_frames: int, fps: float) -> np.ndarray:
    stamps = list(timestamps) + [timestamps[-1]] * max(0, n_units * unit_frames - len(timestamps))
    return np.array([[stamps[u * unit_frames], stamps[u * unit_frames + unit_frames - 1] + 1.0 / fps]
                     for u in range(n_units)], dtype=np.float32)


@dataclasses.dataclass
class GenSpec:
    max_new_tokens: int = 96
    temperature: float = 0.0           # 0 = greedy
    n: int = 1                          # samples when temperature > 0
    ban: Optional[str] = None           # forbid the first token of this word
    seed: int = 0


@dataclasses.dataclass
class BackboneSpec:
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    revision: Optional[str] = None
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    attn: str = "sdpa"
    max_pixels: int = 160 * 28 * 28     # per-frame budget, identical for every system
    min_pixels: int = 32 * 28 * 28
    layer_fracs: Tuple[float, ...] = (0.5, 0.75)
    trust_remote_code: bool = False

    def fingerprint(self) -> Dict[str, Any]:
        return {"model_id": self.model_id, "revision": self.revision, "dtype": self.dtype,
                "load_in_4bit": self.load_in_4bit, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels}


class Backbone:
    """Every model call of every system goes through these methods, so call
    counts, tokens and latency are measured identically for all systems."""

    name = "base"
    hidden_size = 0
    unit_frames = 1

    def __init__(self) -> None:
        self.meter = CostMeter()
        self._peft_model: Any = None

    # primitives implemented by subclasses
    def extract_units(self, frames: Sequence[str], times: Sequence[float], fps: float,
                      game_text: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def lm_loss(self, frames: Sequence[str], fps: float, prompt: str, answer: str) -> Any:
        raise NotImplementedError

    def generate_video(self, frames: Sequence[str], fps: float, prompt: str, gen: GenSpec) -> List[str]:
        raise NotImplementedError

    def first_token_prob(self, frames: Sequence[str], fps: float, prompt: str, word: str) -> float:
        raise NotImplementedError

    def generate_image(self, images: Sequence[Any], prompt: str, gen: GenSpec, system: Optional[str] = None) -> str:
        raise NotImplementedError

    def generate_text(self, prompt: str, gen: GenSpec, system: Optional[str] = None) -> str:
        raise NotImplementedError

    # LoRA plumbing (PEFT); adapters are stored as safetensors by PEFT
    def _wrap(self, module: Any, targets: Any, r: int, alpha: int) -> Any:
        from peft import LoraConfig, get_peft_model

        wrapped = get_peft_model(module, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05, bias="none",
                                                    target_modules=targets, task_type="CAUSAL_LM"))
        n_train = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
        LOGGER.info("LoRA attached: %.2fM trainable parameters", n_train / 1e6)
        self._peft_model = wrapped
        return wrapped

    def _load_peft(self, module: Any, path: Path) -> Any:
        from peft import PeftModel

        self._peft_model = PeftModel.from_pretrained(module, str(safe_path(path)), is_trainable=False)
        return self._peft_model

    def attach_lora(self, r: int, alpha: int) -> None:
        raise NotImplementedError

    def load_adapter(self, path: Path) -> None:
        raise NotImplementedError

    def save_adapter(self, path: Path) -> None:
        if self._peft_model is None:
            raise RuntimeError("no adapter attached")
        self._peft_model.save_pretrained(str(safe_path(path)), safe_serialization=True)

    @contextlib.contextmanager
    def adapter_disabled(self) -> Iterator[None]:
        if self._peft_model is None:
            yield
            return
        with self._peft_model.disable_adapter():
            yield

    def trainable_parameters(self) -> List[Any]:
        if self._peft_model is None:
            return []
        return [p for p in self._peft_model.parameters() if p.requires_grad]

    def train_mode(self, on: bool) -> None:
        raise NotImplementedError


def _seed_torch(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _hf_generate(model: Any, inputs: Dict[str, Any], gen: GenSpec, ban_ids: Optional[List[int]]) -> Any:
    kwargs: Dict[str, Any] = {"max_new_tokens": gen.max_new_tokens}
    if gen.temperature > 0:
        kwargs.update({"do_sample": True, "temperature": gen.temperature, "top_p": 0.95,
                       "num_return_sequences": max(1, gen.n)})
        _seed_torch(gen.seed)
    else:
        kwargs["do_sample"] = False
    if ban_ids:
        kwargs["bad_words_ids"] = [ban_ids]
    return model.generate(**inputs, **kwargs)


class QwenFamily(Backbone):
    """Qwen2.5-VL-3B/7B-Instruct and UI-TARS-1.5-7B (same architecture)."""

    LORA_TARGETS = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"

    def __init__(self, spec: BackboneSpec) -> None:
        super().__init__()
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch, self.spec, self.name = torch, spec, spec.model_id
        if spec.revision is None:
            LOGGER.warning("no --revision given: %s is loaded unpinned; pin a commit for reproducibility", spec.model_id)
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[spec.dtype]
        major = int(re.findall(r"\d+", transformers.__version__)[0])
        kwargs: Dict[str, Any] = {("dtype" if major >= 5 else "torch_dtype"): dtype,
                                  "attn_implementation": spec.attn, "device_map": "auto", "revision": spec.revision}
        if spec.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                                                               bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(spec.model_id, **kwargs).eval()
        self.model = self.base_model
        self.processor = AutoProcessor.from_pretrained(spec.model_id, revision=spec.revision,
                                                       min_pixels=spec.min_pixels, max_pixels=spec.max_pixels)
        self.tokenizer = self.processor.tokenizer
        cfg = self.base_model.config
        vcfg = cfg.vision_config
        self.hidden_size = int((getattr(cfg, "text_config", None) or cfg).hidden_size)
        self.video_token_id = int(cfg.video_token_id)
        self.merge = int(getattr(vcfg, "spatial_merge_size", 2))
        self.factor = int(getattr(vcfg, "patch_size", 14)) * self.merge
        self.unit_frames = int(getattr(vcfg, "temporal_patch_size", 2))
        self.layer_idx = resolve_layers(len(find_decoder_layers(self.base_model)), spec.layer_fracs)
        self._set_budget(spec.min_pixels, spec.max_pixels)

    def _set_budget(self, min_pixels: int, max_pixels: int) -> None:
        """Force one pixel budget onto every sub-processor (newer releases keep
        a separate video processor with different defaults)."""
        for name in ("image_processor", "video_processor"):
            proc = getattr(self.processor, name, None)
            if proc is None:
                continue
            for attr, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
                if hasattr(proc, attr):
                    setattr(proc, attr, value)
            size = getattr(proc, "size", None)
            if isinstance(size, dict) and "longest_edge" in size:
                size.update({"shortest_edge": min_pixels, "longest_edge": max_pixels})
            if hasattr(proc, "do_sample_frames"):
                proc.do_sample_frames = False

    @contextlib.contextmanager
    def _budget(self, min_pixels: int, max_pixels: int) -> Iterator[None]:
        self._set_budget(min_pixels, max_pixels)
        try:
            yield
        finally:
            self._set_budget(self.spec.min_pixels, self.spec.max_pixels)

    def _chat(self, content: List[Dict[str, Any]], system: Optional[str]) -> str:
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user", "content": content})
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _video_inputs(self, frames: Sequence[str], fps: float, text: str) -> Any:
        video = load_uniform_video(frames, self.factor, self.spec.min_pixels, self.spec.max_pixels, self.unit_frames)
        inputs = self.processor(text=[text], videos=[video], fps=[float(fps)], padding=True, return_tensors="pt")
        return inputs.to(self.model.device)

    def _video_text(self, prompt: str, system: Optional[str] = None) -> str:
        return self._chat([{"type": "video", "video": "frames"}, {"type": "text", "text": prompt}], system)

    def _decode(self, out: Any, start: int, box: Dict[str, int]) -> List[str]:
        rows = [out[i, start:] for i in range(out.shape[0])]
        box["out_tokens"] = int(sum(int((r != self.tokenizer.pad_token_id).sum()) for r in rows))
        return [self.tokenizer.decode(r, skip_special_tokens=True).strip() for r in rows]

    def _run_generate(self, inputs: Any, gen: GenSpec, kind: str) -> List[str]:
        start = inputs["input_ids"].shape[1]
        ban = [self.tokenizer.encode(gen.ban, add_special_tokens=False)[0]] if gen.ban else None
        with self.meter.track(kind, start) as box, self.torch.inference_mode():
            return self._decode(_hf_generate(self.model, inputs, gen, ban), start, box)

    def extract_units(self, frames, times, fps, game_text):
        inputs = self._video_inputs(frames, fps, self._video_text(EXTRACT_USER, EXTRACT_SYSTEM.format(**game_text)))
        layers = find_decoder_layers(self.model)
        with self.meter.track("prefill", inputs["input_ids"].shape[1]), self.torch.inference_mode(), \
                self.adapter_disabled(), LayerTap(layers, self.layer_idx) as tap:
            self.model(**inputs, use_cache=False)
        ids, grid = inputs["input_ids"][0], inputs["video_grid_thw"][0].tolist()
        pooled = [qwen_unit_pool(tap.cache[i][0], ids, grid, self.video_token_id, self.merge) for i in self.layer_idx]
        feats = self.torch.cat(pooled, dim=1).cpu().numpy().astype(np.float32)
        ctx = rms_unit(tap.cache[self.layer_idx[-1]][0][-1])
        return feats, unit_time_table(times, feats.shape[0], self.unit_frames, fps), ctx

    def lm_loss(self, frames, fps, prompt, answer):
        inputs = self._video_inputs(frames, fps, self._video_text(prompt) + answer + QWEN_END)
        ids = inputs["input_ids"]
        answer_ids = self.tokenizer(answer + QWEN_END, add_special_tokens=False).input_ids
        n = len(answer_ids)
        if ids[0, -n:].tolist() != answer_ids:
            # Tokenisation did not split cleanly at the turn boundary.
            n = ids.shape[1] - self._video_inputs(frames, fps, self._video_text(prompt))["input_ids"].shape[1]
        labels = self.torch.full_like(ids, -100)
        labels[0, -n:] = ids[0, -n:]
        with self.meter.track("train_step", ids.shape[1]):
            return self.model(**inputs, labels=labels, use_cache=False).loss

    def generate_video(self, frames, fps, prompt, gen):
        return self._run_generate(self._video_inputs(frames, fps, self._video_text(prompt)), gen, "generate_video")

    def first_token_prob(self, frames, fps, prompt, word):
        inputs = self._video_inputs(frames, fps, self._video_text(prompt))
        target = self.tokenizer.encode(word, add_special_tokens=False)[0]
        with self.meter.track("prefill", inputs["input_ids"].shape[1]), self.torch.inference_mode():
            logits = self.model(**inputs, use_cache=False).logits[0, -1].float()
        return float(self.torch.softmax(logits, dim=-1)[target].item())

    def generate_image(self, images, prompt, gen, system=None):
        n_frames = GLIDE_WINDOW
        content = [{"type": "image", "image": "grid"} for _ in images] + [{"type": "text", "text": prompt}]
        # A 2x4 grid carries 8 frames: give it 8x the per-frame budget so each
        # frame is seen at the SAME resolution as in the video systems.
        with self._budget(self.spec.min_pixels, self.spec.max_pixels * n_frames):
            inputs = self.processor(text=[self._chat(content, system)], images=list(images), padding=True,
                                    return_tensors="pt").to(self.model.device)
            return self._run_generate(inputs, gen, "generate_image")[0]

    def generate_text(self, prompt, gen, system=None):
        text = self._chat([{"type": "text", "text": prompt}], system)
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.model.device)
        return self._run_generate(inputs, gen, "generate_text")[0]

    def attach_lora(self, r, alpha):
        self.base_model.gradient_checkpointing_enable()
        self.base_model.enable_input_require_grads()
        self.model = self._wrap(self.base_model, self.LORA_TARGETS, r, alpha)

    def load_adapter(self, path):
        self.model = self._load_peft(self.base_model, path).eval()

    def train_mode(self, on):
        self.model.train(on)


class InternVLFamily(Backbone):
    """InternVL2.5-4B / 8B. Requires --trust-remote-code (explicit opt-in:
    the checkpoint ships its own Python). One 448x448 tile per video frame;
    GliDe grid images use InternVL's dynamic tiling (<= 8 tiles + thumbnail)."""

    def __init__(self, spec: BackboneSpec) -> None:
        super().__init__()
        if not spec.trust_remote_code:
            raise PermissionError(f"{spec.model_id} executes repository code; pass --trust-remote-code "
                                  "and pin --revision to a reviewed commit")
        if spec.revision is None:
            raise ValueError("InternVL requires a pinned --revision when remote code is trusted")
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.spec, self.name = torch, spec, spec.model_id
        self.dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[spec.dtype]
        self.model = AutoModel.from_pretrained(
            spec.model_id, revision=spec.revision, torch_dtype=self.dtype, low_cpu_mem_usage=True,
            trust_remote_code=True, device_map="auto", use_flash_attn=spec.attn == "flash_attention_2").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision,
                                                       trust_remote_code=True, use_fast=False)
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.num_image_token = int(self.model.num_image_token)
        self._conv = importlib.import_module(type(self.model).__module__).get_conv_template
        self._base_lm = self.model.language_model
        self.hidden_size = int(self._base_lm.config.hidden_size)
        self.layer_idx = resolve_layers(len(find_decoder_layers(self._base_lm)), spec.layer_fracs)
        self.unit_frames = 1

    @property
    def device(self) -> Any:
        return next(self.model.vision_model.parameters()).device

    def _to_tensor(self, image: Any) -> Any:
        arr = self.torch.from_numpy(np.asarray(image.convert("RGB").resize((448, 448)), dtype=np.float32) / 255.0)
        mean = self.torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = self.torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (arr.permute(2, 0, 1) - mean) / std

    def _frame_pixels(self, frames: Sequence[str]) -> Any:
        from PIL import Image

        return self.torch.stack([self._to_tensor(Image.fromarray(load_rgb(p))) for p in frames]).to(self.device, self.dtype)

    def _tiles(self, image: Any, max_tiles: int = 8) -> Any:
        """InternVL dynamic tiling: the (cols, rows) grid closest in aspect
        ratio with cols*rows <= max_tiles, plus a global thumbnail."""
        w, h = image.size
        grids = [(c, r) for c in range(1, max_tiles + 1) for r in range(1, max_tiles + 1) if c * r <= max_tiles]
        cols, rows = min(grids, key=lambda g: (abs(g[0] / g[1] - w / h), -g[0] * g[1]))
        big = image.resize((448 * cols, 448 * rows))
        tiles = [big.crop((c * 448, r * 448, (c + 1) * 448, (r + 1) * 448)) for r in range(rows) for c in range(cols)]
        if len(tiles) > 1:
            tiles.append(image.resize((448, 448)))
        return self.torch.stack([self._to_tensor(t) for t in tiles]).to(self.device, self.dtype)

    def _query(self, question: str, patches: Sequence[int], system: Optional[str] = None) -> str:
        template = self._conv(self.model.template)
        template.system_message = system or self.model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        for n in patches:
            query = query.replace("<image>", "<img>" + "<IMG_CONTEXT>" * self.num_image_token * n + "</img>", 1)
        return query

    @staticmethod
    def _frame_block(times: Sequence[float]) -> str:
        return "".join(f"Frame{i + 1} ({t:.1f}s): <image>\n" for i, t in enumerate(times))

    def _embeds(self, ids: Any, pixels: Optional[Any]) -> Any:
        emb = self.model.language_model.get_input_embeddings()(ids).clone()
        if pixels is not None:
            vit = self.model.extract_feature(pixels)
            emb[ids == self.model.img_context_token_id] = vit.reshape(-1, vit.shape[-1]).to(emb.dtype)
        return emb

    def _ids(self, text: str) -> Any:
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def _sep(self) -> str:
        return str(getattr(self._conv(self.model.template), "sep", QWEN_END)).strip() or QWEN_END

    def _video_query(self, frames: Sequence[str], fps: float, prompt: str, system: Optional[str] = None) -> str:
        times = [i / max(fps, 1e-3) for i in range(len(frames))]
        return self._query(self._frame_block(times) + prompt, [1] * len(frames), system)

    def _generate_embeds(self, ids: Any, pixels: Optional[Any], gen: GenSpec, kind: str) -> List[str]:
        ban = [self.tokenizer.encode(gen.ban, add_special_tokens=False)[0]] if gen.ban else None
        with self.meter.track(kind, ids.shape[1]) as box, self.torch.inference_mode():
            emb = self._embeds(ids, pixels)
            inputs = {"inputs_embeds": emb, "attention_mask": self.torch.ones_like(ids),
                      "eos_token_id": self.tokenizer.convert_tokens_to_ids(self._sep())}
            out = _hf_generate(self.model.language_model, inputs, gen, ban)
            box["out_tokens"] = int(out.numel())
            return [self.tokenizer.decode(r, skip_special_tokens=True).strip() for r in out]

    def extract_units(self, frames, times, fps, game_text):
        query = self._query(self._frame_block(times) + EXTRACT_USER, [1] * len(frames), EXTRACT_SYSTEM.format(**game_text))
        ids = self._ids(query)
        layers = find_decoder_layers(self.model.language_model)
        with self.meter.track("prefill", ids.shape[1]), self.torch.inference_mode(), self.adapter_disabled():
            emb = self._embeds(ids, self._frame_pixels(frames))
            with LayerTap(layers, self.layer_idx) as tap:
                self.model.language_model(inputs_embeds=emb, attention_mask=self.torch.ones_like(ids), use_cache=False)
        sel = ids[0] == self.model.img_context_token_id
        pooled = [pool_tokens(tap.cache[i][0][sel].view(len(frames), self.num_image_token, -1)) for i in self.layer_idx]
        feats = self.torch.cat(pooled, dim=1).cpu().numpy().astype(np.float32)
        return feats, unit_time_table(times, len(frames), 1, fps), rms_unit(tap.cache[self.layer_idx[-1]][0][-1])

    def lm_loss(self, frames, fps, prompt, answer):
        prompt_ids = self._ids(self._video_query(frames, fps, prompt))
        ans = self.tokenizer(answer + self._sep(), add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        ids = self.torch.cat([prompt_ids, ans], dim=1)
        labels = self.torch.full_like(ids, -100)
        labels[:, prompt_ids.shape[1]:] = ans
        with self.meter.track("train_step", ids.shape[1]):
            emb = self._embeds(ids, self._frame_pixels(frames))
            return self.model.language_model(inputs_embeds=emb, attention_mask=self.torch.ones_like(ids),
                                             labels=labels, use_cache=False).loss

    def generate_video(self, frames, fps, prompt, gen):
        return self._generate_embeds(self._ids(self._video_query(frames, fps, prompt)), self._frame_pixels(frames),
                                     gen, "generate_video")

    def first_token_prob(self, frames, fps, prompt, word):
        ids = self._ids(self._video_query(frames, fps, prompt))
        target = self.tokenizer.encode(word, add_special_tokens=False)[0]
        with self.meter.track("prefill", ids.shape[1]), self.torch.inference_mode():
            emb = self._embeds(ids, self._frame_pixels(frames))
            logits = self.model.language_model(inputs_embeds=emb, attention_mask=self.torch.ones_like(ids),
                                               use_cache=False).logits[0, -1].float()
        return float(self.torch.softmax(logits, dim=-1)[target].item())

    def generate_image(self, images, prompt, gen, system=None):
        tiles = [self._tiles(im) for im in images]
        query = self._query("<image>\n" * len(images) + prompt, [t.shape[0] for t in tiles], system)
        return self._generate_embeds(self._ids(query), self.torch.cat(tiles), gen, "generate_image")[0]

    def generate_text(self, prompt, gen, system=None):
        return self._generate_embeds(self._ids(self._query(prompt, [], system)), None, gen, "generate_text")[0]

    def _lora_targets(self) -> List[str]:
        names = {n.rsplit(".", 1)[-1] for n, _ in self._base_lm.named_modules()}
        for family in (["wqkv", "wo", "w1", "w2", "w3"],
                       ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
            if all(f in names for f in family):
                return family
        raise RuntimeError("unrecognised InternVL language model; cannot place LoRA")

    def attach_lora(self, r, alpha):
        self._base_lm.gradient_checkpointing_enable()
        self._base_lm.enable_input_require_grads()
        self.model.language_model = self._wrap(self._base_lm, self._lora_targets(), r, alpha)

    def load_adapter(self, path):
        self.model.language_model = self._load_peft(self._base_lm, path)

    def train_mode(self, on):
        self.model.language_model.train(on)


# --------------------------------------------------------------------------- #
# CPU mock: same contract, rule-based answers; numbers obtained with it say
# nothing about any real system
# --------------------------------------------------------------------------- #


MOCK_TELEPORT = "The red square teleports erratically across the screen."


def _change_fraction(a: np.ndarray, b: np.ndarray) -> float:
    return float((np.abs(a - b).max(axis=-1) > 0.25).mean())


class MockBackbone(Backbone):
    name = "mock"
    hidden_size = 48
    unit_frames = 2

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        import torch

        self.torch = torch
        rng = make_rng(seed)
        self.w1 = rng.normal(0, 1.0 / math.sqrt(48), size=(48, 48)).astype(np.float32)
        self.w2 = rng.normal(0, 1.0 / math.sqrt(48), size=(48, 48)).astype(np.float32)
        self._lora = torch.nn.Parameter(torch.zeros(4))
        self.layer_idx = [0, 1]

    @staticmethod
    def _small(path_or_image: Any, size: int = 32) -> np.ndarray:
        from PIL import Image

        image = path_or_image if hasattr(path_or_image, "resize") else Image.fromarray(load_rgb(path_or_image))
        return np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0

    def _jumps(self, frames: Sequence[Any]) -> List[float]:
        small = [self._small(f) for f in frames]
        return [_change_fraction(a, b) for a, b in itertools.pairwise(small)]

    def _tokens(self, path: str) -> np.ndarray:
        arr = self._small(path, 8)
        return np.stack([q.reshape(-1) for q in (arr[:4, :4], arr[:4, 4:], arr[4:, :4], arr[4:, 4:])])

    def extract_units(self, frames, times, fps, game_text):
        with self.meter.track("prefill", 8 * len(frames)):
            paths = list(frames) + [frames[-1]] * (len(frames) % 2)
            toks = np.stack([self._tokens(p) for p in paths])
            h1 = np.tanh(toks @ self.w1)
            h2 = np.tanh(h1 @ self.w2)
            n_units = len(paths) // 2
            feats = self.torch.cat([pool_tokens(self.torch.from_numpy(h.reshape(n_units, 8, -1))) for h in (h1, h2)], dim=1)
            return (feats.numpy(), unit_time_table(times, n_units, 2, fps), h2.mean(axis=(0, 1)).astype(np.float32))

    def lm_loss(self, frames, fps, prompt, answer):
        with self.meter.track("train_step", len(prompt)):
            return (self._lora.sum() - self.torch.tensor(len(answer) / 100.0)) ** 2

    def _glitchy(self, frames: Sequence[Any]) -> bool:
        return max(self._jumps(frames) or [0.0]) > 0.03

    def _report(self, frames: Sequence[str], fps: float) -> str:
        jumps = self._jumps(frames)
        hot = [i for i, j in enumerate(jumps) if j > 0.03]
        if not hot:
            return json.dumps({"no_bugs": True, "bugs": [], "time_nodes": []})
        a, b = hot[0] / fps, (hot[-1] + 2) / fps
        return json.dumps({"no_bugs": False, "bugs": [MOCK_TELEPORT],
                           "time_nodes": [[[round(a, 2), round(b, 2)]]]})

    def generate_video(self, frames, fps, prompt, gen):
        with self.meter.track("generate_video", len(prompt)) as box:
            if '"time_nodes"' in prompt:
                text = self._report(frames, fps)
            else:
                text = ("The red square teleports abruptly across the screen."
                        if gen.ban or self._glitchy(frames) else NO_GLITCH)
            box["out_tokens"] = len(text.split()) * max(1, gen.n)
            return [text] * max(1, gen.n if gen.temperature > 0 else 1)

    def first_token_prob(self, frames, fps, prompt, word):
        with self.meter.track("prefill", len(prompt)):
            return float(1.0 / (1.0 + math.exp(80.0 * (max(self._jumps(frames) or [0.0]) - 0.03))))

    def _grid_tiles(self, image: Any) -> List[Any]:
        w, h = image.size
        tw, th = w // 4, h // 2
        return [image.crop((c * tw, r * th, (c + 1) * tw, (r + 1) * th)) for r in range(2) for c in range(4)]

    def generate_image(self, images, prompt, gen, system=None):
        with self.meter.track("generate_image", len(prompt)) as box:
            tiles = self._grid_tiles(images[0])
            jumps = self._jumps(tiles)
            glitch = max(jumps or [0.0]) > 0.03
            if '"verdict"' in prompt:
                text = self._text_answer(prompt)
            elif "has_glitch" in prompt:
                hot = [i for i, j in enumerate(jumps) if j > 0.03] or [0]
                text = json.dumps({"has_glitch": glitch, "category": "Physics" if glitch else "Other",
                                   "visual_cues": "the red square teleports" if glitch else "smooth motion",
                                   "reasoning": "sudden displacement" if glitch else "nothing abnormal",
                                   "frame_range": [hot[0], min(7, hot[-1] + 1)], "confidence": 0.9 if glitch else 0.2,
                                   "game_context": "A 2D scene where a red square moves steadily to the right."})
            else:
                text = "Yes: the red square jumps to a distant position." if glitch else "No: motion is smooth."
            box["out_tokens"] = len(text.split())
            return text

    @staticmethod
    def _mentions_glitch(text: str) -> bool:
        return any(w in text for w in ("jump", "teleport", "VIOLATION"))

    @staticmethod
    def _answer_plan(_prompt: str) -> str:
        return json.dumps({"tool": "vqa", "question": "Does the red square move discontinuously?", "bbox": None})

    @classmethod
    def _answer_verdict(cls, prompt: str) -> str:
        glitch = cls._mentions_glitch(prompt)
        return json.dumps({"verdict": "glitch" if glitch else "normal", "confidence": 0.9 if glitch else 0.85,
                           "description": MOCK_TELEPORT})

    @classmethod
    def _answer_same(cls, prompt: str) -> str:
        a, _, b = prompt.partition("Observation B:")
        return "yes" if cls._mentions_glitch(a) and cls._mentions_glitch(b.split("Do A and B")[0]) else "no"

    @staticmethod
    def _answer_summary(prompt: str) -> str:
        groups = json.loads(prompt.split("GROUPS_JSON:", 1)[1].strip())
        return json.dumps({"no_bugs": not groups, "bugs": [g["description"] for g in groups],
                           "time_nodes": [g["intervals"] for g in groups]})

    @classmethod
    def _text_answer(cls, prompt: str) -> str:
        routes = (("Return ONE JSON object with the next tool", cls._answer_plan), ('"verdict"', cls._answer_verdict),
                  ("same underlying glitch", cls._answer_same), ("GROUPS_JSON:", cls._answer_summary))
        handler = next((h for marker, h in routes if marker in prompt), None)
        return handler(prompt) if handler else "The square moving smoothly is normal; sudden jumps would be a glitch."

    def generate_text(self, prompt, gen, system=None):
        with self.meter.track("generate_text", len(prompt)) as box:
            text = self._text_answer(prompt)
            box["out_tokens"] = len(text.split())
            return text

    def attach_lora(self, r, alpha):
        # The mock's only trainable tensor (`_lora`) exists from construction;
        # there are no frozen weights to wrap, so attaching is a no-op.
        LOGGER.debug("mock backbone: LoRA r=%d alpha=%d requested (no-op)", r, alpha)

    def save_adapter(self, path):
        target = safe_path(path)
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / "mock_adapter.npy", self._lora.detach().numpy())

    def load_adapter(self, path):
        self._lora.data.copy_(self.torch.from_numpy(np.load(safe_path(Path(path) / "mock_adapter.npy"))))

    def trainable_parameters(self):
        return [self._lora]

    def train_mode(self, on):
        # No dropout or normalisation layers: train and eval behave identically.
        LOGGER.debug("mock backbone train_mode(%s)", on)


def build_backbone(spec: BackboneSpec) -> Backbone:
    key = spec.model_id.lower()
    if key == "mock":
        return MockBackbone()
    if "internvl" in key:
        return InternVLFamily(spec)
    return QwenFamily(spec)


# =========================================================================== #
# 4. feature cache (one early-exit prefill per chunk)
# =========================================================================== #


class Projector:
    """Seeded random orthogonal map D -> P. Johnson-Lindenstrauss: pairwise
    distances are preserved up to (1 +- eps) w.h.p. for P = O(eps^-2 log n),
    which is what the kNN normality bank relies on."""

    def __init__(self, dim_in: int, dim_out: int, seed: int) -> None:
        self.dim_in, self.dim_out = int(dim_in), min(int(dim_out), int(dim_in))
        self.matrix: Optional[np.ndarray] = None
        if self.dim_out < self.dim_in:
            q, _ = np.linalg.qr(make_rng(seed).standard_normal((self.dim_in, self.dim_out)))
            self.matrix = (q * math.sqrt(self.dim_in / self.dim_out)).astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x if self.matrix is None else x @ self.matrix


@dataclasses.dataclass
class ExtractSpec:
    feat_dir: str
    fps: float = DEFAULT_FPS
    chunk_frames: int = 32
    proj_dim: int = 1024
    seed: int = 1337
    overwrite: bool = False


def feature_path(feat_dir: Any, item: VideoItem) -> Path:
    return safe_path(Path(feat_dir) / f"{item.stem}.npz")


def subsample(item: VideoItem, fps: float) -> Tuple[List[str], List[float], float]:
    stride = max(1, round(item.fps / max(fps, 1e-3)))
    return item.frame_paths[::stride], item.timestamps[::stride], item.fps / stride


def _check_meta(feat_dir: Path, meta: Dict[str, Any], overwrite: bool) -> None:
    old = read_json(feat_dir / "meta.json", required=False)
    if old and not overwrite:
        for key in ("backbone", "revision", "proj_dim", "proj_seed", "fps", "chunk_frames", "max_pixels"):
            if old.get(key) != meta.get(key):
                raise RuntimeError(f"feature cache {feat_dir} has {key}={old.get(key)!r}, requested "
                                   f"{meta.get(key)!r}; use a new --feat-dir or --overwrite")
    write_json(feat_dir / "meta.json", meta)


def extract_video(item: VideoItem, backbone: Backbone, spec: ExtractSpec, projector: Projector
                  ) -> Optional[Dict[str, np.ndarray]]:
    frames, stamps, fps = subsample(item, spec.fps)
    unit = max(1, backbone.unit_frames)
    chunk = max(unit, (spec.chunk_frames // unit) * unit)
    feats, ctxs, times = [], [], []
    for s in range(0, len(frames), chunk):
        u_feat, u_time, ctx = backbone.extract_units(frames[s:s + chunk], stamps[s:s + chunk], fps, item.game_text())
        feats.append(projector(u_feat.reshape(-1, u_feat.shape[-1])).reshape(u_feat.shape[0], u_feat.shape[1], -1))
        ctxs.append(np.repeat(projector(ctx[None])[0][None], u_feat.shape[0], axis=0))
        times.append(u_time)
    if not feats:
        return None
    return {"feats": np.concatenate(feats).astype(np.float16), "ctx": np.concatenate(ctxs).astype(np.float16),
            "times": np.concatenate(times).astype(np.float32)}


def extract_corpus(items: Dict[str, VideoItem], backbone: Backbone, spec: ExtractSpec,
                   names: Sequence[str], fingerprint: Dict[str, Any]) -> Dict[str, Any]:
    feat_dir = safe_path(spec.feat_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)
    projector = Projector(backbone.hidden_size, spec.proj_dim, spec.seed)
    meta = {**fingerprint, "backbone": backbone.name, "proj_dim": projector.dim_out, "proj_seed": spec.seed,
            "fps": spec.fps, "chunk_frames": spec.chunk_frames, "layers": getattr(backbone, "layer_idx", None)}
    _check_meta(feat_dir, meta, spec.overwrite)
    before, t0, done = backbone.meter.snapshot(), time.perf_counter(), 0
    for i, name in enumerate(names, 1):
        path = feature_path(feat_dir, items[name])
        if path.exists() and not spec.overwrite:
            continue
        try:
            payload = extract_video(items[name], backbone, spec, projector)
        except (RuntimeError, ValueError, OSError) as exc:
            LOGGER.error("extraction failed for %s: %r", name, exc)
            continue
        if payload is not None:
            np.savez_compressed(path, **payload)
            done += 1
        if i % 100 == 0:
            LOGGER.info("extracted %d/%d videos", i, len(names))
    stats = {"videos": done, "seconds": time.perf_counter() - t0, "peak_gpu_mb": peak_gpu_mb(),
             **CostMeter.delta(backbone.meter.snapshot(), before)}
    write_json(feat_dir / "extract_stats.json", stats)
    LOGGER.info("feature extraction: %s", stats)
    return stats


def load_features(feat_dir: Any, items: Dict[str, VideoItem], names: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for name in names:
        path = feature_path(feat_dir, items[name])
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                out[name] = {k: data[k] for k in ("feats", "ctx", "times")}
    if len(out) < len(names):
        LOGGER.warning("%d/%d requested videos have no cached features", len(names) - len(out), len(names))
    return out


# =========================================================================== #
# 5. NORMA head: self-referential normality + temporal transformer
# =========================================================================== #


class NormalityBank:
    """Whitened PCA + greedy k-center coreset of NORMAL training units.
    Training videos query it with their own game excluded, so the train-time
    distribution of the kNN distance matches the unseen-game test setting."""

    def __init__(self, pca_dim: int = 128, size: int = 4096, k: int = 3, seed: int = 0) -> None:
        self.pca_dim, self.size, self.k, self.seed = pca_dim, size, k, seed
        self.mean = np.zeros(0, dtype=np.float32)
        self.components = np.zeros((0, 0), dtype=np.float32)
        self.bank = np.zeros((0, 0), dtype=np.float32)
        self.bank_game = np.zeros(0, dtype=str)

    @staticmethod
    def _flat(feats: np.ndarray) -> np.ndarray:
        return feats.reshape(feats.shape[0], -1).astype(np.float32)

    def _sample(self, normal_units: Sequence[Tuple[np.ndarray, str]], max_fit: int, rng: np.random.Generator
                ) -> Tuple[np.ndarray, np.ndarray]:
        # Subsample per video BEFORE concatenating (the VGB train pool is
        # ~1e5 units x 4096 dims per fold).
        per_video = max(4, (2 * max_fit) // max(len(normal_units), 1))
        xs, gs = [], []
        for feats, game in normal_units:
            idx = rng.choice(feats.shape[0], min(per_video, feats.shape[0]), replace=False)
            xs.append(self._flat(feats[idx]))
            gs.append(np.full(len(idx), game))
        x, g = np.concatenate(xs), np.concatenate(gs)
        if x.shape[0] > max_fit:
            pick = rng.choice(x.shape[0], max_fit, replace=False)
            x, g = x[pick], g[pick]
        return x, g

    def fit(self, normal_units: Sequence[Tuple[np.ndarray, str]], max_fit: int = 60000) -> None:
        import torch

        rng = make_rng(self.seed)
        x, games = self._sample(normal_units, max_fit, rng)
        xt = torch.from_numpy(x)
        self.mean = xt.mean(dim=0).numpy()
        q = min(self.pca_dim, xt.shape[1], xt.shape[0] - 1)
        _, s, v = torch.pca_lowrank(xt - xt.mean(dim=0), q=q, center=False)
        scale = (s / math.sqrt(max(xt.shape[0] - 1, 1))).clamp_min(1e-6)
        self.components = (v / scale).numpy().astype(np.float32)
        z = self.transform(x)
        keep = self._coreset(z, rng)
        self.bank, self.bank_game = z[keep].astype(np.float32), games[keep]
        LOGGER.info("normality bank: %d normal units -> PCA %d -> coreset %d", x.shape[0],
                    self.components.shape[1], self.bank.shape[0])

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) @ self.components

    def _coreset(self, z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Greedy k-center (Sener & Savarese 2018): a 2-approximation of the
        minimax covering radius, the quantity that bounds kNN error."""
        import torch

        if z.shape[0] <= self.size:
            return np.arange(z.shape[0])
        zt = torch.from_numpy(z)
        chosen = [int(rng.integers(z.shape[0]))]
        dist = torch.cdist(zt, zt[chosen]).squeeze(1)
        for _ in range(self.size - 1):
            nxt = int(torch.argmax(dist))
            chosen.append(nxt)
            dist = torch.minimum(dist, torch.cdist(zt, zt[nxt:nxt + 1]).squeeze(1))
        return np.array(chosen)

    def distance(self, feats: np.ndarray, exclude_game: Optional[str] = None) -> np.ndarray:
        import torch

        z = torch.from_numpy(self.transform(self._flat(feats)))
        bank = torch.from_numpy(self.bank)
        if exclude_game is not None:
            keep = torch.from_numpy(self.bank_game != exclude_game)
            if int(keep.sum()) >= self.k:
                bank = bank[keep]
        k = min(self.k, bank.shape[0])
        knn = torch.topk(torch.cdist(z, bank), k, dim=1, largest=False).values.mean(dim=1) / math.sqrt(z.shape[1])
        return torch.log1p(knn).numpy().astype(np.float32)

    def save(self, path: Any) -> None:
        np.savez(safe_path(path), mean=self.mean, components=self.components, bank=self.bank,
                 bank_game=self.bank_game.astype(str), meta=np.array([self.pca_dim, self.size, self.k, self.seed]))

    @classmethod
    def load(cls, path: Any) -> "NormalityBank":
        with np.load(safe_path(path), allow_pickle=False) as data:
            meta = [int(v) for v in data["meta"]]
            bank = cls(*meta)
            bank.mean, bank.components, bank.bank = data["mean"], data["components"], data["bank"]
            bank.bank_game = data["bank_game"]
        return bank


@dataclasses.dataclass
class HeadConfig:
    n_vectors: int = 4
    in_dim: int = 1024
    d_model: int = 384
    n_layers: int = 3
    n_heads: int = 6
    dropout: float = 0.1
    emb_dim: int = 8
    k_max: int = 4
    srn_clip: float = 6.0
    use_srn: bool = True
    use_bank: bool = True
    use_delta: bool = True
    use_ctx: bool = True
    focal_alpha: float = 0.5
    focal_gamma: float = 2.0
    w_boundary: float = 1.0
    w_count: float = 0.5
    w_embed: float = 0.5
    embed_margin: float = 1.0


def build_head(cfg: HeadConfig) -> Any:
    import torch
    from torch import nn

    class SinusoidalTime(nn.Module):
        """Absolute seconds (periods 0.5 s .. 128 s): the head is agnostic to the
        backbone's unit duration."""

        def __init__(self, d: int) -> None:
            super().__init__()
            periods = torch.logspace(math.log10(0.5), math.log10(128.0), d // 2)
            self.register_buffer("omega", 2.0 * math.pi / periods)
            self.d = d

        def forward(self, sec: Any) -> Any:
            ang = sec.unsqueeze(-1) * self.omega
            pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
            return torch.nn.functional.pad(pe, (0, self.d - pe.shape[-1]))

    class NormaHead(nn.Module):
        def __init__(self, c: HeadConfig) -> None:
            super().__init__()
            self.c = c
            flat, d = c.n_vectors * c.in_dim, c.d_model
            self.in_norm = nn.LayerNorm(flat)
            self.proj = nn.Sequential(nn.Linear(flat, d), nn.LayerNorm(d))
            self.srn_proj = nn.Linear(d, d)
            self.delta_proj = nn.Linear(d, d)
            self.ctx_proj = nn.Sequential(nn.LayerNorm(c.in_dim), nn.Linear(c.in_dim, d))
            self.scalar_mlp = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
            self.time = SinusoidalTime(d)
            self.in_drop = nn.Dropout(c.dropout)
            layer = nn.TransformerEncoderLayer(d, c.n_heads, 4 * d, c.dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, c.n_layers, enable_nested_tensor=False)
            self.out_norm = nn.LayerNorm(d)
            self.act = nn.Linear(d, 1)
            self.bound = nn.Linear(d, 2)
            self.emb = nn.Linear(d, c.emb_dim)
            self.pool = nn.Linear(d, 1)
            self.count = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, c.k_max + 1))

        def srn(self, u: Any, mask: Any) -> Any:
            """Masked coordinate-wise median / MAD over time, per video
            (50 % breakdown point; Hampel 1974, Rousseeuw & Croux 1993)."""
            r = torch.zeros_like(u)
            for b in range(u.shape[0]):
                valid = u[b, mask[b]]
                med = valid.median(dim=0).values
                mad = (valid - med).abs().median(dim=0).values * 1.4826 + 1e-2
                r[b] = ((u[b] - med) / mad).clamp(-self.c.srn_clip, self.c.srn_clip)
            return r * mask.unsqueeze(-1)

        def _inputs(self, u: Any, ctx: Any, bank_dist: Any, times: Any, mask: Any) -> Any:
            c = self.c
            zeros = torch.zeros(u.shape[:2], device=u.device)
            z, e = u, zeros
            if c.use_srn:
                r = self.srn(u, mask)
                # sqrt(x + eps): d/dx sqrt(x) is unbounded at 0 (padded units).
                e = r.pow(2).mean(dim=-1).add(1e-6).sqrt()
                z = z + self.srn_proj(r)
            delta = torch.zeros_like(u)
            delta[:, 1:] = u[:, 1:] - u[:, :-1]
            delta = delta * mask.unsqueeze(-1)
            g = zeros
            if c.use_delta:
                g = delta.pow(2).sum(dim=-1).add(1e-6).sqrt() / math.sqrt(u.shape[-1])
                z = z + self.delta_proj(delta)
            if c.use_ctx:
                z = z + self.ctx_proj(ctx)
            n = bank_dist if c.use_bank else zeros
            centre = times.mean(dim=-1)
            length = (centre * mask).amax(dim=1, keepdim=True).clamp_min(1e-3)
            return z + self.scalar_mlp(torch.stack([e, g, n, centre / length], dim=-1)) + self.time(centre)

        def forward(self, feats: Any, ctx: Any, bank_dist: Any, times: Any, mask: Any) -> Dict[str, Any]:
            u = self.proj(self.in_norm(feats.reshape(feats.shape[0], feats.shape[1], -1)))
            h = self.encoder(self.in_drop(self._inputs(u, ctx, bank_dist, times, mask)), src_key_padding_mask=~mask)
            h = self.out_norm(h)
            weights = torch.softmax(self.pool(h).squeeze(-1).masked_fill(~mask, -1e4), dim=1)
            pooled = (weights.unsqueeze(-1) * h).sum(dim=1)
            bnd = self.bound(h)
            return {"act": self.act(h).squeeze(-1), "start": bnd[..., 0], "end": bnd[..., 1],
                    "emb": torch.nn.functional.normalize(self.emb(h), dim=-1), "count": self.count(pooled)}

    return NormaHead(cfg)


SAMPLE_KEYS = ("feats", "ctx", "times", "bank", "y", "start", "end", "inst")


def collate(samples: Sequence[Dict[str, Any]], device: Any) -> Dict[str, Any]:
    import torch

    B, T = len(samples), max(s["feats"].shape[0] for s in samples)
    V, P = samples[0]["feats"].shape[1:]
    out = {"feats": torch.zeros(B, T, V, P), "ctx": torch.zeros(B, T, samples[0]["ctx"].shape[-1]),
           "times": torch.zeros(B, T, 2), "bank_dist": torch.zeros(B, T), "mask": torch.zeros(B, T, dtype=torch.bool),
           "y": torch.zeros(B, T), "start": torch.zeros(B, T), "end": torch.zeros(B, T),
           "inst": torch.full((B, T), -1, dtype=torch.long), "count": torch.zeros(B, dtype=torch.long)}
    for b, s in enumerate(samples):
        n = s["feats"].shape[0]
        out["feats"][b, :n] = torch.from_numpy(s["feats"].astype(np.float32))
        out["ctx"][b, :n] = torch.from_numpy(s["ctx"].astype(np.float32))
        out["times"][b, :n] = torch.from_numpy(s["times"])
        out["times"][b, n:] = out["times"][b, n - 1]
        out["bank_dist"][b, :n] = torch.from_numpy(s["bank"])
        out["mask"][b, :n] = True
        for key in ("y", "start", "end", "inst"):
            out[key][b, :n] = torch.from_numpy(s[key])
        out["count"][b] = int(s["count"])
    return {k: v.to(device) for k, v in out.items()}


def augment(sample: Dict[str, Any], rng: np.random.Generator, k_max: int) -> Dict[str, Any]:
    """Random temporal crop (>= 60 % of the clip) + multiplicative feature
    noise; the count target is recomputed from the surviving instances."""
    s = dict(sample)
    n = s["feats"].shape[0]
    if n >= 8 and rng.random() < 0.5:
        length = int(rng.integers(int(0.6 * n), n + 1))
        a = int(rng.integers(0, n - length + 1))
        for key in SAMPLE_KEYS:
            s[key] = s[key][a:a + length]
        s["count"] = min(k_max, len({int(i) for i in s["inst"] if i >= 0}) or int(s["y"].any()))
    s["feats"] = s["feats"].astype(np.float32) * (1.0 + rng.normal(0.0, 0.05, size=s["feats"].shape)).astype(np.float32)
    return s


def _focal(logit: Any, y: Any, mask: Any, cfg: HeadConfig) -> Any:
    """Sigmoid focal loss (Lin et al. 2017)."""
    import torch.nn.functional as F

    p = logit.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    p_t = p * y + (1 - p) * (1 - y)
    alpha_t = cfg.focal_alpha * y + (1 - cfg.focal_alpha) * (1 - y)
    return (alpha_t * (1 - p_t).pow(cfg.focal_gamma) * ce * mask).sum() / mask.sum().clamp_min(1.0)


def _boundary(out: Dict[str, Any], batch: Dict[str, Any], mask: Any) -> Any:
    import torch.nn.functional as F

    total = 0.0
    for key in ("start", "end"):
        weight = (1.0 + 4.0 * batch[key]) * mask
        bce = F.binary_cross_entropy_with_logits(out[key], batch[key], reduction="none")
        total = total + (bce * weight).sum() / weight.sum().clamp_min(1.0)
    return total


def _embedding(emb: Any, inst: Any, margin: float) -> Any:
    """Discriminative instance loss (De Brabandere et al. 2017): pull units to
    their event mean, push event means at least `margin` apart."""
    import torch
    import torch.nn.functional as F

    loss, terms = emb.new_zeros(()), 0
    for b in range(emb.shape[0]):
        ids = sorted({int(i) for i in inst[b].tolist() if i >= 0})
        means = []
        for i in ids:
            v = emb[b][inst[b] == i]
            mu = v.mean(dim=0)
            loss = loss + (v - mu).pow(2).sum(dim=-1).mean()
            means.append(mu)
            terms += 1
        if len(means) > 1:
            mu = torch.stack(means)
            dist = (mu.unsqueeze(1) - mu.unsqueeze(0)).pow(2).sum(dim=-1).add(1e-9).sqrt()
            off = ~torch.eye(len(means), dtype=torch.bool, device=mu.device)
            loss = loss + F.relu(margin - dist[off]).pow(2).mean()
    return loss / max(terms, 1)


def head_loss(out: Dict[str, Any], batch: Dict[str, Any], cfg: HeadConfig) -> Tuple[Any, Dict[str, float]]:
    import torch.nn.functional as F

    mask = batch["mask"].float()
    parts = {"act": _focal(out["act"], batch["y"], mask, cfg), "bnd": _boundary(out, batch, mask),
             "cnt": F.cross_entropy(out["count"], batch["count"].clamp(max=cfg.k_max)),
             "emb": _embedding(out["emb"], batch["inst"], cfg.embed_margin)}
    total = parts["act"] + cfg.w_boundary * parts["bnd"] + cfg.w_count * parts["cnt"] + cfg.w_embed * parts["emb"]
    return total, {k: float(v.detach()) for k, v in parts.items()}


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Transitions:
    p11: float
    p01: float
    pi1: float
    prior: float


@dataclasses.dataclass
class DecodeParams:
    temperature: float = 1.0
    none_threshold: float = 0.5
    event_threshold: float = 0.4
    group_sim: float = 0.75
    boundary_radius: int = 2
    boundary_min: float = 0.3
    max_events: int = 4


def fit_transitions(label_seqs: Sequence[np.ndarray], smoothing: float = 1.0) -> Transitions:
    """Laplace-smoothed 2-state Markov chain on NON-overlapping units."""
    c = Counter({"00": smoothing, "01": smoothing, "10": smoothing, "11": smoothing})
    starts, total, pos, n = smoothing, 2.0 * smoothing, 0.0, 0.0
    for seq in label_seqs:
        s = (np.asarray(seq) > 0.5).astype(int)
        if s.size:
            starts, total, pos, n = starts + s[0], total + 1.0, pos + s.sum(), n + s.size
            c.update(f"{a}{b}" for a, b in itertools.pairwise(s.tolist()))

    def clip(v: float) -> float:
        return float(np.clip(v, 1e-4, 1 - 1e-4))

    return Transitions(p11=clip(c["11"] / (c["11"] + c["10"])), p01=clip(c["01"] / (c["01"] + c["00"])),
                       pi1=clip(starts / total), prior=clip(pos / max(n, 1.0)))


def viterbi(llr: np.ndarray, trans: Transitions) -> np.ndarray:
    """Two-state Viterbi; llr_t = log p(x_t|1)/p(x_t|0) (state 0 is the
    reference emission)."""
    n = len(llr)
    if n == 0:
        return np.zeros(0, dtype=int)
    lt = np.log(np.array([[1 - trans.p01, trans.p01], [1 - trans.p11, trans.p11]]))
    delta = np.zeros((n, 2))
    psi = np.zeros((n, 2), dtype=int)
    delta[0] = [math.log(1 - trans.pi1), math.log(trans.pi1) + llr[0]]
    for t in range(1, n):
        cand = delta[t - 1][:, None] + lt                     # [from, to]
        psi[t] = np.argmax(cand, axis=0)
        delta[t] = cand[psi[t], [0, 1]] + np.array([0.0, llr[t]])
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    return path


def runs_of(path: np.ndarray) -> List[Tuple[int, int]]:
    padded = np.concatenate([[0], np.asarray(path, dtype=int), [0]])
    edges = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _refine_boundary(run: Tuple[int, int], ps: np.ndarray, pe: np.ndarray, prm: DecodeParams) -> Tuple[int, int]:
    i, j = run
    n, r = len(ps), prm.boundary_radius
    lo, hi = max(0, i - r), min(n - 1, i + r)
    s_best = lo + int(np.argmax(ps[lo:hi + 1]))
    if ps[s_best] >= prm.boundary_min and s_best <= j:
        i = s_best
    lo, hi = max(0, j - r), min(n - 1, j + r)
    e_best = lo + int(np.argmax(pe[lo:hi + 1]))
    if pe[e_best] >= prm.boundary_min and e_best >= i:
        j = e_best
    return i, j


def _segment(i: int, j: int, p: np.ndarray, emb: np.ndarray) -> Dict[str, Any]:
    v = emb[i:j + 1].mean(axis=0)
    return {"units": (i, j), "score": float(p[i:j + 1].mean()), "emb": v / (np.linalg.norm(v) + 1e-9)}


def _fallback_segment(p: np.ndarray, emb: np.ndarray) -> Dict[str, Any]:
    """Region of half-maximum around the peak (used when Viterbi finds no run
    but the count head says the video has a glitch)."""
    peak = int(np.argmax(p))
    above = p >= 0.5 * p[peak]
    i = peak
    while i > 0 and above[i - 1]:
        i -= 1
    j = peak
    while j < len(p) - 1 and above[j + 1]:
        j += 1
    return _segment(i, j, p, emb)


def _closest_pair(groups: List[List[int]], segs: List[Dict[str, Any]]) -> Tuple[float, int, int]:
    best = (-1.0, 0, 0)
    for a, b in itertools.combinations(range(len(groups)), 2):
        sim = float(np.mean([segs[x]["emb"] @ segs[y]["emb"] for x in groups[a] for y in groups[b]]))
        if sim > best[0]:
            best = (sim, a, b)
    return best


def group_segments(segs: List[Dict[str, Any]], threshold: float) -> List[List[int]]:
    """Average-linkage agglomeration by instance embedding: separated segments
    of one recurring bug become ONE multi-interval event."""
    groups = [[k] for k in range(len(segs))]
    while len(groups) > 1:
        sim, a, b = _closest_pair(groups, segs)
        if sim < threshold:
            break
        groups[a] += groups.pop(b)
    return groups


def _events(groups: List[List[int]], segs: List[Dict[str, Any]], times: np.ndarray, duration: float,
            p_none: float) -> List[Dict[str, Any]]:
    events = []
    for group in groups:
        ivs = clip_intervals([(float(times[segs[k]["units"][0], 0]), float(times[segs[k]["units"][1], 1]))
                              for k in group], duration)
        if ivs:
            seg_score = max(segs[k]["score"] for k in group)
            events.append({"intervals": [(round(a, 3), round(b, 3)) for a, b in ivs], "seg_score": seg_score,
                           "p_any": 1.0 - p_none, "score": seg_score * (1.0 - p_none)})
    return sorted(events, key=lambda e: -e["score"])


def decode_video(raw: Dict[str, np.ndarray], times: np.ndarray, prm: DecodeParams, trans: Transitions,
                 duration: float) -> List[Dict[str, Any]]:
    """
    Hybrid connectionist/HMM decoding (Bourlard & Morgan 1994): the scaled
    likelihood p(x|s)/p(x) = p(s|x)/p(s) gives llr_t = logit p_t - logit pi,
    so the prior enters ONCE, through the transition matrix.
    """
    if len(raw["act"]) == 0:
        return []
    count = np.exp(raw["count"] - raw["count"].max())
    count = count / count.sum()
    p_none = float(count[0])
    if p_none >= prm.none_threshold:
        return []
    p = sigmoid(raw["act"] / max(prm.temperature, 1e-3))
    llr = np.log(p + 1e-6) - np.log(1 - p + 1e-6) - logit(trans.prior)
    ps, pe = sigmoid(raw["start"]), sigmoid(raw["end"])
    segs = [_segment(*_refine_boundary(run, ps, pe, prm), p, raw["emb"]) for run in runs_of(viterbi(llr, trans))]
    segs = segs or [_fallback_segment(p, raw["emb"])]
    events = _events(group_segments(segs, prm.group_sim), segs, times, duration, p_none)
    kept = [e for e in events if e["seg_score"] >= prm.event_threshold] or events[:1]
    k_hat = int(np.argmax(count[1:])) + 1
    return kept[: min(prm.max_events, k_hat)]


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def run_head(model: Any, samples: Sequence[Dict[str, Any]], device: Any, batch_size: int = 16
             ) -> List[Dict[str, np.ndarray]]:
    import torch

    model.eval()
    outs: List[Dict[str, np.ndarray]] = []
    with torch.no_grad():
        for s in range(0, len(samples), batch_size):
            chunk = samples[s:s + batch_size]
            batch = collate(chunk, device)
            out = model(batch["feats"], batch["ctx"], batch["bank_dist"], batch["times"], batch["mask"])
            for b, sample in enumerate(chunk):
                n = sample["feats"].shape[0]
                outs.append({k: out[k][b, :n].float().cpu().numpy() for k in ("act", "start", "end", "emb")}
                            | {"count": out["count"][b].float().cpu().numpy()})
    return outs


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling (Guo et al. 2017): argmin_T NLL on dev units."""
    grid = np.exp(np.linspace(math.log(0.25), math.log(4.0), 41))
    nll = [-np.mean(labels * np.log(sigmoid(logits / t) + 1e-9) + (1 - labels) * np.log(1 - sigmoid(logits / t) + 1e-9))
           for t in grid]
    return float(grid[int(np.argmin(nll))])


@dataclasses.dataclass
class TrainSpec:
    epochs: int = 40
    lr: float = 3e-4
    batch_size: int = 16
    seed: int = 1337


def _train_epoch(model: Any, train: Sequence[Dict[str, Any]], opt: Any, sched: Any, ctx: Dict[str, Any]
                 ) -> Dict[str, float]:
    import torch

    model.train()
    spec, cfg, rng = ctx["spec"], ctx["cfg"], ctx["rng"]
    order = rng.permutation(len(train))
    agg: Counter = Counter()
    for s in range(0, len(order), spec.batch_size):
        chunk = [augment(train[i], rng, cfg.k_max) for i in order[s:s + spec.batch_size]]
        batch = collate(chunk, ctx["device"])
        out = model(batch["feats"], batch["ctx"], batch["bank_dist"], batch["times"], batch["mask"])
        loss, parts = head_loss(out, batch, cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite head loss: {parts}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        agg.update({k: v * len(chunk) for k, v in parts.items()})
    return {k: v / len(train) for k, v in agg.items()}


def train_head(train: Sequence[Dict[str, Any]], cfg: HeadConfig, spec: TrainSpec, device: Any,
               select_fn: Callable[[Any], float]) -> Tuple[Any, List[Dict[str, float]]]:
    """select_fn(model) -> dev selection score (higher is better)."""
    import torch

    torch.manual_seed(spec.seed)
    model = build_head(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=0.05)
    steps = max(1, spec.epochs * math.ceil(len(train) / spec.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=spec.lr, total_steps=steps, pct_start=0.1)
    ctx = {"spec": spec, "cfg": cfg, "rng": make_rng(spec.seed), "device": device}
    history: List[Dict[str, float]] = []
    best_score, best_state = -math.inf, copy.deepcopy(model.state_dict())
    for epoch in range(spec.epochs):
        parts = _train_epoch(model, train, opt, sched, ctx)
        score = float(select_fn(model))
        history.append({"epoch": epoch, "dev_score": score, **parts})
        LOGGER.info("head epoch %2d | %s | dev %.4f", epoch, " ".join(f"{k} {v:.4f}" for k, v in parts.items()), score)
        if score > best_score:
            best_score, best_state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history


def save_head(fdir: Path, suffix: str, model: Any, payload: Dict[str, Any], bank: NormalityBank) -> None:
    """Tensors only (weights_only-loadable) + JSON + pickle-free npz."""
    import torch

    torch.save(model.state_dict(), safe_path(fdir / f"head{suffix}.pt"))
    write_json(fdir / f"head{suffix}.json", payload)
    bank.save(fdir / f"bank{suffix}.npz")


def load_head(fdir: Path, suffix: str, device: Any) -> Tuple[Any, HeadConfig, NormalityBank, Transitions, DecodeParams]:
    import torch

    payload = read_json(fdir / f"head{suffix}.json")
    cfg = HeadConfig(**payload["config"])
    model = build_head(cfg).to(device)
    state = torch.load(safe_path(fdir / f"head{suffix}.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return (model, cfg, NormalityBank.load(fdir / f"bank{suffix}.npz"), Transitions(**payload["transitions"]),
            DecodeParams(**payload["decode"]))


# =========================================================================== #
# 6. evaluation: GliDe protocol, judge, statistics
# =========================================================================== #
#
#   S_ij  judge score in [0,5], normalised by /5 (the paper reports
#         percentages without stating the normalisation)
#   W_ij  = S_ij * tIoU(T_i, T_j)                 tIoU over interval SETS
#   A     = Hungarian matching maximising sum W, pairs with W > 0
#   P = sum_A S / N, R = sum_A S / M, F1;  mIoU = mean_A tIoU
#   P_all = sum_A W / N, R_all = sum_A W / M, F1xIoU = harmonic mean
#   N = #predicted events, M = #gold events, micro-averaged over videos.

SUM_KEYS = ("n_pred", "n_gold", "n_matched", "sum_s", "sum_w", "sum_iou")


@dataclasses.dataclass
class VideoScore:
    """Counts are floats: run-averaged scores of stochastic systems (several
    seeds) have expected, possibly fractional, counts."""

    video_name: str
    game: str
    n_pred: float
    n_gold: float
    n_matched: float
    sum_s: float
    sum_w: float
    sum_iou: float
    ious: List[float]
    pairs: List[Dict[str, Any]]

    def sums(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in SUM_KEYS], dtype=float)


def match_video(pred: Sequence[Dict[str, Any]], gold: Sequence[Dict[str, Any]],
                semantic: Callable[[str, str], float], name: str = "", game: str = "") -> VideoScore:
    from scipy.optimize import linear_sum_assignment

    score = VideoScore(name, game, len(pred), len(gold), 0, 0.0, 0.0, 0.0, [], [])
    if not pred or not gold:
        return score
    iou = np.array([[interval_iou(p["intervals"], g["intervals"]) for g in gold] for p in pred])
    sem = np.array([[semantic(p.get("description", ""), g.get("description", "")) for g in gold] for p in pred])
    w = sem * iou
    for i, j in zip(*linear_sum_assignment(-w), strict=True):
        if w[i, j] <= 0.0:
            continue
        score.n_matched += 1
        score.sum_s += float(sem[i, j])
        score.sum_w += float(w[i, j])
        score.sum_iou += float(iou[i, j])
        score.ious.append(float(iou[i, j]))
        score.pairs.append({"pred": int(i), "gold": int(j), "s": float(sem[i, j]), "iou": float(iou[i, j])})
    return score


def metrics_from_sums(t: np.ndarray) -> Dict[str, float]:
    n_pred, n_gold, n_matched, sum_s, sum_w, sum_iou = (float(v) for v in t)
    p_desc = 100.0 * sum_s / n_pred if n_pred else 0.0
    r_desc = 100.0 * sum_s / n_gold if n_gold else 0.0
    p_all = 100.0 * sum_w / n_pred if n_pred else 0.0
    r_all = 100.0 * sum_w / n_gold if n_gold else 0.0
    return {"precision": p_desc, "recall": r_desc, "f1": harmonic(p_desc, r_desc),
            "miou": sum_iou / n_matched if n_matched else 0.0, "precision_overall": p_all,
            "recall_overall": r_all, "f1_x_iou": harmonic(p_all, r_all),
            "n_pred": n_pred, "n_gold": n_gold, "n_matched": n_matched}


def aggregate(scores: Sequence[VideoScore]) -> Dict[str, float]:
    total = np.sum([s.sums() for s in scores], axis=0) if scores else np.zeros(len(SUM_KEYS))
    return metrics_from_sums(total)


def always_one(_pred: str, _gold: str) -> float:
    return 1.0


def temporal_proxy(pred_by_video: Dict[str, List[Dict[str, Any]]],
                   gold_by_video: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
    """The protocol with S == 1: a judge-free score of localisation alone,
    used ONLY for model selection on dev (never reported as a result)."""
    return aggregate([match_video(pred_by_video.get(k, []), g, always_one) for k, g in gold_by_video.items()])


def recall_at_tiou(scores: Sequence[VideoScore], thresholds: Sequence[float]) -> Dict[str, float]:
    total = max(sum(s.n_gold for s in scores), 1)
    return {f"R@tIoU{t:g}": 100.0 * sum(sum(i >= t for i in s.ious) for s in scores) / total for t in thresholds}


def _cluster_sums(scores: Sequence[VideoScore], cluster: str) -> np.ndarray:
    groups: Dict[str, np.ndarray] = {}
    for s in scores:
        key = s.game if cluster == "game" else s.video_name
        groups[key] = groups.get(key, np.zeros(len(SUM_KEYS))) + s.sums()
    return np.stack([groups[k] for k in sorted(groups)])


def bootstrap_ci(scores: Sequence[VideoScore], metric: str, n_boot: int = 2000, seed: int = 0,
                 cluster: str = "game") -> Dict[str, float]:
    """Cluster bootstrap over games (Field & Welsh 2007): videos of one game
    share engine, style and often the bug, so they are not independent."""
    sums = _cluster_sums(scores, cluster)
    weights = make_rng(seed).multinomial(len(sums), np.full(len(sums), 1.0 / len(sums)), size=n_boot)
    draws = np.array([metrics_from_sums(w @ sums)[metric] for w in weights])
    return {"point": metrics_from_sums(sums.sum(axis=0))[metric], "lo": float(np.quantile(draws, 0.025)),
            "hi": float(np.quantile(draws, 0.975)), "n_clusters": len(sums)}


def paired_test(a: Sequence[VideoScore], b: Sequence[VideoScore], metric: str, n_boot: int = 5000,
                seed: int = 0) -> Dict[str, float]:
    """H0: metric(A) = metric(B) over games. Paired cluster resampling and the
    null-centred two-sided p-value (Hall & Wilson 1991):
        p = (1 + #{|d* - d| >= |d|}) / (1 + B)."""
    common = sorted({s.video_name for s in a} & {s.video_name for s in b})
    a_map, b_map = {s.video_name: s for s in a}, {s.video_name: s for s in b}
    sa = _cluster_sums([a_map[k] for k in common], "game")
    sb = _cluster_sums([b_map[k] for k in common], "game")
    weights = make_rng(seed).multinomial(len(sa), np.full(len(sa), 1.0 / len(sa)), size=n_boot)
    delta = metrics_from_sums(sa.sum(axis=0))[metric] - metrics_from_sums(sb.sum(axis=0))[metric]
    draws = np.array([metrics_from_sums(w @ sa)[metric] - metrics_from_sums(w @ sb)[metric] for w in weights])
    p = (1.0 + np.sum(np.abs(draws - delta) >= abs(delta))) / (1.0 + n_boot)
    return {"delta": float(delta), "p": float(min(1.0, p)), "ci_lo": float(np.quantile(draws, 0.025)),
            "ci_hi": float(np.quantile(draws, 0.975)), "n_videos": len(common), "n_clusters": len(sa)}


def holm(pvalues: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    running, m = 0.0, len(pvalues)
    for rank, (key, p) in enumerate(sorted(pvalues.items(), key=lambda kv: kv[1])):
        running = max(running, min(1.0, (m - rank) * p))
        out[key] = {"p_raw": p, "p_holm": running, "reject": running <= alpha}
    return out


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U / (n1 n0) with average ranks for ties."""
    from scipy.stats import rankdata

    labels = np.asarray(labels).astype(bool)
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if n1 == 0 or n0 == 0:
        return math.nan
    ranks = rankdata(np.asarray(scores, dtype=float))
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    y = np.asarray(labels).astype(float)[np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")]
    if y.sum() <= 0:
        return math.nan
    return float((np.cumsum(y) / np.arange(1, len(y) + 1) * y).sum() / y.sum())


JUDGE_SYSTEM = (
    "You are a strict evaluator for a video game QA benchmark. You compare a system-generated glitch "
    "description against a human-written reference description of the same gameplay video and rate their "
    "semantic agreement. You never reward fluency, length or confidence. You reward only agreement about which "
    "entity fails, in what way it fails, and what kind of software defect it is."
)

JUDGE_PROMPT = """Rate the semantic agreement between the two glitch descriptions below on an integer scale from 0 to 5.

5 = same affected entity, same abnormal behaviour, same defect type; a QA lead would treat them as the same bug report.
4 = same entity and same abnormal behaviour, minor divergence in one detail or in the stated defect type.
3 = same abnormal behaviour but the affected entity is described differently, or the same entity but a partially different failure.
2 = related failure type in the same scene, but the specific behaviour disagrees.
1 = both describe some anomaly, but they are not the same anomaly.
0 = unrelated, or the prediction describes normal gameplay.

REFERENCE (human annotation):
{gold}

PREDICTION (system output):
{pred}

Return ONE JSON object and nothing else:
{{"reasoning": "one sentence", "score": <integer 0-5>}}"""

_WORD = re.compile(r"[a-z][a-z\-']+")
_STOP = frozenset(
    "a an the and or but of in on at to from by with for as is are was were be been it its this that these those "
    "there then than into onto over under through while which who what when where how not no so very more most "
    "some any each other one two game player character video clip screen frame".split()
)


def content_tokens(text: str) -> List[str]:
    return [t for t in _WORD.findall(str(text).lower()) if len(t) > 2 and t not in _STOP]


def unigram_f1(a: str, b: str) -> float:
    ca, cb = Counter(content_tokens(a)), Counter(content_tokens(b))
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    p, r = overlap / sum(ca.values()), overlap / sum(cb.values())
    return 2 * p * r / (p + r)


class Judge:
    """'openai': any OpenAI-compatible endpoint (OpenAI, or a local vLLM
    server via --judge-base-url); 'hf': local causal LM; 'offline':
    unigram F1 (self-tests only; NOT comparable to the paper)."""

    def __init__(self, backend: str, model: str, cache_path: Path, base_url: Optional[str] = None,
                 api_key_env: str = "OPENAI_API_KEY") -> None:
        self.backend, self.model = backend, model
        self.calls = self.fallbacks = 0
        self.cache = {row["key"]: float(row["value"]) for row in read_jsonl(cache_path, required=False)}
        self.cache_path = safe_path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._client: Any = None
        self._tok: Any = None
        self._lm: Any = None
        if backend == "openai":
            import openai

            key = os.environ.get(api_key_env)
            if not key:
                raise RuntimeError(f"set ${api_key_env} (any non-empty value for a local vLLM server)")
            self._client = openai.OpenAI(api_key=key, base_url=base_url)
        elif backend == "hf":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(model)
            self._lm = AutoModelForCausalLM.from_pretrained(model, device_map="auto").eval()
        elif backend != "offline":
            raise ValueError(f"unknown judge backend {backend!r}")

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.backend, "model": self.model, "calls": self.calls, "parse_fallbacks": self.fallbacks}

    def _chat(self, user: str) -> str:
        self.calls += 1
        messages = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]
        if self._client is not None:
            out = self._client.chat.completions.create(model=self.model, temperature=0.0, max_tokens=160,
                                                       messages=messages)
            return out.choices[0].message.content or ""
        import torch

        inputs = self._tok([self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],
                           return_tensors="pt").to(self._lm.device)
        with torch.inference_mode():
            out = self._lm.generate(**inputs, do_sample=False, max_new_tokens=160)
        return self._tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def _llm_score(self, pred: str, gold: str) -> float:
        raw = ""
        for attempt in range(3):
            try:
                raw = self._chat(JUDGE_PROMPT.format(gold=gold, pred=pred))
                break
            except (OSError, RuntimeError, ValueError) as exc:
                LOGGER.warning("judge call failed (%r), retry %d", exc, attempt + 1)
                time.sleep(2 ** attempt)
        match = re.search(r'"score"\s*:\s*([0-5])', raw) or re.search(r"\b([0-5])\b", raw)
        if match:
            return float(match.group(1)) / 5.0
        self.fallbacks += 1
        return unigram_f1(pred, gold)

    def __call__(self, pred: str, gold: str) -> float:
        pred, gold = (pred or "").strip(), (gold or "").strip()
        if not pred or not gold:
            return 0.0
        key = stable_hash(f"{self.backend}|{self.model}|{pred}|{gold}", 24)
        if key not in self.cache:
            self.cache[key] = unigram_f1(pred, gold) if self.backend == "offline" else self._llm_score(pred, gold)
            with self.cache_path.open("a", encoding=UTF8) as sink:
                sink.write(json.dumps({"key": key, "value": self.cache[key]}) + "\n")
        return self.cache[key]


# =========================================================================== #
# 7. describer, calibrator, and the shared report schema
# =========================================================================== #

DESCRIBE_PROMPT = (
    'Gameplay clip from "{game}" ({genre} / {subgenre}), showing {span} of the recording.\n'
    "If the clip shows a software glitch, describe it in one or two sentences: name the affected entity by its "
    "appearance, state what it does wrong, and say where on screen it happens. "
    "If there is no glitch, answer exactly: " + NO_GLITCH
)

# The GliDe summariser schema; used verbatim by vanilla, direct and glide.
REPORT_SCHEMA = ('{{"no_bugs": <true|false>, "bugs": ["<description>", ...], '
                 '"time_nodes": [[[<start_sec>, <end_sec>], ...], ...]}}')

DIRECT_PROMPT = (
    'This is a gameplay video from "{game}" ({genre} / {subgenre}), {duration:.1f} seconds long. '
    "Find every software glitch in it. Return ONE JSON object and nothing else:\n" + REPORT_SCHEMA + "\n"
    "time_nodes[i] lists the time intervals of bugs[i] in seconds."
)


def span_text(intervals: Sequence[Interval]) -> str:
    return ", ".join(f"seconds {a:.1f} to {b:.1f}" for a, b in intervals)


def clip_frames(item: VideoItem, intervals: Sequence[Interval], max_frames: int, margin: float
                ) -> Tuple[List[str], float, List[Interval]]:
    """Frames covering every interval (+ margin), budget split by duration."""
    padded = clip_intervals([(a - margin, b + margin) for a, b in intervals], item.duration)
    total = sum(b - a for a, b in padded) or 1.0
    frames: List[str] = []
    rates: List[float] = []
    for a, b in padded:
        f, _, fps = select_frames(item, a, b, max(2, round(max_frames * (b - a) / total)))
        frames.extend(f)
        rates.append(fps)
    if len(frames) > max_frames:
        frames = [frames[i] for i in np.round(np.linspace(0, len(frames) - 1, num=max_frames)).astype(int)]
    if len(frames) % 2:
        frames.append(frames[-1])
    return frames, float(np.mean(rates)), padded


def _jittered(intervals: Sequence[Interval], duration: float, rng: np.random.Generator, jitter: float) -> List[Interval]:
    out = []
    for a, b in intervals:
        d = max(b - a, 0.5)
        a2 = max(0.0, a + float(rng.uniform(-jitter, jitter)) * d)
        b2 = min(duration or b, b + float(rng.uniform(-jitter, jitter)) * d)
        out.append((a2, b2) if b2 - a2 >= 0.5 else (a, b))
    return merge_intervals(out)


def _free_spans(events: Sequence[Dict[str, Any]], duration: float) -> List[Interval]:
    busy = merge_intervals([iv for ev in events for iv in ev["intervals"]])
    edges = [0.0] + [x for iv in busy for x in iv] + [duration]
    gaps = list(zip(edges[::2], edges[1::2], strict=True))
    return [(a + 0.5, b - 0.5) for a, b in gaps if b - a >= 3.0]


def describer_samples(items: Dict[str, VideoItem], names: Sequence[str], seed: int, neg_ratio: float = 0.33,
                      jitter: float = 0.2) -> List[Dict[str, Any]]:
    """Positives: gold events with jittered boundaries (the describer sees
    the same imperfect crops the localiser hands it at test time).
    Negatives: glitch-free spans, target NO_GLITCH."""
    rng = make_rng(seed)
    pos: List[Dict[str, Any]] = []
    neg: List[Dict[str, Any]] = []
    for name in names:
        item = items[name]
        events = gold_events(item)
        pos.extend({"name": name, "intervals": _jittered(ev["intervals"], item.duration, rng, jitter),
                    "answer": ev["description"]} for ev in events)
        for a, b in _free_spans(events, item.duration):
            length = min(b - a, float(rng.uniform(2.0, 6.0)))
            s = float(rng.uniform(a, b - length))
            neg.append({"name": name, "intervals": [(s, s + length)], "answer": NO_GLITCH})
    neg = [neg[i] for i in rng.permutation(len(neg))[: math.ceil(neg_ratio * len(pos))]]
    samples = pos + neg
    LOGGER.info("describer samples: %d positive, %d NO_GLITCH", len(pos), len(neg))
    return [samples[i] for i in rng.permutation(len(samples))]


def report_json(events: Sequence[Dict[str, Any]]) -> str:
    return json.dumps({"no_bugs": not events, "bugs": [e["description"] for e in events],
                       "time_nodes": [[[round(a, 1), round(b, 1)] for a, b in e["intervals"]] for e in events]},
                      ensure_ascii=False)


def direct_samples(items: Dict[str, VideoItem], names: Sequence[str]) -> List[Dict[str, Any]]:
    return [{"name": n, "intervals": None, "answer": report_json(gold_events(items[n]))} for n in names]


def sample_io(item: VideoItem, intervals: Optional[Sequence[Interval]], max_frames: int, margin: float
              ) -> Tuple[List[str], float, str]:
    """Whole video (intervals None: vanilla / direct) or an event clip."""
    if intervals is None:
        frames, _, fps = select_frames(item, 0.0, item.duration, max_frames)
        return frames, fps, DIRECT_PROMPT.format(duration=item.duration, **item.game_text())
    frames, fps, padded = clip_frames(item, intervals, max_frames, margin)
    return frames, fps, DESCRIBE_PROMPT.format(span=span_text(padded), **item.game_text())


@dataclasses.dataclass
class LMSpec:
    out_dir: str
    epochs: int = 2
    lr: float = 1e-4
    grad_accum: int = 8
    max_frames: int = 16
    margin: float = 1.0
    lora_r: int = 16
    lora_alpha: int = 32
    seed: int = 1337
    max_dev: int = 200


def _lm_loss(backbone: Backbone, items: Dict[str, VideoItem], sample: Dict[str, Any], spec: LMSpec) -> Any:
    frames, fps, prompt = sample_io(items[sample["name"]], sample["intervals"], spec.max_frames, spec.margin)
    return backbone.lm_loss(frames, fps, prompt, sample["answer"])


def _lm_epoch(backbone: Backbone, items: Dict[str, VideoItem], train: Sequence[Dict[str, Any]],
              spec: LMSpec, optim: Tuple[Any, Any, np.random.Generator]) -> float:
    import torch

    opt, sched, rng = optim
    params = backbone.trainable_parameters()
    backbone.train_mode(True)
    order = rng.permutation(len(train))
    running, seen = 0.0, 0
    opt.zero_grad(set_to_none=True)
    for k, idx in enumerate(order, 1):
        loss = _lm_loss(backbone, items, train[int(idx)], spec)
        (loss / spec.grad_accum).backward()
        running, seen = running + float(loss.detach()), seen + 1
        if k % spec.grad_accum == 0 or k == len(order):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
    backbone.train_mode(False)
    return running / max(seen, 1)


def _lm_dev_loss(backbone: Backbone, items: Dict[str, VideoItem], dev: Sequence[Dict[str, Any]], spec: LMSpec) -> float:
    import torch

    with torch.no_grad():
        losses = [float(_lm_loss(backbone, items, s, spec)) for s in dev[: spec.max_dev]]
    return float(np.mean(losses)) if losses else math.inf


def train_lm(backbone: Backbone, items: Dict[str, VideoItem], train: Sequence[Dict[str, Any]],
             dev: Sequence[Dict[str, Any]], spec: LMSpec) -> Dict[str, Any]:
    """LoRA-SFT shared by the NORMA describer and the direct-SFT baseline
    (same recipe, same budget: the comparison isolates the architecture)."""
    import torch

    backbone.attach_lora(spec.lora_r, spec.lora_alpha)
    opt = torch.optim.AdamW(backbone.trainable_parameters(), lr=spec.lr, weight_decay=0.0)
    total = max(1, spec.epochs * math.ceil(len(train) / spec.grad_accum))
    warm = max(1, int(0.03 * total))

    def schedule(step: int) -> float:
        return min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total)))

    optim = (opt, torch.optim.lr_scheduler.LambdaLR(opt, schedule), make_rng(spec.seed))
    best, history, t0 = math.inf, [], time.perf_counter()
    for epoch in range(spec.epochs):
        train_loss = _lm_epoch(backbone, items, train, spec, optim)
        dev_loss = _lm_dev_loss(backbone, items, dev, spec)
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss})
        LOGGER.info("lm epoch %d: train %.4f dev %.4f", epoch, train_loss, dev_loss)
        if dev_loss < best:
            best = dev_loss
            backbone.save_adapter(Path(spec.out_dir))
    summary = {"best_dev_loss": best, "history": history, "train_seconds": time.perf_counter() - t0,
               "peak_gpu_mb": peak_gpu_mb(), "n_train": len(train), "n_dev": len(dev)}
    write_json(Path(spec.out_dir) / TRAIN_SUMMARY, summary)
    return summary


def mbr_select(candidates: Sequence[str], utility: Callable[[str, str], float] = unigram_f1) -> str:
    """Minimum-Bayes-risk choice (Kumar & Byrne 2004; Freitag et al. 2022):
    argmax_i sum_j u(c_i, c_j)."""
    cands = [c for c in candidates if c]
    if len(cands) <= 2:
        return cands[0] if cands else ""
    scores = [sum(utility(a, b) for j, b in enumerate(cands) if j != i) for i, a in enumerate(cands)]
    return cands[int(np.argmax(scores))]


def is_refusal(text: str) -> bool:
    return text.strip().upper().startswith(NO_GLITCH)


@dataclasses.dataclass
class DescribeSpec:
    max_frames: int = 16
    margin: float = 1.0
    mbr_samples: int = 4
    temperature: float = 0.7
    max_new_tokens: int = 96
    seed: int = 0


class Describer:
    def __init__(self, backbone: Backbone, spec: DescribeSpec) -> None:
        self.backbone, self.spec = backbone, spec

    def describe(self, item: VideoItem, intervals: Sequence[Interval]) -> Dict[str, Any]:
        s = self.spec
        frames, fps, prompt = sample_io(item, intervals, s.max_frames, s.margin)
        p_reject = self.backbone.first_token_prob(frames, fps, prompt, NO_GLITCH)
        seed = derived_seed(s.seed, item.name, intervals)
        cands = self.backbone.generate_video(frames, fps, prompt, GenSpec(s.max_new_tokens, 0.0, 1, None, seed))
        if s.mbr_samples > 0:
            cands += self.backbone.generate_video(frames, fps, prompt,
                                                  GenSpec(s.max_new_tokens, s.temperature, s.mbr_samples, None, seed))
        real = [c for c in cands if c and not is_refusal(c)]
        vetoed = not real
        if vetoed:
            # The veto is EVIDENCE (p_reject feeds the dev-fitted calibrator),
            # not a decision: take the best description conditional on a glitch.
            real = self.backbone.generate_video(frames, fps, prompt, GenSpec(s.max_new_tokens, 0.0, 1, NO_GLITCH, seed))
        return {"description": mbr_select(real), "p_reject": float(p_reject), "vetoed": vetoed}


def event_features(event: Dict[str, Any]) -> np.ndarray:
    return np.array([logit(event["seg_score"]), logit(event["p_any"]), logit(1.0 - event.get("p_reject", 0.5))])


def fit_logreg(x: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 50) -> np.ndarray:
    """L2-regularised logistic regression by Newton-Raphson (IRLS)."""
    xb = np.hstack([x, np.ones((x.shape[0], 1))])
    w = np.zeros(xb.shape[1])
    reg = np.r_[np.ones(len(w) - 1), 0.0] * l2
    for _ in range(iters):
        p = sigmoid(xb @ w)
        step = np.linalg.solve((xb * (p * (1 - p))[:, None]).T @ xb + np.diag(reg) + 1e-9 * np.eye(len(w)),
                               xb.T @ (p - y) + reg * w)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def fused_probability(event: Dict[str, Any], weights: Sequence[float]) -> float:
    return float(sigmoid(np.array(float(np.r_[event_features(event), 1.0] @ np.asarray(weights)))))


def event_is_tp(intervals: Sequence[Interval], gold: Sequence[Dict[str, Any]], tiou: float = 0.3) -> int:
    return int(any(interval_iou(intervals, g["intervals"]) >= tiou for g in gold))


def parse_report(text: str, duration: float) -> List[Dict[str, Any]]:
    """Shared post-processing for every report-producing system: JSON parse,
    interval normalisation, clipping to the video, empty entries dropped."""
    parsed = extract_json_object(text) or {}
    bugs = parsed.get("bugs") or []
    nodes = parsed.get("time_nodes") or []
    bugs = bugs if isinstance(bugs, list) else [bugs]
    out = []
    for i, desc in enumerate(bugs):
        ivs = clip_intervals(normalise_intervals(nodes[i] if i < len(nodes) else []), duration)
        if str(desc).strip() and ivs:
            out.append({"description": str(desc).strip(), "intervals": ivs})
    return out


# =========================================================================== #
# 8. GliDe reproduction (arXiv 2604.07818, Sec. 4), same backbone + harness
# =========================================================================== #
#
# Followed from the paper: 4 fps; non-overlapping 8-frame windows stitched
# into one 2x4 grid with frame indices; Scanner fields (has_glitch, coarse
# category, visual_cues, reasoning, frame_range, confidence, game_context)
# with a recall-favouring instruction; game-aware memory summarised from the
# game_context fields; Planner -> Executor -> Reflector loop with tools vqa /
# zoom_in / object_tracking and an Advocate (QA tester) / Skeptic (game
# designer) / Judge (tech lead) debate, <= 5 rounds, early stop at judge
# confidence >= 0.80; greedy first-match event grounding with window-by-window
# forward/backward extension that stops at the first non-matching window;
# Summariser emitting the shared report schema; temperature 0.5 everywhere,
# 512 tokens (scanner, summariser) and 1024 tokens (agents).
#
# Reconstructed (the paper's verbatim prompts are not available to us): the
# prompt wording below. Choices made where the paper is silent, each in
# GliDe's favour: the Judge sees the window image; event boundaries are
# refined with the Scanner's frame_range.
#
# Deviation: object_tracking uses normalised-cross-correlation template
# tracking instead of SAM3; its output feeds the paper's four rule checks
# (position jump > 20 % of the frame diagonal; velocity spike > 500 px/s,
# rescaled to 0.34 diagonals/s, i.e. 500 px/s on a 1280x720 frame; freeze
# < 1 px for >= 4 frames; jitter = direction reversal in >= 50 % of steps).

GLIDE_CATEGORIES = "Physics|Visual|Game Logic|Other"

SCANNER_PROMPT = (
    'You are the Scanner of a video game QA team. The image is a 2x4 grid of 8 consecutive frames from "{game}" '
    "({genre} / {subgenre}), seconds {t0:.1f} to {t1:.1f}; the frame index is written in each tile's top-left "
    "corner (read left to right, top to bottom). Flag anything that could be a software glitch: favour recall, "
    "when in doubt flag it.\nReturn ONE JSON object and nothing else:\n"
    '{{"has_glitch": <true|false>, "category": "' + GLIDE_CATEGORIES + '", "visual_cues": "what you see", '
    '"reasoning": "why", "frame_range": [<first frame index>, <last frame index>], "confidence": <0-1>, '
    '"game_context": "2-4 sentences on what is normal in this game"}}'
)

MEMORY_PROMPT = (
    'Summarise these observations about the game "{game}" into a concise game-aware memory (at most 6 sentences): '
    "what is normal for this title (visual style, physics, mechanics, UI) and what would therefore be "
    "abnormal.\nObservations:\n{notes}"
)

PLANNER_PROMPT = (
    'You are the Planner verifying a suspected glitch in "{game}".\nGame memory:\n{memory}\n\n'
    "Scanner finding (frames {first}-{last}, seconds {t0:.1f}-{t1:.1f}): {cues}\nReasoning: {reasoning}\n\n"
    "Evidence gathered so far:\n{evidence}\n\nTools:\n"
    "- vqa(question): ask a question about the 8-frame window\n"
    "- zoom_in(bbox, question): crop bbox [x0, y0, x1, y1] (fractions of a frame) in every frame, upscale, ask\n"
    "- object_tracking(bbox): track the object in bbox across the frames and test physics rules\n"
    "Return ONE JSON object with the next tool and nothing else:\n"
    '{{"tool": "vqa|zoom_in|object_tracking", "question": "...", "bbox": [x0, y0, x1, y1] or null}}'
)

ADVOCATE_PROMPT = (
    'You are a QA tester. Using the evidence below about a clip from "{game}", argue in two sentences that the '
    "observation IS a genuine software glitch, naming the entity and the rule it violates.\n"
    "Game memory:\n{memory}\n\nEvidence:\n{evidence}"
)

SKEPTIC_PROMPT = (
    'You are the game designer of "{game}". Using the evidence below, argue in two sentences that the observation '
    "is LEGITIMATE gameplay, art direction or a scripted event.\nGame memory:\n{memory}\n\nEvidence:\n{evidence}"
)

JUDGE_AGENT_PROMPT = (
    'You are the technical lead ruling on a suspected glitch in "{game}" (the image is the 8-frame window).\n'
    "Game memory:\n{memory}\n\nEvidence:\n{evidence}\n\nQA tester: {advocate}\nGame designer: {skeptic}\n\n"
    "Return ONE JSON object and nothing else:\n"
    '{{"verdict": "glitch|normal", "confidence": <0-1>, "description": "one or two sentences naming the affected '
    'entity by appearance, what it does wrong and where on screen"}}'
)

SAME_GLITCH_PROMPT = (
    "Observation A: {a}\nObservation B: {b}\n\nDo A and B describe the same underlying glitch, i.e. the affected "
    "entity, the anomaly type and the abnormal behaviour all match (minor wording differences are fine)? "
    "Answer yes or no."
)

SUMMARISER_PROMPT = (
    'Write the final glitch report for the video "{video}" from "{game}". Each group below is one glitch with its '
    "time intervals; merge descriptions of the same glitch into one or two sentences that name the affected "
    "entity, the abnormal behaviour and where it happens. Return ONE JSON object and nothing else:\n"
    + REPORT_SCHEMA + "\nGROUPS_JSON:\n{groups}"
)


def _gray_small(frames: Sequence[str], width: int = 160) -> Tuple[List[np.ndarray], float]:
    from PIL import Image

    arrays = [load_rgb(p) for p in frames]
    h, w = arrays[0].shape[:2]
    scale = width / float(w)
    size = (width, max(8, round(h * scale)))
    grays = [np.asarray(Image.fromarray(a).convert("L").resize(size), dtype=np.float32) / 255.0 for a in arrays]
    return grays, scale


def _ncc_locate(frame: np.ndarray, template: np.ndarray) -> Tuple[float, float]:
    """Global normalised-cross-correlation search over the whole frame. SAM3
    re-detects objects anywhere in the frame; a local search window would
    lose exactly the large displacements the jump rule must catch."""
    th, tw = template.shape
    windows = np.lib.stride_tricks.sliding_window_view(frame, (th, tw))
    t = (template - template.mean()) / (template.std() + 1e-6)
    w = windows - windows.mean(axis=(-1, -2), keepdims=True)
    w = w / (windows.std(axis=(-1, -2), keepdims=True) + 1e-6)
    dy, dx = np.unravel_index(int(np.argmax((w * t).mean(axis=(-1, -2)))), w.shape[:2])
    return dx + tw / 2.0, dy + th / 2.0


def ncc_track(frames: Sequence[str], bbox: Sequence[float]) -> Tuple[np.ndarray, float, float]:
    """Template tracking of the bbox content; returns centres in ORIGINAL
    frame pixels, the frame diagonal, and the downscale factor."""
    grays, scale = _gray_small(frames)
    h, w = grays[0].shape
    x0, y0, x1, y1 = (float(np.clip(v, 0.0, 1.0)) for v in bbox)
    bx0, by0 = int(x0 * w), int(y0 * h)
    bx1, by1 = max(bx0 + 4, int(x1 * w)), max(by0 + 4, int(y1 * h))
    # Pad the box with context: a template that is the object's flat interior
    # has ~zero variance and makes NCC ill-posed; matching needs its edges.
    px, py = max(4, (bx1 - bx0) // 2), max(4, (by1 - by0) // 2)
    tx0, ty0, tx1, ty1 = max(0, bx0 - px), max(0, by0 - py), min(w, bx1 + px), min(h, by1 + py)
    template = grays[0][ty0:ty1, tx0:tx1]
    offset = ((bx0 + bx1) / 2.0 - (tx0 + tx1) / 2.0, (by0 + by1) / 2.0 - (ty0 + ty1) / 2.0)
    located = [_ncc_locate(frame, template) for frame in grays[1:]]
    track = [((bx0 + bx1) / 2.0, (by0 + by1) / 2.0)] + [(x + offset[0], y + offset[1]) for x, y in located]
    return np.array(track) / scale, math.hypot(w / scale, h / scale), scale


def physics_flags(track: np.ndarray, diagonal: float, fps: float, pixel: float = 1.0) -> List[str]:
    """GliDe's four rule checks on a trajectory (pixel = one native pixel)."""
    steps = np.diff(track, axis=0)
    dist = np.linalg.norm(steps, axis=1)
    flags = []
    if (dist > 0.20 * diagonal).any():
        flags.append(f"VIOLATION position jump: {dist.max():.0f} px > 20% of the frame diagonal")
    if (dist * fps > 0.34 * diagonal).any():
        flags.append(f"VIOLATION velocity spike: {dist.max() * fps:.0f} px/s")
    still = np.convolve((dist < pixel).astype(int), np.ones(4, dtype=int), mode="valid")
    if (still >= 4).any():
        flags.append("VIOLATION freeze: < 1 px motion for >= 4 consecutive frames")
    moving = steps[dist >= pixel]
    if len(moving) >= 3:
        reversals = (np.einsum("ij,ij->i", moving[1:], moving[:-1]) < 0).mean()
        if reversals >= 0.5:
            flags.append(f"VIOLATION jitter: direction reverses in {100 * reversals:.0f}% of steps")
    return flags


def consecutive_runs(values: Sequence[int]) -> List[List[int]]:
    runs: List[List[int]] = []
    for v in values:
        if runs and v == runs[-1][-1] + 1:
            runs[-1].append(v)
        else:
            runs.append([v])
    return runs


@dataclasses.dataclass
class GlideSpec:
    temperature: float = GLIDE_TEMPERATURE
    scan_tokens: int = 512
    agent_tokens: int = 1024
    max_rounds: int = GLIDE_MAX_ROUNDS
    stop_conf: float = GLIDE_STOP_CONF
    seed: int = 0


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class GlideRunner:
    def __init__(self, backbone: Backbone, spec: GlideSpec) -> None:
        self.backbone, self.spec = backbone, spec

    def _gen(self, tokens: int, *key: Any) -> GenSpec:
        return GenSpec(tokens, self.spec.temperature, 1, None, derived_seed(self.spec.seed, *key))

    def windows(self, item: VideoItem) -> List[Dict[str, Any]]:
        frames, stamps, fps = subsample(item, DEFAULT_FPS)
        out = []
        for k, s in enumerate(range(0, len(frames), GLIDE_WINDOW)):
            chunk = frames[s:s + GLIDE_WINDOW]
            out.append({"index": k, "first": s, "frames": chunk, "fps": fps,
                        "t0": stamps[s], "t1": min(item.duration, stamps[s + len(chunk) - 1] + 1.0 / fps),
                        "grid": make_grid(chunk, s)})
        return out

    def scan(self, item: VideoItem, win: Dict[str, Any]) -> Dict[str, Any]:
        prompt = SCANNER_PROMPT.format(t0=win["t0"], t1=win["t1"], **item.game_text())
        raw = self.backbone.generate_image([win["grid"]], prompt, self._gen(self.spec.scan_tokens, item.name, "scan", win["index"]))
        parsed = extract_json_object(raw) or {}
        rng = parsed.get("frame_range") if isinstance(parsed.get("frame_range"), list) else []
        first = int(_as_float(rng[0], win["first"])) if rng else win["first"]
        last = int(_as_float(rng[-1], win["first"] + len(win["frames"]) - 1)) if rng else win["first"] + len(win["frames"]) - 1
        return {"has_glitch": bool(parsed.get("has_glitch")), "visual_cues": str(parsed.get("visual_cues", "")),
                "reasoning": str(parsed.get("reasoning", "")), "game_context": str(parsed.get("game_context", "")),
                "confidence": _as_float(parsed.get("confidence"), 0.5), "first": first, "last": max(first, last)}

    def memory(self, item: VideoItem, scans: Sequence[Dict[str, Any]]) -> str:
        notes = "\n".join(f"- {s['game_context']}" for s in scans if s["game_context"]) or "- (none)"
        return self.backbone.generate_text(MEMORY_PROMPT.format(game=item.game, notes=notes),
                                           self._gen(self.spec.scan_tokens, item.name, "memory"))

    def _execute(self, tool: Dict[str, Any], win: Dict[str, Any], key: Tuple[Any, ...]) -> str:
        name = str(tool.get("tool", "vqa"))
        question = str(tool.get("question") or "Is there a software glitch in these frames?")
        raw_box = tool.get("bbox")
        bbox = raw_box if isinstance(raw_box, list) and len(raw_box) == 4 else None
        if name == "object_tracking" and bbox:
            track, diag, _ = ncc_track(win["frames"], bbox)
            flags = physics_flags(track, diag, win["fps"], pixel=1.0)
            return f"object_tracking: {len(track)} positions; " + ("; ".join(flags) or "no physics rule violated")
        image = self._zoom(win, bbox) if name == "zoom_in" and bbox else win["grid"]
        answer = self.backbone.generate_image([image], question, self._gen(self.spec.agent_tokens, *key, "exec"))
        return f"{name}({question}): {answer}"

    @staticmethod
    def _zoom(win: Dict[str, Any], bbox: Sequence[float]) -> Any:
        from PIL import Image

        crops = []
        for path in win["frames"]:
            arr = load_rgb(path)
            h, w = arr.shape[:2]
            x0, y0, x1, y1 = (float(np.clip(v, 0.0, 1.0)) for v in bbox)
            crop = arr[int(y0 * h):max(int(y0 * h) + 2, int(y1 * h)), int(x0 * w):max(int(x0 * w) + 2, int(x1 * w))]
            crops.append(np.asarray(Image.fromarray(crop).resize((w, h), Image.BICUBIC)))
        tmp = [Image.fromarray(c) for c in crops]
        canvas = Image.new("RGB", (tmp[0].width * 4, tmp[0].height * 2))
        for i, im in enumerate(tmp[:8]):
            canvas.paste(im, ((i % 4) * im.width, (i // 4) * im.height))
        return canvas

    def _debate(self, item: VideoItem, win: Dict[str, Any], memory: str, evidence: str, key: Tuple[Any, ...]
                ) -> Dict[str, Any]:
        fields = {"game": item.game, "memory": memory, "evidence": evidence}
        advocate = self.backbone.generate_text(ADVOCATE_PROMPT.format(**fields), self._gen(self.spec.agent_tokens, *key, "adv"))
        skeptic = self.backbone.generate_text(SKEPTIC_PROMPT.format(**fields), self._gen(self.spec.agent_tokens, *key, "skp"))
        raw = self.backbone.generate_image([win["grid"]], JUDGE_AGENT_PROMPT.format(advocate=advocate, skeptic=skeptic, **fields),
                                           self._gen(self.spec.agent_tokens, *key, "judge"))
        parsed = extract_json_object(raw) or {}
        return {"glitch": str(parsed.get("verdict", "")).lower().startswith("glitch"),
                "confidence": float(np.clip(_as_float(parsed.get("confidence"), 0.0), 0.0, 1.0)),
                "description": str(parsed.get("description", "")).strip()}

    def verify(self, item: VideoItem, win: Dict[str, Any], scan: Dict[str, Any], memory: str) -> Dict[str, Any]:
        evidence = [f"scanner: {scan['visual_cues']} ({scan['reasoning']})"]
        verdict: Dict[str, Any] = {"glitch": False, "confidence": 0.0, "description": "", "rounds": 0}
        for rnd in range(self.spec.max_rounds):
            key = (item.name, "verify", win["index"], rnd)
            plan = PLANNER_PROMPT.format(game=item.game, memory=memory, first=scan["first"], last=scan["last"],
                                         t0=win["t0"], t1=win["t1"], cues=scan["visual_cues"],
                                         reasoning=scan["reasoning"], evidence="\n".join(evidence))
            tool = extract_json_object(self.backbone.generate_text(plan, self._gen(self.spec.agent_tokens, *key, "plan"))) or {}
            evidence.append(self._execute(tool, win, key))
            verdict = {**self._debate(item, win, memory, "\n".join(evidence), key), "rounds": rnd + 1}
            if verdict["confidence"] >= self.spec.stop_conf:
                break
        verdict["description"] = verdict["description"] or scan["visual_cues"]
        return verdict

    def same(self, item: VideoItem, a: str, b: str, key: Tuple[Any, ...]) -> bool:
        answer = self.backbone.generate_text(SAME_GLITCH_PROMPT.format(a=a, b=b),
                                             self._gen(16, item.name, "same", *key))
        return answer.strip().lower().startswith("y")

    def _extend(self, item: VideoItem, group: Dict[str, Any], scans: Sequence[Dict[str, Any]]) -> None:
        for direction in (1, -1):
            cursor = (max(group["windows"]) if direction > 0 else min(group["windows"])) + direction
            while 0 <= cursor < len(scans) and cursor not in group["windows"]:
                if not self.same(item, group["description"], scans[cursor]["visual_cues"], ("ext", group["id"], cursor)):
                    break
                group["windows"].add(cursor)
                cursor += direction

    def ground(self, item: VideoItem, confirmed: Dict[int, Dict[str, Any]], scans: Sequence[Dict[str, Any]]
               ) -> List[Dict[str, Any]]:
        groups: List[Dict[str, Any]] = []
        for idx in sorted(confirmed):
            text = confirmed[idx]["description"]
            target = next((g for g in groups if self.same(item, text, g["description"], ("grp", idx, g["id"]))), None)
            if target is None:
                groups.append({"id": len(groups), "description": text, "windows": {idx}, "fragments": [text]})
            else:
                target["windows"].add(idx)
                target["fragments"].append(text)
        for group in groups:
            self._extend(item, group, scans)
        return groups

    @staticmethod
    def group_intervals(group: Dict[str, Any], wins: Sequence[Dict[str, Any]], scans: Sequence[Dict[str, Any]],
                        duration: float) -> List[Interval]:
        ivs = []
        runs = consecutive_runs(sorted(group["windows"]))
        for run in runs:
            first, last = wins[run[0]], wins[run[-1]]
            fps = first["fps"]
            a = first["t0"] + max(0, scans[run[0]]["first"] - first["first"]) / fps
            b = last["t0"] + (min(len(last["frames"]) - 1, scans[run[-1]]["last"] - last["first"]) + 1) / fps
            ivs.append((a, b) if b > a else (first["t0"], last["t1"]))
        return clip_intervals(ivs, duration)

    def summarise(self, item: VideoItem, groups: List[Dict[str, Any]], wins: Sequence[Dict[str, Any]],
                  scans: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        payload = [{"description": " / ".join(dict.fromkeys(g["fragments"])),
                    "intervals": [[round(a, 1), round(b, 1)] for a, b in self.group_intervals(g, wins, scans, item.duration)]}
                   for g in groups]
        raw = self.backbone.generate_text(SUMMARISER_PROMPT.format(video=item.name, game=item.game,
                                                                   groups=json.dumps(payload, ensure_ascii=False)),
                                          self._gen(self.spec.scan_tokens, item.name, "summary"))
        events = parse_report(raw, item.duration)
        if not events and payload:
            # Unparseable summary: fall back to the grounded groups themselves.
            events = [{"description": p["description"], "intervals": clip_intervals(normalise_intervals(p["intervals"]), item.duration)}
                      for p in payload]
        return [e for e in events if e["intervals"]]

    def run(self, item: VideoItem) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        wins = self.windows(item)
        scans = [self.scan(item, w) for w in wins]
        memory = self.memory(item, scans)
        verdicts = {w["index"]: self.verify(item, w, s, memory) for w, s in zip(wins, scans, strict=True) if s["has_glitch"]}
        confirmed = {k: v for k, v in verdicts.items() if v["glitch"]}
        groups = self.ground(item, confirmed, scans)
        events = self.summarise(item, groups, wins, scans) if groups else []
        stats = {"windows": len(wins), "flagged": len(verdicts), "confirmed": len(confirmed),
                 "rounds": sum(v["rounds"] for v in verdicts.values()), "groups": len(groups)}
        return events, stats


# =========================================================================== #
# 9. commands
# =========================================================================== #


def _device() -> Any:
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def backbone_spec(args: argparse.Namespace) -> BackboneSpec:
    return BackboneSpec(model_id=args.model, revision=args.revision, dtype=args.dtype, load_in_4bit=args.load_in_4bit,
                        attn=args.attn, max_pixels=args.max_pixels, min_pixels=args.min_pixels,
                        layer_fracs=tuple(args.layer_fracs), trust_remote_code=args.trust_remote_code)


def fairness_fingerprint(args: argparse.Namespace) -> Dict[str, Any]:
    """Everything that must be identical between systems being compared."""
    return {**backbone_spec(args).fingerprint(), "fps": DEFAULT_FPS, "video_frames": args.video_frames}


def _fold(args: argparse.Namespace) -> Dict[str, Any]:
    return read_json(Path(args.run_dir) / "splits.json")["folds"][args.fold]


def _fold_dir(args: argparse.Namespace) -> Path:
    path = safe_path(Path(args.run_dir) / f"fold{args.fold}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _names(args: argparse.Namespace, part: str) -> List[str]:
    names = _fold(args)[part]
    return names[: args.limit] if getattr(args, "limit", None) else names


# --------------------------------------------------------------------------- #
# data commands
# --------------------------------------------------------------------------- #


def cmd_preprocess(args: argparse.Namespace) -> None:
    by_name, by_stem = build_video_index(Path(args.video_root))
    rows = load_metadata_rows(args.hf_repo, args.split, args.metadata, args.dataset_revision)
    records = [r for r in (record_from_row(row, by_name, by_stem) for row in rows) if r is not None]
    LOGGER.info("%d/%d metadata rows joined to local videos", len(records), len(rows))
    job = CacheJob(str(Path(args.work_dir) / "frames"), DEFAULT_FPS, args.short_side, args.jpeg_quality, args.overwrite)
    videos, failures = cache_all(records[: args.limit] if args.limit else records, job, args.workers)
    videos.sort(key=lambda r: (r["game_name"], r["video_name"]))
    write_jsonl(Path(args.work_dir) / VIDEOS_FILE, videos)
    write_jsonl(Path(args.work_dir) / "preprocess_failures.jsonl", failures)
    LOGGER.info("preprocess: %d videos cached, %d failures", len(videos), len(failures))


def cmd_prepare_generic(args: argparse.Namespace) -> None:
    """Any temporally annotated anomaly dataset (UCF-Crime, XD-Violence, MSAD,
    ...): manifest JSONL rows {video_path, game_name (scene id), genre,
    subgenre, no_bugs, bugs, time_nodes}."""
    rows = read_jsonl(args.manifest)
    by_name = {Path(r["video_path"]).name: safe_path(r["video_path"]) for r in rows}
    records = [r for r in (record_from_row({**row, "video_name": Path(row["video_path"]).name}, by_name, {})
                           for row in rows) if r is not None]
    job = CacheJob(str(Path(args.work_dir) / "frames"), DEFAULT_FPS, args.short_side, args.jpeg_quality, args.overwrite)
    videos, failures = cache_all(records, job, args.workers)
    write_jsonl(Path(args.work_dir) / VIDEOS_FILE, videos)
    write_jsonl(Path(args.work_dir) / "preprocess_failures.jsonl", failures)


def cmd_split(args: argparse.Namespace) -> None:
    write_json(Path(args.run_dir) / "splits.json", make_folds(load_corpus(args.work_dir), args.k, args.dev_frac, args.seed))


def cmd_extract(args: argparse.Namespace) -> None:
    items = load_corpus(args.work_dir)
    backbone = build_backbone(backbone_spec(args))
    names = sorted(items)[: args.limit] if args.limit else sorted(items)
    extract_corpus(items, backbone, ExtractSpec(args.feat_dir, DEFAULT_FPS, args.chunk_frames, args.proj_dim,
                                                args.seed, args.overwrite), names, fairness_fingerprint(args))


# --------------------------------------------------------------------------- #
# NORMA head
# --------------------------------------------------------------------------- #


def head_samples(items: Dict[str, VideoItem], feats: Dict[str, Dict[str, np.ndarray]], names: Sequence[str],
                 bank: NormalityBank, exclude_own_game: bool) -> List[Dict[str, Any]]:
    out = []
    for name in (n for n in names if n in feats):
        item, f = items[name], feats[name]
        events = gold_events(item)
        out.append({"name": name, "game": item.game, "duration": item.duration, "feats": f["feats"], "ctx": f["ctx"],
                    "times": f["times"], "bank": bank.distance(f["feats"], item.game if exclude_own_game else None),
                    "count": min(HeadConfig.k_max, len(events)), **unit_targets(f["times"], events)})
    return out


def _fit_bank(items: Dict[str, VideoItem], feats: Dict[str, Dict[str, np.ndarray]], names: Sequence[str],
              args: argparse.Namespace) -> NormalityBank:
    normal = []
    for name in (n for n in names if n in feats):
        keep = unit_targets(feats[name]["times"], gold_events(items[name]))["y"] < 0.5
        if keep.any():
            normal.append((feats[name]["feats"][keep], items[name].game))
    bank = NormalityBank(args.bank_pca, args.bank_size, args.bank_k, args.seed)
    bank.fit(normal)
    return bank


def _decode_all(raw: Sequence[Dict[str, np.ndarray]], samples: Sequence[Dict[str, Any]], prm: DecodeParams,
                trans: Transitions) -> Dict[str, List[Dict[str, Any]]]:
    return {s["name"]: decode_video(r, s["times"], prm, trans, float(s["duration"]))
            for s, r in zip(samples, raw, strict=True)}


def _selector(dev: Sequence[Dict[str, Any]], trans: Transitions, prm: DecodeParams, device: Any,
              gold: Dict[str, List[Dict[str, Any]]]) -> Callable[[Any], float]:
    labels = np.concatenate([s["y"] for s in dev])

    def select(model: Any) -> float:
        # Primary: judge-free temporal F1xIoU on dev; tie-break: unit AUROC
        # (so epochs in which the default decoder emits nothing still rank).
        raw = run_head(model, dev, device)
        unit_auc = auroc(np.concatenate([r["act"] for r in raw]), labels)
        tie = 0.0 if math.isnan(unit_auc) else unit_auc
        return temporal_proxy(_decode_all(raw, dev, prm, trans), gold)["f1_x_iou"] + tie

    return select


def _tune_decoder(raw: Sequence[Dict[str, np.ndarray]], dev: Sequence[Dict[str, Any]], base: DecodeParams,
                  trans: Transitions, gold: Dict[str, List[Dict[str, Any]]]) -> Tuple[DecodeParams, float]:
    sims = [base.group_sim] if base.group_sim > 1.0 else [0.5, 0.65, 0.8, 0.9, 2.0]
    best, best_prm = -1.0, base
    for none_t, ev_t, gsim in itertools.product([0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [0.2, 0.3, 0.4, 0.5, 0.6], sims):
        prm = dataclasses.replace(base, none_threshold=none_t, event_threshold=ev_t, group_sim=gsim)
        score = temporal_proxy(_decode_all(raw, dev, prm, trans), gold)["f1_x_iou"]
        if score > best:
            best, best_prm = score, prm
    return best_prm, best


def cmd_train_head(args: argparse.Namespace) -> None:
    items, fold, fdir, device = load_corpus(args.work_dir), _fold(args), _fold_dir(args), _device()
    t0 = time.perf_counter()
    feats = load_features(args.feat_dir, items, fold["train"] + fold["dev"])
    bank = _fit_bank(items, feats, fold["train"], args)
    sample = next(iter(feats.values()))["feats"]
    cfg = HeadConfig(n_vectors=sample.shape[1], in_dim=sample.shape[2], d_model=args.d_model, n_layers=args.n_layers,
                     n_heads=args.n_heads, use_srn=not args.no_srn, use_bank=not args.no_bank,
                     use_delta=not args.no_delta, use_ctx=not args.no_ctx)
    train = head_samples(items, feats, fold["train"], bank, True)
    dev = head_samples(items, feats, fold["dev"], bank, False)
    trans = fit_transitions([s["y"] for s in train])
    gold = {s["name"]: gold_events(items[s["name"]]) for s in dev}
    base = DecodeParams(group_sim=2.0 if args.no_group else 0.75)
    model, history = train_head(train, cfg, TrainSpec(args.epochs, args.lr, args.batch_size, args.seed), device,
                                _selector(dev, trans, base, device, gold))
    raw = run_head(model, dev, device)
    logits, labels = np.concatenate([r["act"] for r in raw]), np.concatenate([s["y"] for s in dev])
    prm, dev_score = _tune_decoder(raw, dev, dataclasses.replace(base, temperature=fit_temperature(logits, labels)),
                                   trans, gold)
    report = {"config": dataclasses.asdict(cfg), "transitions": dataclasses.asdict(trans),
              "decode": dataclasses.asdict(prm), "dev_unit_auroc": auroc(logits, labels),
              "dev_unit_ap": average_precision(logits, labels), "dev_temporal_f1_x_iou": dev_score,
              "history": history, "train_seconds": time.perf_counter() - t0, "fold": args.fold}
    save_head(fdir, args.tag_suffix, model, report, bank)
    LOGGER.info("fold %d head: dev unit AUROC %.4f, temporal F1xIoU %.2f, %s", args.fold, report["dev_unit_auroc"],
                dev_score, prm)


def cmd_train_desc(args: argparse.Namespace) -> None:
    items, fold, fdir = load_corpus(args.work_dir), _fold(args), _fold_dir(args)
    backbone = build_backbone(backbone_spec(args))
    if args.direct:
        train, dev = direct_samples(items, fold["train"]), direct_samples(items, fold["dev"])
        spec = LMSpec(str(fdir / "direct"), args.desc_epochs, args.desc_lr, args.grad_accum, args.video_frames,
                      0.0, args.lora_r, args.lora_alpha, args.seed)
    else:
        train = describer_samples(items, fold["train"], args.seed, args.neg_ratio)
        dev = describer_samples(items, fold["dev"], args.seed + 1, args.neg_ratio)
        spec = LMSpec(str(fdir / "describer"), args.desc_epochs, args.desc_lr, args.grad_accum, args.desc_frames,
                      args.margin, args.lora_r, args.lora_alpha, args.seed)
    train_lm(backbone, items, train, dev, spec)


# --------------------------------------------------------------------------- #
# predictions (shared row format + per-file meta)
# --------------------------------------------------------------------------- #


def report_row(item: VideoItem, events: Sequence[Dict[str, Any]], cost: Dict[str, float],
               extra: Dict[str, Any]) -> Dict[str, Any]:
    """GliDe summariser schema + diagnostics. Identical for every system."""
    clean = [{"description": e["description"], "intervals": clip_intervals(e["intervals"], item.duration)}
             for e in events if str(e.get("description", "")).strip()]
    clean = [e for e in clean if e["intervals"]]
    return {"video_name": item.name, "game_name": item.game, "genre": item.genre, "duration": item.duration,
            "no_bugs": not clean, "bugs": [e["description"] for e in clean],
            "time_nodes": [[list(iv) for iv in e["intervals"]] for e in clean], "cost": cost, **extra}


def write_predictions(args: argparse.Namespace, tag: str, rows: List[Dict[str, Any]], backbone: Optional[Backbone],
                      extra_meta: Dict[str, Any]) -> None:
    write_jsonl(Path(args.run_dir) / f"preds_{tag}_fold{args.fold}.jsonl", rows)
    meta = {"tag": tag, "fold": args.fold, "seed": args.seed, "fairness": fairness_fingerprint(args),
            "n_videos": len(rows), "peak_gpu_mb": peak_gpu_mb(),
            "cost_by_kind": backbone.meter.by_kind() if backbone is not None else {}, **extra_meta}
    write_json(Path(args.run_dir) / f"preds_{tag}_fold{args.fold}.meta.json", meta)
    LOGGER.info("%s fold %d: %d videos, %d events", tag, args.fold, len(rows), sum(len(r["bugs"]) for r in rows))


def _describe_events(events: List[Dict[str, Any]], item: VideoItem, describer: Optional[Describer]) -> List[Dict[str, Any]]:
    for ev in events:
        ev.update(describer.describe(item, ev["intervals"]) if describer is not None
                  else {"description": "a software glitch occurs", "p_reject": 0.5})
    # Keep `events` index-aligned with the bugs/time_nodes report_row emits.
    return [e for e in events if e["description"].strip() and clip_intervals(e["intervals"], item.duration)]


def _norma_rows(args: argparse.Namespace, items: Dict[str, VideoItem], part: str, backbone: Optional[Backbone]
                ) -> List[Dict[str, Any]]:
    device = _device()
    model, _, bank, trans, prm = load_head(_fold_dir(args), args.tag_suffix, device)
    names = _names(args, part)
    samples = head_samples(items, load_features(args.feat_dir, items, names), names, bank, False)
    raw = run_head(model, samples, device)
    describer = None if backbone is None else Describer(backbone, DescribeSpec(
        args.desc_frames, args.margin, args.mbr_samples, args.mbr_temperature, args.max_new_tokens, args.seed))
    meter = backbone.meter if backbone is not None else CostMeter()
    rows = []
    for s, r in zip(samples, raw, strict=True):
        item, before = items[s["name"]], meter.snapshot()
        events = _describe_events(decode_video(r, s["times"], prm, trans, item.duration), item, describer)
        rows.append(report_row(item, events, CostMeter.delta(meter.snapshot(), before), {
            "events": [{k: v for k, v in e.items() if k != "description"} for e in events],
            "unit_p": np.round(sigmoid(r["act"] / prm.temperature), 4).tolist(),
            "unit_times": np.round(s["times"], 3).tolist()}))
    return rows


def _fusion_data(rows: List[Dict[str, Any]], gold_map: Dict[str, List[Dict[str, Any]]]) -> Tuple[np.ndarray, np.ndarray]:
    pairs = [(event_features(ev), event_is_tp(ev["intervals"], gold_map[row["video_name"]]))
             for row in rows for ev in row["events"]]
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs], dtype=float)


def _best_threshold(rows: List[Dict[str, Any]], weights: np.ndarray, gold_map: Dict[str, List[Dict[str, Any]]]
                    ) -> Tuple[float, float]:
    scores = []
    for thr in np.linspace(0.0, 0.9, 19):
        preds = {r["video_name"]: [{"intervals": e["intervals"]} for e in r["events"]
                                   if fused_probability(e, weights) >= thr] for r in rows}
        scores.append((temporal_proxy(preds, gold_map)["f1_x_iou"], float(thr)))
    return max(scores, key=lambda st: (st[0], -st[1]))


def _fit_fusion(rows: List[Dict[str, Any]], items: Dict[str, VideoItem]) -> Dict[str, Any]:
    gold_map = {r["video_name"]: gold_events(items[r["video_name"]]) for r in rows}
    x, y = _fusion_data(rows, gold_map)
    weights = fit_logreg(x, y) if len(np.unique(y)) == 2 else np.array([1.0, 0.0, 0.0, 0.0])
    best, best_t = _best_threshold(rows, weights, gold_map)
    LOGGER.info("fusion on %d dev events (%.1f%% TP): w=%s, threshold %.2f, dev tF1xIoU %.2f",
                len(y), 100.0 * float(np.mean(y)) if y.size else 0.0, np.round(weights, 3).tolist(), best_t, best)
    return {"weights": [float(w) for w in weights], "threshold": best_t, "dev_temporal_f1_x_iou": best,
            "n_dev_events": len(y)}


def _apply_fusion(rows: List[Dict[str, Any]], fusion: Optional[Dict[str, Any]]) -> None:
    if fusion is None:
        return
    for row in rows:
        keep = [i for i, ev in enumerate(row["events"]) if fused_probability(ev, fusion["weights"]) >= fusion["threshold"]]
        row["bugs"] = [row["bugs"][i] for i in keep]
        row["time_nodes"] = [row["time_nodes"][i] for i in keep]
        row["events"] = [row["events"][i] for i in keep]
        row["no_bugs"] = not keep


def cmd_predict(args: argparse.Namespace) -> None:
    items, fdir = load_corpus(args.work_dir), _fold_dir(args)
    backbone = None
    if not args.no_describer:
        backbone = build_backbone(backbone_spec(args))
        backbone.load_adapter(fdir / "describer")
    fusion = None
    if not args.no_fusion:
        fusion_path = fdir / f"fusion_{args.tag}.json"
        fusion = read_json(fusion_path, required=False)
        if fusion is None:
            fusion = _fit_fusion(_norma_rows(args, items, "dev", backbone), items)
            write_json(fusion_path, fusion)
    rows = _norma_rows(args, items, "test", backbone)
    _apply_fusion(rows, fusion)
    head = read_json(fdir / f"head{args.tag_suffix}.json")
    train_meta = {"train_seconds": {"head": head.get("train_seconds", 0.0),
                                    "describer": (read_json(fdir / "describer" / TRAIN_SUMMARY, required=False)
                                                  or {}).get("train_seconds", 0.0)},
                  "extract_stats": read_json(Path(args.feat_dir) / "extract_stats.json", required=False)}
    write_predictions(args, args.tag, rows, backbone, {"system": "norma", **train_meta})


def cmd_baseline(args: argparse.Namespace) -> None:
    """vanilla: off-the-shelf backbone, whole video, report JSON. direct: the
    same after LoRA-SFT on the training folds. glide: the reproduction."""
    items, fdir = load_corpus(args.work_dir), _fold_dir(args)
    backbone = build_backbone(backbone_spec(args))
    if args.mode == "direct":
        backbone.load_adapter(fdir / "direct")
    runner = GlideRunner(backbone, GlideSpec(seed=args.seed)) if args.mode == "glide" else None
    rows = []
    for name in _names(args, "test"):
        item, before = items[name], backbone.meter.snapshot()
        if runner is not None:
            events, stats = runner.run(item)
        else:
            frames, fps, prompt = sample_io(item, None, args.video_frames, 0.0)
            text = backbone.generate_video(frames, fps, prompt, GenSpec(args.report_tokens, 0.0, 1, None, args.seed))[0]
            events, stats = parse_report(text, item.duration), {}
        rows.append(report_row(item, events, CostMeter.delta(backbone.meter.snapshot(), before), {"stats": stats}))
    train = (read_json(fdir / "direct" / TRAIN_SUMMARY, required=False) or {}) if args.mode == "direct" else {}
    write_predictions(args, args.tag or args.mode, rows, backbone,
                      {"system": args.mode, "train_seconds": {"lm": train.get("train_seconds", 0.0)}})


# --------------------------------------------------------------------------- #
# evaluation, comparison, cost
# --------------------------------------------------------------------------- #


def _prediction_files(run_dir: Path, tag: str) -> List[Path]:
    files = sorted(safe_path(run_dir).glob(f"preds_{tag}_fold*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no preds_{tag}_fold*.jsonl under {run_dir}")
    return files


def load_predictions(run_dir: Path, tag: str) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    rows: Dict[str, Dict[str, Any]] = {}
    metas = []
    for path in _prediction_files(run_dir, tag):
        rows.update({r["video_name"]: r for r in read_jsonl(path)})
        metas.append(read_json(path.with_suffix(".meta.json"), required=False) or {})
    fingerprints = {json.dumps(m.get("fairness"), sort_keys=True) for m in metas}
    if len(fingerprints) > 1:
        raise RuntimeError(f"{tag}: fold files were produced under different settings: {fingerprints}")
    return rows, metas


def _score_tag(run_dir: Path, tag: str, items: Dict[str, VideoItem], judge: Judge
               ) -> Tuple[List[VideoScore], np.ndarray, np.ndarray]:
    preds, _ = load_predictions(run_dir, tag)
    scores, unit_s, unit_y = [], [], []
    for name in sorted(preds):
        row, item = preds[name], items[name]
        pred = [{"description": d, "intervals": normalise_intervals(t)}
                for d, t in zip(row.get("bugs") or [], row.get("time_nodes") or [], strict=True)]
        gold = gold_events(item)
        scores.append(match_video(pred, gold, judge, name, item.game))
        if row.get("unit_p"):
            unit_s.extend(row["unit_p"])
            unit_y.extend(unit_targets(np.array(row["unit_times"], dtype=np.float32), gold)["y"].tolist())
    return scores, np.array(unit_s), np.array(unit_y)


def _evaluation_payload(tag: str, scores: List[VideoScore], units: Tuple[np.ndarray, np.ndarray],
                        items: Dict[str, VideoItem], judge: Judge, args: argparse.Namespace) -> Dict[str, Any]:
    unit_s, unit_y = units
    genres = defaultdict(list)
    for s in scores:
        genres[items[s.video_name].genre or "unknown"].append(s)
    return {"tag": tag, "n_videos": len(scores), "videos_hash": stable_hash("|".join(s.video_name for s in scores)),
            "judge": judge.describe(), "micro": {**aggregate(scores), **recall_at_tiou(scores, (0.3, 0.5, 0.7))},
            "ci_game_cluster": {m: bootstrap_ci(scores, m, args.n_boot, args.seed) for m in ("f1", "miou", "f1_x_iou")},
            "per_genre": {g: aggregate(v) for g, v in sorted(genres.items())},
            "unit_auroc": auroc(unit_s, unit_y) if unit_s.size else None,
            "unit_ap": average_precision(unit_s, unit_y) if unit_s.size else None}


def cmd_evaluate(args: argparse.Namespace) -> None:
    items = load_corpus(args.work_dir)
    run_dir = Path(args.run_dir)
    judge = Judge(args.judge_backend, args.judge_model,
                  run_dir / f"judge_cache_{args.judge_backend}_{re.sub(r'[^A-Za-z0-9._-]+', '_', args.judge_model)}.jsonl",
                  base_url=args.judge_base_url, api_key_env=args.api_key_env)
    for tag in args.tags:
        scores, unit_s, unit_y = _score_tag(run_dir, tag, items, judge)
        payload = _evaluation_payload(tag, scores, (unit_s, unit_y), items, judge, args)
        micro = payload["micro"]
        write_json(run_dir / f"eval_{tag}.json", payload)
        write_jsonl(run_dir / f"eval_{tag}.per_video.jsonl", [dataclasses.asdict(s) for s in scores])
        LOGGER.info("%s | P %.2f R %.2f F1 %.2f | mIoU %.3f | F1xIoU %.2f | n=%d | judge=%s", tag, micro["precision"],
                    micro["recall"], micro["f1"], micro["miou"], micro["f1_x_iou"], len(scores), args.judge_model)
    if args.judge_backend == "offline":
        LOGGER.warning("offline judge (unigram F1): numbers are NOT comparable to GliDe Table 3")


def _parse_system(spec: str) -> Tuple[str, List[str]]:
    name, _, tags = spec.partition("=")
    return name, [t for t in (tags or name).split(",") if t]


def _mean_scores(run_dir: Path, tags: Sequence[str]) -> List[VideoScore]:
    """Per-video sums averaged over runs (seeds): the expected score of a
    stochastic system, so sampling noise is integrated, not cherry-picked."""
    runs = [{r["video_name"]: VideoScore(**r) for r in read_jsonl(run_dir / f"eval_{t}.per_video.jsonl")} for t in tags]
    common = sorted(set.intersection(*(set(r) for r in runs)))
    out = []
    for name in common:
        sums = dict(zip(SUM_KEYS, (float(v) for v in np.mean([r[name].sums() for r in runs], axis=0)), strict=True))
        out.append(VideoScore(video_name=name, game=runs[0][name].game, n_pred=sums["n_pred"], n_gold=sums["n_gold"],
                              n_matched=sums["n_matched"], sum_s=sums["sum_s"], sum_w=sums["sum_w"],
                              sum_iou=sums["sum_iou"], ious=[], pairs=[]))
    return out


def _check_fair(run_dir: Path, systems: Dict[str, List[str]]) -> None:
    seen: Dict[str, Tuple[str, Any]] = {}
    for name, tags in systems.items():
        for tag in tags:
            evaluation = read_json(run_dir / f"eval_{tag}.json")
            _, metas = load_predictions(run_dir, tag)
            facts = {"fairness": json.dumps(metas[0].get("fairness"), sort_keys=True),
                     "videos": evaluation["videos_hash"], "judge": evaluation["judge"]["model"]}
            for key, value in facts.items():
                if key in seen and seen[key][1] != value:
                    raise RuntimeError(f"unfair comparison: {key} of {name}/{tag} differs from {seen[key][0]}")
                seen.setdefault(key, (f"{name}/{tag}", value))


def _run_summary(run_dir: Path, tags: Sequence[str]) -> str:
    vals = {m: [read_json(run_dir / f"eval_{t}.json")["micro"][m] for t in tags] for m in ("f1", "miou", "f1_x_iou")}
    return " | ".join(f"{np.mean(v):.2f} +- {np.std(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}" for v in vals.values())


def cmd_compare(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    systems = dict(_parse_system(s) for s in args.systems)
    if not args.allow_mismatch:
        _check_fair(run_dir, systems)
    proposed, *baselines = list(systems)
    scores = {name: _mean_scores(run_dir, tags) for name, tags in systems.items()}
    lines = ["| system | runs | F1 | mIoU | F1xIoU |", "|---|---|---|---|---|"]
    lines += [f"| {n} | {len(t)} | {_run_summary(run_dir, t)} |" for n, t in systems.items()]
    rows, raw_p = {}, {}
    for base in baselines:
        for metric in ("f1", "miou", "f1_x_iou"):
            key = f"{proposed} vs {base} | {metric}"
            rows[key] = paired_test(scores[proposed], scores[base], metric, args.n_boot, args.seed)
            raw_p[key] = rows[key]["p"]
    corrected = holm(raw_p, args.alpha)
    lines += ["", "| comparison | metric | delta | 95% CI | p | p (Holm) | reject |", "|---|---|---|---|---|---|---|"]
    for key, r in rows.items():
        comp, metric = key.split(" | ")
        c = corrected[key]
        lines.append(f"| {comp} | {metric} | {r['delta']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | "
                     f"{r['p']:.4f} | {c['p_holm']:.4f} | {'yes' if c['reject'] else 'no'} |")
    text = "\n".join(lines)
    safe_path(run_dir / f"compare_{proposed}.md").write_text(text + "\n", encoding=UTF8)
    write_json(run_dir / f"compare_{proposed}.json", {k: {**rows[k], **corrected[k]} for k in rows})
    print(text)


def _cost_line(run_dir: Path, name: str, tags: Sequence[str]) -> str:
    rows: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []
    for tag in tags:
        preds, m = load_predictions(run_dir, tag)
        rows.extend(preds.values())
        metas.extend(m)
    per_video = {k: float(np.mean([r["cost"].get(k, 0.0) for r in rows])) for k in ("calls", "in_tokens", "out_tokens", "seconds")}
    extract = next((m.get("extract_stats") for m in metas if m.get("extract_stats")), None) or {}
    if extract.get("videos"):
        per_video["calls"] += extract.get("calls", 0.0) / extract["videos"]
        per_video["in_tokens"] += extract.get("in_tokens", 0.0) / extract["videos"]
        per_video["seconds"] += extract.get("seconds", 0.0) / extract["videos"]
    train_s = sum(sum((m.get("train_seconds") or {}).values()) for m in metas) / max(len(tags), 1)
    peak = max((m.get("peak_gpu_mb", 0.0) for m in metas), default=0.0)
    return (f"| {name} | {per_video['calls']:.1f} | {per_video['in_tokens']:.0f} | {per_video['out_tokens']:.0f} | "
            f"{per_video['seconds']:.2f} | {train_s / 3600:.2f} | {peak:.0f} |")


def cmd_cost(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    lines = ["| system | model calls / video | input tokens / video | output tokens / video | seconds / video | "
             "training GPU-h (all folds) | peak GPU MB |", "|---|---|---|---|---|---|---|"]
    lines += [_cost_line(run_dir, *_parse_system(s)) for s in args.systems]
    text = "\n".join(lines)
    safe_path(run_dir / "cost.md").write_text(text + "\n", encoding=UTF8)
    print(text)


# =========================================================================== #
# 10. self-tests
# =========================================================================== #


class SelfTest:
    def __init__(self) -> None:
        self.failures: List[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        if ok:
            LOGGER.info("PASS  %s", name)
        else:
            self.failures.append(f"{name}: {detail}")
            LOGGER.error("FAIL  %s %s", name, detail)


def _same(a: str, b: str) -> float:
    return 1.0 if a == b else 0.0


def _null_scores(rng: np.random.Generator) -> List[VideoScore]:
    return [VideoScore(f"v{i}", f"g{i % 20}", 1, 1, 1, float(rng.random()), float(rng.random() * 0.5), 0.5, [0.5], [])
            for i in range(200)]


def check_estimators(t: SelfTest) -> None:
    t.check("iou_multi", math.isclose(interval_iou([(0, 2), (4, 6)], [(1, 5)]), 2 / 6))
    t.check("merge", merge_intervals([(0, 1), (1, 2), (5, 6)]) == [(0, 2), (5, 6)])
    t.check("json_extract", extract_json_object('x ```json {"a": {"b": 1}, "c": "}"}``` y') == {"a": {"b": 1}, "c": "}"})
    pred = [{"description": "x", "intervals": [(0, 10)]}, {"description": "y", "intervals": [(20, 30)]}]
    m = aggregate([match_video(pred, pred[::-1], _same, "v", "g")])
    t.check("hungarian", math.isclose(m["f1"], 100.0) and math.isclose(m["f1_x_iou"], 100.0), json.dumps(m))
    fp = aggregate([match_video(pred[:1], [], _same)])
    t.check("fp_on_clean_video", math.isclose(fp["precision"], 0.0) and math.isclose(fp["n_pred"], 1.0))
    trans = Transitions(0.85, 0.05, 0.1, 0.2)
    path = viterbi(np.array([-3, -3, 4, 4, -3, -3, 8, -3.0]), trans)
    t.check("viterbi", path.tolist() == [0, 0, 1, 1, 0, 0, 1, 0], str(path))
    t.check("runs", runs_of(path) == [(2, 3), (6, 6)], str(runs_of(path)))
    rng = make_rng(0)
    x = rng.normal(size=(2000, 2))
    y = (rng.random(2000) < sigmoid(1.5 * x[:, 0] - 0.5)).astype(float)
    w = fit_logreg(x, y, l2=1e-4)
    t.check("logreg", abs(w[0] - 1.5) < 0.25 and abs(w[1]) < 0.2 and abs(w[2] + 0.5) < 0.2, str(w))
    t.check("auroc", math.isclose(auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])), 1.0)
            and math.isclose(auroc(np.array([0.5, 0.5]), np.array([0, 1])), 0.5))
    h = holm({"a": 0.001, "b": 0.02, "c": 0.9})
    t.check("holm", h["a"]["reject"] and not h["c"]["reject"] and h["a"]["p_holm"] <= h["b"]["p_holm"])
    t.check("mbr", mbr_select(["red car floats", "red car floats upward", "blue tree"]).startswith("red"))
    rejections = sum(paired_test(_null_scores(make_rng(100 + k)), _null_scores(make_rng(200 + k)), "f1", 400, k)["p"] < 0.05
                     for k in range(40))
    t.check("paired_test_size", rejections <= 6, f"{rejections}/40 false rejections at alpha=0.05")
    try:
        safe_path(os.path.join(os.sep, "etc", "passwd"))
        t.check("path_guard", False, "path outside the allowed roots was accepted")
    except PermissionError:
        t.check("path_guard", True)


def _teleport_frames(root: Path, n: int = 8) -> List[str]:
    from PIL import Image

    paths = []
    for i in range(n):
        img = np.full((90, 160, 3), 40, dtype=np.uint8)
        x = 20 + 3 * i if i != 4 else 130
        img[40:56, x:x + 16] = (230, 30, 30)
        path = root / f"track_{i}.jpg"
        Image.fromarray(img).save(path, quality=95)
        paths.append(str(path))
    return paths


def check_tracker_and_decoder(t: SelfTest, root: Path) -> None:
    import av

    frames = _teleport_frames(root)
    track, diag, _ = ncc_track(frames, [20 / 160, 40 / 90, 36 / 160, 56 / 90])
    flags = physics_flags(track, diag, DEFAULT_FPS)
    t.check("tracker_detects_teleport", any("position jump" in f for f in flags), str(flags))
    smooth = [f for i, f in enumerate(frames) if i != 4]
    flags = physics_flags(ncc_track(smooth, [20 / 160, 40 / 90, 36 / 160, 56 / 90])[0], diag, DEFAULT_FPS)
    t.check("tracker_quiet_on_smooth_motion", not any("jump" in f for f in flags), str(flags))
    video = root / "tiny.mp4"
    with av.open(str(video), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(30):
            arr = np.full((48, 64, 3), (i * 8) % 255, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    decoded, stamps, duration = decode_uniform_frames(video, DEFAULT_FPS, 48)
    t.check("decoder_grid", math.isclose(stamps[0], 0.0, abs_tol=1e-9) and len(decoded) == 12
            and abs(duration - 3.0) < 0.3, f"{len(decoded)} frames, stamps {stamps[:3]}, duration {duration}")


def check_qwen_modules(t: SelfTest) -> None:
    """LayerTap, early exit, unit pooling and LoRA placement against a REAL
    Qwen2.5-VL implementation (tiny random config, nothing downloaded)."""
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    except ImportError:
        LOGGER.warning("SKIP  qwen module checks (transformers / peft not installed)")
        return
    cfg = Qwen2_5_VLConfig(
        text_config={"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 6, "num_attention_heads": 4,
                     "num_key_value_heads": 2, "vocab_size": 152000,
                     "rope_scaling": {"type": "mrope", "mrope_section": [2, 3, 3]}},
        vision_config={"depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 64,
                       "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2,
                       "fullatt_block_indexes": [1], "window_size": 56})
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(cfg).eval()
    t_, h_, w_ = 3, 4, 6
    per = (h_ // 2) * (w_ // 2)
    ids = torch.tensor([[151644, cfg.vision_start_token_id] + [cfg.video_token_id] * (t_ * per)
                        + [cfg.vision_end_token_id, 100, 101]])
    inputs = {"input_ids": ids, "pixel_values_videos": torch.randn(t_ * h_ * w_, 3 * 2 * 14 * 14),
              "video_grid_thw": torch.tensor([[t_, h_, w_]]), "use_cache": False}
    layers = find_decoder_layers(model)
    calls: Counter = Counter()
    probe = layers[-1].register_forward_hook(lambda *_: calls.update(["deep"]))
    with torch.inference_mode():
        with LayerTap(layers, [2, 3]) as tap:
            model(**inputs)
        full = model(**inputs, output_hidden_states=True)
    probe.remove()
    t.check("qwen_early_exit", calls["deep"] == 1, f"deepest layer ran {calls['deep']} times (1 = the full pass only)")
    t.check("qwen_tap_equals_hidden_states", torch.allclose(tap.cache[3][0], full.hidden_states[4][0], atol=1e-5))
    pooled = qwen_unit_pool(tap.cache[3][0], ids[0], [t_, h_, w_], cfg.video_token_id, 2)
    block = tap.cache[3][0][ids[0] == cfg.video_token_id].view(t_, per, -1)[1].float()
    block = block / block.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    t.check("qwen_unit_pool", tuple(pooled.shape) == (t_, 2, 64) and torch.allclose(pooled[1, 0], block.mean(dim=0), atol=1e-5))
    peft_model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=QwenFamily.LORA_TARGETS,
                                                  task_type="CAUSAL_LM"))
    names = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    t.check("qwen_lora_language_model_only", len(names) == 6 * 7 * 2 and not any("visual" in n for n in names),
            f"{len(names)} trainable tensors")


def synthetic_corpus(root: Path, seed: int = 0) -> Path:
    """Stage-1 layout. Every game has its own background (a per-game nuisance
    a global threshold cannot absorb); a glitch is a short departure from the
    game's own dynamics (a teleporting sprite or a flash); recurring
    two-interval bugs and clean videos are mixed in."""
    from PIL import Image

    rng = make_rng(seed)
    work = root / "work"
    rows = []
    for g, v in itertools.product(range(6), range(8)):
        game = f"Game {g}"
        bg = rng.uniform(0.1, 0.9, size=3)
        name = f"g{g}_v{v}.mp4"
        spans, bugs = _synthetic_bugs(v % 4, rng)
        fdir = work / "frames" / game.replace(" ", "_") / Path(name).stem
        fdir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(40):
            img = _synthetic_frame(i / DEFAULT_FPS, i, bg, spans, bugs, rng)
            Image.fromarray(img).save(fdir / f"{i:06d}.jpg", quality=92)
            paths.append(str(fdir / f"{i:06d}.jpg"))
        marker = fdir / "_frames.json"
        marker.write_text(json.dumps({"frame_paths": paths, "timestamps": [i / DEFAULT_FPS for i in range(40)],
                                      "duration": 10.0, "target_fps": DEFAULT_FPS}), encoding=UTF8)
        rows.append({"video_name": name, "game_name": game, "genre": "Action" if g % 2 else "Simulation",
                     "subgenre": "Synthetic", "duration": 10.0, "no_bugs": not bugs, "bugs": bugs,
                     "bug_spans": spans, "frames_marker": str(marker)})
    write_jsonl(work / VIDEOS_FILE, rows)
    return work


def _synthetic_bugs(kind: int, rng: np.random.Generator) -> Tuple[List[List[List[float]]], List[str]]:
    spans: List[List[List[float]]] = []
    bugs: List[str] = []
    if kind in (1, 3):
        a = float(rng.uniform(1.0, 5.0))
        spans.append([[a, a + float(rng.uniform(1.5, 3.0))]])
        bugs.append(MOCK_TELEPORT)
    if kind == 2:
        a, b = float(rng.uniform(0.5, 2.5)), float(rng.uniform(6.0, 7.5))
        spans.append([[a, a + 1.5], [b, b + 1.5]])
        bugs.append("The red square repeatedly teleports across the screen.")
    if kind == 3:
        c = float(rng.uniform(7.0, 8.0))
        spans.append([[c, c + 1.5]])
        bugs.append("The whole scene flashes bright white.")
    return spans, bugs


def _synthetic_frame(t: float, i: int, bg: np.ndarray, spans: Sequence[Sequence[Sequence[float]]],
                     bugs: Sequence[str], rng: np.random.Generator) -> np.ndarray:
    def active(word: str) -> bool:
        return any(lo <= t < hi for k, s in enumerate(spans) for lo, hi in s if word in bugs[k])

    img = np.ones((64, 64, 3)) * bg + rng.normal(0, 0.02, size=(64, 64, 3))
    x, y = (float(rng.uniform(0, 50)), float(rng.uniform(0, 50))) if active("teleport") else ((10 + 1.5 * i) % 50, 30.0)
    img[int(y):int(y) + 12, int(x):int(x) + 12] = [0.9, 0.1, 0.1]
    if active("flashes"):
        img = img + 0.6
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _ns(base: Dict[str, Any], **kw: Any) -> argparse.Namespace:
    return argparse.Namespace(**{**base, **kw})


def run_e2e(t: SelfTest, root: Path) -> None:
    work = synthetic_corpus(root)
    base: Dict[str, Any] = {"work_dir": str(work), "run_dir": str(root / "run"), "feat_dir": str(root / "feats"), "seed": 0,
            "model": "mock", "revision": None, "dtype": "bfloat16", "load_in_4bit": False, "attn": "sdpa",
            "max_pixels": 160 * 28 * 28, "min_pixels": 32 * 28 * 28, "layer_fracs": [0.5, 0.75],
            "trust_remote_code": False, "video_frames": 40, "limit": None, "tag_suffix": "", "margin": 0.5,
            "desc_frames": 8}
    cmd_split(_ns(base, k=3, dev_frac=0.25))
    cmd_extract(_ns(base, chunk_frames=16, proj_dim=64, overwrite=False))
    head = {"bank_pca": 16, "bank_size": 256, "bank_k": 3, "d_model": 64, "n_layers": 2, "n_heads": 4, "epochs": 25,
            "lr": 2e-3, "batch_size": 4, "no_srn": False, "no_bank": False, "no_delta": False, "no_ctx": False,
            "no_group": False}
    lm = {"neg_ratio": 0.5, "desc_epochs": 1, "desc_lr": 1e-3, "grad_accum": 2, "lora_r": 4, "lora_alpha": 8}
    for fold in range(3):
        cmd_train_head(_ns(base, fold=fold, **head))
        cmd_train_desc(_ns(base, fold=fold, direct=False, **lm))
        cmd_train_desc(_ns(base, fold=fold, direct=True, **lm))
        cmd_predict(_ns(base, fold=fold, tag="norma", no_describer=False, no_fusion=False, mbr_samples=2,
                        mbr_temperature=0.7, max_new_tokens=32))
        for mode, tag, seed in (("vanilla", "vanilla", 0), ("direct", "direct", 0), ("glide", "glide_s0", 0),
                                ("glide", "glide_s1", 1)):
            cmd_baseline(_ns(base, fold=fold, mode=mode, tag=tag, report_tokens=128, seed=seed))
    tags = ["norma", "vanilla", "direct", "glide_s0", "glide_s1"]
    cmd_evaluate(_ns(base, tags=tags, judge_backend="offline", judge_model="unigram-f1", judge_base_url=None,
                     api_key_env="OPENAI_API_KEY", n_boot=200))
    systems = ["norma=norma", "glide=glide_s0,glide_s1", "direct=direct", "vanilla=vanilla"]
    cmd_compare(_ns(base, systems=systems, allow_mismatch=False, n_boot=500, alpha=0.05))
    cmd_cost(_ns(base, systems=systems))
    norma = read_json(Path(base["run_dir"]) / "eval_norma.json")
    t.check("e2e_unit_auroc", (norma["unit_auroc"] or 0.0) > 0.85, f"unit AUROC {norma['unit_auroc']}")
    t.check("e2e_norma_localises", norma["micro"]["miou"] > 0.45 and norma["micro"]["f1_x_iou"] > 25.0,
            json.dumps({k: round(v, 3) for k, v in norma["micro"].items()}))
    glide_rows = [r for f in range(3) for r in read_jsonl(Path(base["run_dir"]) / f"preds_glide_s0_fold{f}.jsonl")]
    t.check("e2e_glide_pipeline_runs", sum(r["stats"]["confirmed"] for r in glide_rows) > 0
            and all(r["cost"]["calls"] > 0 for r in glide_rows))
    _check_fairness_gate(t, base)


def _check_fairness_gate(t: SelfTest, base: Dict[str, Any]) -> None:
    run = Path(base["run_dir"])
    meta_path = run / "preds_vanilla_fold0.meta.json"
    meta = read_json(meta_path)
    tampered = {**meta, "fairness": {**meta["fairness"], "max_pixels": 1}}
    for f in range(3):
        write_json(run / f"preds_vanilla_fold{f}.meta.json", {**read_json(run / f"preds_vanilla_fold{f}.meta.json"),
                                                               "fairness": tampered["fairness"]})
    try:
        cmd_compare(_ns(base, systems=["norma=norma", "vanilla=vanilla"], allow_mismatch=False, n_boot=50, alpha=0.05))
        t.check("fairness_gate", False, "a pixel-budget mismatch was not refused")
    except RuntimeError as exc:
        t.check("fairness_gate", "unfair comparison" in str(exc), str(exc))


def cmd_selftest(args: argparse.Namespace) -> int:
    t = SelfTest()
    root = Path(tempfile.mkdtemp(prefix="norma_selftest_"))
    allow_root(root)
    try:
        check_estimators(t)
        check_tracker_and_decoder(t, root)
        check_qwen_modules(t)
        run_e2e(t, root)
    finally:
        if args.keep:
            LOGGER.info("self-test artefacts kept in %s", root)
        else:
            shutil.rmtree(root, ignore_errors=True)
    if t.failures:
        LOGGER.error("%d self-tests FAILED:\n  %s", len(t.failures), "\n  ".join(t.failures))
        return 1
    LOGGER.info("all self-tests passed")
    return 0


# =========================================================================== #
# 11. CLI
# =========================================================================== #


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--work-dir", default="./vgb_work", help="frame cache + videos.jsonl")
    p.add_argument("--run-dir", default="./norma_run")
    p.add_argument("--feat-dir", default="./norma_feats")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("-v", "--verbose", action="count", default=1)


def _add_model(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="Qwen/Qwen2.5-VL-{3B,7B}-Instruct | ByteDance-Seed/UI-TARS-1.5-7B | "
                        "OpenGVLab/InternVL2_5-{4B,8B} | mock")
    p.add_argument("--revision", default=None, help="pin the checkpoint commit (identical for all systems)")
    p.add_argument("--trust-remote-code", action="store_true", help="required for InternVL2.5 (with --revision)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--max-pixels", type=int, default=160 * 28 * 28, help="per-frame pixel budget (all systems)")
    p.add_argument("--min-pixels", type=int, default=32 * 28 * 28)
    p.add_argument("--layer-fracs", type=float, nargs="+", default=[0.5, 0.75])
    p.add_argument("--video-frames", type=int, default=48, help="frame cap for whole-video prompts (vanilla/direct)")


def _add_fold(p: argparse.ArgumentParser) -> None:
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--desc-frames", type=int, default=16)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--tag-suffix", default="", help="keeps ablation heads side by side, e.g. _nosrn")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preprocess", help="decode VGB to the shared 4 fps frame cache")
    _add_common(p)
    p.add_argument("--video-root", required=True)
    p.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    p.add_argument("--split", default="train")
    p.add_argument("--metadata", default=None, help="local JSON/JSONL annotations instead of the HF dataset")
    p.add_argument("--dataset-revision", default=None)
    p.add_argument("--short-side", type=int, default=360)
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_preprocess)

    p = sub.add_parser("prepare-generic", help="cache any temporally annotated anomaly dataset")
    _add_common(p)
    p.add_argument("--manifest", required=True)
    p.add_argument("--short-side", type=int, default=360)
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_prepare_generic)

    p = sub.add_parser("split", help="game-disjoint, genre-stratified K folds")
    _add_common(p)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--dev-frac", type=float, default=0.12)
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("extract", help="one early-exit prefill per chunk -> feature cache")
    _add_common(p)
    _add_model(p)
    p.add_argument("--chunk-frames", type=int, default=32)
    p.add_argument("--proj-dim", type=int, default=1024)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("train-head", help="NORMA head for one fold")
    _add_common(p)
    _add_fold(p)
    for flag, default in (("epochs", 40), ("batch-size", 16), ("d-model", 384), ("n-layers", 3), ("n-heads", 6),
                          ("bank-pca", 128), ("bank-size", 4096), ("bank-k", 3)):
        p.add_argument(f"--{flag}", type=int, default=default)
    p.add_argument("--lr", type=float, default=3e-4)
    for flag in ("srn", "bank", "delta", "ctx", "group"):
        p.add_argument(f"--no-{flag}", action="store_true", help=f"ablation: disable {flag}")
    p.set_defaults(func=cmd_train_head)

    p = sub.add_parser("train-desc", help="LoRA-SFT describer (or --direct baseline) for one fold")
    _add_common(p)
    _add_model(p)
    _add_fold(p)
    p.add_argument("--direct", action="store_true")
    p.add_argument("--neg-ratio", type=float, default=0.33)
    p.add_argument("--desc-epochs", type=int, default=2)
    p.add_argument("--desc-lr", type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.set_defaults(func=cmd_train_desc)

    p = sub.add_parser("predict", help="NORMA: dev -> fused calibrator -> test reports")
    _add_common(p)
    _add_model(p)
    _add_fold(p)
    p.add_argument("--tag", default="norma")
    p.add_argument("--no-describer", action="store_true")
    p.add_argument("--no-fusion", action="store_true")
    p.add_argument("--mbr-samples", type=int, default=4)
    p.add_argument("--mbr-temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("baseline", help="vanilla | direct | glide reports for one fold")
    _add_common(p)
    _add_model(p)
    _add_fold(p)
    p.add_argument("--mode", choices=["vanilla", "direct", "glide"], required=True)
    p.add_argument("--tag", default=None, help="default: the mode; use e.g. glide_s1 with --seed 1")
    p.add_argument("--report-tokens", type=int, default=512)
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("evaluate", help="GliDe protocol for one or more tags")
    _add_common(p)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--judge-backend", default="openai", choices=["openai", "hf", "offline"])
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--judge-base-url", default=None)
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--n-boot", type=int, default=2000)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("compare", help="paired cluster-bootstrap tests; first system is the proposed one")
    _add_common(p)
    p.add_argument("--systems", nargs="+", required=True, help="name=tag[,tag...] e.g. glide=glide_s0,glide_s1")
    p.add_argument("--allow-mismatch", action="store_true", help="skip the fairness gate (not recommended)")
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("cost", help="per-video calls / tokens / latency and training GPU-hours")
    _add_common(p)
    p.add_argument("--systems", nargs="+", required=True)
    p.set_defaults(func=cmd_cost)

    p = sub.add_parser("selftest", help="estimators, decoder, tracker, Qwen modules, CPU end-to-end")
    _add_common(p)
    p.add_argument("--keep", action="store_true")
    p.set_defaults(func=cmd_selftest)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    set_seed(args.seed)
    init_allowed_roots()
    result = args.func(args)
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())
