#!/usr/bin/env python3
"""
VideoGlitchBench -- Stage 1 (Preprocess) + Stage 2 (Context + Scanner)

Stage 1  preprocess : uniform temporal resampling of every gameplay video to a
                      fixed grid (default 4 fps), JPEG frame cache with a
                      fingerprinted config, and overlapping window construction
                      (default 8 frames/window, stride 4).
Stage 2a context    : per-game reference note built ONLY from pixels of a pool
                      of videos of that game. Cached on disk and injected as a
                      RAG-style prefix by the scanner.
Stage 2b scan       : per-window screening ->
                      {has_glitch, p_glitch, findings[], evidence}.

Metadata source : datasets.load_dataset("SandyZheng33/VideoGlitchBench")
Pixel source    : local mp4 files under --video-root
Model           : Qwen/Qwen2.5-VL-*-Instruct via transformers (official API)

=============================================================================
DESIGN NOTES -- what changed relative to the previous revision and why
=============================================================================

(A) GEOMETRY.  qwen_vl_utils.process_vision_info is no longer used.  When it
    is handed a *list of frame files* it calls smart_resize independently on
    every frame, so a pool that mixes a 16:9 and a 4:3 recording produces
    tensors of shape [3,336,616] and [3,336,448] and torch.stack raises
    "stack expects each tensor to be equal size".  That is exactly what
    happens in Stage 2a, where the context pool deliberately spans several
    videos of one game.  Enforcing a constant size *inside* a single video is
    not sufficient; the invariant has to hold over the frame list actually
    handed to the model.  Here the geometry is decided once per call:

        modal (h,w) over the list  ->  smart_resize(h, w, factor, min, max)
        ->  every frame letterboxed (aspect preserved, zero padding) to it.

    factor = patch_size * merge_size is read off the loaded processor rather
    than hardcoded to 28, so a checkpoint with a different vision config still
    yields a legal grid.  Letterboxing rather than stretching keeps angles and
    trajectories metric, which matters because several taxonomy classes
    (physics, clipping, animation) are judged from geometry.

(B) CALIBRATION.  p_glitch is a likelihood read-out, not a self-report --
    verbalised confidences from instruction-tuned models are badly calibrated
    and mode-seeking (Kadavath et al. 2022; Lin et al. 2022; Xiong et al.
    ICLR 2024).  But a likelihood read-out is only valid at a position where
    the two tokens being contrasted are genuine candidate continuations.  The
    previous revision asked for "ONLY a JSON object", so the first generated
    token was "{", and it then scored YES vs NO at that position: a
    measurement of two tokens neither of which the model was ever going to
    emit.  Here the screening head is a single teacher-forced FORWARD pass on
    a prompt that ends in "Answer with exactly one word: YES or NO.", and the
    probability is read at the first answer position:

        p = sigmoid( logsumexp z[yes_ids] - logsumexp z[no_ids] )

    No generation is needed for screening.  The structured description pass
    runs only for windows at or above --threshold (or for all of them under
    --always-describe), which makes the two-pass design strictly cheaper than
    the one-pass design it replaces, because most windows are negative.

(C) TAXONOMY.  The fine taxonomy is the label space; the coarse GliDe
    taxonomy is a derived view, emitted alongside it as `category_coarse`.
    The previous revision mapped fine -> coarse and then validated the coarse
    label against the fine vocabulary, so every clipping / rendering /
    lighting finding was silently dropped before it was ever written out.

(D) CONTEXT.  The scanner prompt now actually contains {context}.  Previously
    render_context() was computed, passed to .format(), ignored for want of a
    placeholder, and the row was still stamped "used_context": true.

(E) CACHE.  The frame cache stores a fingerprint of (fps, short_side, quality,
    codec revision).  A mismatch invalidates rather than silently reusing
    frames decoded under different settings.

Ground-truth fields are carried through the manifest for later evaluation.
The scanner asserts it never reads them.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import multiprocessing
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("vgb")

DEFAULT_HF_REPO = "SandyZheng33/VideoGlitchBench"
DEFAULT_VIDEO_ROOT = "/home/tahir/VLM-Project/VideoGlitchBench/raw/"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v", ".flv")

# Bump when the decode path changes in a way that invalidates cached frames.
FRAME_CACHE_VERSION = 3

# Fine-grained label space. This is the vocabulary the model is asked for and
# the vocabulary that is validated against.
GLITCH_TAXONOMY: Tuple[str, ...] = (
    "physics",            # impossible motion, floating, jitter, ragdoll explosion
    "clipping",           # geometry interpenetration, falling through world
    "rendering",          # missing/stretched/flickering textures, z-fighting
    "lighting",           # broken shadows, light leaks, black/blown-out surfaces
    "animation",          # T-pose, broken rig, frozen or looping animation
    "agent_behaviour",    # NPC/AI pathfinding, stuck or nonsensical actions
    "ui_hud",             # corrupted HUD, overlapping or missing UI elements
    "state_logic",        # invalid state transition, duplicated/vanishing objects
)

# Coarse view used by GliDe-style scoring. DERIVED, never used for validation.
COARSE_TAXONOMY: Tuple[str, ...] = ("Physics", "Visual", "Animation", "Game Logic", "Other")

FINE_TO_COARSE: Dict[str, str] = {
    "physics": "Physics",
    "clipping": "Visual",
    "rendering": "Visual",
    "lighting": "Visual",
    "animation": "Animation",
    "agent_behaviour": "Game Logic",
    "ui_hud": "Game Logic",
    "state_logic": "Game Logic",
}

# Common synonyms the model emits; folded into the fine vocabulary.
CATEGORY_ALIASES: Dict[str, str] = {
    "visual": "rendering",
    "texture": "rendering",
    "textures": "rendering",
    "graphical": "rendering",
    "graphics": "rendering",
    "render": "rendering",
    "shadow": "lighting",
    "light": "lighting",
    "collision": "clipping",
    "clip": "clipping",
    "geometry": "clipping",
    "physic": "physics",
    "motion": "physics",
    "ai": "agent_behaviour",
    "npc": "agent_behaviour",
    "agent": "agent_behaviour",
    "behaviour": "agent_behaviour",
    "behavior": "agent_behaviour",
    "ui": "ui_hud",
    "hud": "ui_hud",
    "interface": "ui_hud",
    "logic": "state_logic",
    "state": "state_logic",
    "game logic": "state_logic",
    "game_logic": "state_logic",
    "anim": "animation",
}


# --------------------------------------------------------------------------- #
# utilities
# --------------------------------------------------------------------------- #


def setup_logging(verbosity: int = 1) -> None:
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if level > logging.DEBUG:
        # HF/urllib3 chatter drowns the actual pipeline log at INFO.
        for noisy in ("urllib3", "filelock", "huggingface_hub", "datasets", "PIL"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sha1_short(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"required manifest not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
    return rows


def transformers_version() -> Tuple[int, ...]:
    try:
        import transformers

        parts = re.findall(r"\d+", transformers.__version__)[:3]
        return tuple(int(p) for p in parts)
    except Exception:  # noqa: BLE001
        return (0,)


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class VideoRecord:
    uid: int
    game_name: str
    genre: str
    subgenre: str
    video_name: str
    video_path: str
    no_bugs: bool
    bugs: List[str]
    # bug_spans[i] holds every interval belonging to bugs[i]; the alignment
    # between a description and its own temporal support must survive ingest,
    # otherwise per-bug recovery cannot be scored at all.
    bug_spans: List[List[Tuple[float, float]]]
    gt_intervals: List[Tuple[float, float]]   # flattened union, analysis only
    bug_categories: List[str] = dataclasses.field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["bug_spans"] = [[list(iv) for iv in spans] for spans in self.bug_spans]
        payload["gt_intervals"] = [list(iv) for iv in self.gt_intervals]
        return payload


def _as_python(value: Any) -> Any:
    """datasets may hand back numpy scalars / arrays / JSON strings."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_as_python(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_as_python(v) for v in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                return _as_python(json.loads(stripped))
            except json.JSONDecodeError:
                return value
    return value


def _collect_intervals(node: Any) -> List[Tuple[float, float]]:
    """Recurse to every [start, end] leaf below `node`, depth-agnostic."""
    out: List[Tuple[float, float]] = []

    def walk(item: Any) -> None:
        if item is None or not isinstance(item, (list, tuple)):
            return
        if len(item) == 2 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in item
        ):
            start, end = float(item[0]), float(item[1])
            if end < start:
                start, end = end, start
            out.append((start, end))
            return
        for child in item:
            walk(child)

    walk(node)
    return out


def merge_intervals(
    intervals: Sequence[Tuple[float, float]], gap: float = 0.0
) -> List[Tuple[float, float]]:
    """Union of possibly overlapping intervals. Concurrent bugs routinely
    overlap in time, so the flattened list is NOT a partition."""
    if not intervals:
        return []
    ordered = sorted((float(a), float(b)) for a, b in intervals)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def parse_time_nodes(
    raw: Any, n_bugs: int, video_name: str = ""
) -> Tuple[List[List[Tuple[float, float]]], List[Tuple[float, float]]]:
    """
    `time_nodes` is a list OVER BUGS, each element a list of [start, end] pairs:
        [[[3, 11]]]                        1 bug,  1 interval
        [[[3, 11], [20, 24]], [[40, 45]]]  2 bugs, 3 intervals
    Returns (bug_spans aligned to bugs[], flattened union of all intervals).

    If the outer arity does not match len(bugs) the alignment is unrecoverable,
    so we say so loudly and fall back to a single unattributed group rather
    than silently pairing the wrong description with the wrong span.
    """
    raw = _as_python(raw)
    if raw is None:
        return [], []
    groups = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    bug_spans = [_collect_intervals(group) for group in groups]
    bug_spans = [spans for spans in bug_spans if spans]

    flat = [iv for spans in bug_spans for iv in spans]
    if n_bugs and len(bug_spans) != n_bugs:
        LOGGER.warning(
            "time_nodes arity %d != len(bugs) %d for %s; bug<->span alignment "
            "dropped for this row",
            len(bug_spans), n_bugs, video_name or "<unknown>",
        )
        bug_spans = [flat] if flat else []
    return bug_spans, flat


def _first_present(row: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def build_video_index(video_root: Path) -> Dict[str, Path]:
    """basename -> path, recursive. Duplicate basenames are a hard error,
    because the metadata joins on basename and an ambiguous join silently
    pairs annotations with the wrong pixels."""
    if not video_root.is_dir():
        raise NotADirectoryError(f"--video-root does not exist: {video_root}")
    index: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(video_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if path.name in index:
            duplicates[path.name] = [index[path.name], path]
            continue
        index[path.name] = path
    if duplicates:
        sample = {k: [str(p) for p in v] for k, v in list(duplicates.items())[:5]}
        raise RuntimeError(
            "ambiguous video basenames under --video-root, cannot join on "
            f"video_name: {sample}"
        )
    if not index:
        raise RuntimeError(f"no video files found under {video_root}")
    LOGGER.info("indexed %d video files under %s", len(index), video_root)
    return index


def _normalize_video_name(video_name: Any) -> str:
    """Normalise a video reference from str / dict / datasets Video object."""
    if video_name is None:
        return ""
    if isinstance(video_name, dict):
        return os.path.basename(str(video_name.get("path", "")))
    for attr in ("path", "filename"):
        val = getattr(video_name, attr, None)
        if isinstance(val, str) and val:
            return os.path.basename(val)
    return os.path.basename(str(video_name))


def _resolve_video_path(
    video_name: str, index: Dict[str, Path], video_root: Path
) -> Optional[Path]:
    path = index.get(video_name)
    if path is not None:
        return path
    stem = Path(video_name).stem
    for name, candidate in index.items():
        if Path(name).stem == stem:
            return candidate
    for ext in VIDEO_SUFFIXES:
        candidate = video_root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_records(
    hf_repo: str,
    split: str,
    video_root: Path,
    limit: Optional[int] = None,
    games: Optional[Sequence[str]] = None,
    require_video: bool = True,
) -> List[VideoRecord]:
    from datasets import load_dataset

    LOGGER.info("load_dataset(%r, split=%r)", hf_repo, split)

    dset = None
    last_error: Optional[BaseException] = None
    for attempt in (split, "train", "test", "validation", None):
        try:
            if attempt is None:
                loaded = load_dataset(hf_repo)
                dset = next(iter(loaded.values())) if hasattr(loaded, "values") else loaded
            else:
                dset = load_dataset(hf_repo, split=attempt)
            LOGGER.info("loaded with split=%s", attempt or "<first available>")
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            LOGGER.debug("split %r failed: %s", attempt, exc)
            dset = None

    if dset is None:
        raise RuntimeError(f"could not load {hf_repo}: {last_error!r}")

    LOGGER.info("loaded %d rows; columns=%s", len(dset), list(dset.column_names))

    index = build_video_index(video_root)
    wanted = {g.lower() for g in games} if games else None

    records: List[VideoRecord] = []
    missing: List[str] = []
    errors: List[Dict[str, Any]] = []

    for row_idx, row in enumerate(dset):
        try:
            video_name = _normalize_video_name(
                _first_present(row, "video_name", "video", "file_name")
            )
            if not video_name:
                LOGGER.warning("row %d: no resolvable video name", row_idx)
                continue

            game_name = str(_as_python(_first_present(row, "game_name", "game", default="unknown")))
            if wanted is not None and game_name.lower() not in wanted:
                continue

            path = _resolve_video_path(video_name, index, video_root)
            if path is None:
                missing.append(video_name)
                if require_video:
                    continue

            no_bugs = bool(_as_python(_first_present(row, "no_bugs", default=False)))
            bugs_raw = _as_python(_first_present(row, "bugs", default=[])) or []
            if not isinstance(bugs_raw, list):
                bugs_raw = [bugs_raw]
            bugs = [str(b).strip() for b in bugs_raw if str(b).strip()]

            bug_spans, intervals = parse_time_nodes(
                _first_present(row, "time_nodes", default=[]), len(bugs), video_name
            )

            bug_cats = _as_python(_first_present(row, "bug_categories", default=[])) or []
            bug_categories = [str(c) for c in bug_cats] if isinstance(bug_cats, list) else []
            if len(bug_categories) != len(bugs):
                bug_categories = []

            records.append(
                VideoRecord(
                    uid=int(_as_python(_first_present(row, "id", "uid", default=len(records)))),
                    game_name=game_name,
                    genre=str(_as_python(_first_present(row, "genre", default=""))),
                    subgenre=str(_as_python(_first_present(row, "subgenre", default=""))),
                    video_name=video_name,
                    video_path=str(path) if path is not None else "",
                    no_bugs=no_bugs,
                    bugs=bugs,
                    bug_spans=bug_spans,
                    gt_intervals=intervals,
                    bug_categories=bug_categories,
                )
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("error processing row %d: %r", row_idx, exc)
            errors.append({"row": row_idx, "error": repr(exc)})

    if missing:
        LOGGER.warning(
            "%d annotated videos have no file under --video-root (first 5: %s)",
            len(missing), missing[:5],
        )
    if errors:
        LOGGER.warning("%d rows raised during ingest", len(errors))

    records.sort(key=lambda r: (r.game_name, r.video_name))
    if limit is not None:
        records = records[:limit]
    if not records:
        raise RuntimeError("no usable records after joining metadata with local videos")

    multi_bug = sum(len(r.bugs) > 1 for r in records)
    multi_span = sum(any(len(s) > 1 for s in r.bug_spans) for r in records)
    LOGGER.info(
        "%d records over %d games (%d no_bugs, %d with >1 bug, %d with a "
        "multi-interval bug, %.2f intervals/video mean)",
        len(records), len({r.game_name for r in records}),
        sum(r.no_bugs for r in records), multi_bug, multi_span,
        float(np.mean([len(r.gt_intervals) for r in records])),
    )
    return records


# --------------------------------------------------------------------------- #
# geometry: the single place where frame size is decided
# --------------------------------------------------------------------------- #


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 64 * 28 * 28,
    max_pixels: int = 256 * 28 * 28,
) -> Tuple[int, int]:
    """
    Qwen2-VL / Qwen2.5-VL resize rule, reimplemented so the pipeline owns it.

    Postconditions (all required by the ViT patch embedding + spatial merge):
      h % factor == 0 and w % factor == 0
      min_pixels <= h*w <= max_pixels  (subject to the factor lattice)
      aspect ratio preserved to within one factor step
    The function is idempotent on its own output, so re-running it inside the
    HF processor is a no-op and no second resample occurs.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"degenerate frame size {height}x{width}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"aspect ratio {height}x{width} exceeds 200:1")

    h_bar = max(factor, int(round(height / factor)) * factor)
    w_bar = max(factor, int(round(width / factor)) * factor)

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, int(math.floor(height / beta / factor)) * factor)
        w_bar = max(factor, int(math.floor(width / beta / factor)) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = max(factor, int(math.ceil(height * beta / factor)) * factor)
        w_bar = max(factor, int(math.ceil(width * beta / factor)) * factor)
    return int(h_bar), int(w_bar)


def _letterbox(array: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize preserving aspect ratio, then centre-pad with zeros."""
    from PIL import Image

    h, w = array.shape[:2]
    if (h, w) == (out_h, out_w):
        return array
    scale = min(out_h / float(h), out_w / float(w))
    new_h = max(1, min(out_h, int(round(h * scale))))
    new_w = max(1, min(out_w, int(round(w * scale))))
    resized = np.asarray(
        Image.fromarray(array).resize((new_w, new_h), Image.BICUBIC), dtype=np.uint8
    )
    if (new_h, new_w) == (out_h, out_w):
        return resized
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    top = (out_h - new_h) // 2
    left = (out_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def _stretch(array: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    from PIL import Image

    if array.shape[:2] == (out_h, out_w):
        return array
    return np.asarray(
        Image.fromarray(array).resize((out_w, out_h), Image.BICUBIC), dtype=np.uint8
    )


def load_uniform_video(
    frame_paths: Sequence[str],
    factor: int,
    min_pixels: int,
    max_pixels: int,
    temporal_patch: int = 2,
    fit: str = "letterbox",
) -> np.ndarray:
    """
    Load a frame list into ONE contiguous uint8 array of shape (T, H, W, 3)
    with a single (H, W) shared by every frame.

    This is the fix for
        RuntimeError: stack expects each tensor to be equal size,
                      but got [3,336,616] at entry 0 and [3,336,448] at entry 2
    The geometry is chosen from the MODAL raw size of the list -- not from the
    first frame -- so one odd-sized recording in a mixed-resolution context
    pool cannot drag the whole batch to a bad target.

    T is padded up to a multiple of `temporal_patch` by repeating the last
    frame, because the vision tower groups frames in pairs and a dangling
    frame is otherwise silently duplicated deeper in the stack where it is
    harder to reason about.
    """
    from PIL import Image

    if not frame_paths:
        raise ValueError("empty frame list")
    if fit not in ("letterbox", "stretch"):
        raise ValueError(f"unknown fit mode {fit!r}")

    raw: List[np.ndarray] = []
    sizes: Counter = Counter()
    for p in frame_paths:
        with Image.open(p) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"frame {p} is not HxWx3 RGB (got {arr.shape})")
        raw.append(arr)
        sizes[arr.shape[:2]] += 1

    (mode_h, mode_w), _ = sizes.most_common(1)[0]
    out_h, out_w = smart_resize(mode_h, mode_w, factor, min_pixels, max_pixels)
    if len(sizes) > 1:
        LOGGER.debug(
            "mixed frame geometry %s -> unified %dx%d via %s",
            dict(sizes), out_h, out_w, fit,
        )

    fitter = _letterbox if fit == "letterbox" else _stretch
    frames = [fitter(arr, out_h, out_w) for arr in raw]

    if temporal_patch > 1:
        while len(frames) % temporal_patch != 0:
            frames.append(frames[-1].copy())

    video = np.stack(frames, axis=0)
    assert video.ndim == 4 and video.shape[1:] == (out_h, out_w, 3), video.shape
    return video


# --------------------------------------------------------------------------- #
# Stage 1: decoding + windowing
# --------------------------------------------------------------------------- #


def _target_from_short_side(height: int, width: int, short_side: int) -> Tuple[int, int]:
    if short_side <= 0:
        return height, width
    scale = short_side / float(min(height, width))
    return max(2, int(round(height * scale))), max(2, int(round(width * scale)))


def decode_uniform_frames(
    video_path: Path, target_fps: float, short_side: int
) -> Tuple[List[np.ndarray], List[float], float, Tuple[int, int]]:
    """
    Single sequential decode pass. For every point of the uniform grid
    t_k = k / target_fps we keep the decoded frame whose presentation
    timestamp is the first at or after t_k.

    Every frame is letterboxed to the geometry derived from the FIRST decoded
    frame, so a stream that changes resolution mid-file (which happens in
    user-uploaded gameplay captures) cannot produce a ragged cache.

    Returns (frames RGB uint8, timestamps, duration, (h, w)).
    """
    try:
        import av
    except ImportError as exc:
        raise ImportError("PyAV is required: pip install av") from exc

    if target_fps <= 0:
        raise ValueError("--fps must be positive")

    frames: List[np.ndarray] = []
    stamps: List[float] = []
    target_h: Optional[int] = None
    target_w: Optional[int] = None

    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise RuntimeError(f"no video stream in {video_path}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = float(stream.time_base) if stream.time_base else 0.0
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
        if duration <= 0.0 and container.duration:
            duration = float(container.duration) / 1_000_000.0

        step = 1.0 / float(target_fps)
        next_t = 0.0

        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base) if time_base else float(len(frames)) * step
            if t + 1e-9 < next_t:
                continue

            array = frame.to_ndarray(format="rgb24")
            height, width = array.shape[:2]

            if target_h is None:
                target_h, target_w = _target_from_short_side(height, width, short_side)
                LOGGER.debug(
                    "%s: first frame %dx%d -> cache geometry %dx%d",
                    video_path.name, height, width, target_h, target_w,
                )

            if (height, width) != (target_h, target_w):
                array = _letterbox(array, target_h, target_w)

            frames.append(array)
            stamps.append(t)
            next_t = (math.floor(t / step) + 1) * step

    if not frames:
        raise RuntimeError(f"decoded zero frames from {video_path}")
    if duration <= 0.0:
        duration = stamps[-1] + 1.0 / float(target_fps)

    assert target_h is not None and target_w is not None
    return frames, stamps, duration, (target_h, target_w)


def make_windows(n_frames: int, window: int, stride: int) -> List[Tuple[int, int]]:
    """Half-open [start, end) index pairs. The tail is snapped back so the last
    window is full-length whenever the clip is at least `window` frames."""
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    if n_frames <= 0:
        return []
    if n_frames <= window:
        return [(0, n_frames)]
    spans = [(s, s + window) for s in range(0, n_frames - window + 1, stride)]
    if spans[-1][1] < n_frames:
        spans.append((n_frames - window, n_frames))
    return spans


def _cache_fingerprint(target_fps: float, short_side: int, jpeg_quality: int) -> str:
    return sha1_short(
        json.dumps(
            {
                "v": FRAME_CACHE_VERSION,
                "fps": round(float(target_fps), 6),
                "short_side": int(short_side),
                "q": int(jpeg_quality),
            },
            sort_keys=True,
        )
    )


def preprocess_one(
    record_json: Dict[str, Any],
    frames_dir: str,
    target_fps: float,
    short_side: int,
    window: int,
    stride: int,
    jpeg_quality: int,
    overwrite: bool,
) -> Dict[str, Any]:
    """Worker body. Returns a per-video summary dict with its window rows."""
    from PIL import Image

    record = VideoRecord(
        **{
            **record_json,
            "bug_spans": [
                [tuple(iv) for iv in spans] for spans in record_json.get("bug_spans", [])
            ],
            "gt_intervals": [tuple(iv) for iv in record_json["gt_intervals"]],
        }
    )
    stem = Path(record.video_name).stem
    out_dir = Path(frames_dir) / re.sub(r"[^\w.-]+", "_", record.game_name) / stem
    done_marker = out_dir / "_frames.json"
    fingerprint = _cache_fingerprint(target_fps, short_side, jpeg_quality)

    cached: Optional[Dict[str, Any]] = None
    if done_marker.exists() and not overwrite:
        try:
            candidate = json.loads(done_marker.read_text())
        except json.JSONDecodeError:
            candidate = None
        if candidate is not None:
            same_config = candidate.get("fingerprint") == fingerprint
            paths_ok = all(Path(p).exists() for p in candidate.get("frame_paths", []))
            if same_config and paths_ok and candidate.get("frame_paths"):
                cached = candidate
            else:
                LOGGER.info(
                    "cache invalid for %s (config_match=%s files_present=%s), re-decoding",
                    record.video_name, same_config, paths_ok,
                )

    if cached is not None:
        frame_paths = cached["frame_paths"]
        stamps = cached["timestamps"]
        duration = cached["duration"]
        frame_hw = tuple(cached.get("frame_hw", (0, 0)))
    else:
        video_path = Path(record.video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"video file not found: {video_path}")

        frames, stamps, duration, frame_hw = decode_uniform_frames(
            video_path, target_fps, short_side
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in out_dir.glob("*.jpg"):
            stale.unlink()
        frame_paths = []
        for i, array in enumerate(frames):
            fp = out_dir / f"{i:06d}.jpg"
            Image.fromarray(array).save(fp, format="JPEG", quality=jpeg_quality)
            frame_paths.append(str(fp))
        done_marker.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "frame_paths": frame_paths,
                    "timestamps": stamps,
                    "duration": duration,
                    "target_fps": target_fps,
                    "short_side": short_side,
                    "jpeg_quality": jpeg_quality,
                    "frame_hw": list(frame_hw),
                },
                ensure_ascii=False,
            )
        )

    spans = make_windows(len(frame_paths), window, stride)
    rows: List[Dict[str, Any]] = []
    for w_idx, (start, end) in enumerate(spans):
        rows.append(
            {
                "window_uid": f"{stem}#{w_idx:04d}",
                "video_name": record.video_name,
                "video_uid": record.uid,
                "game_name": record.game_name,
                "genre": record.genre,
                "subgenre": record.subgenre,
                "window_index": w_idx,
                "frame_start": start,
                "frame_end": end,
                "t_start": float(stamps[start]),
                "t_end": float(stamps[end - 1]) + 1.0 / float(target_fps),
                "fps": float(target_fps),
                "frame_paths": frame_paths[start:end],
            }
        )

    return {
        "video_name": record.video_name,
        "video_uid": record.uid,
        "game_name": record.game_name,
        "genre": record.genre,
        "subgenre": record.subgenre,
        "video_path": record.video_path,
        "duration": duration,
        "n_frames": len(frame_paths),
        "n_windows": len(rows),
        "frame_hw": list(frame_hw),
        "no_bugs": record.no_bugs,
        "bugs": record.bugs,
        "bug_categories": record.bug_categories,
        "bug_spans": [[list(iv) for iv in spans_] for spans_ in record.bug_spans],
        "gt_intervals": [list(iv) for iv in record.gt_intervals],
        "gt_union": [list(iv) for iv in merge_intervals(record.gt_intervals)],
        "_windows": rows,
    }


def cmd_preprocess(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(
        args.hf_repo, args.split, Path(args.video_root), args.limit, args.games
    )

    payloads = [r.to_json() for r in records]
    video_rows: List[Dict[str, Any]] = []
    window_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    job = (str(frames_dir), args.fps, args.short_side, args.window,
           args.stride, args.jpeg_quality, args.overwrite)

    t0 = time.time()
    if args.workers <= 1:
        for done, payload in enumerate(payloads, 1):
            try:
                result = preprocess_one(payload, *job)
            except Exception as exc:  # noqa: BLE001
                failures.append({"video_name": payload["video_name"], "error": repr(exc)})
                LOGGER.error("preprocess failed for %s: %r", payload["video_name"], exc)
                continue
            video_rows.append({k: v for k, v in result.items() if k != "_windows"})
            window_rows.extend(result["_windows"])
            if done % 50 == 0:
                LOGGER.info("preprocessed %d/%d videos", done, len(payloads))
    else:
        # spawn, not fork: PyAV/FFmpeg holds thread state that does not survive
        # fork cleanly and produces intermittent decoder deadlocks.
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = {
                pool.submit(preprocess_one, payload, *job): payload for payload in payloads
            }
            for done, future in enumerate(as_completed(futures), 1):
                payload = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    failures.append({"video_name": payload["video_name"], "error": repr(exc)})
                    LOGGER.error("preprocess failed for %s: %r", payload["video_name"], exc)
                    continue
                video_rows.append({k: v for k, v in result.items() if k != "_windows"})
                window_rows.extend(result["_windows"])
                if done % 50 == 0:
                    LOGGER.info("preprocessed %d/%d videos", done, len(payloads))

    if not window_rows:
        raise RuntimeError("preprocess produced zero windows; see preprocess_failures.jsonl")

    video_rows.sort(key=lambda r: (r["game_name"], r["video_name"]))
    window_rows.sort(key=lambda r: (r["video_name"], r["window_index"]))

    write_jsonl(out_dir / "videos.jsonl", video_rows)
    write_jsonl(out_dir / "windows.jsonl", window_rows)
    if failures:
        write_jsonl(out_dir / "preprocess_failures.jsonl", failures)

    (out_dir / "preprocess_config.json").write_text(
        json.dumps({k: v for k, v in vars(args).items() if k != "func"}, indent=2, default=str)
    )

    geometries = Counter(tuple(r["frame_hw"]) for r in video_rows)
    LOGGER.info("cached frame geometries: %s", dict(geometries))
    LOGGER.info(
        "stage-1 done in %.1fs: %d videos, %d windows, %d failures -> %s",
        time.time() - t0, len(video_rows), len(window_rows), len(failures), out_dir,
    )


def cmd_verify(args: argparse.Namespace) -> None:
    """
    Cheap integrity pass over the frame cache. Run this before spending GPU
    hours: it catches missing files and ragged geometry, which are the two
    failure modes that otherwise surface as an opaque torch.stack error deep
    inside the processor.
    """
    out_dir = Path(args.out_dir)
    windows = read_jsonl(out_dir / "windows.jsonl")
    from PIL import Image

    missing = 0
    per_video_sizes: Dict[str, Counter] = defaultdict(Counter)
    checked = 0
    for win in windows:
        for p in win["frame_paths"]:
            path = Path(p)
            if not path.exists():
                missing += 1
                continue
            if args.deep or checked < args.sample:
                with Image.open(path) as im:
                    per_video_sizes[win["video_name"]][im.size] += 1
                checked += 1

    ragged = {k: dict(v) for k, v in per_video_sizes.items() if len(v) > 1}
    all_sizes = Counter()
    for counter in per_video_sizes.values():
        all_sizes.update(counter)

    LOGGER.info("verified %d frames across %d windows", checked, len(windows))
    LOGGER.info("distinct (w,h) over the whole cache: %s", dict(all_sizes))
    if missing:
        LOGGER.error("%d frame files referenced by windows.jsonl are missing", missing)
    if ragged:
        LOGGER.error("videos with ragged frame geometry: %s", list(ragged)[:10])
    if len(all_sizes) > 1:
        LOGGER.info(
            "the cache spans %d distinct sizes; this is EXPECTED and safe -- "
            "load_uniform_video() unifies geometry per model call",
            len(all_sizes),
        )
    if not missing and not ragged:
        LOGGER.info("cache OK")


# --------------------------------------------------------------------------- #
# VLM backend
# --------------------------------------------------------------------------- #


class QwenVLBackend:
    """
    Explicit wrapper over the official Qwen2.5-VL transformers API.

    qwen_vl_utils is deliberately NOT used. Its process_vision_info() resizes
    every frame of a frame-list video independently, which makes the resulting
    tensors non-stackable whenever the list mixes aspect ratios. All geometry
    is handled here by load_uniform_video().
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        max_pixels: int = 256 * 28 * 28,
        min_pixels: int = 64 * 28 * 28,
        device_map: str = "auto",
        attn_impl: str = "sdpa",
        fit: str = "letterbox",
    ) -> None:
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
        except ImportError as exc:
            raise ImportError(
                "transformers>=4.49 with Qwen2.5-VL support is required: "
                "pip install -U 'transformers>=4.49' accelerate"
            ) from exc

        self.torch = torch
        self.model_id = model_id
        self.fit = fit
        self._video_form: Optional[str] = None   # memoised processor input form
        self._accepts_fps: Optional[bool] = None

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype)
        if torch_dtype is None:
            raise ValueError(f"unsupported dtype {dtype!r}")
        if dtype == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            LOGGER.warning("bfloat16 unsupported on this GPU; falling back to float16")
            torch_dtype = torch.float16

        dtype_key = "dtype" if transformers_version() >= (4, 56) else "torch_dtype"
        kwargs: Dict[str, Any] = {
            dtype_key: torch_dtype,
            "attn_implementation": attn_impl,
            "device_map": device_map,
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        LOGGER.info("loading %s (%s=%s, 4bit=%s, attn=%s)",
                    model_id, dtype_key, dtype, load_in_4bit, attn_impl)
        try:
            self.model = ModelCls.from_pretrained(model_id, **kwargs)
        except (TypeError, ValueError) as exc:
            LOGGER.warning("from_pretrained rejected %s (%s); retrying with the alternate key",
                           dtype_key, exc)
            alt = "torch_dtype" if dtype_key == "dtype" else "dtype"
            kwargs[alt] = kwargs.pop(dtype_key)
            self.model = ModelCls.from_pretrained(model_id, **kwargs)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=min_pixels, max_pixels=max_pixels
        )
        self.tokenizer = self.processor.tokenizer

        self.min_pixels, self.max_pixels = self._configure_pixel_budget(min_pixels, max_pixels)
        self.factor, self.temporal_patch = self._vision_lattice()
        LOGGER.info(
            "vision lattice: factor=%d temporal_patch=%d pixel budget=[%d, %d] "
            "(<= %d merged tokens/frame)",
            self.factor, self.temporal_patch, self.min_pixels, self.max_pixels,
            self.max_pixels // (self.factor * self.factor),
        )

        # Greedy, deterministic. Set once so generate() never trips the
        # "temperature is set but do_sample=False" validation path.
        gen = self.model.generation_config
        gen.do_sample = False
        gen.temperature = None
        gen.top_p = None
        gen.top_k = None
        if gen.pad_token_id is None:
            gen.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    # -- processor plumbing -------------------------------------------------- #

    def _sub_processors(self) -> List[Any]:
        out = []
        for name in ("image_processor", "video_processor"):
            proc = getattr(self.processor, name, None)
            if proc is not None:
                out.append(proc)
        return out or [self.processor]

    def _configure_pixel_budget(self, min_pixels: int, max_pixels: int) -> Tuple[int, int]:
        """
        Force the same budget onto every sub-processor. Newer transformers keep
        a separate video_processor whose defaults differ from image_processor;
        if they disagree, our precomputed geometry is resampled a second time
        and the frames stop being factor-aligned.
        """
        if min_pixels > max_pixels:
            raise ValueError("--min-pixels must not exceed --max-pixels")
        for proc in self._sub_processors():
            for attr, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
                if hasattr(proc, attr):
                    setattr(proc, attr, value)
            size = getattr(proc, "size", None)
            if isinstance(size, dict):
                if "shortest_edge" in size:
                    size["shortest_edge"] = min_pixels
                if "longest_edge" in size:
                    size["longest_edge"] = max_pixels
            # We hand over pre-sampled frames; never let the processor resample.
            if hasattr(proc, "do_sample_frames"):
                proc.do_sample_frames = False
        return min_pixels, max_pixels

    def _vision_lattice(self) -> Tuple[int, int]:
        proc = getattr(self.processor, "image_processor", None) or self.processor
        patch = int(getattr(proc, "patch_size", 14) or 14)
        merge = int(getattr(proc, "merge_size", 2) or 2)
        temporal = int(getattr(proc, "temporal_patch_size", 2) or 2)
        cfg = getattr(self.model, "config", None)
        vision_cfg = getattr(cfg, "vision_config", None)
        if vision_cfg is not None:
            patch = int(getattr(vision_cfg, "patch_size", patch) or patch)
            merge = int(getattr(vision_cfg, "spatial_merge_size", merge) or merge)
            temporal = int(getattr(vision_cfg, "temporal_patch_size", temporal) or temporal)
        return patch * merge, temporal

    # -- token ids used for the binary read-out ------------------------------ #

    def first_token_ids(self, *words: str) -> List[int]:
        ids: set = set()
        for word in words:
            for variant in (word, " " + word, word.lower(), " " + word.lower(),
                            word.upper(), " " + word.upper(),
                            word.capitalize(), " " + word.capitalize()):
                enc = self.tokenizer.encode(variant, add_special_tokens=False)
                if enc:
                    ids.add(int(enc[0]))
        return sorted(ids)

    def binary_token_sets(self) -> Tuple[List[int], List[int]]:
        yes = set(self.first_token_ids("YES"))
        no = set(self.first_token_ids("NO"))
        clash = yes & no
        if clash:
            LOGGER.warning(
                "YES/NO first-token sets overlap on %s; dropping the ambiguous ids",
                [self.tokenizer.decode([c]) for c in clash],
            )
            yes -= clash
            no -= clash
        if not yes or not no:
            raise RuntimeError("could not resolve disjoint YES/NO token ids for this tokenizer")
        LOGGER.info(
            "binary head: YES=%s NO=%s",
            [self.tokenizer.decode([i]) for i in sorted(yes)],
            [self.tokenizer.decode([i]) for i in sorted(no)],
        )
        return sorted(yes), sorted(no)

    # -- input construction --------------------------------------------------- #

    def _chat_text(self, prompt: str, system: Optional[str] = None) -> str:
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({
            "role": "user",
            "content": [
                {"type": "video", "video": "<pre-sampled frames>"},
                {"type": "text", "text": prompt},
            ],
        })
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _call_processor(self, text: str, video: np.ndarray, fps: float):
        """
        Feed one pre-built (T,H,W,3) uint8 video. The accepted container type
        varies across transformers releases, so the first form that produces a
        video_grid_thw is memoised and reused.
        """
        from PIL import Image

        forms: List[Tuple[str, Any]] = [
            ("ndarray_list", [video]),
            ("pil_list", [[Image.fromarray(f) for f in video]]),
            ("ndarray", video),
        ]
        if self._video_form is not None:
            forms = [f for f in forms if f[0] == self._video_form] + \
                    [f for f in forms if f[0] != self._video_form]

        errors: List[str] = []
        for name, videos in forms:
            for with_fps in ((True, False) if self._accepts_fps is not False else (False,)):
                try:
                    extra = {"fps": [float(fps)]} if with_fps else {}
                    inputs = self.processor(
                        text=[text], images=None, videos=videos,
                        padding=True, return_tensors="pt", **extra,
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{name}/fps={with_fps}: {exc!r}")
                    continue
                if "video_grid_thw" not in inputs:
                    errors.append(f"{name}/fps={with_fps}: no video_grid_thw in output")
                    continue
                self._video_form = name
                self._accepts_fps = with_fps
                return inputs
        raise RuntimeError(
            "the processor rejected every video container form; tried:\n  "
            + "\n  ".join(errors)
        )

    def build_inputs(self, frame_paths: Sequence[str], fps: float, prompt: str,
                     system: Optional[str] = None):
        video = load_uniform_video(
            frame_paths,
            factor=self.factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            temporal_patch=self.temporal_patch,
            fit=self.fit,
        )
        text = self._chat_text(prompt, system)
        inputs = self._call_processor(text, video, fps)
        return inputs.to(self.model.device)

    def _oom_retry(self, fn, *fn_args, **fn_kwargs):
        """One retry at half the pixel budget, then give up honestly."""
        torch = self.torch
        try:
            return fn(*fn_args, **fn_kwargs)
        except torch.cuda.OutOfMemoryError:
            old_max = self.max_pixels
            reduced = max(self.min_pixels, old_max // 2)
            if reduced == old_max:
                raise
            LOGGER.warning("CUDA OOM; retrying once at max_pixels %d -> %d", old_max, reduced)
            torch.cuda.empty_cache()
            self._configure_pixel_budget(self.min_pixels, reduced)
            self.max_pixels = reduced
            try:
                return fn(*fn_args, **fn_kwargs)
            finally:
                self._configure_pixel_budget(self.min_pixels, old_max)
                self.max_pixels = old_max
                torch.cuda.empty_cache()

    # -- the two inference primitives ----------------------------------------- #

    def binary_probability(
        self,
        frame_paths: Sequence[str],
        prompt: str,
        fps: float,
        yes_ids: Sequence[int],
        no_ids: Sequence[int],
        system: Optional[str] = None,
    ) -> float:
        """
        Screening head. ONE forward pass, no generation.

        The prompt must end by demanding a single YES/NO word, so that the
        position scored here is the position at which the model would actually
        emit that word. Then

            p = sum_yes exp(z) / (sum_yes exp(z) + sum_no exp(z))
              = sigmoid( logsumexp z[yes] - logsumexp z[no] )

        is a binary-restricted likelihood: an improper but monotone score that
        Platt / isotonic / conformal recalibration can consume. It is NOT a
        calibrated posterior and must not be reported as one without a fitted
        calibrator on held-out windows.
        """
        torch = self.torch

        def _run() -> float:
            inputs = self.build_inputs(frame_paths, fps, prompt, system)
            with torch.inference_mode():
                out = self.model(**inputs, use_cache=False)
            logits = out.logits[0, -1, :].float()
            yes = torch.logsumexp(logits[list(yes_ids)], dim=0)
            no = torch.logsumexp(logits[list(no_ids)], dim=0)
            return float(torch.sigmoid(yes - no).item())

        return self._oom_retry(_run)

    def generate(
        self,
        frame_paths: Sequence[str],
        prompt: str,
        fps: float,
        max_new_tokens: int = 192,
        system: Optional[str] = None,
    ) -> str:
        """Greedy decode of the structured description."""
        torch = self.torch

        def _run() -> str:
            inputs = self.build_inputs(frame_paths, fps, prompt, system)
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                )
            trimmed = out.sequences[:, inputs["input_ids"].shape[1]:]
            return self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

        return self._oom_retry(_run)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """First balanced {...} block, tolerant of ```json fences and trailing prose."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def canonical_category(raw: Any) -> Optional[str]:
    """
    Fold a free-text category onto the FINE vocabulary.

    Validation happens against GLITCH_TAXONOMY only. The coarse GliDe label is
    derived afterwards by FINE_TO_COARSE; it is never used as a filter, which
    is what silently destroyed every clipping/rendering/lighting finding in
    the previous revision.
    """
    if raw is None:
        return None
    key = re.sub(r"[^a-z_ ]+", "", str(raw).strip().lower()).strip()
    if not key:
        return None
    if key in ("none", "no glitch", "no_glitch", "other"):
        return None
    if key in GLITCH_TAXONOMY:
        return key
    if key in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[key]
    key_us = key.replace(" ", "_")
    if key_us in GLITCH_TAXONOMY:
        return key_us
    if key_us in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[key_us]
    for token in re.split(r"[\s_/,|]+", key):
        if token in GLITCH_TAXONOMY:
            return token
        if token in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[token]
    return None


def normalise_findings(
    parsed: Dict[str, Any], n_frames: int, max_findings: int
) -> List[Dict[str, Any]]:
    """
    Accepts the multi-finding schema {"findings": [...]} or a bare single
    finding object. Duplicate (category, evidence) pairs are collapsed;
    distinct categories are kept, because concurrent defects in one window are
    real and common in this benchmark.
    """
    raw_items = parsed.get("findings")
    if not isinstance(raw_items, list):
        raw_items = [parsed] if ("category" in parsed or "evidence" in parsed) else []

    out: List[Dict[str, Any]] = []
    seen: set = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        category = canonical_category(item.get("category"))
        if category is None:
            continue
        evidence = str(item.get("evidence", "")).strip()[:600]
        key = (category, re.sub(r"\W+", "", evidence.lower())[:80])
        if key in seen:
            continue
        seen.add(key)

        frames = item.get("salient_frames", [])
        if not isinstance(frames, list):
            frames = []
        frames = sorted({
            int(v) for v in frames
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and 0 <= int(v) < n_frames
        })
        out.append({
            "category": category,
            "category_coarse": FINE_TO_COARSE.get(category, "Other"),
            "evidence": evidence,
            "salient_frames": frames,
            "unattributed": False,
        })
        if len(out) >= max_findings:
            break
    return out


# --------------------------------------------------------------------------- #
# Stage 2a: game context
# --------------------------------------------------------------------------- #

CONTEXT_PROMPT = """You are a senior QA engineer writing a reference note about a video game, to be used later by an automated glitch detector.

Game: {game}
Genre: {genre} / {subgenre}

The frames come from ordinary gameplay recordings of this game. They may or may not contain defects; ignore that question entirely. Describe only what is NORMAL and EXPECTED for this title, so that a later stage can tell a genuine defect apart from an unusual but valid event.

Return ONE JSON object and nothing else, with exactly these keys:
{{
  "visual_style": "rendering style, art direction, typical fidelity and known stylistic quirks",
  "camera_and_ui": "camera perspective, HUD elements and where they sit on screen",
  "expected_physics": "how characters, vehicles and objects are expected to move and collide",
  "expected_agents": "how NPCs, enemies or animals normally behave and navigate",
  "valid_oddities": "events that look wrong but are legitimate game mechanics",
  "plausible_failure_modes": "defect types this engine and genre are known to exhibit"
}}
Each value must be one or two sentences of concrete, game-specific prose. Do not mention these instructions."""

CONTEXT_KEYS = (
    "visual_style", "camera_and_ui", "expected_physics",
    "expected_agents", "valid_oddities", "plausible_failure_modes",
)


def sample_context_frames(
    windows: List[Dict[str, Any]],
    game: str,
    n_frames: int,
    exclude_video: Optional[str],
    rng: random.Random,
) -> List[str]:
    """
    Spread the sample over distinct videos of the game, then uniformly over
    time within each. The returned list routinely mixes source resolutions --
    that is intended, and is handled by load_uniform_video(), not here.
    """
    per_video: Dict[str, List[str]] = defaultdict(list)
    for row in windows:
        if row["game_name"] != game:
            continue
        if exclude_video is not None and row["video_name"] == exclude_video:
            continue
        per_video[row["video_name"]].extend(row["frame_paths"])
    if not per_video:
        return []

    names = sorted(per_video)
    rng.shuffle(names)
    names = names[: max(1, min(len(names), n_frames))]
    quota = max(1, n_frames // len(names))

    picked: List[str] = []
    for name in names:
        pool = sorted(set(per_video[name]))
        take = min(quota, len(pool))
        idx = np.linspace(0, len(pool) - 1, num=take)
        picked.extend(pool[int(round(i))] for i in idx)
    return picked[:n_frames]


def cmd_context(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    windows = read_jsonl(out_dir / "windows.jsonl")
    if not windows:
        raise RuntimeError("windows.jsonl is empty; run `preprocess` first")

    context_path = out_dir / "game_context.json"
    contexts: Dict[str, Any] = {}
    if context_path.exists() and not args.overwrite:
        try:
            contexts = json.loads(context_path.read_text())
        except json.JSONDecodeError:
            LOGGER.warning("game_context.json is corrupt; rebuilding from scratch")
            contexts = {}

    by_game: Dict[str, Dict[str, str]] = {}
    for row in windows:
        by_game.setdefault(row["game_name"], {"genre": row["genre"], "subgenre": row["subgenre"]})

    todo = [g for g in sorted(by_game) if g not in contexts or not contexts[g].get("parse_ok")]
    LOGGER.info("game context: %d games total, %d to build", len(by_game), len(todo))
    if not todo:
        return

    backend = QwenVLBackend(
        args.model, args.dtype, args.load_in_4bit, args.max_pixels, args.min_pixels,
        attn_impl=args.attn, fit=args.fit,
    )
    rng = random.Random(args.seed)
    n_fail = 0

    for i, game in enumerate(todo, 1):
        frames = sample_context_frames(windows, game, args.context_frames, None, rng)
        if not frames:
            LOGGER.warning("no frames for game %s, skipping", game)
            continue
        prompt = CONTEXT_PROMPT.format(
            game=game, genre=by_game[game]["genre"], subgenre=by_game[game]["subgenre"]
        )
        try:
            raw = backend.generate(frames, prompt, args.fps, max_new_tokens=args.context_tokens)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("context failed for %s: %r", game, exc)
            n_fail += 1
            continue

        parsed = extract_json_object(raw)
        if parsed is not None:
            parsed = {k: str(v) for k, v in parsed.items() if k in CONTEXT_KEYS and v}
            if not parsed:
                parsed = None

        contexts[game] = {
            "game_name": game,
            "genre": by_game[game]["genre"],
            "subgenre": by_game[game]["subgenre"],
            "n_context_frames": len(frames),
            "context": parsed if parsed is not None else {"raw": raw},
            "parse_ok": parsed is not None,
            "model": backend.model_id,
        }
        context_path.write_text(json.dumps(contexts, indent=2, ensure_ascii=False))
        LOGGER.info("[%d/%d] context built for %s (parse_ok=%s)",
                    i, len(todo), game, parsed is not None)

    ok = sum(1 for v in contexts.values() if v.get("parse_ok"))
    LOGGER.info("game context: %d/%d parsed, %d failed -> %s",
                ok, len(contexts), n_fail, context_path)


def render_context(entry: Optional[Dict[str, Any]], max_chars: int) -> str:
    if not entry:
        return "(no reference note available for this game)"
    ctx = entry.get("context", {})
    if "raw" in ctx:
        text = str(ctx["raw"])
    else:
        text = "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in ctx.items())
    return text[:max_chars].strip() or "(empty reference note)"


# --------------------------------------------------------------------------- #
# Stage 2b: scanner
# --------------------------------------------------------------------------- #

SCREEN_PROMPT = """You are screening a short gameplay clip from "{game}" ({genre} / {subgenre}) for software defects.

Reference note on what is NORMAL for this game:
{context}

The clip is {n_frames} frames sampled at {fps:g} fps, covering seconds {t_start:.1f} to {t_end:.1f} of a longer recording.

A DEFECT is a failure of the game software: impossible physics, geometry interpenetration, missing or corrupted textures, broken lighting, frozen or malformed animation, an agent acting in a way the engine clearly did not intend, corrupted UI, or an invalid game state. Unusual but legitimate gameplay, stylistic rendering choices, and normal motion blur are NOT defects.

Does this clip contain a defect? Answer with exactly one word: YES or NO."""

DESCRIBE_PROMPT = """You are analysing a short gameplay clip from "{game}" ({genre} / {subgenre}) that has already been flagged as likely containing a software defect.

Reference note on what is NORMAL for this game:
{context}

The clip is {n_frames} frames sampled at {fps:g} fps, covering seconds {t_start:.1f} to {t_end:.1f} of a longer recording. Frames are indexed 0 to {last_frame}.

Report every distinct defect you can see, at most {max_findings}. Use only these categories:
{taxonomy}

Return ONE JSON object and nothing else:
{{"findings": [{{"category": "<one category above>", "evidence": "one concrete sentence naming what is wrong and where on screen", "salient_frames": [<frame indices>]}}]}}

If on closer inspection there is no defect, return {{"findings": []}}."""


def cmd_scan(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    windows = read_jsonl(out_dir / "windows.jsonl")
    if not windows:
        raise RuntimeError("windows.jsonl is empty; run `preprocess` first")

    leaky = {"gt_intervals", "bugs", "no_bugs", "bug_spans"} & set(windows[0])
    assert not leaky, f"window manifest leaks annotations into the scanner: {leaky}"

    context_path = out_dir / "game_context.json"
    contexts: Dict[str, Any] = {}
    if context_path.exists():
        try:
            contexts = json.loads(context_path.read_text())
        except json.JSONDecodeError:
            LOGGER.error("game_context.json is corrupt; scanning without reference notes")
    if not contexts:
        LOGGER.warning("no usable game_context.json; scanning without reference notes")

    if args.limit_windows is not None:
        windows = windows[: args.limit_windows]

    scan_path = out_dir / "scan.jsonl"
    done: set = set()
    if scan_path.exists() and args.resume:
        done = {row["window_uid"] for row in read_jsonl(scan_path)}
        LOGGER.info("resuming: %d windows already scanned", len(done))
    todo = [w for w in windows if w["window_uid"] not in done]
    LOGGER.info("scanning %d / %d windows", len(todo), len(windows))
    if not todo:
        return

    backend = QwenVLBackend(
        args.model, args.dtype, args.load_in_4bit, args.max_pixels, args.min_pixels,
        attn_impl=args.attn, fit=args.fit,
    )
    yes_ids, no_ids = backend.binary_token_sets()

    taxonomy_str = ", ".join(GLITCH_TAXONOMY)
    mode = "a" if (args.resume and scan_path.exists()) else "w"
    n_parse_fail = 0
    n_described = 0
    n_screen_fail = 0
    t0 = time.time()

    with scan_path.open(mode, encoding="utf-8") as sink:
        for i, win in enumerate(todo, 1):
            entry = contexts.get(win["game_name"])
            fields = {
                "game": win["game_name"],
                "genre": win["genre"] or "unknown",
                "subgenre": win["subgenre"] or "unknown",
                "context": render_context(entry, args.context_chars),
                "n_frames": len(win["frame_paths"]),
                "fps": float(win["fps"]),
                "t_start": float(win["t_start"]),
                "t_end": float(win["t_end"]),
            }

            try:
                p_glitch = backend.binary_probability(
                    win["frame_paths"], SCREEN_PROMPT.format(**fields), win["fps"],
                    yes_ids, no_ids,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("screening failed for %s: %r", win["window_uid"], exc)
                n_screen_fail += 1
                continue

            has_glitch = bool(p_glitch >= args.threshold)
            findings: List[Dict[str, Any]] = []
            raw = ""

            if has_glitch or args.always_describe:
                describe_fields = dict(
                    fields,
                    last_frame=max(0, len(win["frame_paths"]) - 1),
                    max_findings=args.max_findings,
                    taxonomy=taxonomy_str,
                )
                try:
                    raw = backend.generate(
                        win["frame_paths"], DESCRIBE_PROMPT.format(**describe_fields),
                        win["fps"], max_new_tokens=args.scan_tokens,
                    )
                    n_described += 1
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error("description failed for %s: %r", win["window_uid"], exc)
                    raw = ""
                parsed = extract_json_object(raw)
                if parsed is None:
                    n_parse_fail += 1
                    parsed = {}
                findings = normalise_findings(
                    parsed, n_frames=len(win["frame_paths"]), max_findings=args.max_findings
                )

            if has_glitch and not findings:
                # The screening head asserted a defect but the description gave
                # no usable structure. Keep the window alive as unattributed
                # rather than silently converting it to a negative, which would
                # bias recall downward for reasons that have nothing to do with
                # detection.
                findings = [{
                    "category": "state_logic",
                    "category_coarse": "Other",
                    "evidence": "",
                    "salient_frames": [],
                    "unattributed": True,
                }]
            if not has_glitch:
                findings = []

            sink.write(json.dumps({
                "window_uid": win["window_uid"],
                "video_name": win["video_name"],
                "game_name": win["game_name"],
                "window_index": win["window_index"],
                "t_start": win["t_start"],
                "t_end": win["t_end"],
                "has_glitch": has_glitch,
                "p_glitch": p_glitch,
                # decision margin, NOT a calibrated confidence
                "margin": abs(p_glitch - 0.5) * 2.0,
                "n_findings": len(findings),
                "findings": findings,
                "categories": sorted({f["category"] for f in findings}),
                "categories_coarse": sorted({f["category_coarse"] for f in findings}),
                "category": findings[0]["category"] if findings else "none",
                "evidence": findings[0]["evidence"] if findings else "",
                "salient_frames": findings[0]["salient_frames"] if findings else [],
                "raw": raw if args.keep_raw else "",
                "used_context": bool(entry),
                "model": backend.model_id,
                "threshold": args.threshold,
            }, ensure_ascii=False) + "\n")
            sink.flush()

            if i % 25 == 0:
                rate = i / max(time.time() - t0, 1e-9)
                LOGGER.info(
                    "scanned %d/%d (%.2f win/s, eta %.1f min, %d described, "
                    "%d parse failures, %d screen failures)",
                    i, len(todo), rate, (len(todo) - i) / max(rate, 1e-9) / 60.0,
                    n_described, n_parse_fail, n_screen_fail,
                )

    LOGGER.info("stage-2 done in %.1f min -> %s", (time.time() - t0) / 60.0, scan_path)
    summarise_scan(scan_path)


def summarise_scan(scan_path: Path) -> None:
    rows = read_jsonl(scan_path)
    if not rows:
        return
    probs = np.array([r["p_glitch"] for r in rows], dtype=float)
    probs = probs[np.isfinite(probs)]
    positive = sum(bool(r["has_glitch"]) for r in rows)
    counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        for cat in r.get("categories") or []:
            counts[cat] += 1
    multi = sum(int(r.get("n_findings", 0)) > 1 for r in rows)
    unattributed = sum(
        any(f.get("unattributed") for f in r.get("findings", [])) for r in rows
    )
    LOGGER.info(
        "scan summary: %d windows, %.1f%% positive, %d with >1 finding, "
        "%d unattributed, p_glitch mean=%.3f sd=%.3f",
        len(rows), 100.0 * positive / len(rows), multi, unattributed,
        float(probs.mean()) if probs.size else float("nan"),
        float(probs.std()) if probs.size else float("nan"),
    )
    LOGGER.info("category histogram: %s", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
    if probs.size:
        qs = np.quantile(probs, [0.01, 0.25, 0.5, 0.75, 0.99])
        LOGGER.info("p_glitch quantiles (1/25/50/75/99): %s", np.round(qs, 4).tolist())
    if probs.size and probs.std() < 1e-3:
        LOGGER.warning(
            "p_glitch is degenerate (sd<1e-3): the screening head is not "
            "discriminating. Do not calibrate on this; fix the prompt or the "
            "pixel budget first."
        )
    if unattributed > 0.3 * max(positive, 1):
        LOGGER.warning(
            "%d of %d positives are unattributed: the description pass is not "
            "producing parseable JSON often enough. Raise --scan-tokens.",
            unattributed, positive,
        )


# --------------------------------------------------------------------------- #
# analysis-only: window <-> bug alignment
# --------------------------------------------------------------------------- #


def overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def cmd_labels(args: argparse.Namespace) -> None:
    """
    Emit window_labels.jsonl: for every window, which BUGS it overlaps, not
    just whether it overlaps something. A window is positive for bug i if its
    temporal overlap with any span of bug i covers at least --min-cover of the
    window OR at least --min-cover of that span. The second clause matters for
    glitches shorter than one window, which a pure window-IoU rule silently
    drops.

    This file is for evaluation and threshold selection only. The scanner
    never reads it.
    """
    out_dir = Path(args.out_dir)
    videos = {v["video_name"]: v for v in read_jsonl(out_dir / "videos.jsonl")}
    windows = read_jsonl(out_dir / "windows.jsonl")

    rows: List[Dict[str, Any]] = []
    for win in windows:
        video = videos.get(win["video_name"])
        if video is None:
            raise KeyError(f"window references unknown video {win['video_name']}")
        span = (float(win["t_start"]), float(win["t_end"]))
        width = max(span[1] - span[0], 1e-6)
        descriptions = video.get("bugs") or []
        hits: List[Dict[str, Any]] = []
        for bug_idx, spans in enumerate(video.get("bug_spans", [])):
            best = 0.0
            for iv in spans:
                # --- FIX: skip malformed intervals ---
                if not isinstance(iv, (list, tuple)) or len(iv) < 2:
                    continue
                try:
                    a, b = float(iv[0]), float(iv[1])
                except (TypeError, ValueError):
                    continue
                iv_t = (a, b)
                # --- end fix ---
                ov = overlap(span, iv_t)
                if ov <= 0.0:
                    continue
                best = max(best, max(ov / width, ov / max(iv_t[1] - iv_t[0], 1e-6)))
            if best >= args.min_cover:
                hits.append({
                    "bug_index": bug_idx,
                    "coverage": round(best, 4),
                    "description": descriptions[bug_idx] if bug_idx < len(descriptions) else "",
                })
        rows.append({
            "window_uid": win["window_uid"],
            "video_name": win["video_name"],
            "t_start": span[0],
            "t_end": span[1],
            "y": int(bool(hits)),
            "n_bugs_covered": len(hits),
            "bug_hits": hits,
        })

    n = write_jsonl(out_dir / "window_labels.jsonl", rows)
    pos = sum(r["y"] for r in rows)
    multi = sum(r["n_bugs_covered"] > 1 for r in rows)
    LOGGER.info(
        "window_labels.jsonl: %d rows, %d positive (%.1f%%), %d covering >1 bug",
        n, pos, 100.0 * pos / max(n, 1), multi,
    )
    covered: Dict[str, set] = defaultdict(set)
    for r in rows:
        for hit in r["bug_hits"]:
            covered[r["video_name"]].add(hit["bug_index"])
    total_bugs = sum(len(v.get("bug_spans", [])) for v in videos.values())
    reached = sum(len(s) for s in covered.values())
    LOGGER.info(
        "window grid reaches %d/%d annotated bugs (%.1f%%); anything unreached "
        "is unrecoverable by ANY downstream stage at this fps/stride",
        reached, total_bugs, 100.0 * reached / max(total_bugs, 1),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", default="./vgb_work")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("-v", "--verbose", action="count", default=1)


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--attn", default="sdpa",
                        choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28,
                        help="per-frame upper pixel budget after smart_resize")
    parser.add_argument("--min-pixels", type=int, default=64 * 28 * 28)
    parser.add_argument("--fit", default="letterbox", choices=["letterbox", "stretch"],
                        help="how frames of differing aspect ratio are unified; "
                             "letterbox preserves geometry, stretch does not")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("preprocess", help="decode to a fixed fps grid and build windows")
    add_common(pre)
    pre.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    pre.add_argument("--split", default="train")
    pre.add_argument("--video-root", default=DEFAULT_VIDEO_ROOT)
    pre.add_argument("--fps", type=float, default=4.0)
    pre.add_argument("--window", type=int, default=8)
    pre.add_argument("--stride", type=int, default=4)
    pre.add_argument("--short-side", type=int, default=360)
    pre.add_argument("--jpeg-quality", type=int, default=92)
    pre.add_argument("--workers", type=int, default=8)
    pre.add_argument("--limit", type=int, default=None)
    pre.add_argument("--games", nargs="*", default=None)
    pre.add_argument("--overwrite", action="store_true")
    pre.set_defaults(func=cmd_preprocess)

    ver = sub.add_parser("verify", help="check the frame cache before spending GPU time")
    add_common(ver)
    ver.add_argument("--deep", action="store_true", help="open every cached frame")
    ver.add_argument("--sample", type=int, default=2000)
    ver.set_defaults(func=cmd_verify)

    ctx = sub.add_parser("context", help="build the per-game reference note (RAG memory)")
    add_common(ctx)
    add_model_args(ctx)
    ctx.add_argument("--fps", type=float, default=1.0,
                     help="nominal fps reported to the model for the context montage")
    ctx.add_argument("--context-frames", type=int, default=16)
    ctx.add_argument("--context-tokens", type=int, default=512)
    ctx.add_argument("--overwrite", action="store_true")
    ctx.set_defaults(func=cmd_context)

    scan = sub.add_parser("scan", help="screen every window with the VLM")
    add_common(scan)
    add_model_args(scan)
    scan.add_argument("--threshold", type=float, default=0.5)
    scan.add_argument("--max-findings", type=int, default=3)
    scan.add_argument("--scan-tokens", type=int, default=256)
    scan.add_argument("--context-chars", type=int, default=1200)
    scan.add_argument("--limit-windows", type=int, default=None)
    scan.add_argument("--always-describe", action="store_true",
                      help="run the description pass on negatives too (slower)")
    scan.add_argument("--resume", action="store_true")
    scan.add_argument("--keep-raw", action="store_true")
    scan.set_defaults(func=cmd_scan)

    lab = sub.add_parser("labels", help="analysis-only per-window, per-bug ground truth")
    add_common(lab)
    lab.add_argument("--min-cover", type=float, default=0.25)
    lab.set_defaults(func=cmd_labels)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    set_seed(args.seed)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())