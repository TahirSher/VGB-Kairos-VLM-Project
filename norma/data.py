"""
Corpus access on top of the Stage-1 frame cache, gold events, and the
per-unit training targets.

A "unit" is the temporal resolution at which the backbone emits features:
one temporal patch (2 frames) for Qwen2.5-VL / UI-TARS, one frame for
InternVL2.5. Units carry explicit [t0, t1) times, so every target below is
defined in seconds and is independent of the backbone's frame grouping.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import (
    LOGGER,
    Interval,
    import_stage12,
    merge_intervals,
    normalise_intervals,
    read_json,
    read_jsonl,
    write_jsonl,
)

NO_GLITCH = "NO_GLITCH"


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


def _frames_marker(frames_dir: Path, game: str, video_name: str) -> Path:
    # Mirrors the layout written by vgb_stage12.preprocess_one.
    return frames_dir / re.sub(r"[^\w.-]+", "_", game) / Path(video_name).stem / "_frames.json"


def load_corpus(work_dir: Path, require_frames: bool = True) -> Dict[str, VideoItem]:
    """
    Read videos.jsonl + the per-video frame markers produced by
    `vgb_stage12.py preprocess`.

    Timestamps are re-based so the first decoded frame is t=0. Stage 1 stores
    raw pts * time_base; containers with a non-zero stream start_time (edit
    lists, B-frame delay) would otherwise shift every unit against the
    annotation clock and bias tIoU systematically.
    """
    work_dir = Path(work_dir)
    frames_dir = work_dir / "frames"
    items: Dict[str, VideoItem] = {}
    shifted = 0
    missing = 0
    for row in read_jsonl(work_dir / "videos.jsonl"):
        marker = _frames_marker(frames_dir, row["game_name"], row["video_name"])
        payload = read_json(marker, required=False)
        if payload is None or not payload.get("frame_paths"):
            missing += 1
            if require_frames:
                continue
            payload = {"frame_paths": [], "timestamps": [], "target_fps": 4.0}
        stamps = [float(t) for t in payload.get("timestamps", [])]
        if stamps and stamps[0] > 0.05:
            shifted += 1
        if stamps:
            origin = stamps[0]
            stamps = [t - origin for t in stamps]
        bugs = [str(b).strip() for b in (row.get("bugs") or []) if str(b).strip()]
        spans = [normalise_intervals(s) for s in (row.get("bug_spans") or [])]
        items[row["video_name"]] = VideoItem(
            name=row["video_name"],
            game=str(row.get("game_name") or "unknown"),
            genre=str(row.get("genre") or ""),
            subgenre=str(row.get("subgenre") or ""),
            duration=float(row.get("duration") or (stamps[-1] if stamps else 0.0)),
            fps=float(payload.get("target_fps") or 4.0),
            frame_paths=list(payload.get("frame_paths", [])),
            timestamps=stamps,
            no_bugs=bool(row.get("no_bugs")) or not any(spans),
            bugs=bugs,
            bug_spans=spans,
        )
    if missing:
        LOGGER.warning("%d videos in videos.jsonl have no frame cache%s", missing,
                       " and were skipped" if require_frames else "")
    if shifted:
        LOGGER.info("%d videos had a non-zero first pts and were re-based to t=0", shifted)
    if not items:
        raise RuntimeError(f"no usable videos under {work_dir}; run vgb_stage12.py preprocess first")
    return items


def gold_events(item: VideoItem) -> List[Dict[str, Any]]:
    """
    One event per annotated bug, each with its own (possibly multi-interval)
    support. When the annotation arity was unrecoverable at ingest (Stage 1
    then keeps a single unattributed group), the descriptions are joined so no
    gold event is silently lost from the denominator M.
    """
    if item.no_bugs:
        return []
    spans = [s for s in item.bug_spans if s]
    if spans and len(spans) == len(item.bugs):
        return [{"description": d, "intervals": merge_intervals(s)}
                for d, s in zip(item.bugs, spans)]
    if spans:
        flat = merge_intervals([iv for s in spans for iv in s])
        return [{"description": " ".join(item.bugs).strip() or "glitch", "intervals": flat}]
    return []


# --------------------------------------------------------------------------- #
# per-unit targets
# --------------------------------------------------------------------------- #


def unit_targets(
    unit_times: np.ndarray, events: Sequence[Dict[str, Any]], sigma_units: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    y[t]     1 if the centre of unit t lies inside any gold interval
    inst[t]  index of the event owning unit t; -1 background; -2 ambiguous
             (two concurrent events), excluded from the embedding loss
    start[t], end[t]  Gaussian boundary heat-maps (BMN-style soft targets),
             max over every interval boundary, sigma in units
    """
    n = int(unit_times.shape[0])
    centres = unit_times.mean(axis=1)
    y = np.zeros(n, dtype=np.float32)
    inst = np.full(n, -1, dtype=np.int64)
    start = np.zeros(n, dtype=np.float32)
    end = np.zeros(n, dtype=np.float32)
    idx = np.arange(n, dtype=np.float32)
    for k, event in enumerate(events):
        for a, b in event["intervals"]:
            inside = (centres >= a) & (centres < b)
            if not inside.any():
                # Glitch shorter than one unit: claim the unit that contains it.
                inside = np.zeros(n, dtype=bool)
                hit = int(np.argmin(np.abs(centres - 0.5 * (a + b))))
                inside[hit] = True
            y[inside] = 1.0
            clash = inside & (inst >= 0) & (inst != k)
            inst[inside & (inst == -1)] = k
            inst[clash] = -2
            s_idx = float(np.argmin(np.abs(unit_times[:, 0] - a)))
            e_idx = float(np.argmin(np.abs(unit_times[:, 1] - b)))
            start = np.maximum(start, np.exp(-0.5 * ((idx - s_idx) / sigma_units) ** 2))
            end = np.maximum(end, np.exp(-0.5 * ((idx - e_idx) / sigma_units) ** 2))
    return {"y": y, "inst": inst, "start": start, "end": end}


def select_frames(
    item: VideoItem, t0: float, t1: float, max_frames: int, even: bool = True
) -> Tuple[List[str], float]:
    """Uniformly subsample cached frames inside [t0, t1]; returns the frames
    and the effective fps actually represented by the subsample."""
    stamps = np.asarray(item.timestamps, dtype=float)
    keep = np.nonzero((stamps >= t0 - 1e-6) & (stamps <= t1 + 1e-6))[0]
    if keep.size == 0:
        keep = np.array([int(np.argmin(np.abs(stamps - 0.5 * (t0 + t1))))])
    if keep.size > max_frames:
        pick = np.linspace(0, keep.size - 1, num=max_frames)
        keep = keep[np.round(pick).astype(int)]
    keep = list(dict.fromkeys(int(k) for k in keep))
    if even and len(keep) % 2 == 1:
        keep.append(keep[-1])
    span = max(stamps[keep[-1]] - stamps[keep[0]], 1e-3)
    eff_fps = (len(keep) - 1) / span if len(keep) > 1 else item.fps
    return [item.frame_paths[k] for k in keep], float(max(eff_fps, 0.25))


# --------------------------------------------------------------------------- #
# generic (non-game) datasets
# --------------------------------------------------------------------------- #


def prepare_generic(
    manifest: Path, out_dir: Path, fps: float, short_side: int, jpeg_quality: int,
    window: int = 8, stride: int = 8, stage12_path: Optional[str] = None,
) -> None:
    """
    Bring any temporally-annotated anomaly dataset into the Stage-1 layout,
    so NORMA trains or transfers on it unchanged. Manifest rows (JSONL):

      {"video_path": "...", "game_name": "<scene/domain id>", "genre": "...",
       "subgenre": "...", "no_bugs": false, "bugs": ["description", ...],
       "time_nodes": [[[start, end], ...], ...]}

    For surveillance VAD (UCF-Crime, XD-Violence, MSAD) "game_name" is the
    camera / scene id and "genre" the anomaly family; descriptions may be the
    class name when no free text exists.
    """
    stage12 = import_stage12(stage12_path)
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    videos: List[Dict[str, Any]] = []
    windows: List[Dict[str, Any]] = []
    for uid, row in enumerate(read_jsonl(manifest)):
        path = Path(row["video_path"])
        bugs = [str(b) for b in (row.get("bugs") or [])]
        spans, flat = stage12.parse_time_nodes(row.get("time_nodes") or [], len(bugs), path.name)
        record = stage12.VideoRecord(
            uid=int(row.get("uid", uid)), game_name=str(row.get("game_name", "unknown")),
            genre=str(row.get("genre", "")), subgenre=str(row.get("subgenre", "")),
            video_name=path.name, video_path=str(path), no_bugs=bool(row.get("no_bugs", not flat)),
            bugs=bugs, bug_spans=spans, gt_intervals=flat,
        )
        try:
            result = stage12.preprocess_one(
                record.to_json(), str(frames_dir), fps, short_side, window, stride,
                jpeg_quality, False,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("preprocess failed for %s: %r", path, exc)
            continue
        videos.append({k: v for k, v in result.items() if k != "_windows"})
        windows.extend(result["_windows"])
    write_jsonl(out_dir / "videos.jsonl", videos)
    write_jsonl(out_dir / "windows.jsonl", windows)
    LOGGER.info("generic dataset: %d videos -> %s", len(videos), out_dir)
