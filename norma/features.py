"""
One-pass feature extraction and the on-disk feature cache.

Each video is cut into chunks of `chunk_frames` (a multiple of the backbone's
unit size); each chunk costs ONE prefill of the backbone up to the deepest
tapped layer. Per unit we store, for every tapped layer, the RMS-normalised
mean- and max-pooled visual-token states, reduced from D to P dimensions by a
fixed, seeded random orthogonal map. Johnson-Lindenstrauss: pairwise
distances are preserved up to (1 +- eps) w.h.p. for P = O(eps^-2 log n), so
the kNN normality bank built on these vectors keeps its geometry while the
cache shrinks by D/P (3.5x for Qwen2.5-VL-7B at P=1024).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import LOGGER, read_json, write_json
from .data import VideoItem


class Projector:
    def __init__(self, dim_in: int, dim_out: int, seed: int) -> None:
        self.dim_in, self.dim_out = int(dim_in), int(dim_out)
        if self.dim_out >= self.dim_in:
            self.matrix = None
            self.dim_out = self.dim_in
            return
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.standard_normal((self.dim_in, self.dim_out)))
        # Scale so that E||Q^T x||^2 = ||x||^2 for isotropic x.
        self.matrix = (q * math.sqrt(self.dim_in / self.dim_out)).astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x if self.matrix is None else x @ self.matrix


def feature_path(feat_dir: Path, item: VideoItem) -> Path:
    return Path(feat_dir) / f"{item.stem}.npz"


def subsample(item: VideoItem, fps: float) -> Tuple[List[str], List[float], float]:
    stride = max(1, int(round(item.fps / max(fps, 1e-3))))
    return item.frame_paths[::stride], item.timestamps[::stride], item.fps / stride


def extract_corpus(items: Dict[str, VideoItem], backbone: Any, feat_dir: Path, fps: float,
                   chunk_frames: int, proj_dim: int, seed: int, overwrite: bool = False,
                   names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    feat_dir = Path(feat_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)
    projector = Projector(backbone.hidden_size, proj_dim, seed)
    unit = max(1, int(backbone.unit_frames))
    chunk = max(unit, (int(chunk_frames) // unit) * unit)
    meta = {
        "backbone": backbone.name, "hidden_size": backbone.hidden_size,
        "proj_dim": projector.dim_out, "proj_seed": seed, "fps": fps,
        "chunk_frames": chunk, "unit_frames": unit,
        "layers": getattr(backbone, "layer_idx", None),
    }
    old = read_json(feat_dir / "meta.json", required=False)
    if old and not overwrite:
        for key in ("backbone", "proj_dim", "proj_seed", "fps", "chunk_frames"):
            if old.get(key) != meta[key]:
                raise RuntimeError(
                    f"feature cache {feat_dir} was built with {key}={old.get(key)!r}, "
                    f"requested {meta[key]!r}; use a new --feat-dir or --overwrite"
                )
    write_json(feat_dir / "meta.json", meta)

    todo = [items[n] for n in (names or sorted(items))]
    n_done = n_pass = 0
    t0 = time.time()
    for i, item in enumerate(todo, 1):
        path = feature_path(feat_dir, item)
        if path.exists() and not overwrite:
            continue
        frames, stamps, eff_fps = subsample(item, fps)
        if not frames:
            LOGGER.warning("no frames for %s", item.name)
            continue
        feats, ctxs, times = [], [], []
        for s in range(0, len(frames), chunk):
            f_chunk, t_chunk = frames[s:s + chunk], stamps[s:s + chunk]
            game_text = {"game": item.game, "genre": item.genre or "unknown",
                         "subgenre": item.subgenre or "unknown"}
            try:
                u_feat, u_time, ctx = backbone.extract_units(f_chunk, t_chunk, eff_fps, game_text)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("extraction failed for %s chunk %d: %r", item.name, s // chunk, exc)
                feats = []
                break
            n_pass += 1
            feats.append(projector(u_feat.reshape(-1, u_feat.shape[-1])).reshape(
                u_feat.shape[0], u_feat.shape[1], -1))
            ctxs.append(np.repeat(projector(ctx[None])[0][None], u_feat.shape[0], axis=0))
            times.append(u_time)
        if not feats:
            continue
        np.savez_compressed(
            path,
            feats=np.concatenate(feats).astype(np.float16),
            ctx=np.concatenate(ctxs).astype(np.float16),
            times=np.concatenate(times).astype(np.float32),
        )
        n_done += 1
        if i % 50 == 0:
            rate = i / max(time.time() - t0, 1e-9)
            LOGGER.info("extracted %d/%d videos (%.2f vid/s, %d prefills)", i, len(todo), rate, n_pass)
    elapsed = time.time() - t0
    stats = {"videos": n_done, "prefills": n_pass, "seconds": elapsed,
             "seconds_per_video": elapsed / max(n_done, 1)}
    write_json(feat_dir / "extract_stats.json", stats)
    LOGGER.info("feature extraction: %d videos, %d prefills, %.1fs -> %s",
                n_done, n_pass, elapsed, feat_dir)
    return stats


def load_features(feat_dir: Path, items: Dict[str, VideoItem],
                  names: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    missing = 0
    for name in names:
        path = feature_path(feat_dir, items[name])
        if not path.exists():
            missing += 1
            continue
        with np.load(path) as data:
            out[name] = {k: data[k] for k in ("feats", "ctx", "times")}
    if missing:
        LOGGER.warning("%d/%d requested videos have no cached features", missing, len(names))
    return out
