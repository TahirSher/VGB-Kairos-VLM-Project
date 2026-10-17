"""
Shared utilities: logging, seeding, JSON I/O, interval algebra and the
Stage 1/2 bridge.

The interval algebra is the single source of truth for every temporal
quantity in NORMA (labels, decoding, tIoU in the evaluator), so the training
targets and the metric can never disagree about what "overlap" means.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("norma")

Interval = Tuple[float, float]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STAGE12 = REPO_ROOT / "vgb_stage12.py"


# --------------------------------------------------------------------------- #
# logging / seeding / io
# --------------------------------------------------------------------------- #


def setup_logging(verbosity: int = 1) -> None:
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if level > logging.DEBUG:
        for noisy in ("urllib3", "filelock", "huggingface_hub", "datasets", "PIL", "httpx"):
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


def sha1_short(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def read_jsonl(path: Path, required: bool = True) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"required artefact not found: {path}")
        return []
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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_json(path: Path, required: bool = True) -> Any:
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"required artefact not found: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


# --------------------------------------------------------------------------- #
# scalar helpers
# --------------------------------------------------------------------------- #


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def expit(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def harmonic(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


# --------------------------------------------------------------------------- #
# interval algebra
# --------------------------------------------------------------------------- #


def normalise_intervals(raw: Any) -> List[Interval]:
    """Every [a, b] leaf below `raw`, depth-agnostic, ordered, merged."""
    out: List[Interval] = []

    def walk(node: Any) -> None:
        if node is None or not isinstance(node, (list, tuple)):
            return
        if len(node) == 2 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in node
        ):
            a, b = float(node[0]), float(node[1])
            if b < a:
                a, b = b, a
            if b > a:
                out.append((a, b))
            return
        for child in node:
            walk(child)

    walk(raw)
    return merge_intervals(out)


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


def measure(intervals: Sequence[Interval]) -> float:
    return float(sum(b - a for a, b in merge_intervals(intervals)))


def intersect_measure(left: Sequence[Interval], right: Sequence[Interval]) -> float:
    a = merge_intervals(left)
    b = merge_intervals(right)
    total = 0.0
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def interval_iou(left: Sequence[Interval], right: Sequence[Interval]) -> float:
    """tIoU between two interval SETS (multi-interval events), as in GliDe."""
    inter = intersect_measure(left, right)
    union = measure(list(left) + list(right))
    return inter / union if union > 0.0 else 0.0


# --------------------------------------------------------------------------- #
# Stage 1/2 bridge
# --------------------------------------------------------------------------- #


def import_stage12(path: Optional[str] = None) -> Any:
    """Load vgb_stage12.py as a module. NORMA reuses its metadata ingest,
    frame cache, geometry-safe frame loader and Qwen2.5-VL backend instead of
    duplicating them."""
    if "vgb_stage12" in sys.modules:
        return sys.modules["vgb_stage12"]
    candidates = [Path(path)] if path else []
    candidates += [DEFAULT_STAGE12, Path.cwd() / "vgb_stage12.py"]
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("vgb_stage12", candidate)
            module = importlib.util.module_from_spec(spec)
            sys.modules["vgb_stage12"] = module
            spec.loader.exec_module(module)
            LOGGER.debug("loaded Stage 1/2 module from %s", candidate)
            return module
    raise FileNotFoundError(
        "vgb_stage12.py could not be located; pass --stage12 /path/to/vgb_stage12.py"
    )
