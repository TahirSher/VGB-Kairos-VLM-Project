#!/usr/bin/env python3

from __future__ import annotations

import argparse
import bisect
import csv
import dataclasses
import hashlib
import importlib.util
import json
import logging
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

LOGGER = logging.getLogger("kairos")

DEFAULT_SPLIT_DIR = "/home/tahir/VLM-Project/VideoGlitchBench/splits/"
DEFAULT_STAGE12 = "vgb_stage12.py"
SPLIT_NAMES = ("train", "dev", "test")
SPLIT_ALIASES = {
    "train": "train", "training": "train", "trn": "train",
    "dev": "dev", "val": "dev", "valid": "dev", "validation": "dev", "eval": "dev",
    "test": "test", "tst": "test",
}

COARSE = ("Physics", "Visual", "Game Logic", "Other")

FINE_TO_COARSE = {
    "physics": "Physics",
    "clipping": "Physics",
    "rendering": "Visual",
    "lighting": "Visual",
    "animation": "Visual",
    "agent_behaviour": "Game Logic",
    "ui_hud": "Visual",
    "state_logic": "Game Logic",
}

STOPWORDS = frozenset("""
a an the and or but if then than that this these those there here it its it's is are was were be been
being of in on at to from by with without for as into onto over under across through during while
appears appear appearing seems seem seeming looks look looking shows show showing indicates indicate
indicating suggests suggest suggesting which who whom whose what when where how not no nor so such
very more most much many some any each other another same both few several one two three
game games player players character characters object objects scene screen frame frames video clip
""".split())


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
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_json(path: Path, required: bool = True) -> Any:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"required artefact not found: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
# splits
# --------------------------------------------------------------------------- #


def _canonical_split(name: str) -> Optional[str]:
    key = re.sub(r"[^a-z]+", "", name.lower())
    return SPLIT_ALIASES.get(key)


def _names_from_payload(payload: Any) -> List[str]:
    out: List[str] = []

    def walk(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, str):
            out.append(node)
            return
        if isinstance(node, dict):
            for key in ("video_name", "video", "name", "file", "filename", "path", "id"):
                if key in node and isinstance(node[key], str):
                    out.append(node[key])
                    return
            for value in node.values():
                walk(value)
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(payload)
    return out


def _normalise_name(raw: str) -> str:
    raw = raw.strip().strip('"').strip("'")
    if not raw:
        return ""
    return os.path.basename(raw)


def load_splits(split_dir: Path) -> Dict[str, List[str]]:
    if not split_dir.is_dir():
        raise NotADirectoryError(f"--split-dir does not exist: {split_dir}")

    found: Dict[str, List[str]] = {}
    for path in sorted(split_dir.rglob("*")):
        if not path.is_file():
            continue
        split = _canonical_split(path.stem)
        if split is None:
            continue
        suffix = path.suffix.lower()
        names: List[str] = []
        if suffix in (".json",):
            names = _names_from_payload(read_json(path))
        elif suffix in (".jsonl", ".ndjson"):
            names = _names_from_payload(read_jsonl(path))
        elif suffix in (".txt", ".list", ".lst", ""):
            names = [ln for ln in path.read_text(encoding="utf-8").splitlines()]
        elif suffix in (".csv", ".tsv"):
            delim = "\t" if suffix == ".tsv" else ","
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle, delimiter=delim)
                rows = list(reader)
            if not rows:
                continue
            header = [c.strip().lower() for c in rows[0]]
            col = 0
            for cand in ("video_name", "video", "name", "file", "filename"):
                if cand in header:
                    col = header.index(cand)
                    rows = rows[1:]
                    break
            names = [r[col] for r in rows if r and len(r) > col]
        else:
            continue
        cleaned = [_normalise_name(n) for n in names]
        cleaned = [n for n in cleaned if n and not n.startswith("#")]
        if not cleaned:
            continue
        if split in found:
            raise RuntimeError(
                f"two files under {split_dir} both resolve to split '{split}'; "
                "remove the ambiguity rather than letting the loader guess"
            )
        found[split] = cleaned
        LOGGER.info("split %-5s <- %s (%d entries)", split, path.name, len(cleaned))

    if not found:
        raise RuntimeError(
            f"no recognisable split files under {split_dir}; expected files named "
            "train/dev|val/test with extension .json .jsonl .txt .csv"
        )

    missing = [s for s in SPLIT_NAMES if s not in found]
    if missing:
        LOGGER.warning("splits absent from %s: %s", split_dir, missing)

    seen: Dict[str, str] = {}
    for split, names in found.items():
        for name in names:
            if name in seen and seen[name] != split:
                raise RuntimeError(
                    f"video {name!r} appears in both '{seen[name]}' and '{split}'; "
                    "the split files overlap and every downstream number would be leaked"
                )
            seen[name] = split
    LOGGER.info("splits are disjoint over %d unique videos", len(seen))
    return found


def _stem_key(name: str) -> str:
    return os.path.splitext(os.path.basename(str(name)))[0].lower()


def resolve_split_membership(
    split_names: Dict[str, List[str]], available: Sequence[str]
) -> Dict[str, List[str]]:
    by_stem: Dict[str, str] = {}
    for name in available:
        by_stem.setdefault(_stem_key(name), name)

    resolved: Dict[str, List[str]] = {}
    for split, names in split_names.items():
        hits: List[str] = []
        misses = 0
        for raw in names:
            actual = by_stem.get(_stem_key(raw))
            if actual is None:
                misses += 1
                continue
            hits.append(actual)
        resolved[split] = sorted(set(hits))
        LOGGER.info(
            "split %-5s: %d/%d listed videos present in the Stage-1 manifest (%d absent)",
            split, len(resolved[split]), len(names), misses,
        )
        if not resolved[split]:
            raise RuntimeError(
                f"split '{split}' has zero overlap with the Stage-1 manifest; "
                "the manifest was probably built with --limit or a --games filter"
            )
    return resolved


# --------------------------------------------------------------------------- #
# interval algebra
# --------------------------------------------------------------------------- #


def normalise_intervals(raw: Any) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []

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


def merge_intervals(
    intervals: Sequence[Tuple[float, float]], gap: float = 0.0
) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted((float(a), float(b)) for a, b in intervals if b > a)
    if not ordered:
        return []
    merged = [list(ordered[0])]
    for a, b in ordered[1:]:
        if a <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def measure(intervals: Sequence[Tuple[float, float]]) -> float:
    return float(sum(b - a for a, b in merge_intervals(intervals)))


def intersect_measure(
    left: Sequence[Tuple[float, float]], right: Sequence[Tuple[float, float]]
) -> float:
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


def interval_iou(
    left: Sequence[Tuple[float, float]], right: Sequence[Tuple[float, float]]
) -> float:
    inter = intersect_measure(left, right)
    union = measure(list(left) + list(right))
    if union <= 0.0:
        return 0.0
    return inter / union


# --------------------------------------------------------------------------- #
# GLIDE evaluation protocol
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class VideoScore:
    video_name: str
    game_name: str
    n_pred: int
    n_gold: int
    n_matched: int
    sum_s: float
    sum_w: float
    sum_iou: float
    ious: List[float]
    pairs: List[Dict[str, Any]]


def match_video(
    pred: Sequence[Dict[str, Any]],
    gold: Sequence[Dict[str, Any]],
    semantic: Callable[[str, str], float],
    keep: str = "positive",
) -> VideoScore:
    n, m = len(pred), len(gold)
    if n == 0 or m == 0:
        return VideoScore("", "", n, m, 0, 0.0, 0.0, 0.0, [], [])

    s_mat = np.zeros((n, m), dtype=float)
    iou_mat = np.zeros((n, m), dtype=float)
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            s_mat[i, j] = semantic(p["description"], g["description"])
            iou_mat[i, j] = interval_iou(p["intervals"], g["intervals"])
    w_mat = s_mat * iou_mat

    rows, cols = linear_sum_assignment(-w_mat)

    sum_s = sum_w = sum_iou = 0.0
    ious: List[float] = []
    pairs: List[Dict[str, Any]] = []
    for i, j in zip(rows, cols):
        if keep == "positive" and w_mat[i, j] <= 0.0:
            continue
        sum_s += float(s_mat[i, j])
        sum_w += float(w_mat[i, j])
        sum_iou += float(iou_mat[i, j])
        ious.append(float(iou_mat[i, j]))
        pairs.append({
            "pred_index": int(i),
            "gold_index": int(j),
            "s": float(s_mat[i, j]),
            "iou": float(iou_mat[i, j]),
            "w": float(w_mat[i, j]),
        })
    return VideoScore("", "", n, m, len(pairs), sum_s, sum_w, sum_iou, ious, pairs)


def aggregate_micro(scores: Sequence[VideoScore]) -> Dict[str, float]:
    total_n = sum(s.n_pred for s in scores)
    total_m = sum(s.n_gold for s in scores)
    total_a = sum(s.n_matched for s in scores)
    sum_s = sum(s.sum_s for s in scores)
    sum_w = sum(s.sum_w for s in scores)
    sum_iou = sum(s.sum_iou for s in scores)

    p_desc = 100.0 * sum_s / total_n if total_n else 0.0
    r_desc = 100.0 * sum_s / total_m if total_m else 0.0
    f1_desc = harmonic(p_desc, r_desc)
    miou = sum_iou / total_a if total_a else 0.0
    p_all = 100.0 * sum_w / total_n if total_n else 0.0
    r_all = 100.0 * sum_w / total_m if total_m else 0.0
    f1_iou = harmonic(p_all, r_all)
    return {
        "precision_desc": p_desc,
        "recall_desc": r_desc,
        "f1_desc": f1_desc,
        "miou": miou,
        "precision_overall": p_all,
        "recall_overall": r_all,
        "f1_x_iou": f1_iou,
        "n_pred": float(total_n),
        "n_gold": float(total_m),
        "n_matched": float(total_a),
    }


def aggregate_macro(scores: Sequence[VideoScore]) -> Dict[str, float]:
    ps, rs, ious, ws_p, ws_r = [], [], [], [], []
    for s in scores:
        if s.n_pred:
            ps.append(s.sum_s / s.n_pred)
            ws_p.append(s.sum_w / s.n_pred)
        if s.n_gold:
            rs.append(s.sum_s / s.n_gold)
            ws_r.append(s.sum_w / s.n_gold)
        if s.n_matched:
            ious.append(s.sum_iou / s.n_matched)
    p_desc = 100.0 * float(np.mean(ps)) if ps else 0.0
    r_desc = 100.0 * float(np.mean(rs)) if rs else 0.0
    p_all = 100.0 * float(np.mean(ws_p)) if ws_p else 0.0
    r_all = 100.0 * float(np.mean(ws_r)) if ws_r else 0.0
    return {
        "precision_desc": p_desc,
        "recall_desc": r_desc,
        "f1_desc": harmonic(p_desc, r_desc),
        "miou": float(np.mean(ious)) if ious else 0.0,
        "precision_overall": p_all,
        "recall_overall": r_all,
        "f1_x_iou": harmonic(p_all, r_all),
    }


def recall_at_tiou(scores: Sequence[VideoScore], thresholds: Sequence[float]) -> Dict[str, float]:
    total_m = sum(s.n_gold for s in scores)
    out: Dict[str, float] = {}
    for thr in thresholds:
        hit = sum(sum(1 for iou in s.ious if iou >= thr) for s in scores)
        out[f"recall@tIoU{thr:g}"] = 100.0 * hit / total_m if total_m else 0.0
    return out


def multi_interval_recovery(
    scores: Sequence[VideoScore],
    pred_by_video: Dict[str, List[Dict[str, Any]]],
    gold_by_video: Dict[str, List[Dict[str, Any]]],
) -> float:
    hit = tot = 0
    for score in scores:
        gold = gold_by_video.get(score.video_name, [])
        pred = pred_by_video.get(score.video_name, [])
        for pair in score.pairs:
            g = gold[pair["gold_index"]]
            if len(merge_intervals(g["intervals"])) < 2:
                continue
            tot += 1
            p = pred[pair["pred_index"]]
            if len(merge_intervals(p["intervals"])) >= 2:
                hit += 1
    return 100.0 * hit / tot if tot else float("nan")


def bootstrap_ci(
    scores: Sequence[VideoScore],
    statistic: Callable[[Sequence[VideoScore]], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 1337,
) -> Tuple[float, float, float]:
    if not scores:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = statistic(scores)
    idx = np.arange(len(scores))
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.choice(idx, size=len(scores), replace=True)
        draws[b] = statistic([scores[i] for i in pick])
    lo = float(np.quantile(draws, alpha / 2.0))
    hi = float(np.quantile(draws, 1.0 - alpha / 2.0))
    return float(point), lo, hi


def paired_bootstrap_pvalue(
    a: Sequence[VideoScore],
    b: Sequence[VideoScore],
    statistic: Callable[[Sequence[VideoScore]], float],
    n_boot: int = 2000,
    seed: int = 1337,
) -> Tuple[float, float]:
    if len(a) != len(b) or not a:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    observed = statistic(a) - statistic(b)
    idx = np.arange(len(a))
    count = 0
    for _ in range(n_boot):
        pick = rng.choice(idx, size=len(a), replace=True)
        diff = statistic([a[i] for i in pick]) - statistic([b[i] for i in pick])
        if (diff - observed) * np.sign(observed) >= abs(observed) or diff * np.sign(observed) <= 0.0:
            count += 1
    p = (count + 1.0) / (n_boot + 1.0)
    return float(observed), float(min(1.0, p))


def wilcoxon_signed_rank(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    from scipy.stats import wilcoxon

    diffs = [float(a) - float(b) for a, b in zip(x, y)]
    if not diffs or all(abs(d) < 1e-12 for d in diffs):
        return float("nan"), 1.0
    try:
        stat, p = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return float("nan"), 1.0
    return float(stat), float(p)


def holm_bonferroni(pvalues: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
    items = sorted(
        ((k, v) for k, v in pvalues.items() if v == v),
        key=lambda kv: kv[1],
    )
    m = len(items)
    out: Dict[str, Dict[str, Any]] = {}
    running = 0.0
    for rank, (key, p) in enumerate(items):
        adjusted = min(1.0, max(running, (m - rank) * p))
        running = adjusted
        out[key] = {"p_raw": p, "p_holm": adjusted, "reject": adjusted <= alpha}
    for key, p in pvalues.items():
        out.setdefault(key, {"p_raw": p, "p_holm": float("nan"), "reject": False})
    return out


def benjamini_hochberg(pvalues: Sequence[float], q: float) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    keep = np.zeros(n, dtype=bool)
    if passed.any():
        cutoff = np.max(np.nonzero(passed)[0])
        keep[order[: cutoff + 1]] = True
    return keep


# --------------------------------------------------------------------------- #
# lightweight text representation
# --------------------------------------------------------------------------- #


def tokenize(text: str) -> List[str]:
    toks = re.findall(r"[a-z][a-z\-']+", str(text).lower())
    return [t for t in toks if len(t) > 2 and t not in STOPWORDS]


class TfIdf:
    def __init__(self, documents: Sequence[str], min_df: int = 1) -> None:
        counts: Counter = Counter()
        tokenised = [tokenize(d) for d in documents]
        for toks in tokenised:
            counts.update(set(toks))
        self.vocab = {t: i for i, (t, c) in enumerate(sorted(counts.items())) if c >= min_df}
        n_docs = max(len(documents), 1)
        self.idf = np.ones(len(self.vocab), dtype=float)
        for term, idx in self.vocab.items():
            self.idf[idx] = math.log((1.0 + n_docs) / (1.0 + counts[term])) + 1.0

    def transform(self, documents: Sequence[str]) -> np.ndarray:
        mat = np.zeros((len(documents), max(len(self.vocab), 1)), dtype=float)
        for row, doc in enumerate(documents):
            tf: Counter = Counter(tokenize(doc))
            if not tf:
                continue
            norm = float(sum(tf.values()))
            for term, count in tf.items():
                idx = self.vocab.get(term)
                if idx is None:
                    continue
                mat[row, idx] = (count / norm) * self.idf[idx]
        lengths = np.linalg.norm(mat, axis=1, keepdims=True)
        lengths[lengths == 0.0] = 1.0
        return mat / lengths


def cosine_affinity(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((matrix.shape[0], matrix.shape[0]), dtype=float)
    sim = matrix @ matrix.T
    return np.clip(sim, 0.0, 1.0)


def average_linkage(affinity: np.ndarray, threshold: float) -> List[List[int]]:
    n = affinity.shape[0]
    if n == 0:
        return []
    clusters = [[i] for i in range(n)]
    sim = affinity.astype(float).copy()
    np.fill_diagonal(sim, -np.inf)
    active = list(range(n))
    members = {i: [i] for i in range(n)}
    while len(active) > 1:
        best = (-np.inf, -1, -1)
        for ai in range(len(active)):
            for bi in range(ai + 1, len(active)):
                a, b = active[ai], active[bi]
                block = affinity[np.ix_(members[a], members[b])]
                value = float(block.mean())
                if value > best[0]:
                    best = (value, a, b)
        if best[0] < threshold:
            break
        _, a, b = best
        members[a] = members[a] + members[b]
        del members[b]
        active.remove(b)
    clusters = [sorted(members[a]) for a in active]
    clusters.sort(key=lambda c: c[0])
    return clusters


# --------------------------------------------------------------------------- #
# judge backends
# --------------------------------------------------------------------------- #


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = re.sub(r"```(?:json)?", "", str(text)).strip()
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


class JudgeBackend:
    def __init__(self, spec: str, model: str, temperature: float, max_tokens: int,
                 base_url: Optional[str] = None, api_key_env: str = "OPENAI_API_KEY",
                 dtype: str = "bfloat16", load_in_4bit: bool = False) -> None:
        self.spec = spec
        self.model_id = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.calls = 0
        self.prompt_chars = 0
        self.seconds = 0.0
        self._client = None
        self._tok = None
        self._lm = None
        if spec == "openai":
            import openai

            key = os.environ.get(api_key_env)
            if not key:
                raise RuntimeError(f"{api_key_env} is not set but --judge-backend openai was requested")
            self._client = openai.OpenAI(api_key=key, base_url=base_url)
        elif spec == "hf":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            quant = None
            if load_in_4bit:
                from transformers import BitsAndBytesConfig

                quant = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(torch, dtype),
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            self._tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            self._lm = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=getattr(torch, dtype),
                device_map="auto",
                quantization_config=quant,
                trust_remote_code=True,
            )
            self._lm.eval()
        elif spec == "vlm":
            self._lm = "deferred"
        elif spec == "offline":
            pass
        else:
            raise ValueError(f"unknown judge backend {spec!r}")

    def attach_vlm(self, vlm_backend: Any) -> None:
        if self.spec != "vlm":
            return
        self._lm = vlm_backend
        self.model_id = getattr(vlm_backend, "model_id", self.model_id)

    def chat(self, system: str, user: str, max_tokens: Optional[int] = None) -> str:
        budget = int(max_tokens or self.max_tokens)
        self.calls += 1
        self.prompt_chars += len(system) + len(user)
        t0 = time.time()
        try:
            if self.spec == "openai":
                out = self._client.chat.completions.create(
                    model=self.model_id,
                    temperature=self.temperature,
                    max_tokens=budget,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return (out.choices[0].message.content or "").strip()
            if self.spec == "hf":
                return self._chat_hf(system, user, budget)
            if self.spec == "vlm":
                return self._chat_vlm(system, user, budget)
            return ""
        finally:
            self.seconds += time.time() - t0

    def _chat_hf(self, system: str, user: str, budget: int) -> str:
        import torch

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tok([text], return_tensors="pt").to(self._lm.device)
        with torch.inference_mode():
            out = self._lm.generate(
                **inputs,
                do_sample=self.temperature > 0.0,
                temperature=max(self.temperature, 1e-5),
                max_new_tokens=budget,
                pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
            )
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self._tok.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def _chat_vlm(self, system: str, user: str, budget: int) -> str:
        import torch

        backend = self._lm
        if backend == "deferred" or backend is None:
            raise RuntimeError("judge backend 'vlm' was never attached to a loaded VLM")
        processor = backend.processor
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(backend.model.device)
        with torch.inference_mode():
            out = backend.model.generate(
                **inputs,
                do_sample=self.temperature > 0.0,
                temperature=max(self.temperature, 1e-5),
                max_new_tokens=budget,
            )
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def stats(self) -> Dict[str, float]:
        return {
            "calls": float(self.calls),
            "prompt_chars": float(self.prompt_chars),
            "seconds": float(self.seconds),
            "model": self.model_id,
            "backend": self.spec,
        }


class DiskCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: Dict[str, Any] = {}
        self.hits = 0
        self.misses = 0
        if path.exists():
            for row in read_jsonl(path, required=False):
                self.data[row["key"]] = row["value"]
            LOGGER.info("loaded %d cached judge responses from %s", len(self.data), path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")

    def get(self, key: str) -> Optional[Any]:
        if key in self.data:
            self.hits += 1
            return self.data[key]
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        self.data[key] = value
        self._handle.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# scoring judge (sees ground truth, writes nothing back)
# --------------------------------------------------------------------------- #

SCORER_SYSTEM = (
    "You are a strict evaluator for a video game QA benchmark. You compare a "
    "system-generated glitch description against a human-written reference "
    "description of the same gameplay video and rate their semantic agreement. "
    "You never reward fluency, length or confidence. You reward only agreement "
    "about which entity fails, in what way it fails, and what kind of software "
    "defect it is."
)

SCORER_PROMPT = """Rate the semantic agreement between the two glitch descriptions below on an integer scale from 0 to 5.

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


class SemanticScorer:
    def __init__(self, judge: JudgeBackend, cache: DiskCache, scale: float = 5.0,
                 fallback: str = "lexical") -> None:
        self.judge = judge
        self.cache = cache
        self.scale = float(scale)
        self.fallback = fallback
        self.n_fallback = 0
        self._tfidf: Optional[TfIdf] = None

    def fit_lexical(self, corpus: Sequence[str]) -> None:
        self._tfidf = TfIdf(list(corpus) or ["glitch"])

    def _lexical(self, pred: str, gold: str) -> float:
        if self._tfidf is None:
            self.fit_lexical([pred, gold])
        vec = self._tfidf.transform([pred, gold])
        return float(np.clip(vec[0] @ vec[1], 0.0, 1.0))

    def __call__(self, pred: str, gold: str) -> float:
        pred = (pred or "").strip()
        gold = (gold or "").strip()
        if not pred or not gold:
            return 0.0
        key = sha1_short(f"{self.judge.model_id}|{self.scale}|{pred}|{gold}", 24)
        cached = self.cache.get(key)
        if cached is not None:
            return float(cached)

        if self.judge.spec == "offline":
            value = self._lexical(pred, gold)
            self.n_fallback += 1
            self.cache.put(key, value)
            return value

        raw = self.judge.chat(
            SCORER_SYSTEM, SCORER_PROMPT.format(gold=gold, pred=pred), max_tokens=160
        )
        parsed = extract_json_object(raw)
        score = None
        if parsed is not None and "score" in parsed:
            try:
                score = float(parsed["score"])
            except (TypeError, ValueError):
                score = None
        if score is None:
            match = re.search(r"\b([0-5])(?:\.0+)?\b", raw)
            score = float(match.group(1)) if match else None
        if score is None:
            self.n_fallback += 1
            value = self._lexical(pred, gold)
        else:
            value = float(np.clip(score, 0.0, self.scale)) / self.scale
        self.cache.put(key, value)
        return value


# --------------------------------------------------------------------------- #
# cross-game normality memory
# --------------------------------------------------------------------------- #


def random_effects_shrinkage(groups: Dict[str, List[float]]) -> Tuple[float, float, float]:
    sizes = {k: len(v) for k, v in groups.items() if len(v) > 0}
    if not sizes:
        return 0.0, 1.0, 1.0
    all_values = [v for vals in groups.values() for v in vals]
    grand = float(np.mean(all_values))
    k = len(sizes)
    if k < 2:
        return grand, float(np.var(all_values) + 1e-9), 0.0
    ssb = sum(n * (float(np.mean(groups[g])) - grand) ** 2 for g, n in sizes.items())
    ssw = sum(
        float(np.sum((np.asarray(groups[g], dtype=float) - np.mean(groups[g])) ** 2))
        for g in sizes
    )
    n_total = sum(sizes.values())
    df_b = max(k - 1, 1)
    df_w = max(n_total - k, 1)
    msb = ssb / df_b
    msw = ssw / df_w
    n_bar = (n_total - sum(n * n for n in sizes.values()) / n_total) / df_b
    n_bar = max(n_bar, 1.0)
    var_between = max((msb - msw) / n_bar, 0.0)
    tau = msw / var_between if var_between > 0.0 else float("inf")
    return grand, msw, tau


@dataclasses.dataclass
class NormalityLevel:
    key: str
    n: int
    mean: float
    var: float


class CrossGameNormalityMemory:
    def __init__(self) -> None:
        self.levels: Dict[str, Dict[str, NormalityLevel]] = {
            "game": {}, "subgenre": {}, "genre": {}, "global": {}
        }
        self.tau: Dict[str, float] = {}
        self.parent_of: Dict[str, Dict[str, str]] = {"game": {}, "subgenre": {}, "genre": {}}
        self.notes: Dict[str, str] = {}
        self.traps: Dict[str, List[str]] = {}
        self.calibration: List[float] = []
        self.duration: Dict[str, float] = {}
        self.fitted_on: Dict[str, Any] = {}

    @staticmethod
    def _norm(text: Any) -> str:
        value = str(text or "").strip().lower()
        return value or "unknown"

    def fit(
        self,
        windows: Sequence[Dict[str, Any]],
        labels: Dict[str, int],
        videos: Dict[str, Dict[str, Any]],
        contexts: Dict[str, Any],
        trap_per_game: int = 12,
    ) -> None:
        by_game: Dict[str, List[float]] = defaultdict(list)
        by_sub: Dict[str, List[float]] = defaultdict(list)
        by_genre: Dict[str, List[float]] = defaultdict(list)
        every: List[float] = []

        for win in windows:
            if labels.get(win["window_uid"], 0) != 0:
                continue
            score = logit(win["p_glitch"])
            game = self._norm(win["game_name"])
            video = videos.get(win["video_name"], {})
            sub = self._norm(video.get("subgenre") or win.get("subgenre"))
            genre = self._norm(video.get("genre") or win.get("genre"))
            by_game[game].append(score)
            by_sub[sub].append(score)
            by_genre[genre].append(score)
            every.append(score)
            self.parent_of["game"][game] = sub
            self.parent_of["subgenre"][sub] = genre
            self.parent_of["genre"][genre] = "global"

        if not every:
            raise RuntimeError(
                "no negative windows in the training split; the normality memory "
                "cannot be fitted and every downstream calibration would be vacuous"
            )

        for name, groups in (("game", by_game), ("subgenre", by_sub), ("genre", by_genre)):
            _, _, tau = random_effects_shrinkage(groups)
            self.tau[name] = float(tau if math.isfinite(tau) else 1e9)
            for key, vals in groups.items():
                self.levels[name][key] = NormalityLevel(
                    key=key, n=len(vals), mean=float(np.mean(vals)),
                    var=float(np.var(vals) + 1e-9),
                )
        self.levels["global"]["global"] = NormalityLevel(
            key="global", n=len(every), mean=float(np.mean(every)),
            var=float(np.var(every) + 1e-9),
        )
        self.tau["global"] = 0.0

        self.calibration = sorted(
            self.standardise(
                logit(w["p_glitch"]),
                self._norm(w["game_name"]),
                self._norm(videos.get(w["video_name"], {}).get("subgenre") or w.get("subgenre")),
                self._norm(videos.get(w["video_name"], {}).get("genre") or w.get("genre")),
            )
            for w in windows
            if labels.get(w["window_uid"], 0) == 0
        )

        hard: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        for win in windows:
            if labels.get(win["window_uid"], 0) != 0:
                continue
            for finding in win.get("findings") or []:
                evidence = str(finding.get("evidence") or "").strip()
                if len(evidence) < 20:
                    continue
                hard[self._norm(win["game_name"])].append((float(win["p_glitch"]), evidence))
        for game, items in hard.items():
            items.sort(key=lambda kv: -kv[0])
            seen: set = set()
            picked: List[str] = []
            for _, evidence in items:
                key = sha1_short(re.sub(r"\W+", "", evidence.lower())[:90], 8)
                if key in seen:
                    continue
                seen.add(key)
                picked.append(evidence)
                if len(picked) >= trap_per_game:
                    break
            self.traps[game] = picked

        for game, entry in (contexts or {}).items():
            ctx = entry.get("context", {})
            if "raw" in ctx:
                text = str(ctx["raw"])
            else:
                text = "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in ctx.items())
            self.notes[self._norm(game)] = text.strip()

        durations: List[float] = []
        gaps: List[float] = []
        n_intervals: List[int] = []
        for video in videos.values():
            for spans in video.get("bug_spans") or []:
                merged = merge_intervals([tuple(s) for s in spans])
                if not merged:
                    continue
                n_intervals.append(len(merged))
                durations.extend(b - a for a, b in merged)
                for prev, nxt in zip(merged, merged[1:]):
                    gaps.append(nxt[0] - prev[1])
        self.duration = {
            "mean_duration_s": float(np.mean(durations)) if durations else 4.0,
            "median_duration_s": float(np.median(durations)) if durations else 4.0,
            "mean_gap_s": float(np.mean(gaps)) if gaps else 4.0,
            "p_multi_interval": float(np.mean([n > 1 for n in n_intervals])) if n_intervals else 0.0,
            "mean_intervals_per_bug": float(np.mean(n_intervals)) if n_intervals else 1.0,
        }
        self.fitted_on = {
            "n_windows": len(windows),
            "n_negative_windows": len(self.calibration),
            "n_games": len(self.levels["game"]),
            "n_videos": len(videos),
            "tau": dict(self.tau),
        }
        LOGGER.info(
            "XNM fitted: %d games, %d negative calibration windows, tau(game)=%.1f "
            "tau(subgenre)=%.1f, mean glitch duration %.2fs, P(multi-interval)=%.3f",
            len(self.levels["game"]), len(self.calibration), self.tau["game"],
            self.tau["subgenre"], self.duration["mean_duration_s"],
            self.duration["p_multi_interval"],
        )

    def _backoff(self, name: str, key: str) -> Tuple[float, float, int, str]:
        chain: List[Tuple[str, str]] = []
        level, current = name, key
        while level != "global":
            chain.append((level, current))
            current = self.parent_of.get(level, {}).get(current, "global")
            level = {"game": "subgenre", "subgenre": "genre", "genre": "global"}[level]
        chain.append(("global", "global"))

        mean = self.levels["global"]["global"].mean
        var = self.levels["global"]["global"].var
        n_eff = self.levels["global"]["global"].n
        resolved = "global"
        for lvl, ky in reversed(chain[:-1]):
            entry = self.levels[lvl].get(ky)
            if entry is None:
                continue
            tau = self.tau.get(lvl, 1e9)
            lam = entry.n / (entry.n + tau) if math.isfinite(tau) and (entry.n + tau) > 0 else 0.0
            mean = lam * entry.mean + (1.0 - lam) * mean
            var = lam * entry.var + (1.0 - lam) * var
            n_eff = entry.n
            resolved = f"{lvl}:{ky}" if lam > 0.05 else resolved
        return float(mean), float(max(var, 1e-9)), int(n_eff), resolved

    def standardise(self, score: float, game: str, subgenre: str, genre: str) -> float:
        self.parent_of["game"].setdefault(game, subgenre)
        self.parent_of["subgenre"].setdefault(subgenre, genre)
        self.parent_of["genre"].setdefault(genre, "global")
        mean, var, _, _ = self._backoff("game", game)
        return (float(score) - mean) / math.sqrt(var)

    def conformal_p(self, standardised: float) -> float:
        if not self.calibration:
            return 1.0
        idx = bisect.bisect_left(self.calibration, standardised)
        greater_equal = len(self.calibration) - idx
        return (1.0 + greater_equal) / (1.0 + len(self.calibration))

    def note_for(self, game: str, subgenre: str, max_chars: int = 900) -> str:
        game = self._norm(game)
        note = self.notes.get(game, "")
        if not note:
            siblings = [
                self.notes[g] for g, sub in self.parent_of["game"].items()
                if sub == self._norm(subgenre) and g in self.notes
            ]
            note = siblings[0] if siblings else ""
        return note[:max_chars].strip() or "(no normality note available for this title)"

    def traps_for(self, game: str, limit: int = 5) -> List[str]:
        return (self.traps.get(self._norm(game)) or [])[:limit]

    def to_json(self) -> Dict[str, Any]:
        return {
            "levels": {
                lvl: {k: dataclasses.asdict(v) for k, v in entries.items()}
                for lvl, entries in self.levels.items()
            },
            "tau": self.tau,
            "parent_of": self.parent_of,
            "notes": self.notes,
            "traps": self.traps,
            "calibration": self.calibration,
            "duration": self.duration,
            "fitted_on": self.fitted_on,
        }

    @classmethod
    def from_json(cls, payload: Dict[str, Any]) -> "CrossGameNormalityMemory":
        memory = cls()
        memory.levels = {
            lvl: {k: NormalityLevel(**v) for k, v in entries.items()}
            for lvl, entries in payload["levels"].items()
        }
        memory.tau = {k: float(v) for k, v in payload["tau"].items()}
        memory.parent_of = payload["parent_of"]
        memory.notes = payload["notes"]
        memory.traps = payload["traps"]
        memory.calibration = list(payload["calibration"])
        memory.duration = payload["duration"]
        memory.fitted_on = payload.get("fitted_on", {})
        return memory


# --------------------------------------------------------------------------- #
# Stage 1/2 bridge
# --------------------------------------------------------------------------- #


def import_stage12(path: Optional[str]) -> Any:
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path))
    candidates.append(Path(__file__).resolve().parent / DEFAULT_STAGE12)
    candidates.append(Path.cwd() / DEFAULT_STAGE12)
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("vgb_stage12", candidate)
            module = importlib.util.module_from_spec(spec)
            sys.modules["vgb_stage12"] = module
            spec.loader.exec_module(module)
            LOGGER.info("loaded Stage 1/2 module from %s", candidate)
            return module
    raise FileNotFoundError(
        "vgb_stage12.py could not be located; pass --stage12 /path/to/vgb_stage12.py. "
        "Stage 3 reuses its geometry-safe VLM backend rather than duplicating it."
    )


def build_vlm(stage12: Any, args: argparse.Namespace) -> Any:
    backend = stage12.QwenVLBackend(
        args.model, args.dtype, args.load_in_4bit, args.max_pixels, args.min_pixels,
        attn_impl=args.attn, fit=args.fit,
    )
    return backend


# --------------------------------------------------------------------------- #
# evidence assembly
# --------------------------------------------------------------------------- #


def build_window_labels(
    windows: Sequence[Dict[str, Any]], videos: Dict[str, Dict[str, Any]], min_cover: float
) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    for win in windows:
        video = videos.get(win["video_name"])
        if video is None:
            continue
        span = (float(win["t_start"]), float(win["t_end"]))
        width = max(span[1] - span[0], 1e-6)
        positive = 0
        for spans in video.get("bug_spans") or []:
            for iv in spans:
                lo, hi = float(iv[0]), float(iv[1])
                overlap = max(0.0, min(span[1], hi) - max(span[0], lo))
                if overlap <= 0.0:
                    continue
                if max(overlap / width, overlap / max(hi - lo, 1e-6)) >= min_cover:
                    positive = 1
                    break
            if positive:
                break
        labels[win["window_uid"]] = positive
    return labels


def assemble_evidence(
    windows: Sequence[Dict[str, Any]], scans: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    by_uid = {row["window_uid"]: row for row in scans}
    merged: List[Dict[str, Any]] = []
    missing = 0
    for win in windows:
        scan = by_uid.get(win["window_uid"])
        if scan is None:
            missing += 1
            continue
        merged.append({
            "window_uid": win["window_uid"],
            "video_name": win["video_name"],
            "game_name": win["game_name"],
            "genre": win.get("genre", ""),
            "subgenre": win.get("subgenre", ""),
            "window_index": int(win["window_index"]),
            "t_start": float(win["t_start"]),
            "t_end": float(win["t_end"]),
            "fps": float(win["fps"]),
            "n_frames": len(win["frame_paths"]),
            "frame_paths": win["frame_paths"],
            "p_glitch": float(scan["p_glitch"]),
            "has_glitch": bool(scan["has_glitch"]),
            "findings": scan.get("findings") or [],
            "categories": scan.get("categories") or [],
        })
    if missing:
        LOGGER.warning(
            "%d windows in windows.jsonl have no row in scan.jsonl; run Stage 2 to "
            "completion before Stage 3 or the recall ceiling is silently lowered",
            missing,
        )
    merged.sort(key=lambda r: (r["video_name"], r["window_index"]))
    return merged


def group_by_video(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[row["video_name"]].append(row)
    for name in out:
        out[name].sort(key=lambda r: r["window_index"])
    return dict(out)


def calibrate_windows(
    rows: Sequence[Dict[str, Any]], memory: CrossGameNormalityMemory, prior: float
) -> None:
    prior = float(min(max(prior, 1e-4), 1.0 - 1e-4))
    offset = logit(prior)
    for row in rows:
        z = memory.standardise(
            logit(row["p_glitch"]),
            CrossGameNormalityMemory._norm(row["game_name"]),
            CrossGameNormalityMemory._norm(row["subgenre"]),
            CrossGameNormalityMemory._norm(row["genre"]),
        )
        row["z"] = float(z)
        row["conformal_p"] = float(memory.conformal_p(z))
        row["log_odds"] = float(z + offset)
        row["posterior"] = float(expit(row["log_odds"]))


# --------------------------------------------------------------------------- #
# risk-gated selective verification
# --------------------------------------------------------------------------- #


def expected_decision_value(posterior: float, c_fp: float, c_fn: float) -> float:
    return float(min(c_fp * posterior, c_fn * (1.0 - posterior)))


def select_verification_budget(
    rows: Sequence[Dict[str, Any]], budget: int, c_fp: float, c_fn: float, floor: float
) -> List[int]:
    scored = [
        (expected_decision_value(r["posterior"], c_fp, c_fn), i)
        for i, r in enumerate(rows)
    ]
    scored = [(v, i) for v, i in scored if v >= floor]
    scored.sort(key=lambda kv: -kv[0])
    return sorted(i for _, i in scored[:budget])


# --------------------------------------------------------------------------- #
# refining judge (never sees annotations)
# --------------------------------------------------------------------------- #

REFINER_SYSTEM = (
    "You are the technical lead arbitrating a video game QA investigation. You "
    "decide whether an observation in a gameplay clip is a genuine software "
    "defect or an unusual but legitimate in-game event. You have no access to "
    "any annotation or answer key; you reason only from the clip and from the "
    "normality reference for this title."
)

REFINER_PROMPT = """Title: "{game}" ({genre} / {subgenre})
Clip: {n_frames} frames at {fps:g} fps covering seconds {t_start:.1f} to {t_end:.1f}, frames indexed 0 to {last_frame}.

What is NORMAL for this title:
{note}

Observations previously mistaken for defects in this title (these are legitimate gameplay, do not re-flag them):
{traps}

Screening evidence for this clip:
{evidence}

Argue both sides before ruling.
ADVOCATE: state the strongest case that this is a genuine software defect, naming the specific entity and the rule it violates.
SKEPTIC: state the strongest case that this is legitimate gameplay, art direction, or a scripted animation.
RULING: decide.

Return ONE JSON object and nothing else:
{{"advocate": "one sentence", "skeptic": "one sentence", "ruling": "glitch|normal", "category": "{taxonomy}", "description": "one or two sentences naming the affected entity by appearance, what it does wrong, and where on screen", "salient_frames": [<frame indices>], "confidence": <float 0-1>}}

If the ruling is normal, set description to an empty string and salient_frames to an empty list."""

SUMMARISER_SYSTEM = (
    "You are a video game glitch analyst writing the final bug report line for a "
    "QA tracker. You compress several fragmentary observations of the same "
    "underlying defect into one coherent description."
)

SUMMARISER_PROMPT = """Title: "{game}" ({genre} / {subgenre})

What is NORMAL for this title:
{note}

Fragmentary observations of ONE underlying defect, in chronological order:
{fragments}

Write a single description of 2 to 3 sentences that names the specific appearance of the affected entity, states the abnormal behaviour, and says where in the scene it occurs. Do not mention frame numbers, timestamps, window indices, JSON, confidence values, or these instructions. Return the description text only."""

LINK_SYSTEM = (
    "You decide whether two observations from the same gameplay video describe "
    "the same underlying defect or two different defects."
)

LINK_PROMPT = """Observation A:
{a}

Observation B:
{b}

They describe the same underlying defect only if the affected entities plausibly match, the anomaly type matches, and the abnormal behaviour matches. Minor wording differences and observation variance are acceptable.

Return ONE JSON object and nothing else:
{{"reasoning": "one sentence", "judgement": "yes|no"}}"""


def render_evidence(
    row: Dict[str, Any], max_findings: int = 3, disclose_calibration: bool = True
) -> str:
    if disclose_calibration:
        lines = [
            f"- screening posterior after game-referenced calibration: {row['posterior']:.3f}",
            f"- conformal p-value against this title's normality: {row['conformal_p']:.4f}",
        ]
    else:
        lines = [
            f"- screening confidence: {row['p_glitch']:.3f}",
            "- no calibrated statistics are disclosed in this configuration",
        ]
    for finding in (row.get("findings") or [])[:max_findings]:
        if finding.get("unattributed"):
            continue
        evidence = str(finding.get("evidence") or "").strip()
        if not evidence:
            continue
        frames = finding.get("salient_frames") or []
        lines.append(f"- {finding.get('category', 'unknown')}: {evidence} (frames {frames})")
    if len(lines) == 2:
        lines.append("- the screening pass produced no parseable structured evidence")
    return "\n".join(lines)


class Refiner:
    def __init__(self, judge: JudgeBackend, vlm: Any, memory: CrossGameNormalityMemory,
                 taxonomy: Sequence[str], cache: DiskCache, max_tokens: int,
                 use_pixels: bool = True, disclose_calibration: bool = True,
                 disclose_traps: bool = True) -> None:
        self.judge = judge
        self.vlm = vlm
        self.memory = memory
        self.taxonomy = "|".join(taxonomy)
        self.cache = cache
        self.max_tokens = max_tokens
        self.use_pixels = use_pixels
        self.disclose_calibration = bool(disclose_calibration)
        self.disclose_traps = bool(disclose_traps)
        self.vlm_calls = 0

    def _generate(self, row: Dict[str, Any], prompt: str, system: str) -> str:
        if self.use_pixels and self.vlm is not None:
            self.vlm_calls += 1
            return self.vlm.generate(
                row["frame_paths"], prompt, row["fps"],
                max_new_tokens=self.max_tokens, system=system,
            )
        return self.judge.chat(system, prompt, max_tokens=self.max_tokens)

    def verify(self, row: Dict[str, Any]) -> Dict[str, Any]:
        note = self.memory.note_for(row["game_name"], row["subgenre"])
        traps = self.memory.traps_for(row["game_name"]) if self.disclose_traps else []
        prompt = REFINER_PROMPT.format(
            game=row["game_name"],
            genre=row["genre"] or "unknown",
            subgenre=row["subgenre"] or "unknown",
            n_frames=row["n_frames"],
            fps=row["fps"],
            t_start=row["t_start"],
            t_end=row["t_end"],
            last_frame=max(0, row["n_frames"] - 1),
            note=note,
            traps="\n".join(f"- {t}" for t in traps) or "- none recorded",
            evidence=render_evidence(
                row, disclose_calibration=self.disclose_calibration
            ),
            taxonomy=self.taxonomy,
        )
        key = sha1_short(f"refine|{self.judge.model_id}|{row['window_uid']}|{prompt}", 24)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        raw = self._generate(row, prompt, REFINER_SYSTEM)
        parsed = extract_json_object(raw) or {}
        ruling = str(parsed.get("ruling", "")).strip().lower()
        try:
            confidence = float(parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        frames = parsed.get("salient_frames") or []
        frames = sorted({
            int(f) for f in frames
            if isinstance(f, (int, float)) and not isinstance(f, bool)
            and 0 <= int(f) < row["n_frames"]
        })
        result = {
            "ruling": "glitch" if ruling.startswith("glitch") else "normal",
            "category": str(parsed.get("category", "") or "").strip().lower(),
            "description": str(parsed.get("description", "") or "").strip(),
            "salient_frames": frames,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "parsed": bool(parsed),
        }
        self.cache.put(key, result)
        return result

    def link(self, a: str, b: str) -> bool:
        key = sha1_short(f"link|{self.judge.model_id}|{a}|{b}", 24)
        cached = self.cache.get(key)
        if cached is not None:
            return bool(cached)
        raw = self.judge.chat(LINK_SYSTEM, LINK_PROMPT.format(a=a, b=b), max_tokens=120)
        parsed = extract_json_object(raw) or {}
        verdict = str(parsed.get("judgement", "")).strip().lower().startswith("y")
        self.cache.put(key, verdict)
        return verdict

    def summarise(self, game: str, genre: str, subgenre: str, fragments: Sequence[str]) -> str:
        fragments = [f.strip() for f in fragments if f and f.strip()]
        if not fragments:
            return ""
        if len(fragments) == 1:
            return fragments[0]
        note = self.memory.note_for(game, subgenre, max_chars=500)
        prompt = SUMMARISER_PROMPT.format(
            game=game, genre=genre or "unknown", subgenre=subgenre or "unknown",
            note=note,
            fragments="\n".join(f"- {f}" for f in fragments),
        )
        key = sha1_short(f"summary|{self.judge.model_id}|{prompt}", 24)
        cached = self.cache.get(key)
        if cached is not None:
            return str(cached)
        raw = self.judge.chat(SUMMARISER_SYSTEM, prompt, max_tokens=220)
        text = re.sub(r"^```.*?\n|```$", "", raw.strip(), flags=re.S).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            text = max(fragments, key=len)
        self.cache.put(key, text)
        return text


# --------------------------------------------------------------------------- #
# event-level grounding
# --------------------------------------------------------------------------- #


def fit_transition_matrix(
    windows: Sequence[Dict[str, Any]], labels: Dict[str, int], smoothing: float = 1.0
) -> Dict[str, float]:
    counts = {"00": smoothing, "01": smoothing, "10": smoothing, "11": smoothing}
    by_video = group_by_video(windows)
    starts = smoothing
    total_starts = 2.0 * smoothing
    for rows in by_video.values():
        seq = [labels.get(r["window_uid"], 0) for r in rows]
        if not seq:
            continue
        starts += seq[0]
        total_starts += 1.0
        for a, b in zip(seq, seq[1:]):
            counts[f"{a}{b}"] += 1.0
    p11 = counts["11"] / (counts["11"] + counts["10"])
    p01 = counts["01"] / (counts["01"] + counts["00"])
    pi1 = starts / total_starts
    return {
        "p11": float(np.clip(p11, 1e-4, 1.0 - 1e-4)),
        "p01": float(np.clip(p01, 1e-4, 1.0 - 1e-4)),
        "pi1": float(np.clip(pi1, 1e-4, 1.0 - 1e-4)),
    }


def constrained_viterbi(
    emissions: Sequence[float],
    transitions: Dict[str, float],
    clamped_on: Sequence[int] = (),
    clamped_off: Sequence[int] = (),
) -> List[int]:
    n = len(emissions)
    if n == 0:
        return []
    p11 = transitions["p11"]
    p01 = transitions["p01"]
    pi1 = transitions["pi1"]
    log_t = np.array([
        [math.log(1.0 - p01), math.log(p01)],
        [math.log(1.0 - p11), math.log(p11)],
    ], dtype=float)
    log_pi = np.array([math.log(1.0 - pi1), math.log(pi1)], dtype=float)

    penalty = np.zeros((n, 2), dtype=float)
    for idx in clamped_on:
        if 0 <= idx < n:
            penalty[idx, 0] = -1e9
    for idx in clamped_off:
        if 0 <= idx < n:
            penalty[idx, 1] = -1e9

    delta = np.zeros((n, 2), dtype=float)
    psi = np.zeros((n, 2), dtype=int)
    delta[0, 0] = log_pi[0] + 0.0 + penalty[0, 0]
    delta[0, 1] = log_pi[1] + float(emissions[0]) + penalty[0, 1]
    for t in range(1, n):
        for s in range(2):
            scores = delta[t - 1] + log_t[:, s]
            best = int(np.argmax(scores))
            psi[t, s] = best
            emit = float(emissions[t]) if s == 1 else 0.0
            delta[t, s] = scores[best] + emit + penalty[t, s]

    path = [0] * n
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = int(psi[t, path[t]])
    return path


def runs_from_path(path: Sequence[int]) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start = None
    for i, v in enumerate(path):
        if v == 1 and start is None:
            start = i
        elif v == 0 and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(path) - 1))
    return runs


def refine_boundaries(
    rows: Sequence[Dict[str, Any]],
    run: Tuple[int, int],
    salient: Dict[int, List[int]],
    duration: float,
    enabled: bool,
) -> Tuple[float, float]:
    lo_idx, hi_idx = run
    start = float(rows[lo_idx]["t_start"])
    end = float(rows[hi_idx]["t_end"])
    if enabled:
        fps = float(rows[lo_idx]["fps"]) or 4.0
        first = salient.get(lo_idx) or []
        last = salient.get(hi_idx) or []
        if first:
            start = max(start, float(rows[lo_idx]["t_start"]) + min(first) / fps)
        if last:
            end = min(end, float(rows[hi_idx]["t_start"]) + (max(last) + 1) / fps)
    if end <= start:
        end = start + 1.0 / max(float(rows[lo_idx]["fps"]), 1.0)
    start = max(0.0, start)
    if duration > 0.0:
        end = min(end, duration)
        start = min(start, max(duration - 1e-3, 0.0))
    if end <= start:
        end = start + 1e-2
    return round(start, 3), round(end, 3)


def cluster_evidence_texts(
    texts: Sequence[str],
    refiner: Optional[Refiner],
    threshold: float,
    band: Tuple[float, float],
    use_judge: bool,
) -> List[List[int]]:
    if not texts:
        return []
    if len(texts) == 1:
        return [[0]]
    tfidf = TfIdf(list(texts))
    affinity = cosine_affinity(tfidf.transform(list(texts)))
    if use_judge and refiner is not None:
        lo, hi = band
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if lo <= affinity[i, j] <= hi:
                    verdict = refiner.link(texts[i], texts[j])
                    affinity[i, j] = affinity[j, i] = 1.0 if verdict else 0.0
    return average_linkage(affinity, threshold)


# --------------------------------------------------------------------------- #
# prediction modes
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class PredictConfig:
    mode: str = "kairos"
    threshold: float = 0.5
    budget_per_video: int = 4
    c_fp: float = 1.0
    c_fn: float = 2.0
    edv_floor: float = 0.02
    prior: float = 0.25
    fdr_q: float = 0.25
    link_threshold: float = 0.34
    link_band: Tuple[float, float] = (0.12, 0.62)
    affinity_weight: float = 1.2
    emission_shift: float = 0.0
    use_memory: bool = True
    use_gate: bool = True
    use_dp: bool = True
    use_refine_boundaries: bool = True
    use_judge_link: bool = True
    max_events: int = 6
    glide_max_iter: int = 5
    glide_conf: float = 0.80

    def fingerprint(self) -> str:
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True, default=list)
        return sha1_short(payload, 12)


def _window_text(row: Dict[str, Any], verdict: Optional[Dict[str, Any]]) -> str:
    if verdict and verdict.get("description"):
        return verdict["description"]
    parts = [
        str(f.get("evidence") or "") for f in (row.get("findings") or [])
        if not f.get("unattributed")
    ]
    text = " ".join(p for p in parts if p).strip()
    return text or f"unspecified {'/'.join(row.get('categories') or ['anomaly'])} anomaly"


def predict_vanilla(
    rows: Sequence[Dict[str, Any]], video: Dict[str, Any], config: PredictConfig
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    duration = float(video.get("duration") or 0.0)
    positives = [i for i, r in enumerate(rows) if r["p_glitch"] >= config.threshold]
    events: List[Dict[str, Any]] = []
    current: List[int] = []
    for idx in positives:
        if current and idx != current[-1] + 1:
            events.append(current)
            current = []
        current.append(idx)
    if current:
        events.append(current)

    out: List[Dict[str, Any]] = []
    for group in events[: config.max_events]:
        texts = [_window_text(rows[i], None) for i in group]
        start = float(rows[group[0]]["t_start"])
        end = float(rows[group[-1]]["t_end"])
        if duration > 0.0:
            end = min(end, duration)
        if end <= start:
            continue
        out.append({
            "description": max(texts, key=len),
            "intervals": [(round(start, 3), round(end, 3))],
            "category": (rows[group[0]].get("categories") or ["unknown"])[0],
            "n_windows": len(group),
        })
    return out, {"llm_calls": 0.0, "vlm_calls": 0.0, "verified_windows": 0.0}


def predict_glide(
    rows: Sequence[Dict[str, Any]],
    video: Dict[str, Any],
    config: PredictConfig,
    refiner: Refiner,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    duration = float(video.get("duration") or 0.0)
    candidates = [i for i, r in enumerate(rows) if r["p_glitch"] >= config.threshold]
    verdicts: Dict[int, Dict[str, Any]] = {}
    calls = 0
    for idx in candidates:
        confidence = 0.0
        verdict: Dict[str, Any] = {}
        for _ in range(config.glide_max_iter):
            verdict = refiner.verify(rows[idx])
            calls += 1
            confidence = verdict["confidence"]
            if confidence >= config.glide_conf:
                break
        if verdict.get("ruling") == "glitch":
            verdicts[idx] = verdict

    confirmed = sorted(verdicts)
    clusters: List[List[int]] = []
    for idx in confirmed:
        text = _window_text(rows[idx], verdicts[idx])
        placed = False
        for cluster in clusters:
            head = _window_text(rows[cluster[0]], verdicts[cluster[0]])
            calls += 1
            if refiner.link(text, head):
                cluster.append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])

    expanded: List[List[int]] = []
    for cluster in clusters:
        members = set(cluster)
        text = _window_text(rows[cluster[0]], verdicts[cluster[0]])
        cursor = max(cluster) + 1
        while cursor < len(rows):
            calls += 1
            neighbour = _window_text(rows[cursor], None)
            if not refiner.link(text, neighbour):
                break
            members.add(cursor)
            cursor += 1
        cursor = min(cluster) - 1
        while cursor >= 0:
            calls += 1
            neighbour = _window_text(rows[cursor], None)
            if not refiner.link(text, neighbour):
                break
            members.add(cursor)
            cursor -= 1
        expanded.append(sorted(members))

    out: List[Dict[str, Any]] = []
    for cluster in expanded[: config.max_events]:
        fragments = [
            _window_text(rows[i], verdicts.get(i)) for i in cluster if i in verdicts
        ] or [_window_text(rows[cluster[0]], None)]
        description = refiner.summarise(
            rows[cluster[0]]["game_name"], rows[cluster[0]]["genre"],
            rows[cluster[0]]["subgenre"], fragments,
        )
        intervals: List[Tuple[float, float]] = []
        run_start = cluster[0]
        prev = cluster[0]
        for idx in cluster[1:] + [None]:
            if idx is None or idx != prev + 1:
                start = float(rows[run_start]["t_start"])
                end = float(rows[prev]["t_end"])
                if duration > 0.0:
                    end = min(end, duration)
                if end > start:
                    intervals.append((round(start, 3), round(end, 3)))
                if idx is not None:
                    run_start = idx
            if idx is not None:
                prev = idx
        if not intervals:
            continue
        out.append({
            "description": description,
            "intervals": merge_intervals(intervals),
            "category": verdicts.get(cluster[0], {}).get("category", "unknown"),
            "n_windows": len(cluster),
        })
    return out, {
        "llm_calls": float(calls),
        "vlm_calls": float(len(candidates)),
        "verified_windows": float(len(confirmed)),
    }


def predict_kairos(
    rows: Sequence[Dict[str, Any]],
    video: Dict[str, Any],
    config: PredictConfig,
    refiner: Refiner,
    transitions: Dict[str, float],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    duration = float(video.get("duration") or 0.0)
    n = len(rows)
    if n == 0:
        return [], {"llm_calls": 0.0, "vlm_calls": 0.0, "verified_windows": 0.0}

    pvalues = np.array([r["conformal_p"] for r in rows], dtype=float)
    admissible = benjamini_hochberg(pvalues, config.fdr_q)
    if not admissible.any():
        admissible = pvalues <= float(np.min(pvalues))

    if config.use_gate:
        pool = [i for i in range(n) if admissible[i]]
        gated = select_verification_budget(
            [rows[i] for i in pool], config.budget_per_video,
            config.c_fp, config.c_fn, config.edv_floor,
        )
        selected = [pool[i] for i in gated]
        confident = [
            i for i in pool
            if i not in set(selected) and rows[i]["posterior"] >= 1.0 - config.edv_floor
        ]
    else:
        selected = [i for i in range(n) if admissible[i]]
        confident = []

    verdicts: Dict[int, Dict[str, Any]] = {}
    calls = 0
    unparsed = 0
    for idx in selected:
        verdict = refiner.verify(rows[idx])
        calls += 1
        if not verdict["parsed"]:
            unparsed += 1
            if rows[idx]["posterior"] >= 0.5:
                verdicts[idx] = {
                    "ruling": "glitch",
                    "category": (rows[idx].get("categories") or ["unknown"])[0],
                    "description": _window_text(rows[idx], None),
                    "salient_frames": (rows[idx].get("findings") or [{}])[0].get("salient_frames") or [],
                    "confidence": rows[idx]["posterior"],
                    "parsed": False,
                }
            continue
        if verdict["ruling"] == "glitch":
            verdicts[idx] = verdict
    for idx in confident:
        verdicts[idx] = {
            "ruling": "glitch",
            "category": (rows[idx].get("categories") or ["unknown"])[0],
            "description": _window_text(rows[idx], None),
            "salient_frames": (rows[idx].get("findings") or [{}])[0].get("salient_frames") or [],
            "confidence": rows[idx]["posterior"],
            "parsed": False,
        }

    seeds = sorted(verdicts)
    if not seeds:
        return [], {
            "llm_calls": float(calls),
            "vlm_calls": float(len(selected)),
            "verified_windows": 0.0,
            "unparsed_verdicts": float(unparsed),
        }

    texts = [_window_text(rows[i], verdicts[i]) for i in seeds]
    clusters = cluster_evidence_texts(
        texts, refiner, config.link_threshold, config.link_band, config.use_judge_link
    )
    if config.use_judge_link:
        calls += sum(len(c) * (len(c) - 1) // 2 for c in clusters if len(c) > 1)

    window_texts = [_window_text(r, verdicts.get(i)) for i, r in enumerate(rows)]
    tfidf_all = TfIdf(window_texts)
    embed_all = tfidf_all.transform(window_texts)

    out: List[Dict[str, Any]] = []
    for cluster in clusters[: config.max_events]:
        member_idx = [seeds[c] for c in cluster]
        centroid = embed_all[member_idx].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0.0:
            centroid = centroid / norm
        affinity = np.clip(embed_all @ centroid, 0.0, 1.0)

        if config.use_dp:
            emissions = [
                float(rows[i]["log_odds"])
                + config.affinity_weight * (float(affinity[i]) - config.link_threshold)
                + config.emission_shift
                for i in range(n)
            ]
            other_seeds = [s for s in seeds if s not in set(member_idx)]
            path = constrained_viterbi(emissions, transitions, member_idx, other_seeds)
            runs = runs_from_path(path)
        else:
            runs = []
            run_start = member_idx[0]
            prev = member_idx[0]
            for idx in member_idx[1:] + [None]:
                if idx is None or idx != prev + 1:
                    runs.append((run_start, prev))
                    if idx is not None:
                        run_start = idx
                if idx is not None:
                    prev = idx

        salient = {
            i: (verdicts[i].get("salient_frames") or []) for i in member_idx
        }
        intervals = [
            refine_boundaries(rows, run, salient, duration, config.use_refine_boundaries)
            for run in runs
        ]
        intervals = merge_intervals(intervals)
        if not intervals:
            continue

        fragments = [texts[c] for c in cluster]
        description = refiner.summarise(
            rows[member_idx[0]]["game_name"], rows[member_idx[0]]["genre"],
            rows[member_idx[0]]["subgenre"], fragments,
        )
        if len(fragments) > 1:
            calls += 1
        out.append({
            "description": description,
            "intervals": intervals,
            "category": verdicts[member_idx[0]].get("category", "unknown"),
            "n_windows": len(member_idx),
            "n_intervals": len(intervals),
        })

    return out, {
        "llm_calls": float(calls),
        "vlm_calls": float(len(selected)),
        "verified_windows": float(len(seeds)),
        "unparsed_verdicts": float(unparsed),
    }


# --------------------------------------------------------------------------- #
# artefact loading
# --------------------------------------------------------------------------- #


def load_stage12_artefacts(work_dir: Path) -> Dict[str, Any]:
    videos = read_jsonl(work_dir / "videos.jsonl")
    windows = read_jsonl(work_dir / "windows.jsonl")
    scans = read_jsonl(work_dir / "scan.jsonl")
    contexts_path = work_dir / "game_context.json"
    contexts = read_json(contexts_path, required=False) or {}
    if not scans:
        raise RuntimeError(f"{work_dir/'scan.jsonl'} is empty; run Stage 2 `scan` first")
    by_video = {v["video_name"]: v for v in videos}
    for video in videos:
        video["gt_intervals"] = normalise_intervals(video.get("gt_intervals"))
        video["bug_spans"] = [normalise_intervals(s) for s in (video.get("bug_spans") or [])]
    return {
        "videos": by_video,
        "windows": windows,
        "scans": scans,
        "contexts": contexts,
    }


def gold_reports(video: Dict[str, Any]) -> List[Dict[str, Any]]:
    if video.get("no_bugs"):
        return []
    bugs = video.get("bugs") or []
    spans = video.get("bug_spans") or []
    out: List[Dict[str, Any]] = []
    for i, description in enumerate(bugs):
        intervals = spans[i] if i < len(spans) else []
        intervals = merge_intervals(intervals)
        if not str(description).strip():
            continue
        out.append({"description": str(description).strip(), "intervals": intervals})
    if not out and spans:
        merged = merge_intervals([iv for group in spans for iv in group])
        if merged:
            out.append({"description": " ".join(str(b) for b in bugs).strip(), "intervals": merged})
    return out


ABLATIONS: Dict[str, Dict[str, Any]] = {
    "kairos": {},
    "kairos_no_xnm": {"use_memory": False},
    "kairos_no_gate": {"use_gate": False},
    "kairos_no_dp": {"use_dp": False},
    "kairos_no_boundary": {"use_refine_boundaries": False},
    "kairos_no_link": {"use_judge_link": False},
}


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_memory(args: argparse.Namespace) -> None:
    work = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    art = load_stage12_artefacts(work)
    split_names = load_splits(Path(args.split_dir))
    available = sorted(art["videos"])
    splits = resolve_split_membership(split_names, available)
    write_json(out_dir / "splits_resolved.json", splits)

    train = set(splits.get("train", []))
    if not train:
        raise RuntimeError("the train split resolved to zero videos; XNM cannot be fitted")

    train_windows = [w for w in art["windows"] if w["video_name"] in train]
    train_videos = {k: v for k, v in art["videos"].items() if k in train}
    labels = build_window_labels(train_windows, train_videos, args.min_cover)

    scan_by_uid = {r["window_uid"]: r for r in art["scans"]}
    enriched: List[Dict[str, Any]] = []
    for win in train_windows:
        scan = scan_by_uid.get(win["window_uid"])
        if scan is None:
            continue
        enriched.append({
            "window_uid": win["window_uid"],
            "video_name": win["video_name"],
            "game_name": win["game_name"],
            "genre": win.get("genre", ""),
            "subgenre": win.get("subgenre", ""),
            "window_index": int(win["window_index"]),
            "p_glitch": float(scan["p_glitch"]),
            "findings": scan.get("findings") or [],
        })
    if not enriched:
        raise RuntimeError("no train windows have a matching scan.jsonl row")

    memory = CrossGameNormalityMemory()
    memory.fit(enriched, labels, train_videos, art["contexts"], trap_per_game=args.traps_per_game)
    transitions = fit_transition_matrix(enriched, labels)
    LOGGER.info(
        "fitted window transition model: p(1|1)=%.3f p(1|0)=%.3f pi1=%.3f",
        transitions["p11"], transitions["p01"], transitions["pi1"],
    )

    positive_rate = float(np.mean([labels.get(w["window_uid"], 0) for w in enriched]))
    payload = memory.to_json()
    payload["transitions"] = transitions
    payload["train_positive_window_rate"] = positive_rate
    payload["min_cover"] = args.min_cover
    write_json(out_dir / "xnm_memory.json", payload)

    standardised = [
        memory.standardise(
            logit(w["p_glitch"]),
            CrossGameNormalityMemory._norm(w["game_name"]),
            CrossGameNormalityMemory._norm(w["subgenre"]),
            CrossGameNormalityMemory._norm(w["genre"]),
        )
        for w in enriched
    ]
    y = np.array([labels.get(w["window_uid"], 0) for w in enriched], dtype=int)
    z = np.array(standardised, dtype=float)
    if y.sum() and (1 - y).sum():
        auroc = float(
            (np.sum(np.argsort(np.argsort(z))[y == 1]) - y.sum() * (y.sum() - 1) / 2.0)
            / (y.sum() * (1 - y).sum())
        )
        raw = np.array([w["p_glitch"] for w in enriched], dtype=float)
        auroc_raw = float(
            (np.sum(np.argsort(np.argsort(raw))[y == 1]) - y.sum() * (y.sum() - 1) / 2.0)
            / (y.sum() * (1 - y).sum())
        )
        LOGGER.info(
            "train window AUROC: raw p_glitch %.4f -> XNM-standardised %.4f (delta %+.4f)",
            auroc_raw, auroc, auroc - auroc_raw,
        )
        if auroc <= auroc_raw:
            LOGGER.warning(
                "XNM standardisation did not improve window ranking on train; the "
                "per-game offsets are not the dominant nuisance for this backbone"
            )
    LOGGER.info("train positive window rate %.4f -> %s", positive_rate, out_dir / "xnm_memory.json")


def _prior_from_memory(payload: Dict[str, Any], override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    return float(payload.get("train_positive_window_rate") or 0.25)


def cmd_predict(args: argparse.Namespace) -> None:
    work = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    art = load_stage12_artefacts(work)
    splits = read_json(out_dir / "splits_resolved.json")
    payload = read_json(out_dir / "xnm_memory.json")
    memory = CrossGameNormalityMemory.from_json(payload)
    transitions = payload["transitions"]

    if args.split not in splits:
        raise KeyError(f"split {args.split!r} not in splits_resolved.json ({list(splits)})")
    wanted = set(splits[args.split])

    base = PredictConfig(
        mode=args.mode.split("_")[0] if args.mode.startswith("kairos") else args.mode,
        threshold=args.threshold,
        budget_per_video=args.budget,
        c_fp=args.cost_fp,
        c_fn=args.cost_fn,
        edv_floor=args.edv_floor,
        prior=_prior_from_memory(payload, args.prior),
        fdr_q=args.fdr_q,
        link_threshold=args.link_threshold,
        link_band=(args.link_band_lo, args.link_band_hi),
        affinity_weight=args.affinity_weight,
        emission_shift=args.emission_shift,
        max_events=args.max_events,
        glide_max_iter=args.glide_max_iter,
        glide_conf=args.glide_conf,
    )
    for key, value in ABLATIONS.get(args.mode, {}).items():
        setattr(base, key, value)
    config = base
    fingerprint = config.fingerprint()
    LOGGER.info("mode=%s config fingerprint=%s", args.mode, fingerprint)

    if args.split == "test" and args.frozen_config:
        frozen = read_json(Path(args.frozen_config))
        if frozen.get("fingerprint") != fingerprint:
            raise RuntimeError(
                "the configuration used for the test split does not match the frozen "
                f"dev-selected configuration ({frozen.get('fingerprint')} != {fingerprint}); "
                "refusing to run, because tuning on test invalidates the comparison"
            )

    rows_all = assemble_evidence(
        [w for w in art["windows"] if w["video_name"] in wanted], art["scans"]
    )
    if not rows_all:
        raise RuntimeError(f"no window/scan rows for split {args.split!r}")
    calibrate_windows(rows_all, memory, config.prior)
    by_video = group_by_video(rows_all)

    if not config.use_memory:
        memory.notes = {}
        memory.traps = {}

    stage12 = None
    vlm = None
    judge = JudgeBackend(
        args.judge_backend, args.judge_model, args.judge_temperature,
        args.judge_tokens, base_url=args.judge_base_url,
        api_key_env=args.api_key_env, dtype=args.dtype, load_in_4bit=args.load_in_4bit,
    )
    if config.mode != "vanilla" and (args.use_pixels or args.judge_backend == "vlm"):
        stage12 = import_stage12(args.stage12)
        vlm = build_vlm(stage12, args)
        judge.attach_vlm(vlm)

    taxonomy = getattr(stage12, "GLITCH_TAXONOMY", tuple(FINE_TO_COARSE)) if stage12 else tuple(FINE_TO_COARSE)
    cache = DiskCache(out_dir / f"refiner_cache_{args.judge_model.replace('/', '_')}.jsonl")
    refiner = Refiner(
        judge, vlm, memory, taxonomy, cache, args.refine_tokens,
        use_pixels=bool(args.use_pixels and vlm is not None),
        disclose_calibration=(config.mode == "kairos"),
        disclose_traps=(config.mode == "kairos" and config.use_memory),
    )

    predictions: List[Dict[str, Any]] = []
    cost = Counter()
    t0 = time.time()
    names = sorted(by_video)
    if args.limit_videos:
        names = names[: args.limit_videos]

    for i, name in enumerate(names, 1):
        rows = by_video[name]
        video = art["videos"][name]
        if config.mode == "vanilla":
            reports, stats = predict_vanilla(rows, video, config)
        elif config.mode == "glide":
            reports, stats = predict_glide(rows, video, config, refiner)
        else:
            reports, stats = predict_kairos(rows, video, config, refiner, transitions)

        for key, value in stats.items():
            cost[key] += value
        predictions.append({
            "video_name": name,
            "game_name": video.get("game_name", ""),
            "genre": video.get("genre", ""),
            "subgenre": video.get("subgenre", ""),
            "duration": float(video.get("duration") or 0.0),
            "no_bugs": bool(len(reports) == 0),
            "bugs": [r["description"] for r in reports],
            "time_nodes": [[list(iv) for iv in r["intervals"]] for r in reports],
            "categories": [r.get("category", "unknown") for r in reports],
            "n_windows": len(rows),
            "cost": stats,
        })
        if i % 25 == 0:
            rate = i / max(time.time() - t0, 1e-9)
            LOGGER.info(
                "%s/%s: %d/%d videos (%.2f vid/s, eta %.1f min, %d judge calls)",
                args.mode, args.split, i, len(names), rate,
                (len(names) - i) / max(rate, 1e-9) / 60.0, judge.calls,
            )

    pred_path = out_dir / f"pred_{args.mode}_{args.split}.jsonl"
    write_jsonl(pred_path, predictions)
    unparsed = float(cost.get("unparsed_verdicts", 0.0))
    verified = float(cost.get("vlm_calls", 0.0))
    if verified and unparsed > 0.2 * verified:
        LOGGER.warning(
            "%.0f of %.0f refiner verdicts were unparseable (%.1f%%); these fell back "
            "to the calibrated screening decision, so the reported gain understates "
            "the method and overstates the judge. Raise --refine-tokens or switch "
            "--judge-backend before reporting these numbers",
            unparsed, verified, 100.0 * unparsed / verified,
        )
    meta = {
        "mode": args.mode,
        "split": args.split,
        "fingerprint": fingerprint,
        "config": dataclasses.asdict(config),
        "judge": judge.stats(),
        "vlm_calls": refiner.vlm_calls,
        "cost": dict(cost),
        "n_videos": len(names),
        "wall_clock_s": time.time() - t0,
        "cache": {"hits": cache.hits, "misses": cache.misses},
    }
    write_json(out_dir / f"pred_{args.mode}_{args.split}.meta.json", meta)
    cache.close()
    LOGGER.info(
        "%s/%s done in %.1f min: %d videos, %d judge calls (%.2f/video), %d VLM calls -> %s",
        args.mode, args.split, (time.time() - t0) / 60.0, len(names), judge.calls,
        judge.calls / max(len(names), 1), refiner.vlm_calls, pred_path,
    )


def _prediction_reports(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    bugs = row.get("bugs") or []
    nodes = row.get("time_nodes") or []
    for i, description in enumerate(bugs):
        description = str(description or "").strip()
        intervals = merge_intervals(normalise_intervals(nodes[i] if i < len(nodes) else []))
        if not description or not intervals:
            continue
        out.append({"description": description, "intervals": intervals})
    return out


def cmd_evaluate(args: argparse.Namespace) -> None:
    work = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    art = load_stage12_artefacts(work)
    splits = read_json(out_dir / "splits_resolved.json")
    wanted = sorted(set(splits[args.split]))

    pred_path = out_dir / f"pred_{args.mode}_{args.split}.jsonl"
    preds = {r["video_name"]: r for r in read_jsonl(pred_path)}

    judge = JudgeBackend(
        args.scorer_backend, args.scorer_model, 0.0, 200,
        base_url=args.judge_base_url, api_key_env=args.api_key_env,
        dtype=args.dtype, load_in_4bit=args.load_in_4bit,
    )
    if args.scorer_backend == "vlm":
        stage12 = import_stage12(args.stage12)
        judge.attach_vlm(build_vlm(stage12, args))
    cache = DiskCache(out_dir / f"scorer_cache_{args.scorer_model.replace('/', '_')}.jsonl")
    scorer = SemanticScorer(judge, cache, scale=args.judge_scale)

    corpus: List[str] = []
    for name in wanted:
        corpus.extend(g["description"] for g in gold_reports(art["videos"][name]))
        corpus.extend(p["description"] for p in _prediction_reports(preds.get(name, {})))
    scorer.fit_lexical(corpus or ["glitch"])

    scores: List[VideoScore] = []
    pred_by_video: Dict[str, List[Dict[str, Any]]] = {}
    gold_by_video: Dict[str, List[Dict[str, Any]]] = {}
    n_eval = 0
    for name in wanted:
        video = art["videos"][name]
        gold = gold_reports(video)
        pred = _prediction_reports(preds.get(name, {}))
        pred_by_video[name] = pred
        gold_by_video[name] = gold
        score = match_video(pred, gold, scorer, keep=args.match_keep)
        score.video_name = name
        score.game_name = str(video.get("game_name", ""))
        score.n_pred = len(pred)
        score.n_gold = len(gold)
        scores.append(score)
        n_eval += 1
        if n_eval % 100 == 0:
            LOGGER.info("scored %d/%d videos (%d judge calls, %d cache hits)",
                        n_eval, len(wanted), judge.calls, cache.hits)

    micro = aggregate_micro(scores)
    macro = aggregate_macro(scores)
    micro.update(recall_at_tiou(scores, args.tiou))
    micro["multi_interval_recovery"] = multi_interval_recovery(scores, pred_by_video, gold_by_video)

    ci: Dict[str, Any] = {}
    for metric in ("f1_desc", "miou", "f1_x_iou", "precision_desc", "recall_desc"):
        point, lo, hi = bootstrap_ci(
            scores, lambda s, m=metric: aggregate_micro(s)[m],
            n_boot=args.n_boot, alpha=args.alpha, seed=args.seed,
        )
        ci[metric] = {"point": point, "lo": lo, "hi": hi}

    alt = match_all = None
    if args.match_keep == "positive":
        alt = [
            match_video(pred_by_video[s.video_name], gold_by_video[s.video_name], scorer, keep="all")
            for s in scores
        ]
        for a, s in zip(alt, scores):
            a.video_name = s.video_name
        match_all = aggregate_micro(alt)

    per_game: Dict[str, Dict[str, float]] = {}
    by_game: Dict[str, List[VideoScore]] = defaultdict(list)
    for s in scores:
        by_game[s.game_name].append(s)
    for game, group in by_game.items():
        per_game[game] = aggregate_micro(group)

    payload = {
        "mode": args.mode,
        "split": args.split,
        "n_videos": len(scores),
        "scorer": judge.stats(),
        "judge_scale": args.judge_scale,
        "match_keep": args.match_keep,
        "primary_micro": micro,
        "secondary_macro": macro,
        "sensitivity_match_all": match_all,
        "bootstrap_ci": ci,
        "per_game": per_game,
        "scorer_fallbacks": scorer.n_fallback,
        "cache": {"hits": cache.hits, "misses": cache.misses},
    }
    write_json(out_dir / f"eval_{args.mode}_{args.split}.json", payload)
    write_jsonl(out_dir / f"eval_{args.mode}_{args.split}.per_video.jsonl", [
        {
            "video_name": s.video_name,
            "game_name": s.game_name,
            "n_pred": s.n_pred,
            "n_gold": s.n_gold,
            "n_matched": s.n_matched,
            "sum_s": s.sum_s,
            "sum_w": s.sum_w,
            "sum_iou": s.sum_iou,
            "ious": s.ious,
        }
        for s in scores
    ])
    cache.close()

    LOGGER.info(
        "%s/%s | P %.2f R %.2f F1 %.2f | mIoU %.3f | F1xIoU %.2f | n=%d",
        args.mode, args.split, micro["precision_desc"], micro["recall_desc"],
        micro["f1_desc"], micro["miou"], micro["f1_x_iou"], len(scores),
    )
    if scorer.n_fallback:
        LOGGER.warning(
            "%d semantic scores fell back to lexical similarity because the judge "
            "returned no parseable score; these are NOT comparable to the paper",
            scorer.n_fallback,
        )


def _load_per_video(out_dir: Path, mode: str, split: str) -> Dict[str, Dict[str, Any]]:
    path = out_dir / f"eval_{mode}_{split}.per_video.jsonl"
    return {r["video_name"]: r for r in read_jsonl(path, required=False)}


def _rebuild_scores(rows: Sequence[Dict[str, Any]]) -> List[VideoScore]:
    return [
        VideoScore(
            video_name=r["video_name"], game_name=r.get("game_name", ""),
            n_pred=int(r["n_pred"]), n_gold=int(r["n_gold"]), n_matched=int(r["n_matched"]),
            sum_s=float(r["sum_s"]), sum_w=float(r["sum_w"]), sum_iou=float(r["sum_iou"]),
            ious=list(r.get("ious") or []), pairs=[],
        )
        for r in rows
    ]


def cmd_report(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    splits = read_json(out_dir / "splits_resolved.json")
    modes = args.modes
    available_splits = [s for s in SPLIT_NAMES if s in splits]

    table: Dict[str, Dict[str, Any]] = {}
    for split in available_splits:
        for mode in modes:
            path = out_dir / f"eval_{mode}_{split}.json"
            if not path.exists():
                continue
            table[f"{mode}|{split}"] = read_json(path)

    if not table:
        raise RuntimeError(f"no eval_*.json files under {out_dir}; run `evaluate` first")

    significance: Dict[str, Any] = {}
    for split in available_splits:
        raw_p: Dict[str, float] = {}
        deltas: Dict[str, float] = {}
        for baseline in args.baselines:
            base_rows = _load_per_video(out_dir, baseline, split)
            for mode in modes:
                if mode == baseline:
                    continue
                mode_rows = _load_per_video(out_dir, mode, split)
                shared = sorted(set(base_rows) & set(mode_rows))
                if len(shared) < 5:
                    continue
                a = _rebuild_scores([mode_rows[k] for k in shared])
                b = _rebuild_scores([base_rows[k] for k in shared])
                for metric in ("f1_desc", "miou", "f1_x_iou"):
                    key = f"{split}|{mode}_vs_{baseline}|{metric}"
                    delta, p = paired_bootstrap_pvalue(
                        a, b, lambda s, m=metric: aggregate_micro(s)[m],
                        n_boot=args.n_boot, seed=args.seed,
                    )
                    raw_p[key] = p
                    deltas[key] = delta
                x = [
                    (float(np.mean(r["ious"])) if r["ious"] else 0.0)
                    for r in (mode_rows[k] for k in shared)
                ]
                y = [
                    (float(np.mean(r["ious"])) if r["ious"] else 0.0)
                    for r in (base_rows[k] for k in shared)
                ]
                stat, p = wilcoxon_signed_rank(x, y)
                key = f"{split}|{mode}_vs_{baseline}|per_video_miou_wilcoxon"
                raw_p[key] = p
                deltas[key] = float(np.mean(x) - np.mean(y))
        corrected = holm_bonferroni(raw_p, alpha=args.alpha)
        significance[split] = {
            k: {**v, "delta": deltas.get(k, float("nan"))} for k, v in corrected.items()
        }

    lines: List[str] = []
    lines.append("# VideoGlitchBench Stage 3 (KAIROS) evaluation report")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(
        "GLIDE protocol (arXiv 2604.07818, Sec. 5.2): LLM-as-judge semantic score "
        "S in [0,5], joint score W = S x IoU, Hungarian one-to-one matching, "
        "P_desc = sum(S)/N, R_desc = sum(S)/M, F1_desc = harmonic mean; "
        "mIoU = sum(IoU)/|A|; overall P = sum(W)/N, R = sum(W)/M, F1xIoU = harmonic mean."
    )
    lines.append("")
    lines.append(
        "Assumption made explicit: the paper reports percentages, which requires "
        "S to be normalised to [0,1]. This report divides S by "
        f"{args.judge_scale:g}. Matched pairs with W = 0 are "
        f"{'excluded from' if args.match_keep == 'positive' else 'retained in'} |A|; "
        "the alternative is reported as a sensitivity row where available."
    )
    lines.append("")
    lines.append(
        "The refining judge never receives ground-truth descriptions or timestamps. "
        "The scoring judge receives them and writes nothing back into the predictions. "
        "All hyperparameters were fitted on train and selected on dev."
    )
    lines.append("")

    header = (
        "| Split | Method | Precision (%) | Recall (%) | F1 (%) | mIoU | F1xIoU (%) | "
        "N pred | M gold |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines.append("## Main results")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for split in available_splits:
        for mode in modes:
            entry = table.get(f"{mode}|{split}")
            if entry is None:
                continue
            m = entry["primary_micro"]
            lines.append(
                f"| {split} | {mode} | {m['precision_desc']:.2f} | {m['recall_desc']:.2f} | "
                f"{m['f1_desc']:.2f} | {m['miou']:.3f} | {m['f1_x_iou']:.2f} | "
                f"{int(m['n_pred'])} | {int(m['n_gold'])} |"
            )
    lines.append("")

    lines.append("## Bootstrap 95% confidence intervals (video-level resampling)")
    lines.append("")
    lines.append("| Split | Method | F1 (%) | mIoU | F1xIoU (%) |")
    lines.append("|---|---|---|---|---|")
    for split in available_splits:
        for mode in modes:
            entry = table.get(f"{mode}|{split}")
            if entry is None:
                continue
            ci = entry["bootstrap_ci"]
            lines.append(
                f"| {split} | {mode} | "
                f"{ci['f1_desc']['point']:.2f} [{ci['f1_desc']['lo']:.2f}, {ci['f1_desc']['hi']:.2f}] | "
                f"{ci['miou']['point']:.3f} [{ci['miou']['lo']:.3f}, {ci['miou']['hi']:.3f}] | "
                f"{ci['f1_x_iou']['point']:.2f} [{ci['f1_x_iou']['lo']:.2f}, {ci['f1_x_iou']['hi']:.2f}] |"
            )
    lines.append("")

    lines.append("## Temporal detail")
    lines.append("")
    tiou_keys = sorted({
        k for entry in table.values() for k in entry["primary_micro"] if k.startswith("recall@")
    })
    lines.append("| Split | Method | " + " | ".join(tiou_keys) + " | multi-interval recovery (%) |")
    lines.append("|---|---|" + "---|" * (len(tiou_keys) + 1))
    for split in available_splits:
        for mode in modes:
            entry = table.get(f"{mode}|{split}")
            if entry is None:
                continue
            m = entry["primary_micro"]
            cells = " | ".join(f"{m.get(k, float('nan')):.2f}" for k in tiou_keys)
            lines.append(
                f"| {split} | {mode} | {cells} | {m.get('multi_interval_recovery', float('nan')):.2f} |"
            )
    lines.append("")

    lines.append("## Significance (Holm-Bonferroni corrected within split)")
    lines.append("")
    lines.append("| Split | Comparison | Metric | Delta | p (raw) | p (Holm) | reject H0 |")
    lines.append("|---|---|---|---|---|---|---|")
    for split, entries in significance.items():
        for key, value in sorted(entries.items()):
            _, comparison, metric = key.split("|")
            lines.append(
                f"| {split} | {comparison} | {metric} | {value['delta']:+.3f} | "
                f"{value['p_raw']:.4f} | {value['p_holm']:.4f} | "
                f"{'yes' if value['reject'] else 'no'} |"
            )
    lines.append("")

    lines.append("## Inference cost")
    lines.append("")
    lines.append("| Split | Method | judge calls | judge calls / video | VLM calls | wall clock (min) |")
    lines.append("|---|---|---|---|---|---|")
    for split in available_splits:
        for mode in modes:
            meta_path = out_dir / f"pred_{mode}_{split}.meta.json"
            if not meta_path.exists():
                continue
            meta = read_json(meta_path)
            n = max(int(meta.get("n_videos", 1)), 1)
            calls = float(meta.get("judge", {}).get("calls", 0.0))
            lines.append(
                f"| {split} | {mode} | {calls:.0f} | {calls / n:.2f} | "
                f"{meta.get('vlm_calls', 0)} | {float(meta.get('wall_clock_s', 0.0)) / 60.0:.1f} |"
            )
    lines.append("")

    lines.append("## Cross-game generalisation")
    lines.append("")
    lines.append("| Split | Method | games | F1 median | F1 IQR | worst-decile F1 |")
    lines.append("|---|---|---|---|---|---|")
    for split in available_splits:
        for mode in modes:
            entry = table.get(f"{mode}|{split}")
            if entry is None:
                continue
            values = sorted(v["f1_desc"] for v in entry["per_game"].values())
            if not values:
                continue
            q1, q3 = float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))
            worst = float(np.mean(values[: max(1, len(values) // 10)]))
            lines.append(
                f"| {split} | {mode} | {len(values)} | {float(np.median(values)):.2f} | "
                f"{q1:.2f}-{q3:.2f} | {worst:.2f} |"
            )
    lines.append("")

    lines.append("## Reference numbers from the GLIDE paper (Table 3, test protocol)")
    lines.append("")
    lines.append("| Model | Precision (%) | Recall (%) | F1 (%) | mIoU | F1xIoU (%) |")
    lines.append("|---|---|---|---|---|---|")
    lines.append("| Qwen2.5-VL-7B-Instruct (vanilla) | 10.41 | 14.74 | 12.20 | 0.30 | 4.11 |")
    lines.append("| Qwen2.5-VL-7B-Instruct + GLIDE | 34.55 | 45.09 | 39.12 | 0.53 | 19.46 |")
    lines.append("| Qwen2.5-VL-3B-Instruct (vanilla) | 11.08 | 26.53 | 15.63 | 0.31 | 4.81 |")
    lines.append("| Qwen2.5-VL-3B-Instruct + GLIDE | 32.36 | 39.38 | 35.52 | 0.48 | 17.29 |")
    lines.append("| claude-3.5-haiku (best proprietary) | 21.66 | 32.54 | 26.01 | 0.47 | 12.91 |")
    lines.append("")
    lines.append(
        "These are transcribed from the published paper, not reproduced here. They are "
        "comparable to the in-house rows only insofar as the split, the semantic judge "
        "and the S-normalisation match; the paper does not release the judge model or "
        "the split lists, so the in-house `glide` row is the controlled comparison."
    )
    lines.append("")

    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    write_json(out_dir / "report_significance.json", significance)
    LOGGER.info("wrote %s", report_path)


# --------------------------------------------------------------------------- #
# numerical self-test
# --------------------------------------------------------------------------- #


def cmd_selftest(args: argparse.Namespace) -> None:
    failures: List[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        if condition:
            LOGGER.info("PASS  %s", name)
        else:
            failures.append(f"{name}: {detail}")
            LOGGER.error("FAIL  %s %s", name, detail)

    a = [(0.0, 10.0)]
    b = [(5.0, 15.0)]
    check("iou_single", abs(interval_iou(a, b) - 5.0 / 15.0) < 1e-12, str(interval_iou(a, b)))
    c = [(0.0, 2.0), (4.0, 6.0)]
    d = [(1.0, 5.0)]
    check("iou_multi", abs(interval_iou(c, d) - 2.0 / 6.0) < 1e-12, str(interval_iou(c, d)))
    check("iou_disjoint", interval_iou([(0.0, 1.0)], [(2.0, 3.0)]) == 0.0)
    check("iou_identical", abs(interval_iou(c, c) - 1.0) < 1e-12)
    check("merge_touching", merge_intervals([(0.0, 1.0), (1.0, 2.0)]) == [(0.0, 2.0)])

    def perfect(p: str, g: str) -> float:
        return 1.0 if p == g else 0.0

    pred = [
        {"description": "x", "intervals": [(0.0, 10.0)]},
        {"description": "y", "intervals": [(20.0, 30.0)]},
    ]
    gold = [
        {"description": "y", "intervals": [(20.0, 30.0)]},
        {"description": "x", "intervals": [(0.0, 10.0)]},
    ]
    score = match_video(pred, gold, perfect)
    micro = aggregate_micro([score])
    check("hungarian_permutation", score.n_matched == 2 and abs(micro["f1_desc"] - 100.0) < 1e-9,
          json.dumps(micro))
    check("hungarian_miou", abs(micro["miou"] - 1.0) < 1e-9, str(micro["miou"]))
    check("hungarian_f1xiou", abs(micro["f1_x_iou"] - 100.0) < 1e-9, str(micro["f1_x_iou"]))

    half = match_video(
        [{"description": "x", "intervals": [(0.0, 10.0)]}],
        [{"description": "x", "intervals": [(0.0, 20.0)]}],
        perfect,
    )
    hm = aggregate_micro([half])
    check("iou_scaling", abs(hm["miou"] - 0.5) < 1e-9 and abs(hm["f1_x_iou"] - 50.0) < 1e-9,
          json.dumps(hm))

    empty_gold = match_video([{"description": "x", "intervals": [(0.0, 1.0)]}], [], perfect)
    agg = aggregate_micro([empty_gold])
    check("false_positive_on_clean_video", agg["precision_desc"] == 0.0 and agg["f1_desc"] == 0.0)

    rng = np.random.default_rng(0)
    memory = CrossGameNormalityMemory()
    memory.levels["global"]["global"] = NormalityLevel("global", 1000, 0.0, 1.0)
    memory.tau = {"game": 10.0, "subgenre": 10.0, "genre": 10.0, "global": 0.0}
    calib = sorted(rng.normal(0.0, 1.0, 2000).tolist())
    memory.calibration = calib
    fresh = rng.normal(0.0, 1.0, 20000)
    ps = np.array([memory.conformal_p(float(v)) for v in fresh])
    for alpha in (0.05, 0.1, 0.2):
        rate = float(np.mean(ps <= alpha))
        check(f"conformal_validity_alpha{alpha}", rate <= alpha + 0.02, f"empirical {rate:.4f}")

    groups = {
        "a": list(rng.normal(1.0, 1.0, 200)),
        "b": list(rng.normal(-1.0, 1.0, 200)),
        "c": list(rng.normal(0.0, 1.0, 5)),
    }
    grand, msw, tau = random_effects_shrinkage(groups)
    check("shrinkage_finite", math.isfinite(tau) and tau > 0.0, f"tau={tau}")
    lam_big = 200.0 / (200.0 + tau)
    lam_small = 5.0 / (5.0 + tau)
    check("shrinkage_monotone", lam_big > lam_small, f"{lam_big:.3f} vs {lam_small:.3f}")

    transitions = {"p11": 0.85, "p01": 0.05, "pi1": 0.1}
    emissions = [-3.0, -3.0, 4.0, 4.0, -3.0, -3.0, 8.0, -3.0]
    path = constrained_viterbi(emissions, transitions)
    check("viterbi_recovers_runs", path == [0, 0, 1, 1, 0, 0, 1, 0], str(path))
    runs = runs_from_path(path)
    check("viterbi_multi_interval", runs == [(2, 3), (6, 6)], str(runs))

    weak = constrained_viterbi([-3.0, -3.0, 4.0, 4.0, -3.0, -3.0, 4.0, -3.0], transitions)
    check("viterbi_suppresses_isolated_spike", runs_from_path(weak) == [(2, 3)],
          str(runs_from_path(weak)))
    entry_cost = math.log(transitions["p01"]) - math.log(1.0 - transitions["p01"])
    exit_cost = math.log(1.0 - transitions["p11"]) - math.log(transitions["p11"])
    check("viterbi_entry_bar_is_the_duration_prior",
          -(entry_cost + exit_cost) > 4.0 and -(entry_cost + exit_cost) < 8.0,
          f"bar={-(entry_cost + exit_cost):.3f} nats")

    clamped = constrained_viterbi(emissions, transitions, clamped_on=[0])
    check("viterbi_clamp_on", clamped[0] == 1, str(clamped))
    clamped_off = constrained_viterbi(emissions, transitions, clamped_off=[2])
    check("viterbi_clamp_off", clamped_off[2] == 0, str(clamped_off))

    flat = constrained_viterbi([0.0] * 6, {"p11": 0.5, "p01": 0.5, "pi1": 0.5})
    check("viterbi_degenerate_stable", len(flat) == 6)

    edv_mid = expected_decision_value(0.5, 1.0, 1.0)
    edv_edge = expected_decision_value(0.99, 1.0, 1.0)
    check("edv_peaks_at_boundary", edv_mid > edv_edge, f"{edv_mid} vs {edv_edge}")
    rows = [{"posterior": p} for p in (0.01, 0.5, 0.45, 0.99, 0.6)]
    picked = select_verification_budget(rows, 2, 1.0, 1.0, 0.0)
    check("edv_budget_picks_boundary", set(picked) == {1, 2}, str(picked))

    p_uniform = list(np.linspace(0.01, 0.99, 100))
    keep = benjamini_hochberg(p_uniform, 0.1)
    check("bh_monotone", keep.sum() <= 20, f"selected {int(keep.sum())}")
    keep_all = benjamini_hochberg([1e-9] * 10, 0.05)
    check("bh_all_significant", keep_all.all())

    aff = np.array([
        [1.0, 0.9, 0.1],
        [0.9, 1.0, 0.1],
        [0.1, 0.1, 1.0],
    ])
    clusters = average_linkage(aff, 0.5)
    check("linkage_splits", sorted(len(c) for c in clusters) == [1, 2], str(clusters))
    check("linkage_order_invariant",
          average_linkage(aff[::-1, ::-1], 0.5) and
          sorted(len(c) for c in average_linkage(aff[::-1, ::-1], 0.5)) == [1, 2])

    holm = holm_bonferroni({"a": 0.001, "b": 0.02, "c": 0.9}, alpha=0.05)
    check("holm_ordering", holm["a"]["p_holm"] <= holm["b"]["p_holm"] <= holm["c"]["p_holm"],
          json.dumps(holm))
    check("holm_reject", holm["a"]["reject"] and not holm["c"]["reject"])

    tfidf = TfIdf(["the red car floats above the road", "a blue truck clips through a wall"])
    vec = tfidf.transform(["red car floating above road", "blue truck clipping wall"])
    sim = cosine_affinity(vec)
    check("tfidf_offdiag_low", sim[0, 1] < 0.3, str(sim[0, 1]))

    rows = [
        {"t_start": 0.0, "t_end": 2.0, "fps": 4.0},
        {"t_start": 2.0, "t_end": 4.0, "fps": 4.0},
    ]
    start, end = refine_boundaries(rows, (0, 1), {0: [2], 1: [1]}, 10.0, True)
    check("boundary_refinement", abs(start - 0.5) < 1e-6 and abs(end - 2.5) < 1e-6,
          f"{start},{end}")
    start, end = refine_boundaries(rows, (0, 1), {}, 10.0, True)
    check("boundary_no_salient", abs(start - 0.0) < 1e-6 and abs(end - 4.0) < 1e-6,
          f"{start},{end}")

    check("logit_expit_roundtrip", abs(expit(logit(0.73)) - 0.73) < 1e-9)
    check("harmonic_zero", harmonic(0.0, 5.0) == 0.0)

    if failures:
        LOGGER.error("%d self-tests FAILED", len(failures))
        for line in failures:
            LOGGER.error("  %s", line)
        raise SystemExit(1)
    LOGGER.info("all self-tests passed")


def cmd_all(args: argparse.Namespace) -> None:
    cmd_memory(args)
    for split in args.splits:
        for mode in args.modes:
            sub = argparse.Namespace(**vars(args))
            sub.split = split
            sub.mode = mode
            cmd_predict(sub)
            cmd_evaluate(sub)
    cmd_report(args)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-dir", default="./vgb_work")
    parser.add_argument("--out-dir", default="./vgb_stage3")
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--stage12", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("-v", "--verbose", action="count", default=1)


def add_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--min-pixels", type=int, default=64 * 28 * 28)
    parser.add_argument("--fit", default="letterbox", choices=["letterbox", "stretch"])


def add_judge(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--judge-backend", default="vlm", choices=["vlm", "hf", "openai", "offline"])
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-tokens", type=int, default=512)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    mem = sub.add_parser("memory", help="fit the cross-game normality memory on TRAIN only")
    add_common(mem)
    mem.add_argument("--min-cover", type=float, default=0.25)
    mem.add_argument("--traps-per-game", type=int, default=12)
    mem.set_defaults(func=cmd_memory)

    pred = sub.add_parser("predict", help="produce temporally grounded reports for one split")
    add_common(pred)
    add_model(pred)
    add_judge(pred)
    pred.add_argument("--split", default="test", choices=list(SPLIT_NAMES))
    pred.add_argument("--mode", default="kairos",
                      choices=["vanilla", "glide"] + list(ABLATIONS))
    pred.add_argument("--threshold", type=float, default=0.5)
    pred.add_argument("--budget", type=int, default=4)
    pred.add_argument("--cost-fp", type=float, default=1.0)
    pred.add_argument("--cost-fn", type=float, default=2.0)
    pred.add_argument("--edv-floor", type=float, default=0.02)
    pred.add_argument("--prior", type=float, default=None)
    pred.add_argument("--fdr-q", type=float, default=0.25)
    pred.add_argument("--link-threshold", type=float, default=0.34)
    pred.add_argument("--link-band-lo", type=float, default=0.12)
    pred.add_argument("--link-band-hi", type=float, default=0.62)
    pred.add_argument("--affinity-weight", type=float, default=1.2)
    pred.add_argument("--emission-shift", type=float, default=0.0)
    pred.add_argument("--max-events", type=int, default=6)
    pred.add_argument("--glide-max-iter", type=int, default=5)
    pred.add_argument("--glide-conf", type=float, default=0.80)
    pred.add_argument("--refine-tokens", type=int, default=384)
    pred.add_argument("--use-pixels", action="store_true", default=True)
    pred.add_argument("--no-pixels", dest="use_pixels", action="store_false")
    pred.add_argument("--limit-videos", type=int, default=None)
    pred.add_argument("--frozen-config", default=None)
    pred.set_defaults(func=cmd_predict)

    ev = sub.add_parser("evaluate", help="GLIDE-protocol scoring for one split and mode")
    add_common(ev)
    add_model(ev)
    ev.add_argument("--split", default="test", choices=list(SPLIT_NAMES))
    ev.add_argument("--mode", default="kairos")
    ev.add_argument("--scorer-backend", default="openai",
                    choices=["openai", "hf", "vlm", "offline"])
    ev.add_argument("--scorer-model", default="gpt-4o-mini")
    ev.add_argument("--judge-base-url", default=None)
    ev.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ev.add_argument("--judge-scale", type=float, default=5.0)
    ev.add_argument("--match-keep", default="positive", choices=["positive", "all"])
    ev.add_argument("--tiou", type=float, nargs="*", default=[0.3, 0.5, 0.7])
    ev.add_argument("--n-boot", type=int, default=2000)
    ev.add_argument("--alpha", type=float, default=0.05)
    ev.set_defaults(func=cmd_evaluate)

    rep = sub.add_parser("report", help="assemble the cross-split, cross-method report")
    add_common(rep)
    rep.add_argument("--modes", nargs="*",
                     default=["vanilla", "glide", "kairos", "kairos_no_xnm",
                              "kairos_no_gate", "kairos_no_dp", "kairos_no_boundary"])
    rep.add_argument("--baselines", nargs="*", default=["vanilla", "glide"])
    rep.add_argument("--judge-scale", type=float, default=5.0)
    rep.add_argument("--match-keep", default="positive", choices=["positive", "all"])
    rep.add_argument("--n-boot", type=int, default=2000)
    rep.add_argument("--alpha", type=float, default=0.05)
    rep.set_defaults(func=cmd_report)

    allc = sub.add_parser("all", help="memory -> predict -> evaluate -> report over every split")
    add_common(allc)
    add_model(allc)
    add_judge(allc)
    allc.add_argument("--splits", nargs="*", default=["train", "dev", "test"])
    allc.add_argument("--modes", nargs="*", default=["vanilla", "glide", "kairos"])
    allc.add_argument("--baselines", nargs="*", default=["vanilla", "glide"])
    allc.add_argument("--min-cover", type=float, default=0.25)
    allc.add_argument("--traps-per-game", type=int, default=12)
    allc.add_argument("--threshold", type=float, default=0.5)
    allc.add_argument("--budget", type=int, default=4)
    allc.add_argument("--cost-fp", type=float, default=1.0)
    allc.add_argument("--cost-fn", type=float, default=2.0)
    allc.add_argument("--edv-floor", type=float, default=0.02)
    allc.add_argument("--prior", type=float, default=None)
    allc.add_argument("--fdr-q", type=float, default=0.25)
    allc.add_argument("--link-threshold", type=float, default=0.34)
    allc.add_argument("--link-band-lo", type=float, default=0.12)
    allc.add_argument("--link-band-hi", type=float, default=0.62)
    allc.add_argument("--affinity-weight", type=float, default=1.2)
    allc.add_argument("--emission-shift", type=float, default=0.0)
    allc.add_argument("--max-events", type=int, default=6)
    allc.add_argument("--glide-max-iter", type=int, default=5)
    allc.add_argument("--glide-conf", type=float, default=0.80)
    allc.add_argument("--refine-tokens", type=int, default=384)
    allc.add_argument("--use-pixels", action="store_true", default=True)
    allc.add_argument("--no-pixels", dest="use_pixels", action="store_false")
    allc.add_argument("--limit-videos", type=int, default=None)
    allc.add_argument("--frozen-config", default=None)
    allc.add_argument("--scorer-backend", default="openai",
                      choices=["openai", "hf", "vlm", "offline"])
    allc.add_argument("--scorer-model", default="gpt-4o-mini")
    allc.add_argument("--judge-scale", type=float, default=5.0)
    allc.add_argument("--match-keep", default="positive", choices=["positive", "all"])
    allc.add_argument("--tiou", type=float, nargs="*", default=[0.3, 0.5, 0.7])
    allc.add_argument("--n-boot", type=int, default=2000)
    allc.add_argument("--alpha", type=float, default=0.05)
    allc.set_defaults(func=cmd_all)

    st = sub.add_parser("selftest", help="numerical verification of every estimator in this file")
    add_common(st)
    st.set_defaults(func=cmd_selftest)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    set_seed(args.seed)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
