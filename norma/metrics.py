"""
GliDe evaluation protocol (arXiv 2604.07818, adapted from SODA) + statistics.

  S_ij  LLM-judge semantic score in [0,5], normalised to [0,1] by /5
        (the paper reports percentages but does not state the normalisation)
  W_ij  = S_ij * tIoU(T_i, T_j)          tIoU over interval SETS
  A     = Hungarian one-to-one matching maximising sum W, pairs with W > 0
  P_desc = sum_A S / N,  R_desc = sum_A S / M,  F1_desc = harmonic mean
  mIoU  = mean_A tIoU
  P_all = sum_A W / N,   R_all = sum_A W / M,   F1xIoU = harmonic mean
  N = #predicted events, M = #gold events, micro-averaged over all videos.

The paper does not name its judge. Absolute numbers are therefore only
comparable to Table 3 after re-scoring the vanilla backbone with the same
judge you use here; every report states the judge model.

Statistics: videos of one game are NOT independent (same engine, same bug
recurring across uploads), so uncertainty is estimated with a CLUSTER
bootstrap over games (Field & Welsh 2007). Paired comparisons use the
null-centred bootstrap test (Hall & Wilson 1991) with Holm correction.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import LOGGER, harmonic, interval_iou, read_jsonl, sha1_short

SUM_KEYS = ("n_pred", "n_gold", "n_matched", "sum_s", "sum_w", "sum_iou")


@dataclasses.dataclass
class VideoScore:
    video_name: str
    game: str
    n_pred: int
    n_gold: int
    n_matched: int
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

    n, m = len(pred), len(gold)
    if n == 0 or m == 0:
        return VideoScore(name, game, n, m, 0, 0.0, 0.0, 0.0, [], [])
    s_mat = np.zeros((n, m))
    iou_mat = np.zeros((n, m))
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            iou_mat[i, j] = interval_iou(p["intervals"], g["intervals"])
            s_mat[i, j] = semantic(p.get("description", ""), g.get("description", ""))
    w_mat = s_mat * iou_mat
    rows, cols = linear_sum_assignment(-w_mat)
    score = VideoScore(name, game, n, m, 0, 0.0, 0.0, 0.0, [], [])
    for i, j in zip(rows, cols):
        if w_mat[i, j] <= 0.0:
            continue
        score.n_matched += 1
        score.sum_s += float(s_mat[i, j])
        score.sum_w += float(w_mat[i, j])
        score.sum_iou += float(iou_mat[i, j])
        score.ious.append(float(iou_mat[i, j]))
        score.pairs.append({"pred": int(i), "gold": int(j), "s": float(s_mat[i, j]),
                            "iou": float(iou_mat[i, j])})
    return score


def metrics_from_sums(t: np.ndarray) -> Dict[str, float]:
    n_pred, n_gold, n_matched, sum_s, sum_w, sum_iou = (float(v) for v in t)
    p_desc = 100.0 * sum_s / n_pred if n_pred else 0.0
    r_desc = 100.0 * sum_s / n_gold if n_gold else 0.0
    p_all = 100.0 * sum_w / n_pred if n_pred else 0.0
    r_all = 100.0 * sum_w / n_gold if n_gold else 0.0
    return {
        "precision": p_desc, "recall": r_desc, "f1": harmonic(p_desc, r_desc),
        "miou": sum_iou / n_matched if n_matched else 0.0,
        "precision_overall": p_all, "recall_overall": r_all, "f1_x_iou": harmonic(p_all, r_all),
        "n_pred": n_pred, "n_gold": n_gold, "n_matched": n_matched,
    }


def aggregate(scores: Sequence[VideoScore]) -> Dict[str, float]:
    if not scores:
        return metrics_from_sums(np.zeros(len(SUM_KEYS)))
    return metrics_from_sums(np.sum([s.sums() for s in scores], axis=0))


def temporal_proxy(pred_by_video: Dict[str, List[Dict[str, Any]]],
                   gold_by_video: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
    """The protocol with S == 1 for every pair: a judge-free upper bound on
    what localisation alone can deliver. Used for model selection on dev."""
    scores = [match_video(pred_by_video.get(k, []), gold, lambda a, b: 1.0)
              for k, gold in gold_by_video.items()]
    return aggregate(scores)


def recall_at_tiou(scores: Sequence[VideoScore], thresholds: Sequence[float]) -> Dict[str, float]:
    total = sum(s.n_gold for s in scores)
    return {f"R@tIoU{t:g}": 100.0 * sum(sum(i >= t for i in s.ious) for s in scores) / max(total, 1)
            for t in thresholds}


# --------------------------------------------------------------------------- #
# cluster bootstrap and paired tests
# --------------------------------------------------------------------------- #


def _cluster_sums(scores: Sequence[VideoScore], cluster: str) -> Tuple[List[str], np.ndarray]:
    groups: Dict[str, np.ndarray] = {}
    for s in scores:
        key = s.game if cluster == "game" else s.video_name
        groups[key] = groups.get(key, np.zeros(len(SUM_KEYS))) + s.sums()
    keys = sorted(groups)
    return keys, np.stack([groups[k] for k in keys])


def bootstrap_ci(scores: Sequence[VideoScore], metric: str, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0, cluster: str = "game") -> Dict[str, float]:
    keys, sums = _cluster_sums(scores, cluster)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(keys), np.full(len(keys), 1.0 / len(keys)), size=n_boot)
    draws = np.array([metrics_from_sums(w @ sums)[metric] for w in weights])
    return {"point": metrics_from_sums(sums.sum(0))[metric],
            "lo": float(np.quantile(draws, alpha / 2)), "hi": float(np.quantile(draws, 1 - alpha / 2)),
            "cluster": cluster, "n_clusters": len(keys)}


def paired_test(a: Sequence[VideoScore], b: Sequence[VideoScore], metric: str,
                n_boot: int = 5000, seed: int = 0, cluster: str = "game") -> Dict[str, float]:
    """
    H0: metric(A) = metric(B) on the population of games. Resample games
    with replacement (same draw for both systems: paired), form
    delta* = m(A*) - m(B*), and use the null-centred two-sided p-value
        p = (1 + #{|delta* - delta| >= |delta|}) / (1 + B).
    """
    common = sorted({s.video_name for s in a} & {s.video_name for s in b})
    a_map = {s.video_name: s for s in a}
    b_map = {s.video_name: s for s in b}
    a_s = [a_map[k] for k in common]
    b_s = [b_map[k] for k in common]
    keys, sa = _cluster_sums(a_s, cluster)
    _, sb = _cluster_sums(b_s, cluster)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(keys), np.full(len(keys), 1.0 / len(keys)), size=n_boot)
    delta = metrics_from_sums(sa.sum(0))[metric] - metrics_from_sums(sb.sum(0))[metric]
    draws = np.array([metrics_from_sums(w @ sa)[metric] - metrics_from_sums(w @ sb)[metric]
                      for w in weights])
    p = (1.0 + np.sum(np.abs(draws - delta) >= abs(delta))) / (1.0 + n_boot)
    return {"delta": float(delta), "p": float(min(1.0, p)),
            "ci_lo": float(np.quantile(draws, 0.025)), "ci_hi": float(np.quantile(draws, 0.975)),
            "n_videos": len(common), "n_clusters": len(keys)}


def holm(pvalues: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    out, running, m = {}, 0.0, len(items)
    for rank, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        out[key] = {"p_raw": p, "p_holm": running, "reject": running <= alpha}
    return out


# --------------------------------------------------------------------------- #
# ranking metrics (frame-level VAD protocol and unit diagnostics)
# --------------------------------------------------------------------------- #


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U / (n1 n0), average ranks for ties."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    y = np.asarray(labels).astype(float)[order]
    if y.sum() == 0:
        return float("nan")
    precision = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


# --------------------------------------------------------------------------- #
# semantic judges
# --------------------------------------------------------------------------- #

JUDGE_SYSTEM = (
    "You are a strict evaluator for a video game QA benchmark. You compare a "
    "system-generated glitch description against a human-written reference "
    "description of the same gameplay video and rate their semantic agreement. "
    "You never reward fluency, length or confidence. You reward only agreement "
    "about which entity fails, in what way it fails, and what kind of software "
    "defect it is."
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
    "a an the and or but of in on at to from by with for as is are was were be been it its this that "
    "these those there then than into onto over under through while which who what when where how not "
    "no so very more most some any each other one two game player character video clip screen frame".split()
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
    """backend: 'openai' (any OpenAI-compatible endpoint, incl. vLLM),
    'hf' (local causal LM), 'offline' (unigram-F1; NOT comparable to the paper)."""

    def __init__(self, backend: str, model: str, cache_path: Path, base_url: Optional[str] = None,
                 api_key_env: str = "OPENAI_API_KEY", dtype: str = "bfloat16") -> None:
        self.backend, self.model = backend, model
        self.calls = self.fallbacks = 0
        self.cache: Dict[str, float] = {}
        self.cache_path = Path(cache_path)
        for row in read_jsonl(self.cache_path, required=False):
            self.cache[row["key"]] = float(row["value"])
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._sink = self.cache_path.open("a", encoding="utf-8")
        if backend == "openai":
            import openai

            key = os.environ.get(api_key_env) or "EMPTY"
            self._client = openai.OpenAI(api_key=key, base_url=base_url)
        elif backend == "hf":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(model)
            self._lm = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=getattr(torch, dtype), device_map="auto").eval()
        elif backend != "offline":
            raise ValueError(f"unknown judge backend {backend!r}")

    def _chat(self, user: str) -> str:
        self.calls += 1
        if self.backend == "openai":
            out = self._client.chat.completions.create(
                model=self.model, temperature=0.0, max_tokens=160,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": user}])
            return out.choices[0].message.content or ""
        import torch

        text = self._tok.apply_chat_template(
            [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        inputs = self._tok([text], return_tensors="pt").to(self._lm.device)
        with torch.inference_mode():
            out = self._lm.generate(**inputs, do_sample=False, max_new_tokens=160)
        return self._tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def __call__(self, pred: str, gold: str) -> float:
        pred, gold = (pred or "").strip(), (gold or "").strip()
        if not pred or not gold:
            return 0.0
        key = sha1_short(f"{self.backend}|{self.model}|{pred}|{gold}", 24)
        if key in self.cache:
            return self.cache[key]
        if self.backend == "offline":
            value = unigram_f1(pred, gold)
        else:
            raw = ""
            for attempt in range(3):
                try:
                    raw = self._chat(JUDGE_PROMPT.format(gold=gold, pred=pred))
                    break
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("judge call failed (%r), retry %d", exc, attempt + 1)
                    time.sleep(2 ** attempt)
            match = re.search(r'"score"\s*:\s*([0-5])', raw) or re.search(r"\b([0-5])\b", raw)
            if match:
                value = float(match.group(1)) / 5.0
            else:
                self.fallbacks += 1
                value = unigram_f1(pred, gold)
        self.cache[key] = value
        self._sink.write(json.dumps({"key": key, "value": value}) + "\n")
        self._sink.flush()
        return value

    def close(self) -> None:
        self._sink.close()
