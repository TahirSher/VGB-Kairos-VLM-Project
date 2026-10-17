"""
Localise-then-describe: a LoRA-SFT describer with a learned veto, MBR
decoding, a stacked confidence calibrator, and the two baselines every
claim must be measured against (vanilla zero-shot, direct SFT).

Describer target: the human bug description for a gold event, or the literal
token NO_GLITCH for a normal segment. P(first token = NO_GLITCH) is a second,
pixel-grounded opinion on every proposal (the learned analogue of GliDe's
debate loop, at the cost of one prefill instead of up to 5 x 3 agent turns).

Training clips are cut around gold events with randomly jittered boundaries,
so the describer sees the same kind of imperfect crops the localiser will
hand it at test time (exposure-bias mitigation).

MBR decoding (Kumar & Byrne 2004; Freitag et al. 2022): among the greedy
output and n sampled candidates, return argmax_i sum_j u(c_i, c_j). Under a
semantic judge this approximates the Bayes-optimal output for the expected
score, rather than the most probable string.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import LOGGER, import_stage12, interval_iou, merge_intervals, write_json
from .data import NO_GLITCH, VideoItem, gold_events, select_frames
from .metrics import unigram_f1

DESCRIBE_PROMPT = (
    'Gameplay clip from "{game}" ({genre} / {subgenre}), showing {span} of the recording.\n'
    "If the clip shows a software glitch, describe it in one or two sentences: name the affected "
    "entity by its appearance, state what it does wrong, and say where on screen it happens. "
    "If there is no glitch, answer exactly: " + NO_GLITCH
)

DIRECT_PROMPT = (
    'This is a gameplay video from "{game}" ({genre} / {subgenre}), {duration:.1f} seconds long. '
    "Find every software glitch in it. Return ONE JSON object and nothing else:\n"
    '{{"no_bugs": <true|false>, "bugs": ["<description>", ...], '
    '"time_nodes": [[[<start_sec>, <end_sec>], ...], ...]}}\n'
    "time_nodes[i] lists the time intervals of bugs[i] in seconds."
)


def _span_text(intervals: Sequence[Tuple[float, float]]) -> str:
    return ", ".join(f"seconds {a:.1f} to {b:.1f}" for a, b in intervals)


def _game_fields(item: VideoItem) -> Dict[str, str]:
    return {"game": item.game, "genre": item.genre or "unknown", "subgenre": item.subgenre or "unknown"}


def clip_frames(item: VideoItem, intervals: Sequence[Tuple[float, float]], max_frames: int,
                margin: float) -> Tuple[List[str], float, List[Tuple[float, float]]]:
    """Frames covering every interval (+ margin), budget split by duration."""
    padded = merge_intervals([(max(0.0, a - margin), min(item.duration or b + margin, b + margin))
                              for a, b in intervals])
    total = sum(b - a for a, b in padded) or 1.0
    frames: List[str] = []
    fps_eff = []
    for a, b in padded:
        budget = max(2, int(round(max_frames * (b - a) / total)))
        f, fps = select_frames(item, a, b, budget)
        frames.extend(f)
        fps_eff.append(fps)
    if len(frames) > max_frames:
        pick = np.round(np.linspace(0, len(frames) - 1, num=max_frames)).astype(int)
        frames = [frames[i] for i in pick]
    if len(frames) % 2:
        frames.append(frames[-1])
    return frames, float(np.mean(fps_eff)), padded


# --------------------------------------------------------------------------- #
# training samples
# --------------------------------------------------------------------------- #


def describer_samples(items: Dict[str, VideoItem], names: Sequence[str], seed: int,
                      neg_ratio: float = 0.33, jitter: float = 0.2) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    pos, neg = [], []
    for name in names:
        item = items[name]
        events = gold_events(item)
        dur = item.duration or (item.timestamps[-1] if item.timestamps else 0.0)
        for ev in events:
            ivs = []
            for a, b in ev["intervals"]:
                d = max(b - a, 0.5)
                a2 = max(0.0, a + rng.uniform(-jitter, jitter) * d)
                b2 = min(dur or b, b + rng.uniform(-jitter, jitter) * d)
                if b2 - a2 < 0.5:
                    a2, b2 = a, b
                ivs.append((a2, b2))
            pos.append({"name": name, "intervals": merge_intervals(ivs), "answer": ev["description"]})
        busy = merge_intervals([iv for ev in events for iv in ev["intervals"]])
        free, cursor = [], 0.0
        for a, b in busy + [(dur, dur)]:
            if a - cursor >= 3.0:
                free.append((cursor + 0.5, a - 0.5))
            cursor = max(cursor, b)
        for a, b in free:
            length = min(b - a, rng.uniform(2.0, 6.0))
            s = rng.uniform(a, b - length)
            neg.append({"name": name, "intervals": [(s, s + length)], "answer": NO_GLITCH})
    rng.shuffle(neg)
    neg = neg[: int(math.ceil(neg_ratio * len(pos)))]
    samples = pos + neg
    rng.shuffle(samples)
    LOGGER.info("describer samples: %d positive, %d NO_GLITCH", len(pos), len(neg))
    return samples


def direct_samples(items: Dict[str, VideoItem], names: Sequence[str]) -> List[Dict[str, Any]]:
    import json

    out = []
    for name in names:
        events = gold_events(items[name])
        target = {"no_bugs": not events, "bugs": [e["description"] for e in events],
                  "time_nodes": [[[round(a, 1), round(b, 1)] for a, b in e["intervals"]] for e in events]}
        out.append({"name": name, "intervals": None, "answer": json.dumps(target, ensure_ascii=False)})
    return out


def _sample_io(item: VideoItem, sample: Dict[str, Any], max_frames: int,
               margin: float) -> Tuple[List[str], float, str]:
    if sample["intervals"] is None:                       # direct / vanilla: whole video
        frames, fps = select_frames(item, 0.0, item.duration or 1e9, max_frames)
        return frames, fps, DIRECT_PROMPT.format(duration=item.duration, **_game_fields(item))
    frames, fps, padded = clip_frames(item, sample["intervals"], max_frames, margin)
    return frames, fps, DESCRIBE_PROMPT.format(span=_span_text(padded), **_game_fields(item))


# --------------------------------------------------------------------------- #
# LoRA SFT loop (shared by the describer and the direct-SFT baseline)
# --------------------------------------------------------------------------- #


def train_lm(backbone: Any, items: Dict[str, VideoItem], train: Sequence[Dict[str, Any]],
             dev: Sequence[Dict[str, Any]], out_dir: Path, epochs: int, lr: float,
             grad_accum: int, max_frames: int, margin: float, lora_r: int, lora_alpha: int,
             seed: int, max_dev: int = 200) -> Dict[str, Any]:
    import torch

    rng = random.Random(seed)
    backbone.attach_lora(lora_r, lora_alpha, 0.05)
    params = backbone.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    total = max(1, epochs * math.ceil(len(train) / grad_accum))
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    dev = list(dev)[:max_dev]
    best, history = float("inf"), []
    order = list(range(len(train)))
    step = 0
    for epoch in range(epochs):
        backbone.train_mode(True)
        rng.shuffle(order)
        running, n_seen, t0 = 0.0, 0, time.time()
        opt.zero_grad(set_to_none=True)
        for k, idx in enumerate(order, 1):
            sample = train[idx]
            frames, fps, prompt = _sample_io(items[sample["name"]], sample, max_frames, margin)
            try:
                loss = backbone.lm_loss(frames, fps, prompt, sample["answer"])
            except torch.cuda.OutOfMemoryError:
                LOGGER.warning("OOM on %s; skipped", sample["name"])
                torch.cuda.empty_cache()
                continue
            (loss / grad_accum).backward()
            running += float(loss.detach())
            n_seen += 1
            if k % grad_accum == 0 or k == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            if k % 200 == 0:
                LOGGER.info("lm epoch %d %d/%d loss %.4f (%.2f samples/s)", epoch, k, len(order),
                            running / max(n_seen, 1), k / max(time.time() - t0, 1e-9))
        backbone.train_mode(False)
        dev_loss = 0.0
        with torch.no_grad():
            for sample in dev:
                frames, fps, prompt = _sample_io(items[sample["name"]], sample, max_frames, margin)
                dev_loss += float(backbone.lm_loss(frames, fps, prompt, sample["answer"]))
        dev_loss /= max(len(dev), 1)
        history.append({"epoch": epoch, "train_loss": running / max(n_seen, 1), "dev_loss": dev_loss})
        LOGGER.info("lm epoch %d: train %.4f dev %.4f", epoch, running / max(n_seen, 1), dev_loss)
        if dev_loss < best:
            best = dev_loss
            backbone.save_adapter(Path(out_dir))
    write_json(Path(out_dir) / "train_history.json", history)
    return {"best_dev_loss": best, "history": history}


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #


def mbr_select(candidates: Sequence[str], utility: Any = unigram_f1) -> str:
    cands = [c for c in candidates if c]
    if len(cands) <= 2:
        return cands[0] if cands else ""
    scores = [sum(utility(a, b) for j, b in enumerate(cands) if j != i) for i, a in enumerate(cands)]
    return cands[int(np.argmax(scores))]


class Describer:
    def __init__(self, backbone: Any, max_frames: int = 16, margin: float = 1.0,
                 n_samples: int = 4, temperature: float = 0.7, max_new_tokens: int = 96) -> None:
        self.backbone = backbone
        self.max_frames, self.margin = max_frames, margin
        self.n_samples, self.temperature, self.max_new_tokens = n_samples, temperature, max_new_tokens
        self.prefills = 0
        self.generations = 0

    def describe(self, item: VideoItem, intervals: Sequence[Tuple[float, float]]) -> Dict[str, Any]:
        sample = {"intervals": list(intervals)}
        frames, fps, prompt = _sample_io(item, sample, self.max_frames, self.margin)
        p_reject = self.backbone.first_token_prob(frames, fps, prompt, NO_GLITCH)
        cands = self.backbone.generate(frames, fps, prompt, self.max_new_tokens,
                                       self.n_samples, self.temperature)
        self.prefills += 1
        self.generations += len(cands)
        real = [c for c in cands if c and not c.strip().upper().startswith("NO_GLITCH")]
        vetoed = not real
        if vetoed:
            # The veto is EVIDENCE (p_reject feeds the dev-calibrated fusion),
            # not a decision: obtain the best description conditional on a
            # glitch and let the calibrated filter decide whether to emit it.
            real = self.backbone.generate(frames, fps, prompt, self.max_new_tokens, 0,
                                          self.temperature, ban=NO_GLITCH)
            self.generations += 1
        return {"description": mbr_select(real), "p_reject": float(p_reject), "vetoed": vetoed,
                "n_candidates": len(cands), "n_rejected": len(cands) - len(real) if not vetoed else len(cands)}


# --------------------------------------------------------------------------- #
# stacked confidence calibrator
# --------------------------------------------------------------------------- #


def event_features(event: Dict[str, Any]) -> np.ndarray:
    lg = lambda p: math.log(max(p, 1e-6) / max(1 - p, 1e-6))  # noqa: E731
    return np.array([lg(event["seg_score"]), lg(event["p_any"]), lg(1.0 - event.get("p_reject", 0.5))])


def fit_logreg(x: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 50) -> np.ndarray:
    """L2-regularised logistic regression by Newton-Raphson (IRLS)."""
    xb = np.hstack([x, np.ones((x.shape[0], 1))])
    w = np.zeros(xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(xb @ w, -30, 30)))
        grad = xb.T @ (p - y) + l2 * np.r_[w[:-1], 0.0]
        hess = (xb * (p * (1 - p))[:, None]).T @ xb + l2 * np.diag(np.r_[np.ones(len(w) - 1), 0.0])
        step = np.linalg.solve(hess + 1e-9 * np.eye(len(w)), grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def fused_probability(event: Dict[str, Any], weights: np.ndarray) -> float:
    z = float(np.r_[event_features(event), 1.0] @ weights)
    return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))


def event_is_tp(event: Dict[str, Any], gold: Sequence[Dict[str, Any]], tiou: float = 0.3) -> int:
    return int(any(interval_iou(event["intervals"], g["intervals"]) >= tiou for g in gold))


# --------------------------------------------------------------------------- #
# baselines: parse a whole-video JSON report
# --------------------------------------------------------------------------- #


def parse_report(text: str, duration: float) -> List[Dict[str, Any]]:
    stage12 = import_stage12()
    parsed = stage12.extract_json_object(text) or {}
    bugs = parsed.get("bugs") or []
    nodes = parsed.get("time_nodes") or []
    if parsed.get("no_bugs") is True and not bugs:
        return []
    from .common import normalise_intervals

    out = []
    for i, desc in enumerate(bugs if isinstance(bugs, list) else [bugs]):
        ivs = normalise_intervals(nodes[i] if i < len(nodes) else [])
        if duration > 0:
            ivs = [(max(0.0, a), min(duration, b)) for a, b in ivs if a < duration]
            ivs = [(a, b) for a, b in ivs if b > a]
        if str(desc).strip() and ivs:
            out.append({"description": str(desc).strip(), "intervals": ivs})
    return out
