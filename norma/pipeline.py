"""
Command implementations. Per fold f of the game-disjoint CV:

  train-head  f   fit normality bank + head on TRAIN, select/calibrate on DEV
  train-desc  f   LoRA-SFT describer on TRAIN, select on DEV loss
  predict     f   DEV: head -> describer -> fit fused calibrator + emit threshold
                  TEST: head -> describer -> fused filter -> GliDe-format reports
  baseline    f   vanilla zero-shot backbone, or direct-SFT (train-direct f)
  evaluate        GliDe protocol over out-of-fold predictions of a tag
  compare         paired cluster-bootstrap tests between tags, Holm-corrected

Nothing fitted ever sees the TEST fold; the predict command refuses to emit
test predictions unless the fusion parameters were fitted on DEV.
"""

from __future__ import annotations

import dataclasses
import itertools
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .common import LOGGER, read_json, read_jsonl, write_json, write_jsonl
from .data import VideoItem, gold_events, load_corpus, unit_targets
from .features import extract_corpus, load_features
from .splits import make_folds


def _device() -> Any:
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fold(run_dir: Path, fold: int) -> Dict[str, Any]:
    return read_json(Path(run_dir) / "splits.json")["folds"][fold]


def _fold_dir(run_dir: Path, fold: int) -> Path:
    path = Path(run_dir) / f"fold{fold}"
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# split / extract
# --------------------------------------------------------------------------- #


def cmd_split(args) -> None:
    items = load_corpus(Path(args.work_dir))
    payload = make_folds(items, args.k, args.dev_frac, args.seed)
    write_json(Path(args.run_dir) / "splits.json", payload)


def cmd_extract(args) -> None:
    from .backbones import build_backbone

    items = load_corpus(Path(args.work_dir))
    backbone = build_backbone(args.model, args)
    fps = args.fps or backbone.default_fps
    extract_corpus(items, backbone, Path(args.feat_dir), fps, args.chunk_frames, args.proj_dim,
                   args.seed, args.overwrite, names=None if not args.limit else sorted(items)[: args.limit])


# --------------------------------------------------------------------------- #
# head
# --------------------------------------------------------------------------- #


def head_samples(items: Dict[str, VideoItem], feats: Dict[str, Dict[str, np.ndarray]],
                 names: Sequence[str], bank: Any, exclude_own_game: bool, k_max: int) -> List[Dict[str, Any]]:
    out = []
    for name in names:
        if name not in feats:
            continue
        item, f = items[name], feats[name]
        events = gold_events(item)
        tg = unit_targets(f["times"], events)
        out.append({
            "name": name, "game": item.game, "duration": item.duration,
            "feats": f["feats"], "ctx": f["ctx"], "times": f["times"],
            "bank": bank.distance(f["feats"], item.game if exclude_own_game else None),
            "count": min(k_max, len(events)), **tg,
        })
    return out


def _decode_all(model: Any, samples: Sequence[Dict[str, Any]], prm: Any, trans: Dict[str, float],
                device: Any) -> Dict[str, List[Dict[str, Any]]]:
    from .head import decode_video, run_head

    raw = run_head(model, samples, device)
    return {s["name"]: decode_video(r, s["times"], prm, trans, s["duration"]) for s, r in zip(samples, raw)}


def _gold_map(items: Dict[str, VideoItem], names: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    return {n: gold_events(items[n]) for n in names}


def cmd_train_head(args) -> None:
    import torch

    from .head import (DecodeParams, HeadConfig, NormalityBank, decode_video, fit_temperature,
                       fit_transitions, run_head, train_head)
    from .metrics import auroc, average_precision, temporal_proxy

    items = load_corpus(Path(args.work_dir))
    fold = _fold(args.run_dir, args.fold)
    fdir = _fold_dir(args.run_dir, args.fold)
    feat_dir = Path(args.feat_dir)
    feats = load_features(feat_dir, items, fold["train"] + fold["dev"])
    device = _device()

    # Normality bank from TRAIN normal units only.
    normal = []
    for name in fold["train"]:
        if name in feats:
            tg = unit_targets(feats[name]["times"], gold_events(items[name]))
            keep = tg["y"] < 0.5
            if keep.any():
                normal.append((feats[name]["feats"][keep], items[name].game))
    bank = NormalityBank(args.bank_pca, args.bank_size, args.bank_k, args.seed)
    bank.fit(normal)

    sample_feats = next(iter(feats.values()))["feats"]
    cfg = HeadConfig(n_vectors=sample_feats.shape[1], in_dim=sample_feats.shape[2],
                     d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                     use_srn=not args.no_srn, use_bank=not args.no_bank,
                     use_delta=not args.no_delta, use_ctx=not args.no_ctx)
    train = head_samples(items, feats, fold["train"], bank, True, cfg.k_max)
    dev = head_samples(items, feats, fold["dev"], bank, False, cfg.k_max)
    trans = fit_transitions([s["y"] for s in train])
    LOGGER.info("unit prior %.3f, p(1|1) %.3f, p(1|0) %.3f", trans["prior"], trans["p11"], trans["p01"])
    gold_dev = _gold_map(items, [s["name"] for s in dev])
    if args.no_group:
        base_prm = DecodeParams(group_sim=2.0)
    else:
        base_prm = DecodeParams()

    dev_labels = np.concatenate([s["y"] for s in dev])

    def select(model):
        # Primary: judge-free temporal F1xIoU on dev. Tie-break: unit AUROC,
        # so epochs where the default decoder emits nothing are still ranked.
        raw = run_head(model, dev, device)
        preds = {s["name"]: decode_video(r, s["times"], base_prm, trans, s["duration"])
                 for s, r in zip(dev, raw)}
        unit_auc = auroc(np.concatenate([r["act"] for r in raw]), dev_labels)
        return temporal_proxy(preds, gold_dev)["f1_x_iou"] + (0.0 if unit_auc != unit_auc else unit_auc)

    model, history = train_head(train, dev, cfg, trans, args.epochs, args.lr, args.batch_size,
                                args.seed, device, select)

    # Calibration + decoder thresholds on DEV.
    raw = run_head(model, dev, device)
    logits = np.concatenate([r["act"] for r in raw])
    labels = np.concatenate([s["y"] for s in dev])
    temp = fit_temperature(logits, labels)
    best, best_prm = -1.0, None
    grid = itertools.product([0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [0.2, 0.3, 0.4, 0.5, 0.6],
                             [2.0] if args.no_group else [0.5, 0.65, 0.8, 0.9, 2.0])
    for none_t, ev_t, gsim in grid:
        prm = dataclasses.replace(base_prm, temperature=temp, none_threshold=none_t,
                                  event_threshold=ev_t, group_sim=gsim)
        preds = {s["name"]: decode_video(r, s["times"], prm, trans, s["duration"]) for s, r in zip(dev, raw)}
        score = temporal_proxy(preds, gold_dev)["f1_x_iou"]
        if score > best:
            best, best_prm = score, prm
    unit_auc = auroc(logits, labels)
    unit_ap = average_precision(logits, labels)
    report = {"fold": args.fold, "dev_unit_auroc": unit_auc, "dev_unit_ap": unit_ap,
              "dev_temporal_f1_x_iou": best, "temperature": temp,
              "decode": dataclasses.asdict(best_prm), "transitions": trans, "history": history,
              "config": dataclasses.asdict(cfg)}
    write_json(fdir / f"head{args.tag_suffix}_report.json", report)
    torch.save({"cfg": dataclasses.asdict(cfg), "state": model.state_dict(), "bank": bank.state(),
                "trans": trans, "decode": dataclasses.asdict(best_prm),
                "feat_meta": read_json(feat_dir / "meta.json", required=False)},
               fdir / f"head{args.tag_suffix}.pt")
    LOGGER.info("fold %d head: dev unit AUROC %.4f AP %.4f, temporal F1xIoU %.2f (T=%.2f, %s)",
                args.fold, unit_auc, unit_ap, best, temp, best_prm)


def _load_head(fdir: Path, suffix: str, device: Any):
    import torch

    from .head import DecodeParams, HeadConfig, NormalityBank, load_model

    ckpt = torch.load(fdir / f"head{suffix}.pt", map_location="cpu", weights_only=False)
    cfg = HeadConfig(**ckpt["cfg"])
    return (load_model(cfg, ckpt["state"], device), cfg, NormalityBank.from_state(ckpt["bank"]),
            ckpt["trans"], DecodeParams(**ckpt["decode"]))


# --------------------------------------------------------------------------- #
# describer
# --------------------------------------------------------------------------- #


def cmd_train_desc(args) -> None:
    from .backbones import build_backbone
    from .describer import describer_samples, direct_samples, train_lm

    items = load_corpus(Path(args.work_dir))
    fold = _fold(args.run_dir, args.fold)
    fdir = _fold_dir(args.run_dir, args.fold)
    backbone = build_backbone(args.model, args)
    if args.direct:
        train, dev = direct_samples(items, fold["train"]), direct_samples(items, fold["dev"])
        out, frames = fdir / "direct", args.direct_frames
    else:
        train = describer_samples(items, fold["train"], args.seed, args.neg_ratio)
        dev = describer_samples(items, fold["dev"], args.seed + 1, args.neg_ratio)
        out, frames = fdir / "describer", args.desc_frames
    result = train_lm(backbone, items, train, dev, out, args.desc_epochs, args.desc_lr,
                      args.grad_accum, frames, args.margin, args.lora_r, args.lora_alpha, args.seed)
    write_json(out / "summary.json", {"fold": args.fold, "n_train": len(train), "n_dev": len(dev),
                                      **{k: v for k, v in result.items() if k != "history"}})


# --------------------------------------------------------------------------- #
# predict
# --------------------------------------------------------------------------- #


def _report_row(item: VideoItem, events: Sequence[Dict[str, Any]], extra: Dict[str, Any]) -> Dict[str, Any]:
    """GliDe summariser schema + NORMA diagnostics."""
    return {
        "video_name": item.name, "game_name": item.game, "genre": item.genre,
        "subgenre": item.subgenre, "duration": item.duration,
        "no_bugs": not events,
        "bugs": [e["description"] for e in events],
        "time_nodes": [[list(iv) for iv in e["intervals"]] for e in events],
        "events": [{k: v for k, v in e.items() if k not in ("description",)} for e in events],
        **extra,
    }


def _run_split(args, items, fold, fdir, split, backbone, device) -> List[Dict[str, Any]]:
    from .describer import Describer
    from .head import run_head, decode_video

    model, cfg, bank, trans, prm = _load_head(fdir, args.tag_suffix, device)
    names = fold[split][: args.limit] if args.limit else fold[split]
    feats = load_features(Path(args.feat_dir), items, names)
    samples = head_samples(items, feats, names, bank, False, cfg.k_max)
    t0 = time.time()
    raw = run_head(model, samples, device)
    head_s = time.time() - t0
    describer = None
    if backbone is not None:
        describer = Describer(backbone, args.desc_frames, args.margin, args.mbr_samples,
                              args.mbr_temperature, args.max_new_tokens)
    rows = []
    for s, r in zip(samples, raw):
        item = items[s["name"]]
        events = decode_video(r, s["times"], prm, trans, item.duration)
        t1 = time.time()
        for ev in events:
            if describer is not None:
                ev.update(describer.describe(item, ev["intervals"]))
            else:
                ev.update({"description": "a software glitch occurs", "p_reject": 0.5})
        p_unit = 1.0 / (1.0 + np.exp(-r["act"] / prm.temperature))
        rows.append(_report_row(item, events, {
            "unit_p": [round(float(v), 4) for v in p_unit],
            "unit_times": s["times"].round(3).tolist(),
            "cost": {"describe_s": time.time() - t1, "events": len(events)},
        }))
    LOGGER.info("%s: head %.3fs for %d videos; describer prefills %s generations %s", split, head_s,
                len(samples), getattr(describer, "prefills", 0), getattr(describer, "generations", 0))
    return rows


def _apply_fusion(rows: List[Dict[str, Any]], fusion: Optional[Dict[str, Any]]) -> None:
    from .describer import fused_probability

    for row in rows:
        kept_b, kept_t, kept_e = [], [], []
        for desc, nodes, ev in zip(row["bugs"], row["time_nodes"], row["events"]):
            ev_full = dict(ev, description=desc)
            if fusion is not None:
                ev["p_fused"] = fused_probability(ev_full, np.array(fusion["weights"]))
                if ev["p_fused"] < fusion["threshold"]:
                    continue
            if not desc:          # describer vetoed every candidate
                continue
            kept_b.append(desc)
            kept_t.append(nodes)
            kept_e.append(ev)
        row["bugs"], row["time_nodes"], row["events"] = kept_b, kept_t, kept_e
        row["no_bugs"] = not kept_b


def _fit_fusion(rows: List[Dict[str, Any]], items: Dict[str, VideoItem]) -> Dict[str, Any]:
    from .describer import event_features, event_is_tp, fit_logreg, fused_probability
    from .metrics import temporal_proxy

    x, y = [], []
    for row in rows:
        gold = gold_events(items[row["video_name"]])
        for nodes, ev in zip(row["time_nodes"], row["events"]):
            x.append(event_features(ev))
            y.append(event_is_tp({"intervals": [tuple(iv) for iv in nodes]}, gold))
    if len(set(y)) < 2:
        LOGGER.warning("fusion: dev events are all %s; using an identity calibrator", y[:1])
        weights = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        weights = fit_logreg(np.array(x), np.array(y, dtype=float))
    gold_map = {r["video_name"]: gold_events(items[r["video_name"]]) for r in rows}
    best, best_t = -1.0, 0.0
    for thr in np.linspace(0.0, 0.9, 19):
        preds = {}
        for row in rows:
            keep = []
            for desc, nodes, ev in zip(row["bugs"], row["time_nodes"], row["events"]):
                if desc and fused_probability(ev, weights) >= thr:
                    keep.append({"intervals": [tuple(iv) for iv in nodes]})
            preds[row["video_name"]] = keep
        score = temporal_proxy(preds, gold_map)["f1_x_iou"]
        if score > best:
            best, best_t = score, float(thr)
    LOGGER.info("fusion fitted on %d dev events (%.1f%% TP): w=%s, emit threshold %.2f (dev tF1xIoU %.2f)",
                len(y), 100.0 * np.mean(y) if y else 0.0, np.round(weights, 3).tolist(), best_t, best)
    return {"weights": weights.tolist(), "threshold": best_t, "dev_temporal_f1_x_iou": best,
            "n_dev_events": len(y)}


def cmd_predict(args) -> None:
    from .backbones import build_backbone

    items = load_corpus(Path(args.work_dir))
    fold = _fold(args.run_dir, args.fold)
    fdir = _fold_dir(args.run_dir, args.fold)
    device = _device()
    backbone = None
    if not args.no_describer:
        backbone = build_backbone(args.model, args)
        backbone.load_adapter(fdir / "describer")
    tag = args.tag
    fusion_path = fdir / f"fusion_{tag}.json"
    fusion = read_json(fusion_path, required=False)
    if fusion is None and not args.no_fusion:
        dev_rows = _run_split(args, items, fold, fdir, "dev", backbone, device)
        write_jsonl(fdir / f"dev_{tag}.jsonl", dev_rows)
        fusion = _fit_fusion(dev_rows, items)
        write_json(fusion_path, fusion)
    rows = _run_split(args, items, fold, fdir, "test", backbone, device)
    _apply_fusion(rows, None if args.no_fusion else fusion)
    write_jsonl(Path(args.run_dir) / f"preds_{tag}_fold{args.fold}.jsonl", rows)
    LOGGER.info("fold %d: %d test videos, %d events -> preds_%s_fold%d.jsonl", args.fold, len(rows),
                sum(len(r["bugs"]) for r in rows), tag, args.fold)


def cmd_baseline(args) -> None:
    """vanilla: off-the-shelf backbone, whole video, JSON report (paper's
    'vanilla' rows). direct: same prompt after LoRA-SFT on TRAIN (train-desc
    --direct). The direct row is the control that separates NORMA's
    architecture from the mere fact of fine-tuning."""
    from .backbones import build_backbone
    from .describer import _sample_io, parse_report

    items = load_corpus(Path(args.work_dir))
    fold = _fold(args.run_dir, args.fold)
    fdir = _fold_dir(args.run_dir, args.fold)
    backbone = build_backbone(args.model, args)
    if args.mode == "direct":
        backbone.load_adapter(fdir / "direct")
    names = fold["test"][: args.limit] if args.limit else fold["test"]
    rows, t0 = [], time.time()
    for i, name in enumerate(names, 1):
        item = items[name]
        frames, fps, prompt = _sample_io(item, {"intervals": None}, args.direct_frames, 0.0)
        text = backbone.generate(frames, fps, prompt, max_new_tokens=args.report_tokens)[0]
        events = parse_report(text, item.duration)
        rows.append(_report_row(item, events, {"raw": text if args.keep_raw else ""}))
        if i % 50 == 0:
            LOGGER.info("%s baseline: %d/%d videos (%.2f vid/s)", args.mode, i, len(names),
                        i / max(time.time() - t0, 1e-9))
    write_jsonl(Path(args.run_dir) / f"preds_{args.mode}_fold{args.fold}.jsonl", rows)


# --------------------------------------------------------------------------- #
# evaluate / compare
# --------------------------------------------------------------------------- #


def _load_preds(run_dir: Path, tag: str) -> Dict[str, Dict[str, Any]]:
    rows = {}
    files = sorted(Path(run_dir).glob(f"preds_{tag}_fold*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no preds_{tag}_fold*.jsonl under {run_dir}")
    for path in files:
        for row in read_jsonl(path):
            rows[row["video_name"]] = row
    LOGGER.info("%s: %d videos from %d fold files", tag, len(rows), len(files))
    return rows


def cmd_evaluate(args) -> None:
    from .common import normalise_intervals
    from .metrics import (Judge, aggregate, auroc, average_precision, bootstrap_ci, match_video,
                          recall_at_tiou)

    items = load_corpus(Path(args.work_dir), require_frames=False)
    preds = _load_preds(args.run_dir, args.tag)
    judge = Judge(args.judge_backend, args.judge_model,
                  Path(args.run_dir) / f"judge_cache_{args.judge_backend}_{args.judge_model.replace('/', '_')}.jsonl",
                  base_url=args.judge_base_url, api_key_env=args.api_key_env)
    scores, per_genre = [], defaultdict(list)
    unit_s, unit_y = [], []
    for name in sorted(preds):
        row, item = preds[name], items[name]
        pred = [{"description": d, "intervals": normalise_intervals(t)}
                for d, t in zip(row.get("bugs") or [], row.get("time_nodes") or [])]
        pred = [p for p in pred if p["description"] and p["intervals"]]
        gold = gold_events(item)
        score = match_video(pred, gold, judge, name, item.game)
        scores.append(score)
        per_genre[item.genre or "unknown"].append(score)
        if row.get("unit_p"):
            tg = unit_targets(np.array(row["unit_times"], dtype=np.float32), gold)
            unit_s.extend(row["unit_p"])
            unit_y.extend(tg["y"].tolist())
    micro = aggregate(scores)
    micro.update(recall_at_tiou(scores, (0.3, 0.5, 0.7)))
    ci = {m: bootstrap_ci(scores, m, args.n_boot, seed=args.seed, cluster="game")
          for m in ("f1", "miou", "f1_x_iou", "precision", "recall")}
    payload = {
        "tag": args.tag, "n_videos": len(scores), "judge": {"backend": args.judge_backend,
                                                           "model": args.judge_model,
                                                           "calls": judge.calls,
                                                           "parse_fallbacks": judge.fallbacks},
        "micro": micro, "ci_game_cluster": ci,
        "per_genre": {g: aggregate(v) for g, v in sorted(per_genre.items())},
        "unit_auroc": auroc(np.array(unit_s), np.array(unit_y)) if unit_s else None,
        "unit_ap": average_precision(np.array(unit_s), np.array(unit_y)) if unit_s else None,
    }
    judge.close()
    write_json(Path(args.run_dir) / f"eval_{args.tag}.json", payload)
    write_jsonl(Path(args.run_dir) / f"eval_{args.tag}.per_video.jsonl",
                [dataclasses.asdict(s) for s in scores])
    LOGGER.info("%s | P %.2f R %.2f F1 %.2f [%.2f, %.2f] | mIoU %.3f | F1xIoU %.2f [%.2f, %.2f] | n=%d | judge=%s",
                args.tag, micro["precision"], micro["recall"], micro["f1"], ci["f1"]["lo"], ci["f1"]["hi"],
                micro["miou"], micro["f1_x_iou"], ci["f1_x_iou"]["lo"], ci["f1_x_iou"]["hi"],
                len(scores), args.judge_model)
    if args.judge_backend == "offline":
        LOGGER.warning("offline judge (unigram F1): numbers are NOT comparable to GliDe Table 3")


def cmd_compare(args) -> None:
    from .metrics import VideoScore, holm, paired_test

    def load(tag):
        return [VideoScore(**r) for r in read_jsonl(Path(args.run_dir) / f"eval_{tag}.per_video.jsonl")]

    systems = {t: load(t) for t in [args.system] + args.baselines}
    raw_p, rows = {}, {}
    for base in args.baselines:
        for metric in ("f1", "miou", "f1_x_iou"):
            key = f"{args.system}_vs_{base}|{metric}"
            res = paired_test(systems[args.system], systems[base], metric, args.n_boot, args.seed)
            raw_p[key] = res["p"]
            rows[key] = res
    corrected = holm(raw_p, args.alpha)
    lines = ["| comparison | metric | delta | 95% CI | p | p (Holm) | reject |", "|---|---|---|---|---|---|---|"]
    for key in sorted(rows):
        comp, metric = key.split("|")
        r, c = rows[key], corrected[key]
        lines.append(f"| {comp} | {metric} | {r['delta']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] "
                     f"| {r['p']:.4f} | {c['p_holm']:.4f} | {'yes' if c['reject'] else 'no'} |")
    text = "\n".join(lines)
    (Path(args.run_dir) / f"compare_{args.system}.md").write_text(text + "\n", encoding="utf-8")
    write_json(Path(args.run_dir) / f"compare_{args.system}.json",
               {k: {**rows[k], **corrected[k]} for k in rows})
    print(text)
