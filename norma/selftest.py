"""
Self-tests: estimator unit checks, a real Qwen2.5-VL module check (tiny
random config, no download), and a full end-to-end CPU run on a synthetic
corpus written in the exact Stage-1 layout.

The synthetic corpus encodes the two properties NORMA's inductive bias is
designed for: (i) every "game" has its own visual offset (a per-game
nuisance that a global threshold cannot absorb), (ii) a glitch is a short
departure from that game's own normal dynamics (a teleporting sprite), with
recurring two-interval bugs and clean videos mixed in.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import List

import numpy as np

from .common import LOGGER, interval_iou, merge_intervals, write_jsonl

FAIL: List[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        LOGGER.info("PASS  %s", name)
    else:
        FAIL.append(f"{name}: {detail}")
        LOGGER.error("FAIL  %s %s", name, detail)


def unit_checks() -> None:
    from .describer import fit_logreg, mbr_select
    from .head import fit_transitions, viterbi
    from .metrics import aggregate, auroc, holm, match_video, paired_test, bootstrap_ci

    check("iou_multi", abs(interval_iou([(0, 2), (4, 6)], [(1, 5)]) - 2 / 6) < 1e-12)
    check("merge", merge_intervals([(0, 1), (1, 2), (5, 6)]) == [(0, 2), (5, 6)])
    same = lambda a, b: 1.0 if a == b else 0.0  # noqa: E731
    s = match_video([{"description": "x", "intervals": [(0, 10)]}, {"description": "y", "intervals": [(20, 30)]}],
                    [{"description": "y", "intervals": [(20, 30)]}, {"description": "x", "intervals": [(0, 10)]}],
                    same, "v", "g")
    m = aggregate([s])
    check("hungarian", abs(m["f1"] - 100) < 1e-9 and abs(m["f1_x_iou"] - 100) < 1e-9, json.dumps(m))
    fp = aggregate([match_video([{"description": "x", "intervals": [(0, 1)]}], [], same)])
    check("fp_on_clean_video", fp["precision"] == 0.0 and fp["n_pred"] == 1)

    trans = {"p11": 0.85, "p01": 0.05, "pi1": 0.1, "prior": 0.2}
    path = viterbi(np.array([-3, -3, 4, 4, -3, -3, 8, -3.0]), trans)
    check("viterbi", path.tolist() == [0, 0, 1, 1, 0, 0, 1, 0], str(path))
    tr = fit_transitions([np.array([0, 0, 1, 1, 1, 0])])
    check("transitions", 0 < tr["p01"] < tr["p11"] < 1, str(tr))

    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 2))
    y = (rng.random(2000) < 1 / (1 + np.exp(-(1.5 * x[:, 0] - 0.5)))).astype(float)
    w = fit_logreg(x, y, l2=1e-4)
    check("logreg_recovers_weights", abs(w[0] - 1.5) < 0.25 and abs(w[1]) < 0.2 and abs(w[2] + 0.5) < 0.2,
          str(w))
    check("auroc_perfect", auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0)
    check("auroc_ties", abs(auroc(np.array([0.5, 0.5]), np.array([0, 1])) - 0.5) < 1e-12)
    h = holm({"a": 0.001, "b": 0.02, "c": 0.9})
    check("holm", h["a"]["reject"] and not h["c"]["reject"] and h["a"]["p_holm"] <= h["b"]["p_holm"])
    check("mbr_consensus", mbr_select(["red car floats", "red car floats upward", "blue tree"]).startswith("red"))

    # Paired test calibration: two systems with identical score
    # distributions must not be "significant" more than ~alpha of the time.
    from .metrics import VideoScore

    rejections = 0
    for trial in range(40):
        r = np.random.default_rng(100 + trial)
        mk = lambda: [VideoScore(f"v{i}", f"g{i % 20}", 1, 1, 1, float(r.random()), float(r.random() * 0.5),  # noqa: E731
                                 0.5, [0.5], []) for i in range(200)]
        rejections += paired_test(mk(), mk(), "f1", n_boot=400, seed=trial)["p"] < 0.05
    check("paired_test_size", rejections <= 6, f"{rejections}/40 false rejections at alpha=0.05")
    ci = bootstrap_ci([VideoScore(f"v{i}", f"g{i % 10}", 1, 1, 1, 0.5, 0.25, 0.5, [0.5], [])
                       for i in range(50)], "f1", n_boot=200)
    check("cluster_ci", ci["lo"] <= ci["point"] <= ci["hi"] and ci["n_clusters"] == 10, json.dumps(ci))


def qwen_module_check() -> None:
    """Runs LayerTap + early exit + qwen_unit_pool against a REAL
    Qwen2.5-VL implementation (random tiny weights)."""
    try:
        import torch
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    except ImportError:
        LOGGER.warning("SKIP  qwen_module_check (transformers not installed)")
        return
    from .backbones import LayerTap, find_decoder_layers, qwen_unit_pool

    cfg = Qwen2_5_VLConfig(
        text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=152000,
                         rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]}),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
                           fullatt_block_indexes=[1], window_size=56),
    )
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(cfg).eval()
    t, h, w = 3, 4, 6                                   # 3 temporal groups, 4x6 patches -> 2x3 merged
    per = (h // 2) * (w // 2)
    vid = cfg.video_token_id
    ids = [151644, 872, 198, cfg.vision_start_token_id] + [vid] * (t * per) + [cfg.vision_end_token_id, 100, 101]
    input_ids = torch.tensor([ids])
    pixel = torch.randn(t * h * w, 3 * 2 * 14 * 14)
    grid = torch.tensor([[t, h, w]])
    layers = find_decoder_layers(model)
    calls = {"n": 0}
    deep = layers[-1].register_forward_hook(lambda *a: calls.__setitem__("n", calls["n"] + 1))
    with torch.inference_mode():
        with LayerTap(layers, [2, 3]) as tap:
            model(input_ids=input_ids, pixel_values_videos=pixel, video_grid_thw=grid, use_cache=False)
        full = model(input_ids=input_ids, pixel_values_videos=pixel, video_grid_thw=grid,
                     use_cache=False, output_hidden_states=True)
    deep.remove()
    check("qwen_early_exit_skips_deep_layers", calls["n"] == 1, f"deepest layer ran {calls['n']} times (1 = full pass only)")
    ref = full.hidden_states[4][0]                      # hidden_states[i+1] == output of layer i
    check("qwen_tap_matches_hidden_states", torch.allclose(tap.cache[3][0], ref, atol=1e-5),
          f"max diff {float((tap.cache[3][0] - ref).abs().max()):.2e}")
    pooled = qwen_unit_pool(tap.cache[3][0], input_ids[0], grid[0].tolist(), vid, 2)
    check("qwen_unit_pool_shape", tuple(pooled.shape) == (t, 2, 64), str(tuple(pooled.shape)))
    # Unit k must be pooled from exactly the k-th block of t-major video tokens.
    mask = input_ids[0] == vid
    blk = tap.cache[3][0][mask].view(t, per, -1)[1].float()
    blk = blk / blk.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    check("qwen_unit_pool_order", torch.allclose(pooled[1, 0], blk.mean(0), atol=1e-5))


def synthetic_corpus(root: Path, n_games: int = 6, per_game: int = 8, seconds: float = 10.0,
                     fps: float = 4.0, seed: int = 0) -> Path:
    from PIL import Image

    rng = np.random.default_rng(seed)
    work = root / "work"
    rows = []
    for g in range(n_games):
        game = f"Game {g}"
        bg = rng.uniform(0.1, 0.9, size=3)
        for v in range(per_game):
            name = f"g{g}_v{v}.mp4"
            stem = Path(name).stem
            fdir = work / "frames" / game.replace(" ", "_") / stem
            fdir.mkdir(parents=True, exist_ok=True)
            n = int(seconds * fps)
            kind = v % 4                                # 0 clean, 1 single, 2 recurring, 3 two bugs
            spans: List[List[List[float]]] = []
            bugs: List[str] = []
            if kind in (1, 3):
                a = float(rng.uniform(1.0, 5.0))
                spans.append([[a, a + float(rng.uniform(1.5, 3.0))]])
                bugs.append("The red square teleports erratically across the screen.")
            if kind == 2:
                a = float(rng.uniform(0.5, 2.5))
                b = float(rng.uniform(6.0, 7.5))
                spans.append([[a, a + 1.5], [b, b + 1.5]])
                bugs.append("The red square repeatedly teleports across the screen.")
            if kind == 3:
                c = float(rng.uniform(7.0, 8.0))
                spans.append([[c, c + 1.5]])
                bugs.append("The whole scene flashes bright white.")
            paths, stamps = [], []
            x, y = 10.0, 30.0
            for i in range(n):
                t = i / fps
                img = np.ones((64, 64, 3)) * bg
                img += rng.normal(0, 0.02, size=img.shape)
                teleport = any(lo <= t < hi for k, s in enumerate(spans) for lo, hi in s
                               if "teleport" in bugs[k])
                flash = any(lo <= t < hi for k, s in enumerate(spans) for lo, hi in s
                            if "flashes" in bugs[k])
                x = (x + 1.5) % 50
                px, py = (rng.uniform(0, 50), rng.uniform(0, 50)) if teleport else (x, y)
                img[int(py):int(py) + 12, int(px):int(px) + 12] = [0.9, 0.1, 0.1]
                if flash:
                    img = np.clip(img + 0.6, 0, 1)
                path = fdir / f"{i:06d}.jpg"
                Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path, quality=92)
                paths.append(str(path))
                stamps.append(t)
            (fdir / "_frames.json").write_text(json.dumps(
                {"frame_paths": paths, "timestamps": stamps, "duration": seconds, "target_fps": fps}))
            rows.append({"video_name": name, "game_name": game, "genre": "Action" if g % 2 else "Simulation",
                         "subgenre": "Synthetic", "duration": seconds, "no_bugs": kind == 0,
                         "bugs": bugs, "bug_spans": spans,
                         "gt_intervals": [iv for s in spans for iv in s]})
    write_jsonl(work / "videos.jsonl", rows)
    return work


def end_to_end(tmp: Path) -> None:
    from . import pipeline
    from .common import read_json, read_jsonl

    work = synthetic_corpus(tmp)
    run = tmp / "run"
    feat = tmp / "feats"
    base = dict(work_dir=str(work), run_dir=str(run), feat_dir=str(feat), seed=0, model="mock",
                dtype="bfloat16", load_in_4bit=False, attn="sdpa", max_pixels=0, min_pixels=0,
                fit="letterbox", layer_fracs=(0.5, 0.75), stage12=None)
    ns = lambda **kw: argparse.Namespace(**{**base, **kw})  # noqa: E731
    pipeline.cmd_split(ns(k=3, dev_frac=0.25))
    pipeline.cmd_extract(ns(fps=4.0, chunk_frames=16, proj_dim=64, overwrite=False, limit=None))
    head_kw = dict(bank_pca=16, bank_size=256, bank_k=3, d_model=64, n_layers=2, n_heads=4,
                   epochs=25, lr=2e-3, batch_size=4, no_srn=False, no_bank=False, no_delta=False,
                   no_ctx=False, no_group=False, tag_suffix="")
    pred_kw = dict(no_describer=False, no_fusion=False, tag="norma", tag_suffix="", limit=None,
                   desc_frames=8, margin=0.5, mbr_samples=2, mbr_temperature=0.7, max_new_tokens=32)
    for fold in range(3):
        pipeline.cmd_train_head(ns(fold=fold, **head_kw))
        pipeline.cmd_train_desc(ns(fold=fold, direct=False, neg_ratio=0.5, desc_frames=8, direct_frames=16,
                                   margin=0.5, desc_epochs=1, desc_lr=1e-3, grad_accum=2, lora_r=4,
                                   lora_alpha=8))
        pipeline.cmd_predict(ns(fold=fold, **pred_kw))
    # Trivial baseline: one event spanning the whole clip, same description.
    rows = []
    for fold in range(3):
        for r in read_jsonl(run / f"preds_norma_fold{fold}.jsonl"):
            rows.append({**r, "bugs": ["The red square teleports erratically across the screen."],
                         "time_nodes": [[[0.0, r["duration"]]]], "events": [], "unit_p": []})
    write_jsonl(run / "preds_whole_fold0.jsonl", rows)
    ev_kw = dict(judge_backend="offline", judge_model="unigram-f1", judge_base_url=None,
                 api_key_env="OPENAI_API_KEY", n_boot=200)
    pipeline.cmd_evaluate(ns(tag="norma", **ev_kw))
    pipeline.cmd_evaluate(ns(tag="whole", **ev_kw))
    pipeline.cmd_compare(ns(system="norma", baselines=["whole"], n_boot=500, alpha=0.05))

    norma = read_json(run / "eval_norma.json")
    whole = read_json(run / "eval_whole.json")
    check("e2e_unit_auroc", (norma["unit_auroc"] or 0) > 0.85, f"unit AUROC {norma['unit_auroc']}")
    check("e2e_beats_whole_clip_baseline", norma["micro"]["f1_x_iou"] > whole["micro"]["f1_x_iou"] + 5,
          f"{norma['micro']['f1_x_iou']:.2f} vs {whole['micro']['f1_x_iou']:.2f}")
    check("e2e_miou", norma["micro"]["miou"] > 0.45, f"mIoU {norma['micro']['miou']:.3f}")
    multi = 0
    for fold in range(3):
        multi += sum(any(len(t) > 1 for t in r["time_nodes"]) for r in read_jsonl(run / f"preds_norma_fold{fold}.jsonl"))
    LOGGER.info("e2e: %d videos predicted with a multi-interval event", multi)
    LOGGER.info("e2e NORMA: %s", {k: round(v, 3) for k, v in norma["micro"].items()})


def run(keep: bool = False) -> int:
    unit_checks()
    qwen_module_check()
    tmp = Path(tempfile.mkdtemp(prefix="norma_selftest_"))
    try:
        end_to_end(tmp)
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            LOGGER.info("self-test artefacts kept in %s", tmp)
    if FAIL:
        LOGGER.error("%d self-tests FAILED:\n  %s", len(FAIL), "\n  ".join(FAIL))
        return 1
    LOGGER.info("all self-tests passed")
    return 0
