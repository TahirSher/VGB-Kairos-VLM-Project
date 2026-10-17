"""
NORMA -- Normality-Referenced Multimodal Anomaly grounding.

Pipeline (run `vgb_stage12.py preprocess` first; it builds the 4 fps frame
cache NORMA reads):

  split        game-disjoint, genre-stratified K folds (+ dev games per fold)
  extract      one early-exit prefill per chunk -> per-unit hidden-state cache
  train-head   SRN normality head per fold (CPU-cheap; features are cached)
  train-desc   LoRA-SFT describer per fold (--direct: direct-SFT baseline)
  predict      dev -> fused calibrator -> test reports (GliDe schema)
  baseline     vanilla zero-shot / direct-SFT reports
  evaluate     GliDe protocol + game-cluster bootstrap CIs
  compare      paired cluster-bootstrap tests, Holm-corrected
  prepare-generic   bring any temporally annotated anomaly dataset into the layout
  selftest     unit checks + real Qwen2.5-VL module check + CPU end-to-end run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .common import set_seed, setup_logging


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--work-dir", default="./vgb_work", help="Stage-1 output dir (videos.jsonl, frames/)")
    p.add_argument("--run-dir", default="./norma_run")
    p.add_argument("--feat-dir", default="./norma_feats/qwen7b")
    p.add_argument("--stage12", default=None, help="path to vgb_stage12.py")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("-v", "--verbose", action="count", default=1)


def _model(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="Qwen/Qwen2.5-VL-{3B,7B}-Instruct | ByteDance-Seed/UI-TARS-1.5-7B | "
                        "OpenGVLab/InternVL2_5-{4B,8B} | mock")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--max-pixels", type=int, default=160 * 28 * 28)
    p.add_argument("--min-pixels", type=int, default=32 * 28 * 28)
    p.add_argument("--fit", default="letterbox", choices=["letterbox", "stretch"])
    p.add_argument("--layer-fracs", type=float, nargs="+", default=[0.5, 0.75],
                   help="decoder depths to tap, as fractions of the layer count")


def _fold(p: argparse.ArgumentParser) -> None:
    p.add_argument("--fold", type=int, required=True)


def _desc(p: argparse.ArgumentParser) -> None:
    p.add_argument("--desc-frames", type=int, default=16)
    p.add_argument("--direct-frames", type=int, default=48)
    p.add_argument("--margin", type=float, default=1.0, help="context seconds around an event")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("split")
    _common(p)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--dev-frac", type=float, default=0.12)

    p = sub.add_parser("extract")
    _common(p)
    _model(p)
    p.add_argument("--fps", type=float, default=None, help="default: 4 (Qwen family), 2 (InternVL)")
    p.add_argument("--chunk-frames", type=int, default=32)
    p.add_argument("--proj-dim", type=int, default=1024)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("train-head")
    _common(p)
    _fold(p)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--bank-pca", type=int, default=128)
    p.add_argument("--bank-size", type=int, default=4096)
    p.add_argument("--bank-k", type=int, default=3)
    for flag in ("srn", "bank", "delta", "ctx", "group"):
        p.add_argument(f"--no-{flag}", action="store_true", help=f"ablation: disable {flag}")
    p.add_argument("--tag-suffix", default="", help="e.g. _nosrn, to keep ablation heads side by side")

    p = sub.add_parser("train-desc")
    _common(p)
    _model(p)
    _fold(p)
    _desc(p)
    p.add_argument("--direct", action="store_true", help="train the direct-SFT baseline instead")
    p.add_argument("--neg-ratio", type=float, default=0.33)
    p.add_argument("--desc-epochs", type=int, default=2)
    p.add_argument("--desc-lr", type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)

    p = sub.add_parser("predict")
    _common(p)
    _model(p)
    _fold(p)
    _desc(p)
    p.add_argument("--tag", default="norma")
    p.add_argument("--tag-suffix", default="")
    p.add_argument("--no-describer", action="store_true", help="ablation: localisation only")
    p.add_argument("--no-fusion", action="store_true", help="ablation: no stacked calibrator")
    p.add_argument("--mbr-samples", type=int, default=4, help="0 = greedy only")
    p.add_argument("--mbr-temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("baseline")
    _common(p)
    _model(p)
    _fold(p)
    _desc(p)
    p.add_argument("--mode", choices=["vanilla", "direct"], required=True)
    p.add_argument("--report-tokens", type=int, default=384)
    p.add_argument("--keep-raw", action="store_true")
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("evaluate")
    _common(p)
    p.add_argument("--tag", required=True)
    p.add_argument("--judge-backend", default="openai", choices=["openai", "hf", "offline"])
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--judge-base-url", default=None, help="OpenAI-compatible endpoint, e.g. a vLLM server")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--n-boot", type=int, default=2000)

    p = sub.add_parser("compare")
    _common(p)
    p.add_argument("--system", default="norma")
    p.add_argument("--baselines", nargs="+", default=["vanilla", "direct"])
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--alpha", type=float, default=0.05)

    p = sub.add_parser("prepare-generic")
    _common(p)
    p.add_argument("--manifest", required=True)
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--short-side", type=int, default=360)
    p.add_argument("--jpeg-quality", type=int, default=92)

    p = sub.add_parser("selftest")
    _common(p)
    p.add_argument("--keep", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    set_seed(args.seed)
    from . import pipeline

    if args.command == "selftest":
        from .selftest import run

        return run(args.keep)
    if args.command == "prepare-generic":
        from .data import prepare_generic

        prepare_generic(Path(args.manifest), Path(args.work_dir), args.fps, args.short_side,
                        args.jpeg_quality, stage12_path=args.stage12)
        return 0
    if args.command == "train-desc":
        args.layer_fracs = tuple(args.layer_fracs)
    handler = {
        "split": pipeline.cmd_split, "extract": pipeline.cmd_extract,
        "train-head": pipeline.cmd_train_head, "train-desc": pipeline.cmd_train_desc,
        "predict": pipeline.cmd_predict, "baseline": pipeline.cmd_baseline,
        "evaluate": pipeline.cmd_evaluate, "compare": pipeline.cmd_compare,
    }[args.command]
    handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
