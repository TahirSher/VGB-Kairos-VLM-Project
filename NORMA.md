# NORMA — Normality-Referenced Multimodal Anomaly grounding

A trained, backbone-agnostic VLM model for **open-ended, temporally grounded
game-glitch detection** on VideoGlitchBench, built to be compared with GliDe
(arXiv 2604.07818) under a protocol that makes the comparison valid.

Everything is in **one script, `norma_vgb.py`** (Python ≥ 3.10). It contains:
- preprocessing and the frame cache;
- the four systems (vanilla, direct-SFT, a GliDe reproduction, NORMA), all running on one shared backbone adapter;
- evaluation, statistics, cost accounting and self-tests.

It no longer depends on `vgb_stage12.py` or the KAIROS file.

---

## 0. Corrections to the premises

| Premise | What is actually true | Consequence |
|---|---|---|
| "beat **GLIDE+**" | The paper's method is **GliDe**. "GLIDE+" appears nowhere in it. | The targets are the GliDe rows of Table 3 (below). |
| GliDe is a model to out-train | GliDe is **training-free**: a prompting/agentic pipeline (scanner → planner/executor/reflector debate with SAM3 tracking → grounder → summarizer). | A trained model beating a training-free one is expected, not a contribution. The control that matters is **direct LoRA-SFT of the same backbone**. NORMA must beat that too. |
| VideoGlitchBench has a train/test split | It ships **one split** (5,238 videos, 120 games). GliDe reports on the whole set. | We use **game-disjoint 5-fold CV**: every video gets an out-of-fold prediction from a model that never saw its game, so the scored set equals GliDe's. |
| Paper numbers are directly comparable | The paper **does not name its LLM judge**. | Re-score the vanilla backbone with *your* judge and report the deltas. Absolute numbers from different judges are not comparable. |
| Ground truth has glitch categories | GT is free text plus intervals. The 8-class "fine taxonomy" in `vgb_stage12.py` is your own. | Category accuracy cannot be evaluated on VGB, so NORMA does not optimise it. |

**GliDe targets (Table 3, P / R / F1 / mIoU / F1×IoU):**

| Backbone | vanilla | +GliDe |
|---|---|---|
| Qwen2.5-VL-3B | 11.08 / 26.53 / 15.63 / 0.31 / 4.81 | 32.36 / 39.38 / 35.52 / 0.48 / 17.29 |
| Qwen2.5-VL-7B | 10.41 / 14.74 / 12.20 / 0.30 / 4.11 | 34.55 / 45.09 / 39.12 / 0.53 / 19.46 |
| InternVL2.5-4B | 8.62 / 22.93 / 12.53 / 0.18 / 1.92 | 29.15 / 36.87 / 32.56 / 0.52 / 15.27 |
| InternVL2.5-8B | 11.90 / 23.36 / 15.77 / 0.25 / 4.06 | 32.64 / 40.97 / 36.33 / 0.50 / 17.04 |
| UI-TARS-1.5-7B | 19.08 / 24.93 / 21.62 / 0.40 / 9.11 | 34.31 / 49.28 / 40.45 / 0.51 / 17.02 |

Dataset facts relevant to the design:
- Clips average 19.1 s (maximum 60 s), with 1.03 bugs per video (maximum 6).
- About 90% of videos contain a glitch. They are r/GamePhysics clips: the glitch is the reason the clip exists.
- Descriptions are GPT-4o drafts edited by humans, averaging 36 tokens.

---

## 1. Critical review of the existing code

Severity: **C** = invalidates a reported number, **M** = biases it, **m** = minor.

### `vgb_stage12.py`

| Sev | Where | Issue | Fix |
|---|---|---|---|
| **C** | `:1585` | The game context is built with `exclude_video=None`, from frames of *every* video of the game, including test videos and their glitch segments. The note's `valid_oddities` field can describe the very glitch under evaluation as normal. The notes are not split-aware, and Stage 3 reuses them. | Build notes from training-split videos only. NORMA replaces text notes with a statistical normality reference (§2). |
| **M** | `:728` | Timestamps are `pts * time_base` with no `stream.start_time` subtracted. Containers with a non-zero start shift every window against the annotation clock, which biases tIoU. | NORMA re-bases to the first frame (`norma_vgb.py`, `_timed_frames`). |
| **M** | `:1769-1770` | The unattributed fallback is labelled `state_logic` with coarse label `Other`, which contradicts `FINE_TO_COARSE` (`state_logic → Game Logic`). | — |
| **M** | `:127-136` vs Stage 3 `:40-49` | The two files disagree on `FINE_TO_COARSE` (clipping: Visual vs Physics; animation: Animation vs Visual; ui_hud: Game Logic vs Visual). The coarse sets differ too, and Stage 12 has an "Animation" class the paper lacks. | One mapping. Moot on VGB, which has no GT categories. |
| m | `:434` | `load_records` silently falls back to another split if the requested one fails. | Fail instead. |
| m | design | Windows are overlapping 8-frame *videos* (stride 4). GliDe uses **non-overlapping** 8-frame windows stitched into a 2×4 grid image. Your "GliDe" numbers are therefore not GliDe's input setting. | — |

The Stage 1/2 file also does several things well, and NORMA reuses them:
- **YES/NO likelihood read-out:** p_glitch is read from token logits instead of the model's verbal confidence.
- **Geometry unification:** frames of different aspect ratios are fitted to one shape before batching.
- **Leak assertion:** the scanner checks that window rows carry no ground-truth fields.
- **Frame cache:** decoded frames are cached under a fingerprint of the decode settings.

### `vgb_stage3_kairos(XNM-BOUNDARY).py`

| Sev | Where | Issue | Fix |
|---|---|---|---|
| **C** | `:1361`, `:1997` | `log_odds = z + logit(prior)` treats a z-score of the logit under negatives as a log-likelihood ratio. Under an equal-variance Gaussian model the LLR is `d·z − d²/2` with `d = (μ₁−μ₀)/σ₀`, so the formula is wrong unless d = 1 and the offset is ignored. Downstream, the "posterior" drives the cost-based gate and the 0.98 bypass, and the same quantity *plus the prior* is used as an HMM emission. The prior is then **counted twice**: once in the emission and once in the transitions. | Hybrid HMM: emission `logit p(s|x) − logit π` (Bourlard & Morgan 1994), with p from a calibrated discriminative model (`norma_vgb.py`, `decode_video`). |
| **C** | `:2250` vs `:2253` | The `kairos_no_xnm` ablation still applies XNM (z, posterior, conformal p) because `calibrate_windows` runs **before** `use_memory` is checked. Only notes and traps are removed, so the ablation table misattributes gains. | — |
| **C** | `:1819`, `:1523` | The in-house "glide" baseline repeats `refiner.verify` up to 5× with an *identical* prompt. Calls after the first are cache hits returning the same verdict, but each is still counted in `calls`. GliDe's iterations do nothing, and GliDe is charged for them. There is also no tool use, no SAM3 and no 3-role debate. **It is not GliDe.** | Compare against published GliDe numbers re-anchored by your own vanilla row, plus a direct-SFT control. |
| **C** | `:553` | The paired bootstrap p-value ORs two different rejection events (the shifted test *and* sign-crossing). That is not a valid test of any stated null. | Null-centred bootstrap (Hall & Wilson 1991), clustered by game (`norma_vgb.py`, `paired_test`). |
| **M** | `:1075`, `:1919` | The conformal calibration set is made of *overlapping* windows (stride = window/2) from the same videos, so exchangeability does not hold. BH runs per video at q = 0.25, and the fallback forces at least one candidate in **every** video, including clean ones. | — |
| **M** | `:1404` | "Traps": scanner evidence from train windows labelled negative under a 25%-coverage rule is injected as *"legitimate gameplay, do not re-flag"*. A window next to a real glitch, with coverage under 0.25, can describe that glitch. This teaches the refiner to suppress recurring real bugs of that game. | — |
| **M** | `:1458` | The calibrated posterior and p-value are shown to the LLM refiner, which anchors on the number (anchoring bias). | — |
| m | `:2586` | The report hard-codes "All hyperparameters were fitted on train and selected on dev" whether or not that happened. | — |
| m | stats | Bootstrap is over videos. Videos of one game are correlated (same engine, often the same bug), so the CIs are too narrow for a cross-game claim. | Game-cluster bootstrap. |

**Scientific verdict on KAIROS/XNM.** The pipeline is training-free post-processing of one frozen
scalar, `p_glitch`, per window. Standardisation, conformal p-values, BH and HMM smoothing can
**re-rank and threshold** that scalar, but they cannot add discriminative information the scalar
does not contain. Each component is also an established technique: score normalisation as in
speaker-verification Z-norm, conformal outlier tests (Bates et al. 2023), BH, and HMM smoothing.
The combination is engineering, not a strong novelty, and its ceiling is the zero-shot window
AUROC of the backbone.

---

## 2. NORMA

### 2.1 Research gap it targets

GliDe names two difficulties:
1. **A glitch versus an unusual but legitimate event.** Normality is game-specific.
2. **Brief, extended or recurring glitches**, which call for multi-interval grounding.

GliDe answers both with text, an LLM-written game memory and greedy textual window linking, and pays for it in many generated tokens. NORMA answers both **in representation space**, with a learned model and one prefill per chunk.

### 2.2 Architecture

```
frames ──► frozen VLM (prefill, early exit at layer L*) ──► per-unit hidden-state stats
                                                           │   (mean+max of RMS-normed visual
                                                           │    tokens, 2 mid-depth layers, JL→P)
                         ┌─────────────────────────────────┤
  self-referential:  r_t = (u_t − median_t u)/(1.4826·MAD)  e_t = ‖r_t‖/√d
  temporal surprise: g_t = ‖u_t − u_{t−1}‖/√d
  extra-video:       n_t = log(1 + kNN dist to coreset of NORMAL train units)  (leave-game-out)
                         └──► temporal transformer ──► actionness a_t, boundaries (s_t, e_t),
                                                       instance embedding v_t, event count K
  decode: hybrid-HMM Viterbi on llr_t = logit p_t − logit π  → runs → boundary refinement
          → average-linkage grouping by v → multi-interval events
  describe: LoRA-SFT VLM on the event's frames → P(NO_GLITCH) + MBR description
  emit: stacked logistic calibrator on [logit seg_score, logit P(any), logit (1 − P(reject))],
        threshold chosen on dev
```

### 2.3 Why each piece is principled

| Component | Justification |
|---|---|
| Hidden states, not generated text | Intermediate representations linearly encode more than decoded answers (linear-probe evidence). Reading them costs one prefill and no decoding. |
| Task text in the **system** turn | Decoder-only VLMs use causal attention. Qwen's template places the video before the user text, so visual tokens cannot attend to a user prompt. Only a system turn conditions them. |
| Early exit after the deepest tapped layer | Layers deeper than L* do not affect the stored features, so computing them is wasted. Verified against real Qwen2.5-VL modules in `selftest`. |
| RMS-norm before pooling | Massive-activation channels (Sun et al. 2024) would otherwise dominate max-pooling. |
| Johnson–Lindenstrauss projection D→P | Preserves pairwise distances, which is what the kNN bank relies on, while shrinking the cache by D/P. |
| Median/MAD reference (SRN) | 50% breakdown point (Hampel 1974; Rousseeuw & Croux 1993). The video's own frames estimate "normal for this game" on an unseen title without any text memory. This is the statistical counterpart of GliDe's game-aware memory. |
| kNN coreset bank (PatchCore; Roth et al. 2022) | Covers clips where the glitch is the majority of frames, where the median breaks down. Training-time distances exclude the video's own game, so the train-time distribution of n_t matches the unseen-game test setting. |
| Focal loss (Lin et al. 2017), soft boundary targets (BMN) | Handle class imbalance and sparse boundaries. |
| Discriminative instance embedding (De Brabandere et al. 2017) | Groups non-contiguous segments of one recurring bug into one event with several intervals, which addresses GliDe challenge 2. |
| Hybrid HMM decoding | The prior enters once, through the transitions. Transitions are fitted on non-overlapping units, so window overlap does not inflate them. |
| Temperature scaling (Guo et al. 2017), stacked LR calibrator | Calibrated scores are needed because decisions are thresholded. |
| Describer veto treated as evidence | P(NO_GLITCH) is a fusion feature, not a hard drop. When the describer refuses, the best description conditional on a glitch is generated (NO_GLITCH banned), and the calibrated filter decides whether to emit it. |
| MBR decoding (Kumar & Byrne 2004; Freitag et al. 2022) | Approximates the output with the best expected judge score rather than the most probable string. |
| Jittered training crops | Exposure-bias mitigation: the describer is trained on the imperfect boundaries the localiser will hand it. |

### 2.4 What is novel, and what must be proven

Claimed contributions, to be supported by ablations:
1. **The first trained model for open-ended, temporally grounded glitch detection on VGB**, with a game-disjoint protocol.
2. **Self-referential normality probing of a frozen VLM.** Glitch-ness is modelled as deviation from the video's own robust reference plus a leave-game-out normal memory. This replaces token-expensive textual game memory.
3. **Localise-then-describe with a learned, calibrated veto.** It replaces multi-agent debate at a fixed, small number of model calls.
4. **Multi-interval event grouping** via instance embeddings.

Required ablations, all available as flags:
- `--no-srn`, `--no-bank`, `--no-delta`, `--no-ctx` and `--no-group` on `train-head`.
- `--no-describer`, `--no-fusion` and `--mbr-samples 0` on `predict`.
- `--layer-fracs` for which layers are probed.

Contributions 2 to 4 are claims only if their ablations hold up under the paired, game-clustered test.

---

## 3. Fair comparison with GliDe: every aspect, enforced in code

| Aspect | What is held identical | Where it is enforced |
|---|---|---|
| Videos | All 5,238. Trained systems use out-of-fold predictions from 5 game-disjoint folds, so the union of test folds is GliDe's evaluation set. Training-free systems (vanilla, glide) are run fold by fold on the same test lists. | `make_folds`, `assert_disjoint`; `compare` checks a hash of the evaluated video list |
| Frames | One PyAV decoder and one 4 fps grid (GliDe's rate), with t = 0 at the first displayed frame, shared by every system | `decode_uniform_frames`, `subsample` |
| Pixels | One per-frame budget (`--max-pixels`). GliDe's 2×4 grid image gets 8× that budget, so each of its frames is seen at the same resolution. | `QwenFamily.generate_image`; budget recorded in the fairness fingerprint |
| Backbone | Same checkpoint, `--revision` and dtype for every system | `fairness_fingerprint`; `compare` refuses mismatches |
| Input format | GliDe: non-overlapping 8-frame 2×4 grids with frame indices. vanilla/direct: the whole video with the same frame cap. NORMA: 8 s chunks. | `GlideRunner.windows`, `make_grid`, `sample_io` |
| GliDe algorithm | Scanner fields and recall bias, game memory, Planner/Executor/Reflector with vqa / zoom_in / object_tracking, Advocate/Skeptic/Judge, ≤ 5 rounds, stop at confidence ≥ 0.80, first-match grounding with extension, summariser, T = 0.5, 512/1024 tokens | `GlideRunner` (see the deviations below) |
| Output | One report schema for all systems, with the same clipping and merging | `REPORT_SCHEMA`, `parse_report`, `report_row` |
| Judge | One named judge, prompt, normalisation (S/5) and cache | `Judge`; `compare` refuses different judges |
| Randomness | Stochastic systems (glide at T = 0.5, NORMA's MBR sampling) are run with several seeds. Per-video scores are averaged over runs before testing. | `cmd_compare` (`name=tag1,tag2,…`), `_mean_scores` |
| Training data | GliDe trains on nothing. `direct` gets exactly NORMA's training videos and LoRA recipe, which isolates the architecture from the act of fine-tuning. | `train_lm` (shared by describer and direct) |
| Cost | Every model call goes through one `CostMeter`: calls, input/output tokens, seconds and peak GPU memory per video for every system. NORMA's feature extraction is amortised into its per-video cost, and training GPU-hours are reported. | `Backbone.meter`, `cmd_cost` |
| Statistics | 95% CIs from a game-clustered bootstrap. Paired, null-centred cluster-bootstrap tests, Holm-corrected. | `bootstrap_ci`, `paired_test`, `holm` |

**GliDe reproduction: the honest limits.**
- **Prompts.** The paper's verbatim prompts are not available, so the wording is reconstructed from its description.
- **Tracking.** SAM3 is replaced by global normalised-cross-correlation template tracking, which feeds the paper's four physics rules. The velocity threshold is rescaled from 500 px/s to 0.34 frame-diagonals/s.
- **Choices in GliDe's favour,** where the paper is silent: the Judge sees the window image, and boundaries use the Scanner's `frame_range`.
- **Reporting.** Report your reproduction's numbers next to the paper's Table 3. If they fall far below the paper's, say so rather than claiming a win.

## 4. SonarCloud findings on PR #1 (39) and their resolution in `norma_vgb.py`

| # | Rule (file:line in PR #1) | Resolution |
|---|---|---|
| 1 | Weak hashing, SHA-1 (`common.py:62`) | `stable_hash` uses SHA-256, for cache keys only |
| 2–4 | Path traversal from CLI arguments in `read_jsonl`/`write_jsonl` (`common.py:72,86,88`) | Every file access goes through `safe_path`: `realpath` plus `commonpath` containment in the working dir or `NORMA_DATA_ROOTS`. A path outside these roots raises `PermissionError` (self-tested). |
| 5 | Unsafe call: pickle `torch.load(weights_only=False)` (`pipeline.py:194`) | The checkpoint is a tensor-only `state_dict` loaded with `weights_only=True`, plus JSON and a pickle-free `.npz` (`allow_pickle=False`). Adapters are PEFT safetensors. |
| 6–11 | PRNG hotspots, `random` (`describer.py`, `splits.py`) | The `random` module is gone. All randomness uses a seeded `np.random.Generator(PCG64)` via `make_rng`. **This is a cryptography rule.** Seeded generators are *required* here for reproducible folds, crops and bootstrap draws. If Sonar still lists them as hotspots, mark them *Safe* with that justification. |
| 12 | `torch.unique` without `dim` (`head.py:375`) | Removed; ids come from a Python set |
| 13–14 | `x != x` NaN check, identical operands and an always-false condition (`pipeline.py:154`) | `math.isnan` |
| 15–16 | Float equality (`selftest.py:52,66`) | `math.isclose`; AUROC tie handling now uses `scipy.stats.rankdata` |
| 17 | Duplicated `"<|im_end|>"` literal | `QWEN_END`; also `UTF8`, `VIDEOS_FILE`, `TRAIN_SUMMARY`, `MOCK_TELEPORT` |
| 18 | Empty method (`backbones.py:644`) | Documented no-op plus a log line; base-class methods raise `NotImplementedError` |
| 19–26 | Cognitive complexity > 15 in 9 functions | Every function is now ≤ 15 (checked with `complexipy`, which implements Sonar's definition) |
| 27 | `decode_video` argument type mismatch | Typed `Transitions` and `DecodeParams` dataclasses; `mypy --check-untyped-defs` passes with no issues |
| 28 | `train_lm` has 14 parameters | Takes an `LMSpec` dataclass; no function has more than 13 parameters |
| 29 | Unused parameter `trans` (`head.py:600`) | Removed |
| 30–39 | `set()`/`dict()` constructors, nested `with`, unused `step` | Literals and comprehensions, a combined `with`, the unused variable removed (ruff C4/SIM/F are clean) |

Also fixed while consolidating, beyond Sonar's list:
- **Decoder timestamps:** frames are re-based to t = 0 at the first displayed frame.
- **No silent split fallback:** the requested dataset split is used verbatim.
- **No test-pixel leakage:** NORMA's normality reference is statistical (the video's own median plus a bank built from training games only); there are no per-game text notes built from test videos.
- **Name collision:** a CLI helper had shadowed a command helper; the CLI builders are now `_add_*`.
- **NCC tracker:** the search is global, and the template is padded with context.

## 5. Running

```bash
pip install -r requirements-norma.txt
export NORMA_DATA_ROOTS=/path/to/VideoGlitchBench          # videos may live outside the working dir
python norma_vgb.py selftest                                 # CPU, about a minute

M=Qwen/Qwen2.5-VL-7B-Instruct; REV=<commit>; W=vgb_work; R=runs/qwen7b; F=feats/qwen7b
COMMON="--work-dir $W --run-dir $R --feat-dir $F --model $M --revision $REV"
python norma_vgb.py preprocess --work-dir $W --video-root /path/to/VideoGlitchBench/raw
python norma_vgb.py split   --work-dir $W --run-dir $R --k 5
python norma_vgb.py extract $COMMON
for f in 0 1 2 3 4; do
  python norma_vgb.py train-head --work-dir $W --run-dir $R --feat-dir $F --fold $f
  python norma_vgb.py train-desc $COMMON --fold $f
  python norma_vgb.py train-desc $COMMON --fold $f --direct
  for s in 0 1 2; do python norma_vgb.py predict  $COMMON --fold $f --tag norma_s$s --seed $s; done
  for s in 0 1 2; do python norma_vgb.py baseline $COMMON --fold $f --mode glide --tag glide_s$s --seed $s; done
  python norma_vgb.py baseline $COMMON --fold $f --mode direct
  python norma_vgb.py baseline $COMMON --fold $f --mode vanilla
done
python norma_vgb.py evaluate --work-dir $W --run-dir $R \
    --tags norma_s0 norma_s1 norma_s2 glide_s0 glide_s1 glide_s2 direct vanilla \
    --judge-backend openai --judge-model Qwen/Qwen2.5-72B-Instruct --judge-base-url http://localhost:8000/v1
SYS="norma=norma_s0,norma_s1,norma_s2 glide=glide_s0,glide_s1,glide_s2 direct=direct vanilla=vanilla"
python norma_vgb.py compare --run-dir $R --systems $SYS
python norma_vgb.py cost    --run-dir $R --systems $SYS
```

Other backbones:
- `Qwen/Qwen2.5-VL-3B-Instruct` and `ByteDance-Seed/UI-TARS-1.5-7B` use the Qwen path as above.
- `OpenGVLab/InternVL2_5-4B` and `OpenGVLab/InternVL2_5-8B` additionally need `--trust-remote-code` and a pinned `--revision`, since these checkpoints execute their own code.

Ablations:
- `train-head` flags: `--no-srn`, `--no-bank`, `--no-delta`, `--no-ctx`, `--no-group`, each with `--tag-suffix`.
- `predict` flags: `--no-describer`, `--no-fusion`, `--mbr-samples 0`.

## 6. Honest status

**Verified here** with `python norma_vgb.py selftest`, 24 checks:
- **Estimators:** interval algebra, Hungarian matching, Viterbi, IRLS logistic regression, AUROC with ties, Holm correction.
- **Paired-test size:** at most 6/40 false rejections at α = 0.05.
- **Path guard:** a path outside the allowed roots is rejected.
- **Decoder:** a real MP4 decodes to a 4 fps grid starting at t = 0.
- **NCC tracker and physics rules:** a teleport is detected, and smooth motion stays quiet.
- **Real Qwen2.5-VL modules** (tiny random config): layer taps, early exit, unit pooling, LoRA placed on the language model only.
- **Fairness gate:** a mismatched comparison is refused.
- **CPU end-to-end run** of all four systems through the mock backbone.

Static checks: `mypy --check-untyped-defs` passes, ruff (including the bandit rule set) is clean, and no function exceeds cognitive complexity 15.

**Not verified here** (no GPU and no Hugging Face access):
- Loading the real checkpoints; InternVL's remote-code path.
- Wall-clock cost.
- Any VGB number.

The metric values printed by the self-test come from a rule-based mock, and they say nothing about any system. **Whether NORMA beats GliDe is decided only by running §5 and applying the claim criteria below.**

To claim "NORMA beats GliDe" for a backbone, all of the following must hold:
- (a) NORMA beats `direct`, significantly after Holm correction.
- (b) NORMA beats the in-harness `glide`, significantly.
- (c) NORMA's gain over `vanilla` exceeds GliDe's gain over vanilla as reported in the paper.
