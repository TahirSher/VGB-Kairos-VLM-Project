"""
Backbone adapters: one interface over Qwen2.5-VL-3B/7B, UI-TARS-1.5-7B
(a Qwen2.5-VL derivative), InternVL2.5-4B/8B, and a CPU mock for self-tests.

Every adapter exposes four primitives:

  extract_units(frames, times, fps, game_text) -> per-unit hidden-state statistics
      ONE forward pass, NO generation, early exit after the deepest tapped
      decoder layer (the remaining layers are never computed).
  lm_loss(frames, fps, prompt, answer)          -> teacher-forced CE on answer tokens
  generate(frames, fps, prompt, n, temperature) -> list of strings
  first_token_prob(frames, fps, prompt, word)   -> P(first answer token == word[0])

Why hidden states and not generated text: a frozen VLM's intermediate
representations linearly encode far more than its decoded answers expose
(linear-probe results on LLMs/VLMs; e.g. Burns et al. 2023, Gurnee & Tegmark
2024), and reading them costs one prefill instead of hundreds of decoded
tokens per window as in GliDe's scanner.

Causal-attention constraint (applies to every decoder-only VLM): a token only
attends to tokens before it. Qwen's chat template places the video BEFORE the
user text, so visual-token states are blind to a user prompt. The task
description is therefore placed in the SYSTEM turn for extraction, which
precedes the video.
"""

from __future__ import annotations

import contextlib
import importlib
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .common import LOGGER, import_stage12

EXTRACT_SYSTEM = (
    "You are a video game QA engineer. The following gameplay footage is from "
    "\"{game}\" ({genre} / {subgenre}). Watch it for software glitches: physics "
    "violations, clipping through geometry, rendering, texture or lighting "
    "errors, broken animation, nonsensical agent behaviour, corrupted UI, or "
    "invalid game state. Unusual but legitimate gameplay is not a glitch."
)
EXTRACT_USER = "Is there a software glitch in this footage, and when does it happen?"


class EarlyExit(Exception):
    """Raised by the deepest layer tap to stop the forward pass."""


class LayerTap:
    """Forward hooks on chosen decoder layers; optionally aborts the forward
    pass right after the deepest one, so deeper layers cost nothing."""

    def __init__(self, layers: Any, indices: Sequence[int], early_exit: bool = True) -> None:
        self.layers = layers
        self.indices = sorted(set(int(i) for i in indices))
        self.early_exit = early_exit
        self.cache: Dict[int, Any] = {}
        self._handles: List[Any] = []

    def __enter__(self) -> "LayerTap":
        deepest = self.indices[-1]
        for idx in self.indices:
            def hook(_module, _inp, out, idx=idx):
                hidden = out[0] if isinstance(out, (tuple, list)) else out
                self.cache[idx] = hidden.detach()
                if self.early_exit and idx == deepest:
                    raise EarlyExit()
            self._handles.append(self.layers[idx].register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return exc_type is EarlyExit


def find_decoder_layers(root: Any) -> Any:
    """The longest ModuleList named '*layers' outside the vision tower."""
    import torch

    best, best_len = None, -1
    for name, module in root.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or not name.endswith("layers"):
            continue
        if any(tag in name for tag in ("visual", "vision", "vit")):
            continue
        if len(module) > best_len:
            best, best_len = module, len(module)
    if best is None:
        raise RuntimeError("could not locate the language-model decoder layers")
    return best


def resolve_layers(n_layers: int, fractions: Sequence[float]) -> List[int]:
    out = sorted({min(n_layers - 1, max(0, int(round(f * (n_layers - 1))))) for f in fractions})
    if not out:
        raise ValueError("no feature layers resolved")
    return out


def pool_tokens(tokens: Any) -> Any:
    """
    tokens: [U, N, D] hidden states of the N visual tokens of each of U units.
    Each token is RMS-normalised first: decoder hidden states carry a few
    "massive activation" channels (Sun et al. 2024) that would otherwise
    dominate max-pooling. Mean-pool keeps the scene gist; max-pool keeps a
    small, localised defect from being averaged away.
    Returns [U, 2, D] float32.
    """
    import torch

    x = tokens.float()
    x = x / x.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    return torch.stack([x.mean(dim=1), x.amax(dim=1)], dim=1)


class Backbone:
    name: str = "base"
    hidden_size: int = 0
    unit_frames: int = 1
    default_fps: float = 4.0

    # -- feature extraction ------------------------------------------------- #
    def extract_units(self, frame_paths: Sequence[str], timestamps: Sequence[float],
                      fps: float, game_text: Dict[str, str]
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    # -- describer primitives ------------------------------------------------ #
    def lm_loss(self, frame_paths: Sequence[str], fps: float, prompt: str, answer: str) -> Any:
        raise NotImplementedError

    def generate(self, frame_paths: Sequence[str], fps: float, prompt: str,
                 max_new_tokens: int = 96, n_samples: int = 0,
                 temperature: float = 0.7, ban: Optional[str] = None) -> List[str]:
        """Greedy output first, then n_samples samples. `ban` forbids the
        first token of that word (used to obtain the describer's best
        description conditional on there being a glitch)."""
        raise NotImplementedError

    def first_token_prob(self, frame_paths: Sequence[str], fps: float, prompt: str,
                         word: str) -> float:
        raise NotImplementedError

    # -- adapters ------------------------------------------------------------ #
    def attach_lora(self, r: int, alpha: int, dropout: float) -> None:
        raise NotImplementedError

    def save_adapter(self, path: Path) -> None:
        raise NotImplementedError

    def load_adapter(self, path: Path) -> None:
        raise NotImplementedError

    @contextlib.contextmanager
    def adapter_disabled(self) -> Iterator[None]:
        yield

    def trainable_parameters(self) -> List[Any]:
        raise NotImplementedError

    def train_mode(self, on: bool) -> None:
        pass


# --------------------------------------------------------------------------- #
# shared LoRA plumbing (PEFT)
# --------------------------------------------------------------------------- #


class _PeftMixin:
    _peft_model: Any = None

    def _wrap(self, module: Any, targets: Any, r: int, alpha: int, dropout: float) -> Any:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                            target_modules=targets, task_type="CAUSAL_LM")
        wrapped = get_peft_model(module, config)
        n_train = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in wrapped.parameters())
        LOGGER.info("LoRA attached: %.2fM trainable / %.2fB total (%.3f%%)",
                    n_train / 1e6, n_all / 1e9, 100.0 * n_train / max(n_all, 1))
        self._peft_model = wrapped
        return wrapped

    def _load(self, module: Any, path: Path) -> Any:
        from peft import PeftModel

        wrapped = PeftModel.from_pretrained(module, str(path), is_trainable=False)
        self._peft_model = wrapped
        LOGGER.info("loaded LoRA adapter from %s", path)
        return wrapped

    def save_adapter(self, path: Path) -> None:
        if self._peft_model is None:
            raise RuntimeError("no adapter attached")
        Path(path).mkdir(parents=True, exist_ok=True)
        self._peft_model.save_pretrained(str(path))

    @contextlib.contextmanager
    def adapter_disabled(self) -> Iterator[None]:
        if self._peft_model is None:
            yield
            return
        with self._peft_model.disable_adapter():
            yield

    def trainable_parameters(self) -> List[Any]:
        if self._peft_model is None:
            return []
        return [p for p in self._peft_model.parameters() if p.requires_grad]


# --------------------------------------------------------------------------- #
# Qwen2.5-VL family (Qwen2.5-VL-3B/7B-Instruct, UI-TARS-1.5-7B)
# --------------------------------------------------------------------------- #

QWEN_LORA_TARGETS = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def qwen_unit_pool(hidden: Any, input_ids: Any, grid_thw: Sequence[int],
                   video_token_id: int, merge: int) -> Any:
    """
    Gather the video-token states of one sequence and fold them into units.
    Qwen2.5-VL emits merged video tokens in (t, h/merge, w/merge) row-major
    order (window-attention reordering is undone inside the vision tower), so
    a reshape to [t, (h/merge)*(w/merge), D] is exact.
    """
    t, h, w = (int(v) for v in grid_thw)
    per = (h // merge) * (w // merge)
    mask = input_ids == video_token_id
    tokens = hidden[mask]
    if tokens.shape[0] != t * per:
        raise RuntimeError(f"video token count {tokens.shape[0]} != t*h*w/merge^2 = {t * per}")
    return pool_tokens(tokens.view(t, per, -1))


class QwenFamily(_PeftMixin, Backbone):
    def __init__(self, model_id: str, dtype: str = "bfloat16", load_in_4bit: bool = False,
                 max_pixels: int = 160 * 28 * 28, min_pixels: int = 32 * 28 * 28,
                 attn: str = "sdpa", fit: str = "letterbox",
                 layer_fracs: Sequence[float] = (0.5, 0.75),
                 stage12_path: Optional[str] = None) -> None:
        import torch

        self.torch = torch
        self.stage12 = import_stage12(stage12_path)
        self.vlm = self.stage12.QwenVLBackend(
            model_id, dtype, load_in_4bit, max_pixels, min_pixels, attn_impl=attn, fit=fit
        )
        self.name = model_id
        self.base_model = self.vlm.model
        cfg = self.base_model.config
        text_cfg = getattr(cfg, "text_config", None) or cfg
        self.hidden_size = int(getattr(text_cfg, "hidden_size"))
        self.video_token_id = int(getattr(cfg, "video_token_id"))
        self.merge = int(getattr(cfg.vision_config, "spatial_merge_size", 2))
        self.unit_frames = int(self.vlm.temporal_patch)
        n_layers = len(find_decoder_layers(self.base_model))
        self.layer_idx = resolve_layers(n_layers, layer_fracs)
        self.default_fps = 4.0
        self.tokenizer = self.vlm.tokenizer
        LOGGER.info("Qwen-family backbone %s: hidden %d, %d decoder layers, tapping %s, "
                    "unit = %d frames", model_id, self.hidden_size, n_layers, self.layer_idx,
                    self.unit_frames)

    @property
    def model(self) -> Any:
        return self.vlm.model

    def _inputs(self, frame_paths: Sequence[str], fps: float, text: str) -> Any:
        video = self.stage12.load_uniform_video(
            frame_paths, factor=self.vlm.factor, min_pixels=self.vlm.min_pixels,
            max_pixels=self.vlm.max_pixels, temporal_patch=self.vlm.temporal_patch,
            fit=self.vlm.fit,
        )
        return self.vlm._call_processor(text, video, fps).to(self.model.device)

    def extract_units(self, frame_paths, timestamps, fps, game_text):
        torch = self.torch
        system = EXTRACT_SYSTEM.format(**game_text)
        text = self.vlm._chat_text(EXTRACT_USER, system)
        inputs = self._inputs(frame_paths, fps, text)
        layers = find_decoder_layers(self.model)
        with torch.inference_mode(), self.adapter_disabled():
            with LayerTap(layers, self.layer_idx) as tap:
                self.model(**inputs, use_cache=False)
        ids = inputs["input_ids"][0]
        grid = inputs["video_grid_thw"][0].tolist()
        pooled = [qwen_unit_pool(tap.cache[i][0], ids, grid, self.video_token_id, self.merge)
                  for i in self.layer_idx]
        feats = torch.cat(pooled, dim=1).cpu().numpy()            # [U, 2L, D]
        ctx = tap.cache[self.layer_idx[-1]][0][-1].float()
        ctx = (ctx / ctx.pow(2).mean().add(1e-6).sqrt()).cpu().numpy()
        n_units = feats.shape[0]
        stamps = list(timestamps) + [timestamps[-1]] * (n_units * self.unit_frames - len(timestamps))
        times = np.array([
            [stamps[u * self.unit_frames], stamps[u * self.unit_frames + self.unit_frames - 1] + 1.0 / fps]
            for u in range(n_units)
        ], dtype=np.float32)
        return feats.astype(np.float32), times, ctx.astype(np.float32)

    def _answer_suffix(self) -> str:
        return "<|im_end|>"

    def lm_loss(self, frame_paths, fps, prompt, answer):
        torch = self.torch
        text = self.vlm._chat_text(prompt) + answer + self._answer_suffix()
        inputs = self._inputs(frame_paths, fps, text)
        ans_ids = self.tokenizer(answer + self._answer_suffix(), add_special_tokens=False).input_ids
        ids = inputs["input_ids"]
        n = len(ans_ids)
        if ids.shape[1] <= n or ids[0, -n:].tolist() != ans_ids:
            # Tokenisation did not split cleanly at the turn boundary: fall
            # back to measuring the prompt length directly.
            prompt_len = self._inputs(frame_paths, fps, self.vlm._chat_text(prompt))["input_ids"].shape[1]
            n = ids.shape[1] - prompt_len
        labels = torch.full_like(ids, -100)
        labels[0, -n:] = ids[0, -n:]
        out = self.model(**inputs, labels=labels, use_cache=False)
        return out.loss

    def generate(self, frame_paths, fps, prompt, max_new_tokens=96, n_samples=0, temperature=0.7,
                 ban=None):
        torch = self.torch
        inputs = self._inputs(frame_paths, fps, self.vlm._chat_text(prompt))
        start = inputs["input_ids"].shape[1]
        if ban:
            inputs = dict(inputs, bad_words_ids=[[self.tokenizer.encode(ban, add_special_tokens=False)[0]]])
        outs: List[str] = []
        with torch.inference_mode():
            greedy = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
            outs.append(self.tokenizer.decode(greedy[0, start:], skip_special_tokens=True).strip())
            if n_samples > 0:
                try:
                    sampled = self.model.generate(
                        **inputs, do_sample=True, temperature=temperature, top_p=0.95,
                        num_return_sequences=n_samples, max_new_tokens=max_new_tokens,
                    )
                    rows = [sampled[i, start:] for i in range(sampled.shape[0])]
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("num_return_sequences unsupported (%r); sampling sequentially", exc)
                    rows = [self.model.generate(**inputs, do_sample=True, temperature=temperature,
                                                top_p=0.95, max_new_tokens=max_new_tokens)[0, start:]
                            for _ in range(n_samples)]
                outs.extend(self.tokenizer.decode(r, skip_special_tokens=True).strip() for r in rows)
        return outs

    def first_token_prob(self, frame_paths, fps, prompt, word):
        torch = self.torch
        inputs = self._inputs(frame_paths, fps, self.vlm._chat_text(prompt))
        target = self.tokenizer.encode(word, add_special_tokens=False)[0]
        with torch.inference_mode():
            logits = self.model(**inputs, use_cache=False).logits[0, -1].float()
        return float(torch.softmax(logits, dim=-1)[target].item())

    def attach_lora(self, r, alpha, dropout):
        self.base_model.gradient_checkpointing_enable()
        self.base_model.enable_input_require_grads()
        self.vlm.model = self._wrap(self.base_model, QWEN_LORA_TARGETS, r, alpha, dropout)

    def load_adapter(self, path):
        self.vlm.model = self._load(self.base_model, Path(path))
        self.vlm.model.eval()

    def train_mode(self, on: bool) -> None:
        self.model.train(on)


# --------------------------------------------------------------------------- #
# InternVL2.5 family (InternVL2.5-4B / 8B; trust_remote_code)
# --------------------------------------------------------------------------- #

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVLFamily(_PeftMixin, Backbone):
    """
    Uses InternVL's own pieces (extract_feature, conv template, IMG_CONTEXT
    splicing) but drives the language model directly, which is what makes
    layer taps, teacher-forced loss and first-token probabilities possible.
    One 448x448 tile per frame (no dynamic tiling): 256 tokens per frame.
    """

    def __init__(self, model_id: str, dtype: str = "bfloat16", load_in_4bit: bool = False,
                 attn: str = "sdpa", layer_fracs: Sequence[float] = (0.5, 0.75),
                 image_size: int = 448) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
        kwargs: Dict[str, Any] = dict(torch_dtype=torch_dtype, low_cpu_mem_usage=True,
                                      trust_remote_code=True, device_map="auto",
                                      use_flash_attn=(attn == "flash_attention_2"))
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        self.model = AutoModel.from_pretrained(model_id, **kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.num_image_token = int(self.model.num_image_token)
        self._conv = importlib.import_module(type(self.model).__module__).get_conv_template
        self.dtype = torch_dtype
        self.image_size = image_size
        self.name = model_id
        lm_cfg = self.model.language_model.config
        self.hidden_size = int(lm_cfg.hidden_size)
        n_layers = len(find_decoder_layers(self.model.language_model))
        self.layer_idx = resolve_layers(n_layers, layer_fracs)
        self.unit_frames = 1
        self.default_fps = 2.0
        self._base_lm = self.model.language_model
        LOGGER.info("InternVL backbone %s: hidden %d, %d layers, tapping %s",
                    model_id, self.hidden_size, n_layers, self.layer_idx)

    @property
    def device(self) -> Any:
        return next(self.model.vision_model.parameters()).device

    def _pixels(self, frame_paths: Sequence[str]) -> Any:
        from PIL import Image

        torch = self.torch
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tiles = []
        for p in frame_paths:
            with Image.open(p) as im:
                im = im.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
                arr = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
            tiles.append((arr - mean) / std)
        return torch.stack(tiles).to(self.device, self.dtype)

    def _template(self) -> Any:
        template = self._conv(self.model.template)
        template.system_message = self.model.system_message
        return template

    def _query(self, question: str, n_frames: int, system: Optional[str] = None) -> str:
        template = self._template()
        if system:
            template.system_message = system
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        image_tokens = "<img>" + "<IMG_CONTEXT>" * self.num_image_token + "</img>"
        for _ in range(n_frames):
            query = query.replace("<image>", image_tokens, 1)
        return query

    @staticmethod
    def _frame_block(timestamps: Sequence[float]) -> str:
        return "".join(f"Frame{i + 1} ({t:.1f}s): <image>\n" for i, t in enumerate(timestamps))

    def _embeds(self, input_ids: Any, pixel_values: Any) -> Any:
        vit = self.model.extract_feature(pixel_values)
        emb = self.model.language_model.get_input_embeddings()(input_ids).clone()
        sel = input_ids == self.model.img_context_token_id
        emb[sel] = vit.reshape(-1, vit.shape[-1]).to(emb.dtype)
        return emb

    def _encode(self, text: str) -> Any:
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def _stamps(self, frame_paths: Sequence[str], fps: float) -> List[float]:
        return [i / max(fps, 1e-3) for i in range(len(frame_paths))]

    def extract_units(self, frame_paths, timestamps, fps, game_text):
        torch = self.torch
        system = EXTRACT_SYSTEM.format(**game_text)
        query = self._query(self._frame_block(timestamps) + EXTRACT_USER, len(frame_paths), system)
        ids = self._encode(query)
        with torch.inference_mode(), self.adapter_disabled():
            emb = self._embeds(ids, self._pixels(frame_paths))
            layers = find_decoder_layers(self.model.language_model)
            with LayerTap(layers, self.layer_idx) as tap:
                self.model.language_model(inputs_embeds=emb, attention_mask=torch.ones_like(ids),
                                          use_cache=False)
        sel = ids[0] == self.model.img_context_token_id
        pooled = []
        for i in self.layer_idx:
            tokens = tap.cache[i][0][sel]
            pooled.append(pool_tokens(tokens.view(len(frame_paths), self.num_image_token, -1)))
        feats = torch.cat(pooled, dim=1).cpu().numpy()
        ctx = tap.cache[self.layer_idx[-1]][0][-1].float()
        ctx = (ctx / ctx.pow(2).mean().add(1e-6).sqrt()).cpu().numpy()
        times = np.array([[t, t + 1.0 / fps] for t in timestamps], dtype=np.float32)
        return feats.astype(np.float32), times, ctx.astype(np.float32)

    def _sep(self) -> str:
        sep = getattr(self._template(), "sep", "<|im_end|>")
        return str(sep).strip() or "<|im_end|>"

    def lm_loss(self, frame_paths, fps, prompt, answer):
        torch = self.torch
        stamps = self._stamps(frame_paths, fps)
        query = self._query(self._frame_block(stamps) + prompt, len(frame_paths))
        prompt_ids = self._encode(query)
        ans_ids = self.tokenizer(answer + self._sep(), add_special_tokens=False,
                                 return_tensors="pt").input_ids.to(self.device)
        ids = torch.cat([prompt_ids, ans_ids], dim=1)
        labels = torch.full_like(ids, -100)
        labels[:, prompt_ids.shape[1]:] = ans_ids
        emb = self._embeds(ids, self._pixels(frame_paths))
        out = self.model.language_model(inputs_embeds=emb, attention_mask=torch.ones_like(ids),
                                        labels=labels, use_cache=False)
        return out.loss

    def generate(self, frame_paths, fps, prompt, max_new_tokens=96, n_samples=0, temperature=0.7,
                 ban=None):
        torch = self.torch
        stamps = self._stamps(frame_paths, fps)
        ids = self._encode(self._query(self._frame_block(stamps) + prompt, len(frame_paths)))
        eos = self.tokenizer.convert_tokens_to_ids(self._sep())
        extra = {"bad_words_ids": [[self.tokenizer.encode(ban, add_special_tokens=False)[0]]]} if ban else {}
        outs: List[str] = []
        with torch.inference_mode():
            emb = self._embeds(ids, self._pixels(frame_paths))
            mask = torch.ones_like(ids)
            for k in range(1 + max(0, n_samples)):
                sample = k > 0
                gen = self.model.language_model.generate(
                    inputs_embeds=emb, attention_mask=mask, max_new_tokens=max_new_tokens,
                    do_sample=sample, temperature=temperature if sample else None,
                    top_p=0.95 if sample else None, eos_token_id=eos, **extra,
                )
                outs.append(self.tokenizer.decode(gen[0], skip_special_tokens=True).strip())
        return outs

    def first_token_prob(self, frame_paths, fps, prompt, word):
        torch = self.torch
        stamps = self._stamps(frame_paths, fps)
        ids = self._encode(self._query(self._frame_block(stamps) + prompt, len(frame_paths)))
        target = self.tokenizer.encode(word, add_special_tokens=False)[0]
        with torch.inference_mode():
            emb = self._embeds(ids, self._pixels(frame_paths))
            logits = self.model.language_model(inputs_embeds=emb, attention_mask=torch.ones_like(ids),
                                               use_cache=False).logits[0, -1].float()
        return float(torch.softmax(logits, dim=-1)[target].item())

    def _lora_targets(self) -> List[str]:
        names = {n.rsplit(".", 1)[-1] for n, _ in self._base_lm.named_modules()}
        for family in (("wqkv", "wo", "w1", "w2", "w3"),
                       ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
            if all(f in names for f in family):
                return list(family)
        raise RuntimeError("unrecognised InternVL language model; cannot place LoRA")

    def attach_lora(self, r, alpha, dropout):
        if hasattr(self._base_lm, "gradient_checkpointing_enable"):
            self._base_lm.gradient_checkpointing_enable()
        self._base_lm.enable_input_require_grads()
        self.model.language_model = self._wrap(self._base_lm, self._lora_targets(), r, alpha, dropout)

    def load_adapter(self, path):
        self.model.language_model = self._load(self._base_lm, Path(path))

    def train_mode(self, on: bool) -> None:
        self.model.language_model.train(on)


# --------------------------------------------------------------------------- #
# CPU mock (self-tests only)
# --------------------------------------------------------------------------- #


class MockBackbone(Backbone):
    """
    Deterministic pixel features with the same output contract as a real
    backbone. It lets the full pipeline (extract -> head -> decode -> describe
    -> evaluate) run on a CPU in seconds. Its describer is rule-based; numbers
    obtained with it say nothing about NORMA's accuracy.
    """

    def __init__(self, hidden: int = 48, seed: int = 0) -> None:
        import torch

        self.torch = torch
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 1.0 / math.sqrt(48), size=(48, hidden)).astype(np.float32)
        self.w2 = rng.normal(0, 1.0 / math.sqrt(hidden), size=(hidden, hidden)).astype(np.float32)
        self.hidden_size = hidden
        self.unit_frames = 2
        self.name = "mock"
        self.default_fps = 4.0
        self._lora = torch.nn.Parameter(torch.zeros(4))

    @staticmethod
    def _tokens(path: str) -> np.ndarray:
        from PIL import Image

        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB").resize((8, 8)), dtype=np.float32) / 255.0
        quads = [arr[:4, :4], arr[:4, 4:], arr[4:, :4], arr[4:, 4:]]
        return np.stack([q.reshape(-1) for q in quads])               # [4, 48]

    def extract_units(self, frame_paths, timestamps, fps, game_text):
        torch = self.torch
        paths = list(frame_paths) + [frame_paths[-1]] * (len(frame_paths) % 2)
        stamps = list(timestamps) + [timestamps[-1]] * (len(paths) - len(timestamps))
        toks = np.stack([self._tokens(p) for p in paths])               # [F, 4, 48]
        h1 = np.tanh(toks @ self.w1)
        h2 = np.tanh(h1 @ self.w2)
        n_units = len(paths) // 2
        feats = []
        for h in (h1, h2):
            unit_tok = torch.from_numpy(h.reshape(n_units, 8, -1))
            feats.append(pool_tokens(unit_tok))
        feats = torch.cat(feats, dim=1).numpy()
        times = np.array([[stamps[2 * u], stamps[2 * u + 1] + 1.0 / fps] for u in range(n_units)],
                         dtype=np.float32)
        return feats, times, h2.mean(axis=(0, 1)).astype(np.float32)

    def _jump(self, frame_paths) -> float:
        toks = np.stack([self._tokens(p).reshape(-1) for p in frame_paths])
        if len(toks) < 2:
            return 0.0
        return float(np.abs(np.diff(toks, axis=0)).mean(axis=1).max())

    def lm_loss(self, frame_paths, fps, prompt, answer):
        target = self.torch.tensor(len(answer) / 100.0)
        return (self._lora.sum() - target) ** 2

    def generate(self, frame_paths, fps, prompt, max_new_tokens=96, n_samples=0, temperature=0.7,
                 ban=None):
        text = ("The red square teleports abruptly across the screen."
                if ban or self._jump(frame_paths) > 0.05 else "NO_GLITCH")
        return [text] * (1 + max(0, n_samples))

    def first_token_prob(self, frame_paths, fps, prompt, word):
        return float(1.0 / (1.0 + math.exp(40.0 * (self._jump(frame_paths) - 0.05))))

    def attach_lora(self, r, alpha, dropout):
        pass

    def save_adapter(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.torch.save({"lora": self._lora.detach()}, Path(path) / "mock_adapter.pt")

    def load_adapter(self, path):
        state = self.torch.load(Path(path) / "mock_adapter.pt")
        self._lora.data.copy_(state["lora"])

    def trainable_parameters(self):
        return [self._lora]


def build_backbone(model_id: str, args: Any) -> Backbone:
    key = model_id.lower()
    fracs = tuple(getattr(args, "layer_fracs", (0.5, 0.75)))
    if key == "mock":
        return MockBackbone()
    if "internvl" in key:
        return InternVLFamily(model_id, args.dtype, args.load_in_4bit, args.attn, fracs)
    # Qwen2.5-VL-3B/7B-Instruct and UI-TARS-1.5-7B share the Qwen2.5-VL architecture.
    return QwenFamily(model_id, args.dtype, args.load_in_4bit, args.max_pixels, args.min_pixels,
                      args.attn, args.fit, fracs, getattr(args, "stage12", None))
