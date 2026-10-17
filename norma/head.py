"""
NORMA localisation head: Self-Referential Normality (SRN) + a small temporal
transformer over frozen-VLM unit features, trained with dense targets, and a
correctly specified hybrid-HMM decoder.

---------------------------------------------------------------------------
Model (per video, T units, V pooled vectors per unit, P dims each)
---------------------------------------------------------------------------
u_t   = LN(W_p vec(x_t))                                       content
m     = median_t u_t,  s = 1.4826 * median_t |u_t - m| + eps   robust reference
r_t   = clip((u_t - m) / s, -c, c)                             SRN deviation
e_t   = ||r_t|| / sqrt(d)                                      SRN magnitude
g_t   = ||u_t - u_{t-1}|| / sqrt(d)                            temporal surprise
n_t   = log(1 + mean_k ||PCAw(x_t) - b_(k)||), b in normal bank, kNN distance
z_t   = u_t + W_r r_t + W_d (u_t - u_{t-1}) + W_c ctx_t + MLP([e,g,n,t/T]) + PE(sec_t)
h     = TransformerEncoder(z)
heads: actionness a_t, boundaries (s_t, e_t), instance embedding v_t (unit
       norm), event count K via attention pooling.

Why SRN is principled: a glitch is a deviation from what is normal FOR THIS
GAME, and the cheapest unbiased estimate of "normal for this game" available
at test time on an unseen title is the video itself. The coordinate-wise
median / MAD has a 50% breakdown point (Hampel 1974; Rousseeuw & Croux
1993), so the reference is unaffected by the glitch whenever the glitch
covers under half of the clip. When it covers more, the extra-video kNN
distance n_t (PatchCore-style memory, Roth et al. CVPR 2022) still carries
the signal; the transformer learns how to weigh the two. This replaces
GliDe's text-only game memory with a statistical one that costs no tokens.

---------------------------------------------------------------------------
Decoder (fixes the double-counted prior in KAIROS)
---------------------------------------------------------------------------
A discriminative head outputs posteriors p(s_t=1 | x). An HMM needs
likelihoods. Hybrid connectionist/HMM theory (Bourlard & Morgan 1994) gives
the scaled likelihood p(x|s) / p(x) = p(s|x) / p(s), so the per-unit
emission log-ratio is

    llr_t = logit p(s_t=1|x) - logit pi,     pi = training base rate,

and the prior enters ONCE, through the transition matrix. Transitions are
estimated on non-overlapping units, so they are not inflated by window
overlap.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import LOGGER, merge_intervals

# --------------------------------------------------------------------------- #
# extra-video normality bank
# --------------------------------------------------------------------------- #


class NormalityBank:
    """Whitened-PCA + greedy k-center coreset of NORMAL training units.
    Distances for TRAINING videos exclude the video's own game, so the
    training-time distribution of n_t matches test time (unseen game)."""

    def __init__(self, pca_dim: int = 128, size: int = 4096, k: int = 3, seed: int = 0) -> None:
        self.pca_dim, self.size, self.k, self.seed = pca_dim, size, k, seed
        self.mean: Optional[np.ndarray] = None
        self.components: Optional[np.ndarray] = None
        self.bank: Optional[np.ndarray] = None
        self.bank_game: Optional[np.ndarray] = None

    @staticmethod
    def _flat(feats: np.ndarray) -> np.ndarray:
        return feats.reshape(feats.shape[0], -1).astype(np.float32)

    def fit(self, normal_units: Sequence[Tuple[np.ndarray, str]], max_fit: int = 60000) -> None:
        import torch

        rng = np.random.default_rng(self.seed)
        # Subsample per video BEFORE concatenating: the full VGB train pool is
        # ~1e5 units x 4096 dims (~2 GB float32) per fold.
        per_video = max(4, (2 * max_fit) // max(len(normal_units), 1))
        picked = []
        for f, g in normal_units:
            idx = rng.choice(f.shape[0], min(per_video, f.shape[0]), replace=False)
            picked.append((self._flat(f[idx]), g))
        x = np.concatenate([f for f, _ in picked])
        games = np.concatenate([np.full(f.shape[0], g, dtype=object) for f, g in picked])
        if x.shape[0] > max_fit:
            pick = rng.choice(x.shape[0], max_fit, replace=False)
            x, games = x[pick], games[pick]
        xt = torch.from_numpy(x)
        self.mean = xt.mean(0).numpy()
        q = min(self.pca_dim, xt.shape[1], xt.shape[0] - 1)
        _, s, v = torch.pca_lowrank(xt - xt.mean(0), q=q, center=False)
        scale = s / math.sqrt(max(xt.shape[0] - 1, 1))
        self.components = (v / scale.clamp_min(1e-6)).numpy().astype(np.float32)   # whitening
        z = self.transform_flat(x)
        self.bank, self.bank_game = self._coreset(z, games, rng)
        LOGGER.info("normality bank: %d normal units -> PCA %d -> coreset %d", x.shape[0],
                    self.components.shape[1], self.bank.shape[0])

    def transform_flat(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) @ self.components

    def _coreset(self, z: np.ndarray, games: np.ndarray, rng) -> Tuple[np.ndarray, np.ndarray]:
        """Greedy k-center (Sener & Savarese 2018): 2-approximation of the
        minimax covering radius, the quantity that bounds kNN error."""
        import torch

        if z.shape[0] <= self.size:
            return z.astype(np.float32), games
        zt = torch.from_numpy(z)
        chosen = [int(rng.integers(z.shape[0]))]
        dist = torch.cdist(zt, zt[chosen]).squeeze(1)
        for _ in range(self.size - 1):
            nxt = int(torch.argmax(dist))
            chosen.append(nxt)
            dist = torch.minimum(dist, torch.cdist(zt, zt[nxt:nxt + 1]).squeeze(1))
        return z[chosen].astype(np.float32), games[chosen]

    def distance(self, feats: np.ndarray, exclude_game: Optional[str] = None) -> np.ndarray:
        import torch

        z = torch.from_numpy(self.transform_flat(self._flat(feats)))
        bank = torch.from_numpy(self.bank)
        if exclude_game is not None:
            keep = torch.from_numpy(self.bank_game != exclude_game)
            if int(keep.sum()) >= self.k:
                bank = bank[keep]
        d = torch.cdist(z, bank)
        k = min(self.k, bank.shape[0])
        knn = torch.topk(d, k, largest=False).values.mean(1) / math.sqrt(z.shape[1])
        return torch.log1p(knn).numpy().astype(np.float32)

    def state(self) -> Dict[str, Any]:
        return {"pca_dim": self.pca_dim, "size": self.size, "k": self.k, "seed": self.seed,
                "mean": self.mean, "components": self.components, "bank": self.bank,
                "bank_game": None if self.bank_game is None else self.bank_game.tolist()}

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "NormalityBank":
        bank = cls(state["pca_dim"], state["size"], state["k"], state["seed"])
        bank.mean, bank.components, bank.bank = state["mean"], state["components"], state["bank"]
        bank.bank_game = np.array(state["bank_game"], dtype=object)
        return bank


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class HeadConfig:
    n_vectors: int = 4
    in_dim: int = 1024
    d_model: int = 384
    n_layers: int = 3
    n_heads: int = 6
    dropout: float = 0.1
    emb_dim: int = 8
    k_max: int = 4
    srn_clip: float = 6.0
    use_srn: bool = True
    use_bank: bool = True
    use_delta: bool = True
    use_ctx: bool = True
    # loss
    focal_alpha: float = 0.5
    focal_gamma: float = 2.0
    w_boundary: float = 1.0
    w_count: float = 0.5
    w_embed: float = 0.5
    embed_margin: float = 1.0


def _build_model(cfg: HeadConfig) -> Any:
    import torch
    from torch import nn

    class SinusoidalTime(nn.Module):
        """Encodes absolute seconds (periods 0.5 s ... 128 s), so the head is
        agnostic to the backbone's unit duration and to the sampling fps."""

        def __init__(self, d: int) -> None:
            super().__init__()
            half = d // 2
            periods = torch.logspace(math.log10(0.5), math.log10(128.0), half)
            self.register_buffer("omega", 2.0 * math.pi / periods)
            self.d = d

        def forward(self, sec):
            ang = sec.unsqueeze(-1) * self.omega
            pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
            if pe.shape[-1] < self.d:
                pe = torch.nn.functional.pad(pe, (0, self.d - pe.shape[-1]))
            return pe

    class NormaHead(nn.Module):
        def __init__(self, c: HeadConfig) -> None:
            super().__init__()
            self.c = c
            flat = c.n_vectors * c.in_dim
            d = c.d_model
            self.in_norm = nn.LayerNorm(flat)
            self.proj = nn.Sequential(nn.Linear(flat, d), nn.LayerNorm(d))
            self.srn_proj = nn.Linear(d, d)
            self.delta_proj = nn.Linear(d, d)
            self.ctx_proj = nn.Sequential(nn.LayerNorm(c.in_dim), nn.Linear(c.in_dim, d))
            self.scalar_mlp = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
            self.time = SinusoidalTime(d)
            self.in_drop = nn.Dropout(c.dropout)
            layer = nn.TransformerEncoderLayer(d, c.n_heads, 4 * d, c.dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, c.n_layers, enable_nested_tensor=False)
            self.out_norm = nn.LayerNorm(d)
            self.act = nn.Linear(d, 1)
            self.bound = nn.Linear(d, 2)
            self.emb = nn.Linear(d, c.emb_dim)
            self.pool = nn.Linear(d, 1)
            self.count = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, c.k_max + 1))

        def srn(self, u, mask):
            """Masked coordinate-wise median / MAD over time, per video."""
            r = torch.zeros_like(u)
            for b in range(u.shape[0]):
                valid = u[b, mask[b]]
                if valid.shape[0] == 0:
                    continue
                med = valid.median(dim=0).values
                mad = (valid - med).abs().median(dim=0).values * 1.4826 + 1e-2
                r[b] = ((u[b] - med) / mad).clamp(-self.c.srn_clip, self.c.srn_clip)
            return r * mask.unsqueeze(-1)

        def forward(self, feats, ctx, bank_dist, times, mask):
            c = self.c
            B, T = feats.shape[:2]
            u = self.proj(self.in_norm(feats.reshape(B, T, -1)))
            d = u.shape[-1]
            z = u
            zeros = torch.zeros(B, T, device=u.device)
            e = zeros
            if c.use_srn:
                r = self.srn(u, mask)
                # sqrt(x + eps): d/dx sqrt(x) is unbounded at 0 (padded units), which
                # would turn inf * 0 into NaN in the backward pass.
                e = r.pow(2).mean(-1).add(1e-6).sqrt()
                z = z + self.srn_proj(r)
            delta = torch.zeros_like(u)
            delta[:, 1:] = u[:, 1:] - u[:, :-1]
            delta = delta * mask.unsqueeze(-1)
            g = delta.pow(2).sum(-1).add(1e-6).sqrt() / math.sqrt(d) if c.use_delta else zeros
            if c.use_delta:
                z = z + self.delta_proj(delta)
            if c.use_ctx:
                z = z + self.ctx_proj(ctx)
            n = bank_dist if c.use_bank else zeros
            centre = times.mean(-1)
            length = (centre * mask).amax(dim=1, keepdim=True).clamp_min(1e-3)
            scalars = torch.stack([e, g, n, centre / length], dim=-1)
            z = z + self.scalar_mlp(scalars) + self.time(centre)
            h = self.encoder(self.in_drop(z), src_key_padding_mask=~mask)
            h = self.out_norm(h)
            pool_logit = self.pool(h).squeeze(-1).masked_fill(~mask, -1e4)
            pooled = (torch.softmax(pool_logit, dim=1).unsqueeze(-1) * h).sum(1)
            emb = torch.nn.functional.normalize(self.emb(h), dim=-1)
            bnd = self.bound(h)
            return {
                "act": self.act(h).squeeze(-1),
                "start": bnd[..., 0],
                "end": bnd[..., 1],
                "emb": emb,
                "count": self.count(pooled),
            }

    return NormaHead(cfg)


# --------------------------------------------------------------------------- #
# batching
# --------------------------------------------------------------------------- #


def collate(samples: Sequence[Dict[str, Any]], device: Any) -> Dict[str, Any]:
    import torch

    B = len(samples)
    T = max(s["feats"].shape[0] for s in samples)
    V, P = samples[0]["feats"].shape[1:]
    feats = torch.zeros(B, T, V, P)
    ctx = torch.zeros(B, T, samples[0]["ctx"].shape[-1])
    times = torch.zeros(B, T, 2)
    bank = torch.zeros(B, T)
    mask = torch.zeros(B, T, dtype=torch.bool)
    y = torch.zeros(B, T)
    start = torch.zeros(B, T)
    end = torch.zeros(B, T)
    inst = torch.full((B, T), -1, dtype=torch.long)
    count = torch.zeros(B, dtype=torch.long)
    for b, s in enumerate(samples):
        n = s["feats"].shape[0]
        feats[b, :n] = torch.from_numpy(s["feats"].astype(np.float32))
        ctx[b, :n] = torch.from_numpy(s["ctx"].astype(np.float32))
        times[b, :n] = torch.from_numpy(s["times"])
        if n < T:
            times[b, n:] = times[b, n - 1]
        bank[b, :n] = torch.from_numpy(s["bank"])
        mask[b, :n] = True
        if "y" in s:
            y[b, :n] = torch.from_numpy(s["y"])
            start[b, :n] = torch.from_numpy(s["start"])
            end[b, :n] = torch.from_numpy(s["end"])
            inst[b, :n] = torch.from_numpy(s["inst"])
            count[b] = int(s["count"])
    out = {"feats": feats, "ctx": ctx, "times": times, "bank_dist": bank, "mask": mask,
           "y": y, "start": start, "end": end, "inst": inst, "count": count}
    return {k: v.to(device) for k, v in out.items()}


def augment(sample: Dict[str, Any], rng: np.random.Generator, k_max: int) -> Dict[str, Any]:
    """Random temporal crop (keeps >= 60% of the clip) + feature noise.
    The count target is recomputed from the instances that survive the crop."""
    s = dict(sample)
    n = s["feats"].shape[0]
    if n >= 8 and rng.random() < 0.5:
        length = int(rng.integers(int(0.6 * n), n + 1))
        a = int(rng.integers(0, n - length + 1))
        for key in ("feats", "ctx", "times", "bank", "y", "start", "end", "inst"):
            s[key] = s[key][a:a + length]
        s["count"] = min(k_max, len({int(i) for i in s["inst"] if i >= 0}) or int(s["y"].any()))
    noise = rng.normal(0.0, 0.05, size=s["feats"].shape).astype(np.float32)
    s["feats"] = s["feats"].astype(np.float32) * (1.0 + noise)
    return s


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #


def norma_loss(out: Dict[str, Any], batch: Dict[str, Any], cfg: HeadConfig) -> Tuple[Any, Dict[str, float]]:
    import torch
    import torch.nn.functional as F

    mask = batch["mask"].float()
    n_valid = mask.sum().clamp_min(1.0)

    # Sigmoid focal loss (Lin et al. 2017) on actionness.
    logit = out["act"]
    y = batch["y"]
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    p_t = p * y + (1 - p) * (1 - y)
    alpha_t = cfg.focal_alpha * y + (1 - cfg.focal_alpha) * (1 - y)
    l_act = (alpha_t * (1 - p_t).pow(cfg.focal_gamma) * ce * mask).sum() / n_valid

    # Soft-target boundary BCE, up-weighting the sparse peaks.
    l_bnd = 0.0
    for key in ("start", "end"):
        target = batch[key]
        weight = (1.0 + 4.0 * target) * mask
        l_bnd = l_bnd + (F.binary_cross_entropy_with_logits(out[key], target, reduction="none")
                         * weight).sum() / weight.sum().clamp_min(1.0)

    l_cnt = F.cross_entropy(out["count"], batch["count"].clamp(max=cfg.k_max))

    # Discriminative instance-embedding loss (De Brabandere et al. 2017):
    # pull units to their event mean, push event means at least `margin` apart.
    l_emb = out["act"].new_zeros(())
    n_terms = 0
    for b in range(out["emb"].shape[0]):
        inst = batch["inst"][b]
        ids = [int(i) for i in torch.unique(inst) if int(i) >= 0]
        if not ids:
            continue
        means = []
        for i in ids:
            v = out["emb"][b][inst == i]
            mu = v.mean(0)
            l_emb = l_emb + (v - mu).pow(2).sum(-1).mean()
            means.append(mu)
            n_terms += 1
        if len(means) > 1:
            mu = torch.stack(means)
            dist = (mu.unsqueeze(1) - mu.unsqueeze(0)).pow(2).sum(-1).add(1e-9).sqrt()
            off = ~torch.eye(len(means), dtype=torch.bool, device=mu.device)
            l_emb = l_emb + F.relu(cfg.embed_margin - dist[off]).pow(2).mean()
    l_emb = l_emb / max(n_terms, 1)

    total = l_act + cfg.w_boundary * l_bnd + cfg.w_count * l_cnt + cfg.w_embed * l_emb
    return total, {"act": float(l_act.detach()), "bnd": float(l_bnd.detach()), "cnt": float(l_cnt.detach()),
                   "emb": float(l_emb.detach())}


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class DecodeParams:
    temperature: float = 1.0
    none_threshold: float = 0.5
    event_threshold: float = 0.4
    group_sim: float = 0.75
    boundary_radius: int = 2
    boundary_min: float = 0.3
    max_events: int = 4
    fallback: bool = True


def fit_transitions(label_seqs: Sequence[np.ndarray], smoothing: float = 1.0) -> Dict[str, float]:
    c = {"00": smoothing, "01": smoothing, "10": smoothing, "11": smoothing}
    starts, total = smoothing, 2.0 * smoothing
    pos, n = 0.0, 0.0
    for seq in label_seqs:
        seq = (np.asarray(seq) > 0.5).astype(int)
        if seq.size == 0:
            continue
        starts += seq[0]
        total += 1.0
        pos += seq.sum()
        n += seq.size
        for a, b in zip(seq[:-1], seq[1:]):
            c[f"{a}{b}"] += 1.0
    clip = lambda v: float(np.clip(v, 1e-4, 1 - 1e-4))  # noqa: E731
    return {"p11": clip(c["11"] / (c["11"] + c["10"])), "p01": clip(c["01"] / (c["01"] + c["00"])),
            "pi1": clip(starts / total), "prior": clip(pos / max(n, 1.0))}


def viterbi(llr: np.ndarray, trans: Dict[str, float]) -> np.ndarray:
    """Two-state Viterbi. llr_t = log p(x_t|1)/p(x_t|0) (scaled likelihood);
    state-0 emission is the reference (0)."""
    n = len(llr)
    if n == 0:
        return np.zeros(0, dtype=int)
    lt = np.log(np.array([[1 - trans["p01"], trans["p01"]], [1 - trans["p11"], trans["p11"]]]))
    delta = np.zeros((n, 2))
    psi = np.zeros((n, 2), dtype=int)
    delta[0] = [math.log(1 - trans["pi1"]), math.log(trans["pi1"]) + llr[0]]
    for t in range(1, n):
        for s in (0, 1):
            cand = delta[t - 1] + lt[:, s]
            psi[t, s] = int(np.argmax(cand))
            delta[t, s] = cand[psi[t, s]] + (llr[t] if s == 1 else 0.0)
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    return path


def _runs(path: np.ndarray) -> List[Tuple[int, int]]:
    runs, start = [], None
    for i, v in enumerate(path):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(path) - 1))
    return runs


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def decode_video(raw: Dict[str, np.ndarray], times: np.ndarray, prm: DecodeParams,
                 trans: Dict[str, float], duration: float = 0.0) -> List[Dict[str, Any]]:
    """raw: per-unit logits act/start/end, emb [T,E], count logits [K+1]."""
    n = len(raw["act"])
    if n == 0:
        return []
    p = _sigmoid(raw["act"] / max(prm.temperature, 1e-3))
    ps, pe = _sigmoid(raw["start"]), _sigmoid(raw["end"])
    cnt = np.exp(raw["count"] - raw["count"].max())
    cnt = cnt / cnt.sum()
    p_none = float(cnt[0])
    if p_none >= prm.none_threshold:
        return []
    k_hat = int(np.argmax(cnt[1:])) + 1

    eps = 1e-6
    prior = trans["prior"]
    llr = np.log(p + eps) - np.log(1 - p + eps) - math.log(prior / (1 - prior))
    runs = _runs(viterbi(llr, trans))

    segs = []
    for i, j in runs:
        r = prm.boundary_radius
        lo, hi = max(0, i - r), min(n - 1, i + r)
        s_best = lo + int(np.argmax(ps[lo:hi + 1]))
        if ps[s_best] >= prm.boundary_min and s_best <= j:
            i = s_best
        lo, hi = max(0, j - r), min(n - 1, j + r)
        e_best = lo + int(np.argmax(pe[lo:hi + 1]))
        if pe[e_best] >= prm.boundary_min and e_best >= i:
            j = e_best
        emb = raw["emb"][i:j + 1].mean(0)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        segs.append({"units": (i, j), "score": float(p[i:j + 1].mean()), "emb": emb})

    if not segs and prm.fallback:
        peak = int(np.argmax(p))
        i = j = peak
        while i > 0 and p[i - 1] >= 0.5 * p[peak]:
            i -= 1
        while j < n - 1 and p[j + 1] >= 0.5 * p[peak]:
            j += 1
        emb = raw["emb"][i:j + 1].mean(0)
        segs.append({"units": (i, j), "score": float(p[i:j + 1].mean()),
                     "emb": emb / (np.linalg.norm(emb) + 1e-9)})

    # Average-linkage grouping of segments by instance embedding: segments of
    # one recurring bug become ONE event with several intervals.
    groups = [[k] for k in range(len(segs))]
    while len(groups) > 1:
        best, pair = -1.0, None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                sim = float(np.mean([segs[x]["emb"] @ segs[y]["emb"]
                                     for x in groups[a] for y in groups[b]]))
                if sim > best:
                    best, pair = sim, (a, b)
        if best < prm.group_sim:
            break
        a, b = pair
        groups[a] = groups[a] + groups[b]
        del groups[b]

    events = []
    for group in groups:
        ivs = []
        for k in group:
            i, j = segs[k]["units"]
            a, b = float(times[i, 0]), float(times[j, 1])
            if duration > 0:
                b = min(b, duration)
            if b > a:
                ivs.append((round(a, 3), round(b, 3)))
        if not ivs:
            continue
        events.append({
            "intervals": merge_intervals(ivs),
            "seg_score": max(segs[k]["score"] for k in group),
            "p_any": 1.0 - p_none,
            "score": max(segs[k]["score"] for k in group) * (1.0 - p_none),
        })
    events.sort(key=lambda e: -e["score"])
    kept = [e for e in events if e["seg_score"] >= prm.event_threshold]
    if not kept and events and prm.fallback:
        kept = events[:1]
    return kept[: min(prm.max_events, max(1, k_hat))]


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def run_head(model: Any, samples: Sequence[Dict[str, Any]], device: Any,
             batch_size: int = 16) -> List[Dict[str, np.ndarray]]:
    import torch

    model.eval()
    outs: List[Dict[str, np.ndarray]] = []
    with torch.no_grad():
        for s in range(0, len(samples), batch_size):
            chunk = samples[s:s + batch_size]
            batch = collate(chunk, device)
            out = model(batch["feats"], batch["ctx"], batch["bank_dist"], batch["times"], batch["mask"])
            for b, sample in enumerate(chunk):
                n = sample["feats"].shape[0]
                outs.append({
                    "act": out["act"][b, :n].float().cpu().numpy(),
                    "start": out["start"][b, :n].float().cpu().numpy(),
                    "end": out["end"][b, :n].float().cpu().numpy(),
                    "emb": out["emb"][b, :n].float().cpu().numpy(),
                    "count": out["count"][b].float().cpu().numpy(),
                })
    return outs


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling (Guo et al. 2017): argmin_T NLL on dev units."""
    best_t, best_nll = 1.0, float("inf")
    for t in np.exp(np.linspace(math.log(0.25), math.log(4.0), 41)):
        p = _sigmoid(logits / t)
        nll = -np.mean(labels * np.log(p + 1e-9) + (1 - labels) * np.log(1 - p + 1e-9))
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t


def train_head(train: Sequence[Dict[str, Any]], dev: Sequence[Dict[str, Any]], cfg: HeadConfig,
               trans: Dict[str, float], epochs: int, lr: float, batch_size: int, seed: int,
               device: Any, select_fn: Any) -> Tuple[Any, List[Dict[str, float]]]:
    """select_fn(model) -> dev selection score (higher is better)."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = _build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = max(1, epochs * math.ceil(len(train) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    history: List[Dict[str, float]] = []
    best_score, best_state = -float("inf"), None
    order = np.arange(len(train))
    for epoch in range(epochs):
        model.train()
        rng.shuffle(order)
        agg: Dict[str, float] = {}
        for s in range(0, len(order), batch_size):
            chunk = [augment(train[i], rng, cfg.k_max) for i in order[s:s + batch_size]]
            batch = collate(chunk, device)
            out = model(batch["feats"], batch["ctx"], batch["bank_dist"], batch["times"], batch["mask"])
            loss, parts = norma_loss(out, batch, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite head loss at epoch {epoch}: {parts}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v * len(chunk)
        score = float(select_fn(model))
        row = {"epoch": epoch, "dev_score": score, **{k: v / len(train) for k, v in agg.items()}}
        history.append(row)
        LOGGER.info("head epoch %2d | act %.4f bnd %.4f cnt %.4f emb %.4f | dev %.4f",
                    epoch, row["act"], row["bnd"], row["cnt"], row["emb"], score)
        if score > best_score:
            best_score, best_state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history


def load_model(cfg: HeadConfig, state: Dict[str, Any], device: Any) -> Any:
    model = _build_model(cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    return model
