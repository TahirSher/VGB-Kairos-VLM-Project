"""
Game-disjoint, genre-stratified K-fold protocol.

VideoGlitchBench ships no train/test split, and GliDe (training-free) reports
on all 5,238 videos. A trained model can only be compared fairly if every
video receives an OUT-OF-FOLD prediction from a model that never saw its
game. Out-of-fold predictions concatenated over K folds cover exactly the set
GliDe was scored on, and game-disjointness makes the number a cross-game
generalisation estimate rather than a within-game memorisation estimate
(videos of one game share engine, art style and often the very same bug).

Inside each training partition a further game-disjoint dev subset is carved
out for early stopping, calibration and decoder thresholds. The test fold is
never touched by any fitting step.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

from .common import LOGGER
from .data import VideoItem


def make_folds(items: Dict[str, VideoItem], k: int, dev_frac: float, seed: int) -> Dict[str, object]:
    if k < 2:
        raise ValueError("k must be >= 2")
    games: Dict[str, List[str]] = defaultdict(list)
    genre_of: Dict[str, str] = {}
    for name, item in items.items():
        games[item.game].append(name)
        genre_of.setdefault(item.game, item.genre or "unknown")
    if len(games) < k:
        raise ValueError(f"only {len(games)} games; cannot build {k} game-disjoint folds")

    rng = random.Random(seed)
    by_genre: Dict[str, List[str]] = defaultdict(list)
    for game, genre in genre_of.items():
        by_genre[genre].append(game)

    fold_games: List[List[str]] = [[] for _ in range(k)]
    fold_size = [0] * k
    fold_genre: List[Counter] = [Counter() for _ in range(k)]
    # Largest-first greedy bin packing inside every genre: balances both the
    # genre mix and the video count of each fold.
    for genre in sorted(by_genre):
        pool = sorted(by_genre[genre], key=lambda g: (-len(games[g]), rng.random()))
        for game in pool:
            target = min(range(k), key=lambda f: (fold_genre[f][genre], fold_size[f], rng.random()))
            fold_games[target].append(game)
            fold_size[target] += len(games[game])
            fold_genre[target][genre] += len(games[game])

    folds = []
    for f in range(k):
        test_games = sorted(fold_games[f])
        rest = sorted(g for i, gs in enumerate(fold_games) if i != f for g in gs)
        rng_f = random.Random(seed * 1000 + f)
        rng_f.shuffle(rest)
        n_dev_target = dev_frac * sum(len(games[g]) for g in rest)
        dev_games: List[str] = []
        acc = 0
        for game in rest:
            if acc >= n_dev_target:
                break
            dev_games.append(game)
            acc += len(games[game])
        train_games = sorted(set(rest) - set(dev_games))
        folds.append({
            "fold": f,
            "test": sorted(v for g in test_games for v in games[g]),
            "dev": sorted(v for g in dev_games for v in games[g]),
            "train": sorted(v for g in train_games for v in games[g]),
            "test_games": test_games,
            "dev_games": sorted(dev_games),
            "train_games": train_games,
        })
        LOGGER.info("fold %d: train %d / dev %d / test %d videos (%d / %d / %d games)",
                    f, len(folds[-1]["train"]), len(folds[-1]["dev"]), len(folds[-1]["test"]),
                    len(train_games), len(dev_games), len(test_games))

    assert_disjoint(folds)
    return {"k": k, "seed": seed, "dev_frac": dev_frac, "folds": folds}


def assert_disjoint(folds: Sequence[Dict[str, object]]) -> None:
    covered: Counter = Counter()
    for fold in folds:
        tr, dv, te = (set(fold[s]) for s in ("train", "dev", "test"))
        if tr & dv or tr & te or dv & te:
            raise RuntimeError(f"fold {fold['fold']}: partitions overlap")
        gtr, gdv, gte = (set(fold[s]) for s in ("train_games", "dev_games", "test_games"))
        if gtr & gte or gdv & gte or gtr & gdv:
            raise RuntimeError(f"fold {fold['fold']}: a game crosses partitions")
        covered.update(te)
    if any(c != 1 for c in covered.values()):
        raise RuntimeError("every video must be in exactly one test fold")
