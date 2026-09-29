"""Cached preview features, per-window labels, sampled budgets and the solved teacher.

``mf-cache-features`` writes one pair of files per shard:

* ``shard<i>of<n>_feats.npy``: float16 preview features, ``(total windows, 2048)``;
* ``shard<i>of<n>_meta.npz``: ``labels`` ``(total windows, 2)`` as (image forged, audio forged),
  ``offsets`` ``(videos + 1,)`` into the shard, and per-video ``keys``, ``modify_type``,
  ``video_file`` and ``source``.

The features are memory-mapped, never concatenated, so several training processes on one host
share a single page cache.

One dataset item is one video with ``k`` budgets and, for each budget, the **whole** optimal-action
table of the knapsack teacher over (window, remaining budget). Stage 1 queries that table at
whatever state the router's own rollout reaches (DAgger).
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils import data

from ..router.oracle import solve_knapsack, uniform_costs
from ..router.reward import reward_matrix
from .budgets import BUDGETS_PER_VIDEO, MAX_FRACTION, fixed_budgets, sample_budget


def load_feature_shards(cache_dir: str) -> Dict:
    """Memory-map every feature shard and index its videos."""
    paths = sorted(glob.glob(os.path.join(cache_dir, "shard*_feats.npy")))
    if not paths:
        raise ValueError(f"no feature shards (shard*_feats.npy) under {cache_dir}")
    shards, labels, keys, mtypes, files, sources = [], [], [], [], [], []
    vid_shard, vid_lo, vid_hi = [], [], []
    for s, path in enumerate(paths):
        shards.append(np.load(path, mmap_mode="r"))
        with np.load(path.replace("_feats.npy", "_meta.npz"), allow_pickle=False) as meta:
            off = meta["offsets"]
            labels.append(meta["labels"])
            keys.append(meta["keys"])
            mtypes.append(meta["modify_type"])
            files.append(meta["video_file"])
            sources.append(meta["source"])
        vid_shard.extend([s] * (len(off) - 1))
        vid_lo.extend(off[:-1].tolist())
        vid_hi.extend(off[1:].tolist())
    lab_off = np.cumsum([0] + [int(h - l) for l, h in zip(vid_lo, vid_hi)])
    return {"shards": shards, "labels": np.concatenate(labels),
            "vid_shard": np.asarray(vid_shard, np.int64),
            "vid_lo": np.asarray(vid_lo, np.int64), "vid_hi": np.asarray(vid_hi, np.int64),
            "lab_off": np.asarray(lab_off, np.int64),
            "keys": np.concatenate(keys), "modify_type": np.concatenate(mtypes),
            "file": np.concatenate(files), "source": np.concatenate(sources)}


def identity_of(video_file: str) -> str:
    """VoxCeleb2 speaker identity of a video, the axis the train/held-out split is disjoint on."""
    return str(video_file).split("/")[0]


def split_indices(store: Dict, fraction: float = 0.15, seed: int = 0,
                  heldout_sources: Optional[Sequence[str]] = None):
    """Identity-disjoint split: ``(train indices, held-out indices, held-out identities)``.

    ``heldout_sources`` forces the speakers of the benchmark videos (the ones the detectors were
    scored on) into the held-out side, so the router is evaluated on speakers it never saw.
    """
    ids = np.asarray([identity_of(f) for f in store["file"]])
    uniq = np.asarray(sorted(set(ids.tolist())))
    rng = np.random.default_rng(seed)
    held = set(uniq[rng.permutation(len(uniq))[: max(1, int(len(uniq) * fraction))]].tolist())
    if heldout_sources:
        forced = set(str(s) for s in heldout_sources)
        held |= set(ids[np.isin(store["source"].astype(str), list(forced))].tolist())
    is_held = np.asarray([i in held for i in ids])
    return np.where(~is_held)[0], np.where(is_held)[0], len(held)


class FeatureDataset(data.Dataset):
    """One item = one video: features, labels, ``k`` budgets and their solved teacher tables."""

    def __init__(self, store: Dict, index: Optional[Sequence[int]] = None,
                 false_alarm: float = 0.25, seed: int = 0,
                 budgets_per_video: int = BUDGETS_PER_VIDEO, max_fraction: float = MAX_FRACTION,
                 rhos: Optional[Sequence[float]] = None):
        """``rhos`` switches to the fixed evaluation grid; otherwise budgets are sampled."""
        self.store = store
        self.index = list(range(len(store["keys"]))) if index is None else list(index)
        self.k = int(budgets_per_video)
        self.false_alarm = float(false_alarm)
        self.max_fraction = float(max_fraction)
        self.seed = int(seed)
        self.rhos = None if rhos is None else list(rhos)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Re-seed the budget draw so each epoch sees new budgets for the same video."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.index)

    def video(self, i: int):
        j = self.index[i]
        shard = self.store["shards"][int(self.store["vid_shard"][j])]
        feats = np.asarray(shard[int(self.store["vid_lo"][j]):int(self.store["vid_hi"][j])])
        lo = int(self.store["lab_off"][j])
        return feats, self.store["labels"][lo:lo + feats.shape[0]], str(self.store["keys"][j])

    def __getitem__(self, i: int) -> Dict:
        feats, lab, key = self.video(i)
        n = int(feats.shape[0])
        image_forged, audio_forged = lab[:, 0] > 0, lab[:, 1] > 0
        if self.rhos is not None:
            budgets = fixed_budgets(n, self.rhos)
        else:
            rng = np.random.default_rng((self.seed, self.epoch, self.index[i]))
            budgets = sample_budget(n, rng, n=self.k, max_fraction=self.max_fraction)
        reward = reward_matrix(audio_forged, image_forged, self.false_alarm)
        cost = uniform_costs(n)
        tables, best = [], []
        for b in budgets:
            sol = solve_knapsack(reward, cost, int(b))
            tables.append(sol.actions.astype(np.int8))           # (n, b + 1)
            best.append(sol.optimal_value)
        return {"feats": torch.from_numpy(feats.astype(np.float32)),
                "audio_forged": torch.from_numpy(audio_forged.astype(np.float32)),
                "image_forged": torch.from_numpy(image_forged.astype(np.float32)),
                "budgets": torch.from_numpy(budgets), "expert_tables": tables,
                "oracle_value": torch.tensor(best, dtype=torch.float32),
                "n_windows": n, "key": key}


def collate(batch: List[Dict]) -> Dict:
    """Pad to the longest video and flatten (video, budget) into the batch axis.

    Each video contributes ``k`` rows with the same features and different budgets. ``expert``
    holds the teacher's optimal action at every (window, remaining budget); -1 marks cells outside
    a row's own table.
    """
    k = len(batch[0]["budgets"])
    if any(len(b["budgets"]) != k for b in batch):
        raise ValueError("every item must carry the same number of budgets")
    n_max = max(b["n_windows"] for b in batch)
    b_max = max(int(b["budgets"].max()) for b in batch)
    rows, dim = len(batch) * k, batch[0]["feats"].shape[1]
    feats = torch.zeros(rows, n_max, dim)
    mask = torch.zeros(rows, n_max, dtype=torch.bool)
    af, imf = torch.zeros(rows, n_max), torch.zeros(rows, n_max)
    budgets, oracle = torch.zeros(rows), torch.zeros(rows)
    expert = torch.full((rows, n_max, b_max + 1), -1, dtype=torch.long)
    keys: List[str] = []
    for i, item in enumerate(batch):
        n = item["n_windows"]
        for j in range(k):
            r = i * k + j
            feats[r, :n] = item["feats"]
            mask[r, :n] = True
            af[r, :n] = item["audio_forged"]
            imf[r, :n] = item["image_forged"]
            budgets[r] = float(item["budgets"][j])
            oracle[r] = item["oracle_value"][j]
            table = item["expert_tables"][j]
            expert[r, :n, : table.shape[1]] = torch.from_numpy(table.astype(np.int64))
            keys.append(item["key"])
    return {"feats": feats, "mask": mask, "audio_forged": af, "image_forged": imf,
            "budgets": budgets, "oracle_value": oracle, "expert": expert, "keys": keys}
