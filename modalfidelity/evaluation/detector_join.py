"""Detector-scored evaluation of the router and the baselines (Fig. 3, Table 1).

The router is trained on labels, never on detector outputs. This evaluation closes the gap: for
each benchmark video it takes the actions a method chooses under budget ``B = round(rho * T)``,
reads the real detectors' scores on the acquired streams, thresholds them at the fixed operating
points tau^m, and scores the window predictions against the window labels.

Two inputs are joined by window index:

* the **benchmark** (``data/benchmark_windows.npz``): per-window detector scores, labels and the
  unbudgeted gate's recorded actions for the 1,999 held-out benchmark videos;
* the **preview-feature cache**: the router's input for the same videos on the same 1 s / 0.24 s
  window grid.

The alignment is checked per video; a video whose window counts differ is skipped and counted.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from ..baselines.fixed import (MULTIMODAL, force_modality, predict, random_actions,
                               uniform_actions)
from ..constants import AUDIO, IMAGE, RHOS
from ..paths import saved_data_dir
from ..router.oracle import solve_knapsack, uniform_costs
from ..router.policy import load_router
from ..router.reward import reward_matrix
from .flops import charge, load_cost_model
from .metrics import Confusion

REPORTED = ("random", "uniform", "image_only", "audio_only", "multimodal", "router", "oracle",
            "unbudgeted_gate")


def load_benchmark(path: Optional[str] = None, thresholds: Optional[str] = None) -> Dict:
    """Benchmark windows grouped by video, with per-window detector decisions."""
    path = path or os.path.join(saved_data_dir(), "benchmark_windows.npz")
    thresholds = thresholds or os.path.join(saved_data_dir(), "benchmark_thresholds.json")
    d = dict(np.load(path, allow_pickle=False))
    thr = json.load(open(thresholds))["thresholds"]
    key = np.char.add(np.char.add(d["source"].astype(str), "|"), d["variant"].astype(str))
    order = np.argsort(d["start_frame"], kind="stable")
    by_video = {str(k): order[key[order] == k] for k in np.unique(key)}
    fired = {AUDIO: d["score_aasist"] > thr["aasist"], IMAGE: d["score_gend"] > thr["gend"],
             MULTIMODAL: d["score_avhalign"] > thr["avhalign"]}
    gate = np.where(d["unbudgeted_gate_matched"], d["unbudgeted_gate__action"], 0)
    return {"by_video": by_video, "fired": fired, "any_forged": d["any_forged"].astype(bool),
            "unbudgeted_gate": gate}


def load_feature_cache(cache_dir: str) -> Dict:
    """Memory-map the preview-feature shards (``shard*_feats.npy`` + ``shard*_meta.npz``).

    Each ``_meta.npz`` holds per-video ``offsets`` into its shard, per-window ``labels`` ordered
    (image, audio), and the video's ``source`` and ``modify_type``.
    """
    stems = sorted(glob.glob(os.path.join(cache_dir, "shard*_feats.npy")))
    if not stems:
        raise FileNotFoundError(f"no feature shards under {cache_dir}")
    shards, labels, sources, mtypes, vid_shard, lo, hi = [], [], [], [], [], [], []
    for s, path in enumerate(stems):
        shards.append(np.load(path, mmap_mode="r"))
        with np.load(path.replace("_feats.npy", "_meta.npz"), allow_pickle=False) as meta:
            off = meta["offsets"]
            labels.append(meta["labels"])
            sources.append(meta["source"])
            mtypes.append(meta["modify_type"])
        vid_shard += [s] * (len(off) - 1)
        lo += off[:-1].tolist()
        hi += off[1:].tolist()
    lab_off = np.cumsum([0] + [h - l for l, h in zip(lo, hi)])
    return {"shards": shards, "labels": np.concatenate(labels), "lab_off": lab_off,
            "vid_shard": np.asarray(vid_shard), "lo": np.asarray(lo), "hi": np.asarray(hi),
            "source": np.concatenate(sources).astype(str),
            "modify_type": np.concatenate(mtypes).astype(str)}


def evaluate(checkpoint: str, cache_dir: str, bench: Dict, false_alarm: float,
             seed: int, rhos: Sequence[float] = RHOS, device: str = "cpu",
             cost_model: Optional[Dict[str, float]] = None, max_videos: Optional[int] = None) -> Dict:
    """Run every method at every budget and return per-method metrics and costs.

    ``seed`` seeds the random baseline (one generator for the whole pass, drawn video by video,
    budget by budget, in cache order), so results are reproducible exactly.
    """
    router = load_router(checkpoint, device)
    store = load_feature_cache(cache_dir)
    cost_model = cost_model or load_cost_model()
    conf = {m: Confusion(len(rhos)) for m in REPORTED}
    calls = {m: np.zeros(len(rhos)) for m in REPORTED}
    gflops = {m: np.zeros(len(rhos)) for m in REPORTED}
    windows = np.zeros(len(rhos))
    rng = np.random.default_rng(seed)
    matched = skipped = 0

    for v in range(len(store["source"])):
        k = f"{store['source'][v]}|{store['modify_type'][v]}"
        if k not in bench["by_video"]:
            continue
        idx = bench["by_video"][k]
        n = int(store["hi"][v] - store["lo"][v])
        if len(idx) != n:
            skipped += 1
            continue
        if max_videos is not None and matched >= max_videos:
            break
        matched += 1
        feats = np.asarray(store["shards"][int(store["vid_shard"][v])][store["lo"][v]:store["hi"][v]])
        lab = store["labels"][store["lab_off"][v]:store["lab_off"][v] + n]
        image_f, audio_f = lab[:, 0] > 0, lab[:, 1] > 0
        fired = {a: f[idx] for a, f in bench["fired"].items()}
        truth = bench["any_forged"][idx]
        cost = uniform_costs(n)
        true_reward = reward_matrix(audio_f, image_f, false_alarm)

        for bi, rho in enumerate(rhos):
            budget = int(max(1, round(rho * n)))
            with torch.no_grad():
                f = torch.from_numpy(feats.astype(np.float32))[None].to(device)
                r_acts = router.rollout(f, torch.tensor([float(budget)], device=device))[
                    "actions"][0].cpu().numpy()
            acts = {"oracle": solve_knapsack(true_reward, cost, budget).rollout()[0],
                    "router": r_acts,
                    "random": random_actions(n, budget, cost, rng),
                    "uniform": uniform_actions(n, budget, cost),
                    "audio_only": force_modality(r_acts, AUDIO, budget, 1.0),
                    "image_only": force_modality(r_acts, IMAGE, budget, 1.0),
                    "multimodal": force_modality(r_acts, MULTIMODAL, budget, 2.0),
                    "unbudgeted_gate": bench["unbudgeted_gate"][idx]}
            for m in REPORTED:
                conf[m].add(bi, predict(acts[m], fired), truth)
                c, g = charge(acts[m], cost_model, multimodal=(m == "multimodal"))
                calls[m][bi] += c
                gflops[m][bi] += g
            windows[bi] += n

    out = {"checkpoint": os.path.basename(os.path.dirname(os.path.abspath(checkpoint)))
                         + "/" + os.path.basename(checkpoint),
           "rhos": list(rhos), "videos": matched, "videos_skipped_misaligned": skipped,
           "windows": windows.tolist(), "false_alarm": false_alarm, "seed": seed}
    per = {m: conf[m].rates() for m in REPORTED}
    for metric in ("accuracy", "balanced_accuracy", "recall", "fpr"):
        out[metric] = {m: per[m][metric] for m in REPORTED}
    out["calls_per_window"] = {m: (calls[m] / np.maximum(windows, 1)).tolist() for m in REPORTED}
    out["gflops_per_window"] = {m: (gflops[m] / np.maximum(windows, 1)).tolist() for m in REPORTED}
    out["cost_model_gflops_per_window"] = cost_model
    return out
