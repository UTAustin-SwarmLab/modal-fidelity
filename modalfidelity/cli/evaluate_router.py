"""Evaluate a router checkpoint and the baselines with the real detectors (Fig. 3, Table 1).

    mf-evaluate-router --checkpoint checkpoints/router/seed0/router_final.pt --out outputs/eval/seed0

Writes ``<out>/detector_accuracy.json``: accuracy, balanced accuracy, recall and false-positive
rate per method and budget, plus detector calls and GFLOPs per window. Refuses to overwrite.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from ..constants import RHOS
from ..evaluation.detector_join import evaluate, load_benchmark
from ..evaluation.flops import load_cost_model
from ..paths import checkpoint, saved_data_dir


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="router checkpoint (.pt) holding the LSTM and head weights")
    p.add_argument("--features", default=checkpoint("preview_features"),
                   help="preview-feature cache directory (shard*_feats.npy + shard*_meta.npz); "
                        "default: $MODALFIDELITY_DATA/preview_features")
    p.add_argument("--benchmark", default=os.path.join(saved_data_dir(), "benchmark_windows.npz"),
                   help="per-window detector scores and labels of the benchmark videos")
    p.add_argument("--thresholds", default=os.path.join(saved_data_dir(), "benchmark_thresholds.json"),
                   help="JSON with the detectors' fixed operating points tau^m")
    p.add_argument("--rhos", type=float, nargs="+", default=list(RHOS),
                   help="budgets as a fraction rho of each video's windows")
    p.add_argument("--false-alarm", type=float, default=None,
                   help="reward penalty p for the oracle; default: the value stored in the checkpoint")
    p.add_argument("--seed", type=int, default=None,
                   help="seed of the random baseline; default: the router's training seed from the "
                        "checkpoint, which is what the paper's numbers use")
    p.add_argument("--cost-model", default=None,
                   help="optional JSON overriding GFLOPs per window: preview, audio, image, multimodal")
    p.add_argument("--max-videos", type=int, default=None,
                   help="evaluate only the first N matched videos (smoke tests)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="torch device for the router rollout")
    p.add_argument("--out", required=True, help="output directory (must not already hold results)")
    p.add_argument("--overwrite", action="store_true", help="allow replacing an existing result")
    a = p.parse_args(argv)

    dest = os.path.join(a.out, "detector_accuracy.json")
    if os.path.exists(dest) and not a.overwrite:
        raise SystemExit(f"{dest} exists; choose a new --out or pass --overwrite")
    blob = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    false_alarm = a.false_alarm if a.false_alarm is not None else float(blob.get("false_alarm", 0.25))
    seed = a.seed if a.seed is not None else int(blob.get("seed", 0))

    res = evaluate(a.checkpoint, a.features, load_benchmark(a.benchmark, a.thresholds),
                   false_alarm=false_alarm, seed=seed, rhos=a.rhos, device=a.device,
                   cost_model=load_cost_model(a.cost_model), max_videos=a.max_videos)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(dest, "w"), indent=1)

    print(f"{res['videos']} videos, {res['videos_skipped_misaligned']} skipped; balanced accuracy:")
    print(f"{'rho':>6} " + " ".join(f"{m:>15}" for m in res["balanced_accuracy"]))
    for i, rho in enumerate(res["rhos"]):
        print(f"{rho:>6.2f} " + " ".join(f"{res['balanced_accuracy'][m][i]:>15.4f}"
                                         for m in res["balanced_accuracy"]))
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
