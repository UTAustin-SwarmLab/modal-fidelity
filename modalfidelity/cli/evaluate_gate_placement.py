"""Build Table 1: gating before the detectors (the router) against after them (late MoE).

    mf-evaluate-gate-placement --late-moe outputs/late_moe/*/best.pt --embeddings <emb_dir> \
        --router-eval outputs/eval/seed0/detector_accuracy.json --out outputs/table1

Late MoE rows: both detectors on every window, a window predicted forged if either output
exceeds 0.5, balanced accuracy over the benchmark windows averaged over seeds per gate position.
Router rows: balanced accuracy, detector calls and GFLOPs per window from an ``mf-evaluate-router``
result. Writes ``<out>/table1_gate_placement.json`` in the same format as
``data/table1_gate_placement.json``; refuses to overwrite.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from ..baselines.late_moe import load_embeddings, load_late_moe, video_embeddings
from ..evaluation.flops import load_cost_model
from ..evaluation.metrics import binary_rates
from ..paths import checkpoint, saved_data_dir


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--late-moe", nargs="+", required=True,
                   help="late-MoE checkpoints (any mix of gate positions and seeds)")
    p.add_argument("--embeddings", default=checkpoint("detector_embeddings"),
                   help="detector-embedding cache covering the benchmark videos")
    p.add_argument("--router-eval", required=True,
                   help="detector_accuracy.json written by mf-evaluate-router")
    p.add_argument("--benchmark", default=os.path.join(saved_data_dir(), "benchmark_windows.npz"),
                   help="per-window labels of the benchmark videos")
    p.add_argument("--cost-model", default=None,
                   help="optional JSON overriding GFLOPs per window: preview, audio, image, multimodal")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="torch device")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--overwrite", action="store_true", help="allow replacing an existing table")
    a = p.parse_args(argv)
    dest = os.path.join(a.out, "table1_gate_placement.json")
    if os.path.exists(dest) and not a.overwrite:
        raise SystemExit(f"{dest} exists; choose a new --out or pass --overwrite")

    d = np.load(a.benchmark, allow_pickle=False)
    key = np.char.add(np.char.add(d["source"].astype(str), "|"), d["variant"].astype(str))
    order = np.argsort(d["start_frame"], kind="stable")
    by_video = {str(k): order[key[order] == k] for k in np.unique(key)}
    store = load_embeddings(a.embeddings)
    emb_index = {f"{s}|{t}": v for v, (s, t) in enumerate(zip(store["source"], store["modify_type"]))}
    shared = sorted(k for k in by_video if k in emb_index
                    and store["hi"][emb_index[k]] - store["lo"][emb_index[k]] == len(by_video[k]))
    truth = np.concatenate([d["any_forged"][by_video[k]] for k in shared]).astype(bool)

    per_pos = defaultdict(list)
    for path in a.late_moe:
        model = load_late_moe(path, a.device)
        preds = []
        for k in shared:
            ea, ei = video_embeddings(store, emb_index[k])
            with torch.no_grad():
                pr = torch.sigmoid(model(torch.from_numpy(ea).to(a.device),
                                         torch.from_numpy(ei).to(a.device))[0]).cpu().numpy()
            preds.append((pr[:, 0] > 0.5) | (pr[:, 1] > 0.5))
        bal = binary_rates(np.concatenate(preds), truth)["balanced_accuracy"]
        per_pos[model.gate_at].append(bal)
        print(f"{os.path.relpath(path)}: {model.gate_at} balanced {bal:.4f}", flush=True)

    cm = load_cost_model(a.cost_model)
    r = json.load(open(a.router_eval))
    table = {"videos": len(shared), "windows": int(len(truth)),
             "gflops_per_window": {"preview": cm["preview"], "audio_detector": cm["audio"],
                                   "image_detector": cm["image"]},
             "late_moe": {pos: {"balanced_accuracy": float(np.mean(v)), "n_seeds": len(v)}
                          for pos, v in sorted(per_pos.items())},
             "late_moe_calls_per_window": 2.0,
             "late_moe_gflops_per_window": cm["audio"] + cm["image"],
             "router": {"rho": r["rhos"], "balanced_accuracy": r["balanced_accuracy"]["router"],
                        "calls_per_window": r["calls_per_window"]["router"],
                        "gflops_per_window": r["gflops_per_window"]["router"],
                        "seed": r["seed"], "false_alarm": r["false_alarm"]}}
    os.makedirs(a.out, exist_ok=True)
    json.dump(table, open(dest, "w"), indent=1)
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
