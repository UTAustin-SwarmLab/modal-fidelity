"""Table 1: where the gate goes. Late MoE gates run both detectors on every window; the router
decides before any detector runs."""
from __future__ import annotations

import json
import os

PAPER_RHOS = (0.20, 0.30, 0.50)


def load(data_dir: str) -> dict:
    return json.load(open(os.path.join(data_dir, "table1_gate_placement.json")))


def rows(t: dict) -> list:
    """``[(placement, variant, calls/window, GFLOPs/window, accuracy)]`` in the paper's order."""
    out = [("Late MoE" if k == 0 else "", pos, t["late_moe_calls_per_window"],
            t["late_moe_gflops_per_window"], t["late_moe"][pos]["balanced_accuracy"])
           for k, pos in enumerate(("dense", "feature", "output"))]
    r = t["router"]
    for k, rho in enumerate(PAPER_RHOS):
        i = r["rho"].index(rho)
        out.append(("Input (ours)" if k == 0 else "", f"rho={rho:.2f}", r["calls_per_window"][i],
                    r["gflops_per_window"][i], r["balanced_accuracy"][i]))
    return out


def as_markdown(t: dict) -> str:
    lines = ["| Placement | | Calls/win. (↓) | GFLOPs/win. (↓) | Acc. (↑) |",
             "|---|---|---:|---:|---:|"]
    best = max(r[4] for r in rows(t))
    for place, var, calls, gflops, acc in rows(t):
        a = f"{acc:.4f}".lstrip("0")
        lines.append(f"| {place} | {var.replace('rho', 'ρ')} | {calls:.3f} | {gflops:.1f} | "
                     f"{'**' + a + '**' if acc == best else a} |")
    return "\n".join(lines)
