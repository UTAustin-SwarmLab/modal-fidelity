"""Every number the paper states about Figs. 2-3 and Table 1, recomputed from ``data/``.

:func:`check` returns one row per claim: what the paper says, the value recomputed here, and
whether the paper's rounding of the recomputed value matches.
"""
from __future__ import annotations

import numpy as np

from . import fig2, fig3, table1


def _balanced(pred, label) -> float:
    return float(((pred & label).sum() / label.sum() + (~pred & ~label).sum() / (~label).sum()) / 2)


def compute(data_dir: str) -> dict:
    out = {}
    # ---- Fig. 3 (mean of 4 seeds, balanced accuracy)
    runs = fig3.load(data_dir)
    rhos = runs[0]["rhos"]
    c = {m: v[0] for m, v in fig3.curves(runs).items()}
    s1 = {m: v[0] for m, v in fig3.curves(fig3.load(data_dir, "budget_curves_stage1")).items()}
    i20, i30 = rhos.index(0.2), rhos.index(0.3)
    out["max gap to oracle (pp)"] = float(((c["oracle"] - c["router"]) * 100).max())
    out["min share of oracle accuracy"] = float((c["router"] / c["oracle"]).min())
    out["router accuracy at rho=0.30"] = float(c["router"][i30])
    out["gain over audio-only at rho=0.20 (pp)"] = float((c["router"] - c["audio_only"])[i20] * 100)
    out["gain over image-only at rho=0.20 (pp)"] = float((c["router"] - c["image_only"])[i20] * 100)
    gain = (c["router"] - s1["router"]) * 100
    out["stage-2 gain, min over budgets (pp)"] = float(gain.min())
    out["stage-2 gain, max over budgets (pp)"] = float(gain.max())
    # ---- Fig. 2 (window level, no budget cap)
    d, thr = fig2.load(data_dir)
    y = d["any_forged"]
    for m in ("unbudgeted_gate", "audio_only", "image_only", "multimodal", "oracle"):
        act, margin = d[f"{m}__action"], d[f"{m}__margin"]
        pred = margin > 0 if m == "multimodal" else (act > 0) & (margin > 0)
        sel = d["unbudgeted_gate_matched"] if m == "unbudgeted_gate" else np.ones(len(y), bool)
        out[f"Fig. 2 accuracy, {m}"] = _balanced(pred[sel], y[sel])
    out["router at rho=0.30 below the unbudgeted gate (pp)"] = (
        (out["Fig. 2 accuracy, unbudgeted_gate"] - out["router accuracy at rho=0.30"]) * 100)
    med = fig2.medians(fig2.rescaled_margins(d, thr))
    out["Fig. 2 median, audio edits, gate"] = med[("audio only", "unbudgeted_gate")]
    out["Fig. 2 median, audio edits, image detector"] = med[("audio only", "image_only")]
    out["Fig. 2 median, image edits, gate"] = med[("image only", "unbudgeted_gate")]
    out["Fig. 2 median, image edits, audio detector"] = med[("image only", "audio_only")]
    # ---- Table 1
    t = table1.load(data_dir)
    r = t["router"]
    j20, j50 = r["rho"].index(0.2), r["rho"].index(0.5)
    late = [v["balanced_accuracy"] for v in t["late_moe"].values()]
    out["GFLOPs ratio late/router at rho=0.20"] = t["late_moe_gflops_per_window"] / r["gflops_per_window"][j20]
    out["calls ratio late/router at rho=0.20"] = t["late_moe_calls_per_window"] / r["calls_per_window"][j20]
    out["GFLOPs ratio late/router at rho=0.50"] = t["late_moe_gflops_per_window"] / r["gflops_per_window"][j50]
    out["accuracy gain over best late gate at rho=0.50 (pp)"] = (r["balanced_accuracy"][j50] - max(late)) * 100
    out["spread among late gates (pp)"] = (max(late) - min(late)) * 100
    return out


#: (claim, paper's wording, decimals the paper rounds to, how the paper bounds it)
PAPER = {
    "max gap to oracle (pp)": ("within 2.6 pp of the oracle at every budget", 1, "round"),
    "min share of oracle accuracy": ("retains over 96% of the oracle's accuracy", 0.96, "above"),
    "router accuracy at rho=0.30": ("0.815 at rho=0.30", 3, "round"),
    "router at rho=0.30 below the unbudgeted gate (pp)": ("within 0.4 pp of the 0.819 of the unbudgeted gate", 1, "round"),
    "gain over audio-only at rho=0.20 (pp)": ("6.6 pp", 1, "round"),
    "gain over image-only at rho=0.20 (pp)": ("8.1 pp", 1, "round"),
    "stage-2 gain, min over budgets (pp)": ("1.9 pp", 1, "round"),
    "stage-2 gain, max over budgets (pp)": ("6.5 pp", 1, "round"),
    "Fig. 2 accuracy, unbudgeted_gate": ("0.819", 3, "round"),
    "Fig. 2 accuracy, audio_only": ("0.666", 3, "round"),
    "Fig. 2 accuracy, image_only": ("0.635", 3, "round"),
    "Fig. 2 accuracy, multimodal": ("0.718", 3, "round"),
    "Fig. 2 accuracy, oracle": ("0.839", 3, "round"),
    "Fig. 2 median, audio edits, gate": ("+0.44", 2, "round"),
    "Fig. 2 median, audio edits, image detector": ("-0.25", 2, "round"),
    "Fig. 2 median, image edits, gate": ("+0.18", 2, "round"),
    "Fig. 2 median, image edits, audio detector": ("-0.15", 2, "round"),
    "GFLOPs ratio late/router at rho=0.20": ("15.9x", 1, "round"),
    "calls ratio late/router at rho=0.20": ("15.4x", 1, "round"),
    "GFLOPs ratio late/router at rho=0.50": ("10.6x", 1, "round"),
    "accuracy gain over best late gate at rho=0.50 (pp)": ("11 pp", 0, "round"),
    "spread among late gates (pp)": ("under 0.6 pp", 0.6, "below"),
}


def _paper_value(text: str) -> float:
    import re
    return float(re.findall(r"[-+]?\d*\.?\d+", text.replace("+", ""))[0])


def check(data_dir: str) -> list:
    """``[(claim, paper wording, recomputed, ok)]``."""
    got = compute(data_dir)
    rows = []
    for claim, (text, rule, kind) in PAPER.items():
        v = got[claim]
        if kind == "above":
            ok = v > rule
        elif kind == "below":
            ok = v < rule
        else:
            ok = round(v, rule) == _paper_value(text)
        rows.append((claim, text, v, bool(ok)))
    return rows
