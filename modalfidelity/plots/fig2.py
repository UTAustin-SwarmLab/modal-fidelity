"""Fig. 2: per-window confidence of each method by forgery type, with no budget cap.

Every window is affordable, but at most one stream per window is read. For each method the
plotted value is the margin ``s - tau`` of the detector it read, rescaled per detector so its
strongest positive evidence maps to +1 and its strongest negative evidence to -1; 0 stays the
decision boundary. The unbudgeted gate is plotted on the windows it was scored on.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .style import COLOR, LABEL, legend_rows, save, use_paper_style

METHODS = ["random", "uniform", "image_only", "audio_only", "multimodal", "unbudgeted_gate",
           "oracle"]
GROUPS = [("audio only", lambda a, i: a & ~i), ("image only", lambda a, i: i & ~a),
          ("bimodal", lambda a, i: a & i), ("clean", lambda a, i: ~a & ~i)]
#: which detector an action reads: 1 audio (W2V2-AASIST), 2 image (GenD), 3 multimodal (AVH-Align)
DETECTOR_OF_ACTION = {1: "aasist", 2: "gend", 3: "avhalign"}
BOUNDARY = "#a50026"


def palette() -> dict:
    """Fig. 2 colours: entries 0, 1, 2, 3, 4 and 6 of seaborn's "colorblind" palette, with the
    unbudgeted gate (a baseline) in neutral grey."""
    import seaborn as sns
    cb = sns.color_palette("colorblind", 10)
    return {"random": cb[0], "uniform": cb[1], "image_only": cb[2], "audio_only": cb[3],
            "multimodal": cb[4], "unbudgeted_gate": (0.58, 0.58, 0.58), "oracle": cb[6]}


def load(data_dir: str):
    d = dict(np.load(os.path.join(data_dir, "benchmark_windows.npz"), allow_pickle=False))
    thr = json.load(open(os.path.join(data_dir, "benchmark_thresholds.json")))["thresholds"]
    return d, thr


def rescaled_margins(d, thr):
    """``{(group, method): values}``: per-window margins, rescaled per detector and side."""
    pos, neg = {}, {}
    for act, det in DETECTOR_OF_ACTION.items():
        margin = d[f"score_{det}"] - thr[det]
        pos[act], neg[act] = float(margin.max()), float(abs(margin.min()))

    a, i = d["audio_forged"], d["image_forged"]
    out = {}
    for gname, member in GROUPS:
        in_group = member(a, i)
        for m in METHODS:
            act, margin = d[f"{m}__action"], d[f"{m}__margin"]
            if m == "multimodal":
                act = np.full(len(act), 3)
            read = np.ones(len(act), bool) if m == "multimodal" else act > 0
            if m == "unbudgeted_gate":
                read = read & d["unbudgeted_gate_matched"]
            sel = read & in_group
            vals, acts = margin[sel], act[sel]
            p = np.array([pos.get(int(x), 1.0) for x in acts])
            n = np.array([neg.get(int(x), 1.0) for x in acts])
            out[(gname, m)] = np.where(vals >= 0, vals / p, vals / n)
    return out


def medians(values) -> dict:
    return {k: float(np.median(v)) for k, v in values.items() if len(v)}


def plot(data_dir: str, out_png: str, overwrite: bool = False):
    import pandas as pd
    use_paper_style(font_scale=3 * 0.75 * 2.0)
    import matplotlib.pyplot as plt
    import seaborn as sns

    d, thr = load(data_dir)
    values = rescaled_margins(d, thr)
    rows = [{"group": g, "method": LABEL[m], "margin": v}
            for g, _ in GROUPS for m in METHODS for v in values[(g, m)]]
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(26, 22))
    sns.boxplot(data=df, x="group", y="margin", hue="method", order=[g for g, _ in GROUPS],
                hue_order=[LABEL[m] for m in METHODS], palette=[palette()[m] for m in METHODS],
                showfliers=False, saturation=1, linewidth=1.1, ax=ax)
    ax.axhline(0, xmin=0, xmax=1, color=BOUNDARY, linestyle="--", linewidth=3.2, zorder=3)
    ax.text(1.008, 0, "decision boundary", transform=ax.get_yaxis_transform(), rotation=90,
            va="center", ha="left", fontsize=3 * 11.5 * 2.0, color=BOUNDARY, clip_on=False)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([g for g, _ in GROUPS])
    ax.tick_params(axis="x", labelsize=60)
    ax.set_xlabel("Forgery Type", fontweight="bold")
    ax.set_ylabel("Confidence Score", fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    order = legend_rows(len(handles), ncol=2)
    if ax.legend_:
        ax.legend_.remove()
    fig.legend([handles[k] for k in order], [labels[k] for k in order], title="",
               loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False,
               fontsize=3 * 12 * 2.0, handlelength=2.2, columnspacing=1.6)
    fig.subplots_adjust(left=0.13, right=0.93, top=0.70, bottom=0.13)
    paths = save(fig, out_png, overwrite)
    plt.close(fig)
    return paths
