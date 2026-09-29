"""Fig. 3: detector-scored accuracy against the budget fraction rho, mean of four router seeds.

Lines are the mean over seeds; the shaded band is the min-max range over seeds (visible only for
the router, the one method that differs between seeds). Accuracy is the mean of recall and
specificity over windows.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from .style import COLOR, LABEL, legend_rows, save, use_paper_style

METHODS = ["random", "uniform", "image_only", "audio_only", "multimodal", "router", "oracle"]
#: (line width, z-order) of each method, before the x3 line scale
WIDTH_Z = {"oracle": (2.2, 5), "router": (3.2, 6), "audio_only": (2.2, 4),
           "image_only": (2.2, 4), "multimodal": (2.2, 4), "random": (2.0, 3),
           "uniform": (2.0, 3)}
#: one dash pattern per family: solid (ours, oracle), dotted (blind), dash-dot (one detector)
DASH = {"router": (), "oracle": (), "random": (1, 2), "uniform": (1, 2),
        "image_only": (6, 2, 1.5, 2), "audio_only": (6, 2, 1.5, 2),
        "multimodal": (6, 2, 1.5, 2)}
LINE_SCALE = 3.0


def load(data_dir: str, subdir: str = "budget_curves"):
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(data_dir, subdir, "*.json")))]
    if not runs:
        raise FileNotFoundError(f"no seed files in {os.path.join(data_dir, subdir)}")
    return runs


def curves(runs, metric: str = "balanced_accuracy") -> dict:
    """``{method: (mean, min, max)}`` over seeds, each an array over rho."""
    out = {}
    for m in METHODS:
        arr = np.array([r[metric][m] for r in runs])
        out[m] = (arr.mean(0), arr.min(0), arr.max(0))
    return out


def plot(data_dir: str, out_png: str, overwrite: bool = False):
    use_paper_style(font_scale=0.85 * 3.0)
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    runs = load(data_dir)
    rhos = runs[0]["rhos"]
    stats = curves(runs)
    fig, ax = plt.subplots(figsize=(20, 15))
    handles = []
    for m in METHODS:
        mean, lo, hi = stats[m]
        lw, z = WIDTH_Z[m]
        ours = m == "router"
        ln, = ax.plot(rhos, mean, color=COLOR[m], linewidth=lw * LINE_SCALE, zorder=z,
                      marker="o", markersize=6 * LINE_SCALE * (1 if ours else 1.25),
                      markeredgecolor="black" if ours else "white",
                      markeredgewidth=1.1 * LINE_SCALE)
        if ours:   # yellow needs an outline on the light background
            ax.plot(rhos, mean, color="black", linewidth=lw * LINE_SCALE + 3.5, zorder=z - 0.5,
                    solid_capstyle="round")
        if DASH[m]:
            ln.set_dashes(DASH[m])
        if (hi - lo).max() > 1e-9:
            ax.fill_between(rhos, lo, hi, color=COLOR[m], alpha=0.13, zorder=z - 1, linewidth=0)
        fx = ([pe.Stroke(linewidth=lw * LINE_SCALE + 3.5, foreground="black"), pe.Normal()]
              if ours else None)
        handles.append((plt.Line2D([], [], color=COLOR[m], lw=lw * LINE_SCALE,
                                   dashes=DASH[m] or (1, 0), path_effects=fx), LABEL[m]))
    ax.set_xlabel("Budget Fraction (ρ)", fontweight="bold")
    ax.set_ylabel("Accuracy", fontweight="bold")
    ax.set_xlim(0.03, 0.52)
    ax.set_xticks(rhos)
    ax.set_xticklabels([f"{r:g}" for r in rhos])
    order = legend_rows(len(handles), ncol=3)
    fig.legend([handles[k][0] for k in order], [handles[k][1] for k in order],
               loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False,
               fontsize=32, handlelength=2.0, columnspacing=1.0)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.83, bottom=0.12)
    paths = save(fig, out_png, overwrite)
    plt.close(fig)
    return paths
