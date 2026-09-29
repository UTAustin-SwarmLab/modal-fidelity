"""Shared look of the paper's figures: seaborn darkgrid, the Fig. 2 palette, TrueType fonts.

Colours follow the method, never its position, so a method has the same colour in every figure.
"""
from __future__ import annotations

#: display name of each method, in the paper's legend order
LABEL = {"random": "random", "uniform": "uniform (round-robin)",
         "image_only": "image detector only", "audio_only": "audio detector only",
         "multimodal": "multimodal", "router": "ModalFidelity (ours)",
         "unbudgeted_gate": "unbudgeted gate", "oracle": "oracle (upper bound)"}

#: the paper's colours (seaborn "colorblind" palette entries)
COLOR = {"random": "#0173b2", "uniform": "#de8f05", "image_only": "#029e73",
         "audio_only": "#d55e00", "multimodal": "#cc78bc", "router": "#ece133",
         "oracle": "#fbafe4", "unbudgeted_gate": "#949494"}


def use_paper_style(font_scale: float) -> None:
    """Non-interactive backend, TrueType (Type 42) fonts in PDFs, seaborn darkgrid."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42      # IEEE PDF checks reject Type 3 fonts
    matplotlib.rcParams["ps.fonttype"] = 42
    import seaborn as sns
    sns.set_theme(style="darkgrid", context="talk", font_scale=font_scale)


def legend_rows(n: int, ncol: int) -> list:
    """Index order that makes a multi-row legend read left to right, row by row
    (matplotlib fills legends column by column)."""
    nrow = -(-n // ncol)
    rows = [list(range(r * ncol, min((r + 1) * ncol, n))) for r in range(nrow)]
    return [rows[r][c] for c in range(ncol) for r in range(nrow) if c < len(rows[r])]


def save(fig, out_png: str, overwrite: bool = False) -> tuple:
    """Write ``out_png`` (160 dpi) and the matching PDF; refuse to overwrite unless asked."""
    import os
    pdf = os.path.splitext(out_png)[0] + ".pdf"
    for path in (out_png, pdf):
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return out_png, pdf
