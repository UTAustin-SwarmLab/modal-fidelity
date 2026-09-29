"""Measure per-window GFLOPs of the preview encoder and the image detector.

    mf-measure-flops --models preview --out outputs/flops

Counts with ``torch.utils.flop_counter`` (a multiply-add is 2 FLOPs). The preview is measured on
one window (4 RGB frames + one spectrogram); GenD on one 224x224 frame, times the window's 25
frames. The audio (W2V2-AASIST) and multimodal (AVH-Align) costs need those third-party detectors
and are taken from the measured values in ``modalfidelity.evaluation.flops``. Writes one
``<model>.json`` per model; refuses to overwrite.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from ..evaluation.flops import measure_image_detector, measure_preview


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=["preview", "image"], default=["preview"],
                   help="what to measure: the preview encoder and/or the GenD image detector")
    p.add_argument("--gend-repo", default=None,
                   help="path to a clone of the GenD repository (needed for --models image)")
    p.add_argument("--gend-model", default="yermandy/GenD_CLIP_L_14",
                   help="Hugging Face id of the GenD checkpoint")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="torch device")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--overwrite", action="store_true", help="allow replacing existing files")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for name in a.models:
        dest = os.path.join(a.out, f"{name}.json")
        if os.path.exists(dest) and not a.overwrite:
            raise SystemExit(f"{dest} exists; choose a new --out or pass --overwrite")
        if name == "preview":
            rec = measure_preview(a.device)
        else:
            if not a.gend_repo:
                raise SystemExit("--models image needs --gend-repo")
            rec = measure_image_detector(a.gend_repo, a.gend_model, a.device)
        rec.update(model=name, counter="torch FlopCounterMode (2 FLOPs per multiply-add)")
        json.dump(rec, open(dest, "w"), indent=1)
        print(f"{name}: {rec['flops_per_window'] / 1e9:.3f} GFLOPs per window -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
