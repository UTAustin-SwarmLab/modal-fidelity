"""Compute cost in GFLOPs per window (paper Table 1).

The call count treats every detector call as one unit. In FLOPs the two detectors differ by two
orders of magnitude, and the preview, free in call units, is paid on every window. The measured
cost of one 1 s window (``torch.utils.flop_counter``, which counts a multiply-add as 2 FLOPs):

    preview   1.53 GFLOPs  MobileNetV2 preview over 4 frames + a spectrogram, every window
    audio    35.97 GFLOPs  W2V2-AASIST on the window's 1 s of 16 kHz audio
    image  4050.60 GFLOPs  GenD (CLIP ViT-L/14) on the window's 25 frames
    multimodal 78.87 GFLOPs  AVH-Align: three AV-HuBERT passes + the fusion head

The preview and GenD numbers can be re-measured with ``mf-measure-flops``. The W2V2-AASIST and
AVH-Align numbers need those third-party detectors, which this repo cites but does not vendor.
"""
from __future__ import annotations

import json
from typing import Callable, Dict, Optional

import numpy as np

from ..constants import AUDIO, BOTH, IMAGE, WINDOW_FRAMES

#: measured GFLOPs per window (see module docstring)
GFLOPS_PER_WINDOW = {"preview": 1.533245184, "audio": 35.968037504, "image": 4050.5992192,
                     "multimodal": 78.871816448}


def load_cost_model(path: Optional[str] = None) -> Dict[str, float]:
    """The default cost model, optionally overridden by a JSON of the same keys."""
    model = dict(GFLOPS_PER_WINDOW)
    if path:
        model.update({k: float(v) for k, v in json.load(open(path)).items()})
    return model


def charge(actions: np.ndarray, cost_model: Dict[str, float], multimodal: bool = False):
    """Detector calls and GFLOPs for one video's actions; the preview is charged every window.

    With ``multimodal=True`` every acquired window is one AVH-Align call priced at 2 units, as in
    the multimodal baseline.
    """
    a = np.asarray(actions)
    n_windows = len(a)
    if multimodal:
        n = int((a != 0).sum())
        return 2.0 * n, n_windows * cost_model["preview"] + n * cost_model["multimodal"]
    n_audio = int(((a == AUDIO) | (a == BOTH)).sum())
    n_image = int(((a == IMAGE) | (a == BOTH)).sum())
    gflops = (n_windows * cost_model["preview"] + n_audio * cost_model["audio"]
              + n_image * cost_model["image"])
    return float(n_audio + n_image), gflops


def count_flops(fn: Callable[[], object]) -> float:
    """FLOPs of one forward pass. Raises instead of returning 0 for an untraceable module."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        fn()
    total = float(counter.get_total_flops())
    if total <= 0:
        raise RuntimeError("the FLOP counter saw no operations; the module is not traceable")
    return total


def measure_preview(device: str = "cpu") -> Dict[str, float]:
    """FLOPs of the preview encoder on one window (4 RGB frames + one spectrogram)."""
    import torch
    from ..router.preview import PreviewEncoder

    enc = PreviewEncoder().to(device).eval()
    rgb = torch.zeros(1, 12, 160, 160, device=device)     # 4 RGB frames, channels stacked
    spec = torch.zeros(1, 1, 256, 256, device=device)     # one log-mel spectrogram
    f = count_flops(lambda: enc.features([rgb, spec]))
    return {"flops_per_call": f, "calls_per_window": 1.0, "flops_per_window": f}


def measure_image_detector(gend_repo: str, model_id: str = "yermandy/GenD_CLIP_L_14",
                           device: str = "cpu") -> Dict[str, float]:
    """FLOPs of GenD on one 224x224 frame, times the 25 frames of a window."""
    import sys
    import torch

    sys.path.insert(0, gend_repo)
    from src.hf.modeling_gend import GenD  # the GenD repository's HF wrapper

    model = GenD.from_pretrained(model_id).to(device).eval()
    img = torch.zeros(1, 3, 224, 224, device=device)
    f = count_flops(lambda: model(img))
    return {"flops_per_call": f, "calls_per_window": float(WINDOW_FRAMES),
            "flops_per_window": f * WINDOW_FRAMES}
