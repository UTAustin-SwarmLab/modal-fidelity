"""The router's reward R (paper Eq. 2): finding a forgery pays, abstaining pays nothing.

Each stream an action inspects scores on its own: +1 if that stream is forged in the window,
-p if it is authentic. Acquiring nothing earns 0, so the budget is a ceiling, not a target.

| window truth | none | audio | image | both   |
|--------------|------|-------|-------|--------|
| authentic    | 0    | -p    | -p    | -2p    |
| audio forged | 0    | +1    | -p    | +1 - p |
| image forged | 0    | -p    | +1    | +1 - p |
| both forged  | 0    | +1    | +1    | +2     |

The reward uses the per-window, per-stream labels, never detector outputs, so the router learns
where forgeries are and a different detector can be plugged in without retraining.
"""
from __future__ import annotations

import numpy as np

from ..constants import ACTIONS, AUDIO, BOTH, IMAGE, NONE


def reward_matrix(audio_forged, image_forged, false_alarm: float) -> np.ndarray:
    """``reward[t, a]`` for one video, shape ``(T, 4)``."""
    a = np.asarray(audio_forged, dtype=bool).reshape(-1)
    i = np.asarray(image_forged, dtype=bool).reshape(-1)
    if a.shape != i.shape:
        raise ValueError(f"audio {a.shape} and image {i.shape} labels differ in length")
    audio_term = np.where(a, 1.0, -float(false_alarm))
    image_term = np.where(i, 1.0, -float(false_alarm))
    out = np.zeros((a.size, len(ACTIONS)), dtype=np.float64)
    out[:, AUDIO] = audio_term
    out[:, IMAGE] = image_term
    out[:, BOTH] = audio_term + image_term
    return out


def reward_table_torch(stream_forged, false_alarm: float):
    """Torch version of :func:`reward_matrix` for training.

    ``stream_forged`` has a last dimension ordered **(image, audio)**, the order the feature
    cache stores the labels in.
    """
    import torch

    image = stream_forged[..., 0] > 0
    audio = stream_forged[..., 1] > 0
    p = float(false_alarm)
    one = torch.ones_like(audio, dtype=torch.float32)
    audio_term = torch.where(audio, one, torch.full_like(one, -p))
    image_term = torch.where(image, one, torch.full_like(one, -p))
    return torch.stack([torch.zeros_like(audio_term), audio_term, image_term,
                        audio_term + image_term], dim=-1)


def achieved_reward(actions, audio_forged, image_forged, false_alarm: float) -> float:
    """Total reward collected by a realised action sequence."""
    r = reward_matrix(audio_forged, image_forged, false_alarm)
    acts = np.asarray(actions, dtype=np.int64).reshape(-1)
    if acts.size != r.shape[0]:
        raise ValueError(f"{acts.size} actions for {r.shape[0]} windows")
    return float(r[np.arange(acts.size), acts].sum())


__all__ = ["reward_matrix", "reward_table_torch", "achieved_reward", "NONE"]
