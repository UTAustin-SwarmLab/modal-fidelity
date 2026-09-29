"""The baselines of Fig. 3, each spending the same budget B as the router.

* ``random``: windows in random order, a random affordable action each, until B is spent.
* ``uniform``: round-robin from the first window, cycling audio -> none -> image.
* ``audio_only`` / ``image_only`` / ``multimodal``: the router's own windows, with one fixed
  detector forced on each. The multimodal detector (AVH-Align) reads both streams and costs 2
  units, so fewer of the router's windows fit; they are taken in temporal order.
* ``oracle``: the exact knapsack on the true labels (``modalfidelity.router.oracle``).
* ``unbudgeted_gate``: the per-window gate the preview encoder is inherited from; no budget, its
  recorded actions are read from the benchmark.
"""
from __future__ import annotations

import numpy as np

from ..constants import AUDIO, BOTH, IMAGE, NONE

#: action code for the multimodal detector. It is not in the router's action space; it only
#: scores the forced-multimodal baseline.
MULTIMODAL = 4

METHODS = ("random", "uniform", "image_only", "audio_only", "multimodal", "router", "oracle",
           "unbudgeted_gate")


def random_actions(n: int, budget: int, cost: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random windows, a random affordable action each, until the budget runs out."""
    acts = np.zeros(n, dtype=np.int64)
    left = budget
    for t in rng.permutation(n):
        choices = [a for a in (AUDIO, IMAGE, BOTH) if cost[t, a] <= left]
        if not choices:
            break
        a = int(rng.choice(choices))
        acts[t] = a
        left -= cost[t, a]
    return acts


def uniform_actions(n: int, budget: int, cost: np.ndarray) -> np.ndarray:
    """Round-robin from the first window: audio, none, image, audio, ... until B is spent."""
    acts = np.zeros(n, dtype=np.int64)
    left = budget
    for t in range(n):
        a = (AUDIO, NONE, IMAGE)[t % 3]
        if a != NONE and cost[t, a] <= left:
            acts[t] = a
            left -= cost[t, a]
    return acts


def force_modality(router_actions: np.ndarray, action: int, budget: float,
                   unit_cost: float = 1.0) -> np.ndarray:
    """Keep the windows the router acquired and force one detector on each, in temporal order,
    while it fits in the budget."""
    out = np.zeros(len(router_actions), dtype=np.int64)
    left = float(budget)
    for t in np.where(np.asarray(router_actions) != NONE)[0]:
        if unit_cost > left:
            break
        out[t] = action
        left -= unit_cost
    return out


def predict(actions: np.ndarray, fired: dict) -> np.ndarray:
    """Window prediction: forged if any acquired stream's detector fired.

    ``fired`` maps AUDIO, IMAGE and MULTIMODAL to per-window booleans (score above tau^m).
    """
    a = np.asarray(actions)
    pred = np.zeros(len(a), dtype=bool)
    for act in (AUDIO, IMAGE):
        pred |= ((a == act) | (a == BOTH)) & fired[act]
    pred |= (a == MULTIMODAL) & fired[MULTIMODAL]
    return pred
