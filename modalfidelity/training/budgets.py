"""Budgets sampled for training (paper Sec. 2, "Stage 1").

Each video gets **four budgets per epoch**, drawn log-uniformly over ``[1, ceil(0.6 T)]`` and
rounded to integers. Log-uniform puts as much mass on small budgets, where the budget binds, as
on large ones, and keeps ``B`` only weakly correlated with the video length ``T`` (0.37 against
0.69 for ``B = rho * T``), so the router has to read the budget rather than infer it from ``T``.
Because budgets are sampled per video, one router serves every budget.

Evaluation instead uses the fixed grid ``B = round(rho * T)`` for ``rho`` in ``constants.RHOS``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

#: budgets drawn per video per epoch
BUDGETS_PER_VIDEO = 4
#: largest budget drawn, as a fraction of the video's windows
MAX_FRACTION = 0.6


def sample_budget(n_windows: int, rng: np.random.Generator, n: int = BUDGETS_PER_VIDEO,
                  max_fraction: float = MAX_FRACTION, min_budget: int = 1) -> np.ndarray:
    """``n`` log-uniform integer budgets in ``[min_budget, ceil(max_fraction * n_windows)]``."""
    t = int(n_windows)
    if t < 1:
        raise ValueError(f"a video needs at least one window, got {t}")
    hi = max(int(min_budget), int(np.ceil(max_fraction * t)))
    draws = np.exp(rng.uniform(np.log(float(min_budget)), np.log(float(hi)), size=int(n)))
    return np.clip(np.round(draws), min_budget, t).astype(np.int64)


def fixed_budgets(n_windows: int, rhos: Sequence[float]) -> np.ndarray:
    """The evaluation grid ``B = round(rho * T)``, clipped to ``[1, T]``."""
    n = int(n_windows)
    return np.clip(np.round(np.asarray(rhos, float) * n), 1, n).astype(np.int64)
