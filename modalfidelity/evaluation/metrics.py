"""Window-level detection metrics (paper Sec. 3, "Metrics").

A window is predicted forged if any stream the method acquired made its detector fire; a stream
that was never acquired counts as authentic. Accuracy is reported as the mean of recall and
specificity, so the 86.6 % authentic base rate cannot be won by never looking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class Confusion:
    """Running confusion counts, one entry per budget."""
    n_budgets: int
    tp: np.ndarray = field(init=False)
    fn: np.ndarray = field(init=False)
    fp: np.ndarray = field(init=False)
    tn: np.ndarray = field(init=False)

    def __post_init__(self):
        self.tp, self.fn, self.fp, self.tn = (np.zeros(self.n_budgets) for _ in range(4))

    def add(self, i: int, pred: np.ndarray, truth: np.ndarray) -> None:
        pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
        self.tp[i] += float((pred & truth).sum())
        self.fn[i] += float((~pred & truth).sum())
        self.fp[i] += float((pred & ~truth).sum())
        self.tn[i] += float((~pred & ~truth).sum())

    def rates(self) -> Dict[str, List[float]]:
        return rates(self.tp, self.fn, self.fp, self.tn)


def rates(tp, fn, fp, tn) -> Dict[str, List[float]]:
    """Plain accuracy, balanced accuracy (mean of recall and specificity), recall and FPR."""
    tp, fn, fp, tn = (np.asarray(x, dtype=np.float64) for x in (tp, fn, fp, tn))
    recall = tp / np.maximum(tp + fn, 1)
    fpr = fp / np.maximum(fp + tn, 1)
    total = np.maximum(tp + fn + fp + tn, 1)
    return {"accuracy": ((tp + tn) / total).tolist(),
            "balanced_accuracy": ((recall + 1 - fpr) / 2).tolist(),
            "recall": recall.tolist(), "fpr": fpr.tolist()}


def binary_rates(pred, truth) -> Dict[str, float]:
    """The same four numbers for one prediction vector."""
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    out = rates([(pred & truth).sum()], [(~pred & truth).sum()],
                [(pred & ~truth).sum()], [(~pred & ~truth).sum()])
    return {k: v[0] for k, v in out.items()}
