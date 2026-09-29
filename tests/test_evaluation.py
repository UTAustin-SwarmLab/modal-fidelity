"""Metrics, the Fig. 3 baselines and the late-MoE model."""
import numpy as np
import pytest
import torch

from modalfidelity.baselines.fixed import (MULTIMODAL, force_modality, predict, random_actions,
                                           uniform_actions)
from modalfidelity.constants import AUDIO, BOTH, IMAGE, NONE
from modalfidelity.evaluation.metrics import binary_rates
from modalfidelity.router.oracle import uniform_costs


def test_balanced_accuracy_is_mean_of_recall_and_specificity():
    truth = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], bool)
    pred = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0], bool)
    r = binary_rates(pred, truth)
    assert r["recall"] == pytest.approx(0.75)
    assert r["fpr"] == pytest.approx(1 / 6)
    assert r["balanced_accuracy"] == pytest.approx((0.75 + 5 / 6) / 2)
    assert binary_rates(np.zeros(10, bool), truth)["balanced_accuracy"] == pytest.approx(0.5)


@pytest.mark.parametrize("seed", range(10))
def test_blind_baselines_never_exceed_the_budget(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 60))
    cost = uniform_costs(n)
    for budget in (0, 1, 3, n // 2):
        for acts in (random_actions(n, budget, cost, rng), uniform_actions(n, budget, cost)):
            assert cost[np.arange(n), acts].sum() <= budget


def test_uniform_is_round_robin():
    assert uniform_actions(7, 10, uniform_costs(7)).tolist() == \
        [AUDIO, NONE, IMAGE, AUDIO, NONE, IMAGE, AUDIO]


def test_forced_detector_keeps_the_router_windows():
    router = np.array([NONE, AUDIO, NONE, BOTH, IMAGE, NONE])
    assert force_modality(router, IMAGE, budget=3).tolist() == [0, IMAGE, 0, IMAGE, IMAGE, 0]
    assert force_modality(router, MULTIMODAL, budget=4, unit_cost=2).tolist() == \
        [0, MULTIMODAL, 0, MULTIMODAL, 0, 0]


def test_prediction_needs_an_acquired_stream_that_fires():
    fired = {AUDIO: np.array([1, 1, 0, 1], bool), IMAGE: np.array([0, 1, 1, 1], bool),
             MULTIMODAL: np.array([1, 1, 1, 1], bool)}
    assert predict(np.array([NONE, IMAGE, AUDIO, BOTH]), fired).tolist() == \
        [False, True, False, True]


@pytest.mark.parametrize("gate_at,params", [("dense", 16275170), ("feature", 16317444),
                                            ("output", 16321158)])
def test_late_moe_is_parameter_matched(gate_at, params):
    from modalfidelity.baselines.late_moe import build_matched
    model = build_matched(gate_at)
    assert sum(p.numel() for p in model.parameters()) == params
