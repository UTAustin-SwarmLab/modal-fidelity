"""The budget can never be exceeded, and the oracle is exact."""
import numpy as np
import pytest
import torch

from modalfidelity.constants import BOTH, NONE, UNIT_COST
from modalfidelity.router.oracle import brute_force_value, solve_knapsack, uniform_costs
from modalfidelity.router.policy import RouterPolicy
from modalfidelity.router.reward import achieved_reward, reward_matrix


def test_reward_table():
    r = reward_matrix([0, 1, 0, 1], [0, 0, 1, 1], false_alarm=0.25)
    assert r.tolist() == [[0, -0.25, -0.25, -0.5], [0, 1, -0.25, 0.75],
                          [0, -0.25, 1, 0.75], [0, 1, 1, 2]]


@pytest.mark.parametrize("seed", range(20))
def test_oracle_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    t = int(rng.integers(1, 7))
    reward = reward_matrix(rng.random(t) < 0.4, rng.random(t) < 0.4, false_alarm=0.25)
    cost = uniform_costs(t)
    budget = int(rng.integers(0, 2 * t + 1))
    sol = solve_knapsack(reward, cost, budget)
    assert sol.optimal_value == pytest.approx(brute_force_value(reward, cost, budget))
    acts, left = sol.rollout()
    assert cost[np.arange(t), acts].sum() <= budget
    assert achieved_reward(acts, *(reward[:, 1] > 0, reward[:, 2] > 0), 0.25) == \
        pytest.approx(sol.optimal_value)


def test_oracle_prefers_not_spending_on_ties():
    sol = solve_knapsack(np.zeros((3, 4)), uniform_costs(3), 3)
    assert (sol.rollout()[0] == NONE).all()


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("sample", [False, True])
def test_router_never_exceeds_the_budget(seed, sample):
    torch.manual_seed(seed)
    policy = RouterPolicy().eval()
    with torch.no_grad():
        policy.head.bias.copy_(torch.tensor([-5.0, 0.0, 0.0, 5.0]))   # push hard towards "both"
        batch, windows = 8, 30
        feats = torch.randn(batch, windows, 2048)
        budget = torch.randint(0, 20, (batch,)).float()
        g = torch.Generator().manual_seed(seed)
        out = policy.rollout(feats, budget, sample=sample, generator=g)
    unit = torch.tensor(UNIT_COST)
    spent = unit[out["actions"]].sum(1)
    assert (spent <= budget).all()
    assert (out["b_left"] >= 0).all()


def test_padding_costs_nothing():
    policy = RouterPolicy().eval()
    with torch.no_grad():
        policy.head.bias.copy_(torch.tensor([-5.0, 0.0, 0.0, 5.0]))
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, :4] = True
        out = policy.rollout(torch.randn(2, 10, 2048), torch.tensor([20.0, 20.0]), mask=mask)
    assert (out["actions"][:, 4:] == NONE).all()
    assert (out["actions"][:, :4] == BOTH).all()
