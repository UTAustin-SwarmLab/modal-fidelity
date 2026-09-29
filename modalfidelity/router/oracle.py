"""Exact budgeted allocation by dynamic programming (paper Eq. 3).

Choosing one action per window under a total budget is a multiple-choice knapsack. It is solved
exactly over (window, remaining budget): ``V(t, b)`` is the best reward reachable from window
``t`` with ``b`` units left. Fed the true-label reward, the solution is the clairvoyant
**oracle** and the stage-1 teacher.

The whole table is returned, not only the optimal path, so the optimal action is known from
every state, including states the optimal plan never visits. That is what stage-1 DAgger queries.
Ties go to the lowest action index, so NONE wins a tie and the teacher never spends for nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Tuple

import numpy as np

from ..constants import ACTIONS, UNIT_COST

NEG_INF = -np.inf


@dataclass(frozen=True)
class KnapsackSolution:
    values: np.ndarray        # (T + 1, B + 1): V(t, b)
    actions: np.ndarray       # (T, B + 1): optimal action at every state
    cost: np.ndarray          # (T, 4) integer costs
    budget: int

    @property
    def n_windows(self) -> int:
        return int(self.actions.shape[0])

    @property
    def optimal_value(self) -> float:
        return float(self.values[0, self.budget])

    def expert_action(self, t: int, b_left: int) -> int:
        """Optimal action at an arbitrary state ``(t, b_left)``."""
        return int(self.actions[int(t), int(np.clip(b_left, 0, self.budget))])

    def expert_actions_along(self, b_trajectory: np.ndarray) -> np.ndarray:
        """Optimal action at each window of a rolled-out trajectory of remaining budgets."""
        b = np.clip(np.asarray(b_trajectory, dtype=np.int64), 0, self.budget)
        return self.actions[np.arange(len(b)), b]

    def rollout(self) -> Tuple[np.ndarray, np.ndarray]:
        """The optimal plan: ``(actions, budget left before each window)``."""
        b = int(self.budget)
        acts = np.empty(self.n_windows, np.int64)
        left = np.empty(self.n_windows, np.int64)
        for t in range(self.n_windows):
            left[t] = b
            acts[t] = self.actions[t, b]
            b -= int(self.cost[t, acts[t]])
        return acts, left


def uniform_costs(n_windows: int, unit=UNIT_COST) -> np.ndarray:
    """``cost[t, a]``, the same unit price at every window, shape ``(T, 4)``."""
    unit = np.asarray(unit, dtype=np.float64)
    if unit.shape != (len(ACTIONS),) or unit[0] != 0 or (unit < 0).any():
        raise ValueError("unit cost must be 4 non-negative prices with a free NONE")
    return np.tile(unit, (int(n_windows), 1))


def solve_knapsack(reward: np.ndarray, cost: np.ndarray, budget: int) -> KnapsackSolution:
    """Solve the multiple-choice knapsack exactly. ``reward`` and ``cost`` are ``(T, 4)``;
    costs must be non-negative integers."""
    reward = np.asarray(reward, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    if reward.shape != cost.shape or reward.shape[1] != len(ACTIONS):
        raise ValueError(f"reward {reward.shape} and cost {cost.shape} must both be (T, 4)")
    if not np.allclose(cost, np.round(cost)) or (cost < 0).any():
        raise ValueError("costs must be non-negative integers")
    budget = int(budget)
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")

    icost = np.round(cost).astype(np.int64)
    n, n_act = reward.shape
    values = np.zeros((n + 1, budget + 1))
    actions = np.zeros((n, budget + 1), dtype=np.int64)
    for t in range(n - 1, -1, -1):
        nxt = values[t + 1]
        cand = np.full((n_act, budget + 1), NEG_INF)
        for a in range(n_act):
            c = int(icost[t, a])
            if c == 0:
                cand[a] = reward[t, a] + nxt
            elif c <= budget:
                cand[a, c:] = reward[t, a] + nxt[: budget + 1 - c]
        values[t] = cand.max(axis=0)
        actions[t] = cand.argmax(axis=0)
    return KnapsackSolution(values=values, actions=actions, cost=icost, budget=budget)


def brute_force_value(reward: np.ndarray, cost: np.ndarray, budget: int) -> float:
    """Exhaustive best total reward. Exponential in T; for tests only."""
    n, n_act = reward.shape
    best = NEG_INF
    for combo in product(range(n_act), repeat=n):
        if sum(cost[t, a] for t, a in enumerate(combo)) <= budget:
            best = max(best, sum(reward[t, a] for t, a in enumerate(combo)))
    return float(best)
