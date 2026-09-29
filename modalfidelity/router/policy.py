"""The ModalFidelity router phi (paper Sec. 2, "Preview and policy").

At each window ``t`` an LSTM cell receives the preview feature of the window, the previous
action, and the remaining budget ``b_t`` and horizon ``T - t`` as absolute counts. A linear head
scores the four actions (none, audio, image, both). Every action that costs more than ``b_t`` is
masked by setting its logit to a large negative value before the softmax, so the budget can never
be exceeded, trained or not (paper Sec. 2, "Enforcing the budget").

The rollout is a loop over windows because ``b_t`` depends on every earlier decision. The
expensive preview is computed beforehand (``modalfidelity.router.preview``), usually once, into a
feature cache.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import ACTIONS, FEATURE_DIM, INPUT_SCALE, LSTM_HIDDEN, NONE, UNIT_COST

MASKED = -1e9      # not -inf: a fully masked row would make the softmax NaN


class RouterPolicy(nn.Module):
    """Preview features + remaining budget -> one of {none, audio, image, both} per window.

    Only the LSTM cell and the head are trained (2.37 M parameters). ``preview`` is an optional
    frozen preview encoder; without it the router consumes cached features.
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, preview: Optional[nn.Module] = None):
        super().__init__()
        # the attribute names lstm / head / joint_net match the released checkpoints
        self.joint_net = preview
        self.lstm = nn.LSTMCell(feature_dim + len(ACTIONS) + 2, LSTM_HIDDEN)
        self.head = nn.Linear(LSTM_HIDDEN, len(ACTIONS))

    def rollout(self, feats: torch.Tensor, budget: torch.Tensor,
                cost: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                expert_actions: Optional[torch.Tensor] = None, sample: bool = False,
                generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        """Step through a batch of videos, spending budget as it goes.

        Args:
            feats: ``(batch, windows, feature_dim)`` preview features.
            budget: ``(batch,)`` budgets ``B`` in detector calls.
            cost: ``(batch, windows, 4)`` prices; defaults to ``(0, 1, 1, 2)`` everywhere.
            mask: ``(batch, windows)`` bool, False on padding. Padded windows are forced to NONE.
            expert_actions: if given, these actions are taken instead of the router's own
                (teacher forcing); the default is the router's own trajectory.
            sample: sample from the policy instead of taking the argmax (stage 2).

        Returns logits, actions, the budget left *before* each window, log-probabilities of the
        taken actions and the policy entropy, each ``(batch, windows[, 4])``.
        """
        b, w, _ = feats.shape
        dev, dt = feats.device, feats.dtype
        if cost is None:
            cost = torch.tensor(UNIT_COST, device=dev, dtype=dt).view(1, 1, -1).expand(b, w, -1)
        if mask is None:
            mask = torch.ones((b, w), dtype=torch.bool, device=dev)

        b_left = budget.to(dev, dt).clone()
        state = None
        prev = torch.zeros((b, len(ACTIONS)), device=dev, dtype=dt)
        logits_t, actions_t, bleft_t, logp_t, ent_t = [], [], [], [], []
        for t in range(w):
            t_left = torch.full((b,), float(w - t), device=dev, dtype=dt)
            step = torch.cat((feats[:, t], prev, (b_left / INPUT_SCALE).unsqueeze(-1),
                              (t_left / INPUT_SCALE).unsqueeze(-1)), dim=-1)
            state = self.lstm(step) if state is None else self.lstm(step, state)
            logits = self.head(state[0])

            affordable = cost[:, t] <= b_left.unsqueeze(-1) + 1e-6
            affordable[:, NONE] = True
            logits = logits.masked_fill(~affordable, MASKED)

            logp = F.log_softmax(logits, dim=-1)
            probs = logp.exp()
            if expert_actions is not None:
                act = expert_actions[:, t].to(dev).long()
            elif sample:
                act = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            else:
                act = logits.argmax(dim=-1)
            act = torch.where(mask[:, t], act, torch.full_like(act, NONE))

            logits_t.append(logits)
            actions_t.append(act)
            bleft_t.append(b_left.clone())
            logp_t.append(logp.gather(-1, act.unsqueeze(-1)).squeeze(-1))
            ent_t.append(-(probs * logp.clamp_min(MASKED)).sum(-1))

            b_left = (b_left - cost[:, t].gather(-1, act.unsqueeze(-1)).squeeze(-1)).clamp_min(0)
            prev = F.one_hot(act, len(ACTIONS)).to(dt)

        return {"logits": torch.stack(logits_t, 1), "actions": torch.stack(actions_t, 1),
                "b_left": torch.stack(bleft_t, 1), "log_prob": torch.stack(logp_t, 1),
                "entropy": torch.stack(ent_t, 1)}

    def forward(self, feats: torch.Tensor, budget: torch.Tensor, **kw) -> Dict[str, torch.Tensor]:
        return self.rollout(feats, budget, **kw)


def load_router(path: str, device: str = "cpu") -> RouterPolicy:
    """Load a released router checkpoint (LSTM + head) for use on cached preview features."""
    blob = torch.load(path, map_location="cpu", weights_only=True)
    state = {k: v for k, v in blob["model"].items() if not k.startswith("joint_net.")}
    model = RouterPolicy()
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
