"""Stage 1: DAgger distillation of the knapsack teacher (paper Sec. 2, "Stage 1").

The router rolls out its own decisions under a sampled budget. Wherever it lands, mistakes
included, the teacher's table supplies the optimal action from that exact state
``(window, remaining budget)`` as a cross-entropy target. The router thus learns to recover from
its own errors rather than imitate a path it would never reproduce.
"""
from __future__ import annotations

import time
from typing import Dict

import torch
import torch.nn.functional as F

#: paper settings: 12 epochs, Adam at 1e-3, gradient norm clipped at 5
EPOCHS = 12
LR = 1e-3
CLIP_NORM = 5.0


def train_epoch(model, loader, opt, device) -> Dict[str, float]:
    """One epoch of DAgger; returns the mean loss and agreement with the teacher."""
    model.train()
    t0, total, seen, agree = time.time(), 0.0, 0, 0.0
    for b in loader:
        feats, m = b["feats"].to(device), b["mask"].to(device)
        out = model.rollout(feats, b["budgets"].to(device), mask=m, sample=True)
        # query the teacher at the states the router itself reached
        expert = b["expert"].to(device)
        b_left = out["b_left"].long().clamp_min(0).clamp(max=expert.shape[2] - 1)
        target = expert.gather(2, b_left.unsqueeze(-1)).squeeze(-1)
        valid = m & (target >= 0)
        if valid.sum() == 0:
            continue
        loss = F.cross_entropy(out["logits"][valid], target[valid])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        opt.step()
        total += float(loss) * int(valid.sum())
        agree += float((out["actions"][valid] == target[valid]).float().sum())
        seen += int(valid.sum())
    return {"loss": total / max(seen, 1), "teacher_agreement": agree / max(seen, 1),
            "secs": round(time.time() - t0, 1)}
