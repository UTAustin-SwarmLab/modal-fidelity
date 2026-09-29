"""Stage 2: self-critical policy gradient on the real reward (paper Sec. 2, "Stage 2").

The stage-1 teacher is clairvoyant, so stage 1 ends with a router that imitates a foresight it
lacks. Stage 2 starts from the stage-1 weights and optimizes the reward of Eq. 2 directly. For
each (video, budget) the router samples one rollout and acts greedily in another; the greedy
rollout's reward is the baseline, so a sampled decision is reinforced only when it beats what the
router would have done anyway. The baseline runs under the same mask and budget, so it is exactly
as constrained as the sample it judges, and no learned critic is needed.
"""
from __future__ import annotations

import time
from typing import Dict

import torch

#: paper settings: 6 epochs, Adam at 1e-4, entropy bonus 0.01, gradient norm clipped at 5
EPOCHS = 6
LR = 1e-4
ENTROPY = 0.01
CLIP_NORM = 5.0


def batch_reward(actions: torch.Tensor, audio_forged: torch.Tensor, image_forged: torch.Tensor,
                 mask: torch.Tensor, false_alarm: float) -> torch.Tensor:
    """Total reward of each row's realised action sequence, shape ``(rows,)``."""
    a, i, p = audio_forged, image_forged, float(false_alarm)
    audio_term = torch.where(a > 0, torch.ones_like(a), torch.full_like(a, -p))
    image_term = torch.where(i > 0, torch.ones_like(i), torch.full_like(i, -p))
    table = torch.stack([torch.zeros_like(a), audio_term, image_term,
                         audio_term + image_term], dim=-1)
    per_window = table.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return (per_window * mask.float()).sum(dim=1)


def train_epoch(model, loader, opt, device, false_alarm: float,
                entropy: float = ENTROPY) -> Dict[str, float]:
    """One epoch of self-critical policy gradient."""
    model.train()
    t0, total, adv_sum, batches = time.time(), 0.0, 0.0, 0
    for b in loader:
        feats, m = b["feats"].to(device), b["mask"].to(device)
        af, imf = b["audio_forged"].to(device), b["image_forged"].to(device)
        budget = b["budgets"].to(device)
        sampled = model.rollout(feats, budget, mask=m, sample=True)
        with torch.no_grad():
            greedy = model.rollout(feats, budget, mask=m)
            baseline = batch_reward(greedy["actions"], af, imf, m, false_alarm)
        reward = batch_reward(sampled["actions"], af, imf, m, false_alarm)
        adv = (reward - baseline).detach()
        logp = (sampled["log_prob"] * m.float()).sum(1)
        ent = (sampled["entropy"] * m.float()).sum(1).mean()
        loss = -(adv * logp).mean() - entropy * ent
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        opt.step()
        total += float(loss)
        adv_sum += float(adv.mean())
        batches += 1
    return {"loss": total / max(batches, 1), "mean_advantage": adv_sum / max(batches, 1),
            "secs": round(time.time() - t0, 1)}
