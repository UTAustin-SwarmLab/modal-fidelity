"""Train the ModalFidelity router on cached preview features (paper Sec. 2).

Stage 1 distills the exact knapsack teacher with DAgger (12 epochs, Adam 1e-3); stage 2 then
fine-tunes the stage-1 weights with self-critical policy gradient on the real reward (6 epochs,
Adam 1e-4). Each video gets four log-uniform budgets per epoch, so one router serves every budget.
After every epoch the router is scored on a fixed held-out subsample at the budgets of
``constants.RHOS`` (reward as a fraction of the oracle's), and the best epoch is kept.

Written to ``--out``:

* ``stage1_final.pt``: the router after stage 1;
* ``best.pt``: the best epoch over both stages (the released router is this file);
* ``last.pt``, ``stage{1,2}_epoch<k>.pt`` (first epochs), ``metrics.jsonl``, ``summary.json``.

Example (the paper's seed-0 router)::

    mf-train-router --features <cache>/preview_features --out outputs/router_seed0 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from ..constants import ACTIONS, FALSE_ALARM, RHOS, UNIT_COST
from ..paths import saved_data_dir
from ..router.policy import RouterPolicy
from ..training import stage1_dagger, stage2_scst
from ..training.budgets import BUDGETS_PER_VIDEO
from ..training.features import FeatureDataset, collate, load_feature_shards, split_indices


def evaluate(model, store, index, false_alarm, device, batch_size, seed) -> dict:
    """Greedy rollouts at the fixed budget grid: reward, oracle reward and spend per budget."""
    model.eval()
    ds = FeatureDataset(store, index=index, false_alarm=false_alarm, seed=seed, rhos=RHOS)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         num_workers=2, collate_fn=collate)
    k = len(RHOS)
    got, best, spent, n = np.zeros(k), np.zeros(k), np.zeros(k), 0
    unit = torch.tensor(UNIT_COST, device=device)
    with torch.no_grad():
        for b in loader:
            m = b["mask"].to(device)
            out = model.rollout(b["feats"].to(device), b["budgets"].to(device), mask=m)
            r = stage2_scst.batch_reward(out["actions"], b["audio_forged"].to(device),
                                         b["image_forged"].to(device), m, false_alarm)
            cost = (unit[out["actions"]] * m.float()).sum(1)
            got += r.view(-1, k).sum(0).cpu().numpy()
            best += b["oracle_value"].view(-1, k).sum(0).numpy()
            spent += cost.view(-1, k).sum(0).cpu().numpy()
            n += r.numel() // k
    frac = np.divide(got, best, out=np.zeros_like(got), where=best > 0)
    return {"videos": int(n), "rhos": list(RHOS), "reward": got.tolist(),
            "oracle": best.tolist(), "spent": spent.tolist(), "oracle_fraction": frac.tolist(),
            "total_fraction": float(got.sum() / best.sum()) if best.sum() > 0 else 0.0}


def save(path, model, opt, epoch, metrics, args, stage):
    torch.save({"model": model.state_dict(),
                "optimizer": opt.state_dict() if opt is not None else None,
                "epoch": epoch, "stage": stage, "metrics": metrics,
                "false_alarm": args.false_alarm, "seed": args.seed,
                "actions": list(ACTIONS), "unit_cost": list(UNIT_COST)}, path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mf-train-router", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True,
                   help="directory of cached preview features written by mf-cache-features")
    p.add_argument("--out", required=True, help="output directory for this run's checkpoints")
    p.add_argument("--heldout-sources",
                   default=os.path.join(saved_data_dir(), "heldout_sources.json"),
                   help="JSON list of benchmark source clips whose speakers are forced into the "
                        "held-out split (default: data/heldout_sources.json); '' disables it")
    p.add_argument("--false-alarm", type=float, default=FALSE_ALARM,
                   help="penalty p for inspecting an authentic stream (paper: 0.25)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for initialization, split, budgets and sampling (paper: 0-3)")
    p.add_argument("--stage1-epochs", type=int, default=stage1_dagger.EPOCHS,
                   help="DAgger epochs (paper: 12); 0 skips stage 1, use with --init-from")
    p.add_argument("--stage2-epochs", type=int, default=stage2_scst.EPOCHS,
                   help="self-critical policy-gradient epochs (paper: 6)")
    p.add_argument("--batch-size", type=int, default=16, help="videos per batch (paper: 16)")
    p.add_argument("--budgets-per-video", type=int, default=BUDGETS_PER_VIDEO,
                   help="log-uniform budgets drawn per video per epoch (paper: 4)")
    p.add_argument("--lr", type=float, default=stage1_dagger.LR, help="stage-1 Adam learning rate")
    p.add_argument("--stage2-lr", type=float, default=stage2_scst.LR,
                   help="stage-2 Adam learning rate")
    p.add_argument("--entropy", type=float, default=stage2_scst.ENTROPY,
                   help="stage-2 entropy bonus")
    p.add_argument("--init-from", default="",
                   help="start from this router checkpoint (e.g. a stage1_final.pt) instead of a "
                        "fresh initialization")
    p.add_argument("--eval-subsample", type=int, default=2000,
                   help="held-out videos scored after every epoch to pick best.pt; 0 = all")
    p.add_argument("--early-epochs", type=int, default=5,
                   help="also keep the checkpoint of every epoch up to this index")
    p.add_argument("--limit", type=int, default=0,
                   help="use only the first N training and N held-out videos (smoke tests)")
    p.add_argument("--workers", type=int, default=4, help="data-loader worker processes")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="torch device")
    p.add_argument("--overwrite", action="store_true",
                   help="allow writing into an --out directory that already holds checkpoints")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(2)
    if os.path.isdir(args.out) and any(f.endswith(".pt") for f in os.listdir(args.out)) \
            and not args.overwrite:
        raise SystemExit(f"{args.out} already holds checkpoints; choose a new --out or pass "
                         "--overwrite. Nothing was written.")
    os.makedirs(args.out, exist_ok=True)

    store = load_feature_shards(args.features)
    forced = None
    if args.heldout_sources:
        if not os.path.isfile(args.heldout_sources):
            raise SystemExit(f"--heldout-sources {args.heldout_sources} not found; pass '' to "
                             "train without forcing the benchmark speakers out")
        forced = json.load(open(args.heldout_sources))
    tr_idx, he_idx, n_ids = split_indices(store, fraction=0.15, seed=args.seed,
                                          heldout_sources=forced)
    if args.limit:
        tr_idx, he_idx = tr_idx[: args.limit], he_idx[: args.limit]
    print(f"train {len(tr_idx):,} videos | held out {len(he_idx):,} ({n_ids} identities) | "
          f"p={args.false_alarm} seed={args.seed}", flush=True)
    eval_idx = he_idx
    if args.eval_subsample and len(he_idx) > args.eval_subsample:
        pick = np.random.default_rng(12345 + args.seed).permutation(len(he_idx))
        eval_idx = he_idx[np.sort(pick[: args.eval_subsample])]

    device = args.device
    model = RouterPolicy().to(device)
    if args.init_from:
        blob = torch.load(args.init_from, map_location="cpu", weights_only=True)
        model.load_state_dict({k: v for k, v in blob["model"].items()
                               if not k.startswith("joint_net.")})
        print(f"initialized from {args.init_from}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    metrics_path = os.path.join(args.out, "metrics.jsonl")
    history, best_score = [], -1e9

    def after_epoch(stage, epoch, train_stats):
        nonlocal best_score
        ev = evaluate(model, store, eval_idx, args.false_alarm, device, args.batch_size,
                      args.seed)
        row = {"stage": stage, "epoch": epoch, **train_stats, **ev}
        history.append(row)
        with open(metrics_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"{stage} epoch {epoch}: oracle fraction {ev['total_fraction']:.4f} per rho "
              f"{[round(x, 3) for x in ev['oracle_fraction']]}", flush=True)
        if epoch <= args.early_epochs:
            save(os.path.join(args.out, f"{stage}_epoch{epoch}.pt"), model, opt, epoch, row,
                 args, stage)
        if ev["total_fraction"] > best_score:
            best_score = ev["total_fraction"]
            save(os.path.join(args.out, "best.pt"), model, opt, epoch, row, args, stage)
        save(os.path.join(args.out, "last.pt"), model, opt, epoch, row, args, stage)

    ds = FeatureDataset(store, index=tr_idx, false_alarm=args.false_alarm, seed=args.seed,
                        budgets_per_video=args.budgets_per_video)

    def loader():
        return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, collate_fn=collate)

    for epoch in range(args.stage1_epochs):
        ds.set_epoch(epoch)
        after_epoch("stage1", epoch, stage1_dagger.train_epoch(model, loader(), opt, device))
    if args.stage1_epochs:
        save(os.path.join(args.out, "stage1_final.pt"), model, opt, args.stage1_epochs - 1,
             history[-1] if history else {}, args, "stage1")

    for group in opt.param_groups:
        group["lr"] = args.stage2_lr
    for epoch in range(args.stage2_epochs):
        ds.set_epoch(1000 + epoch)
        after_epoch("stage2", epoch, stage2_scst.train_epoch(model, loader(), opt, device,
                                                             args.false_alarm, args.entropy))

    full = evaluate(model, store, he_idx, args.false_alarm, device, args.batch_size, args.seed)
    print(f"final model, all {len(he_idx):,} held-out videos: oracle fraction "
          f"{full['total_fraction']:.4f}", flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"false_alarm": args.false_alarm, "seed": args.seed,
                   "best_total_fraction": best_score, "history": history,
                   "final_full_heldout": full, "eval_subsample": len(eval_idx),
                   "train_videos": len(tr_idx), "heldout_videos": len(he_idx)}, fh, indent=2)
    print(f"done -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
