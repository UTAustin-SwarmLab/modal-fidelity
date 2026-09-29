"""Train one late-MoE baseline of Table 1 on cached detector embeddings.

    mf-train-late-moe --gate-at feature --embeddings <emb_dir> --records <records_dir> \
        --heldout-sources data/heldout_sources.json --out outputs/late_moe/feature_s0

Two independent binary outputs per window (audio forged, image forged), trained with a
positive-weighted BCE (most windows are authentic). The split is by speaker identity, and every
identity that appears in the benchmark videos is held out, so the late models and the router are
scored on videos neither trained on. Refuses to write into a directory that already holds
checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..baselines.late_moe import (POSITIONS, WindowSet, build_matched, identity_split,
                                  load_embeddings, load_records, window_labels)
from ..paths import checkpoint


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gate-at", choices=POSITIONS, required=True,
                   help="where the gate sits: dense (no gate), feature or output")
    p.add_argument("--embeddings", default=checkpoint("detector_embeddings"),
                   help="detector-embedding cache (shard*_aud.npy, shard*_img.npy, shard*_meta.npz)")
    p.add_argument("--records", required=True,
                   help="directory of manifest_shard*.json metadata records (frames and fake segments)")
    p.add_argument("--heldout-sources", default=None,
                   help="JSON list of benchmark video sources whose identities are always held out")
    p.add_argument("--out", required=True, help="run directory for checkpoints and metrics")
    p.add_argument("--epochs", type=int, default=20, help="maximum epochs")
    p.add_argument("--batch-size", type=int, default=1024, help="training batch size (windows)")
    p.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate (cosine schedule)")
    p.add_argument("--patience", type=int, default=5,
                   help="stop after this many epochs without a better held-out balanced accuracy")
    p.add_argument("--seed", type=int, default=0, help="seed for the split and the initialisation")
    p.add_argument("--workers", type=int, default=6, help="data-loader workers")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="torch device")
    a = p.parse_args(argv)

    if os.path.isdir(a.out) and any(f.endswith(".pt") for f in os.listdir(a.out)):
        raise SystemExit(f"{a.out} already holds checkpoints; choose a new --out")
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    store = load_embeddings(a.embeddings)
    recs = {r["key"]: r for r in load_records(a.records)}
    labels = [window_labels(recs[k]) for k in store["keys"]]
    ids = np.asarray([recs[k]["file"].split("/")[0] for k in store["keys"]])
    forced = set(json.load(open(a.heldout_sources))) if a.heldout_sources else set()
    held = identity_split(ids, store["source"], forced, a.seed)

    def windows_of(mask):
        return [(v, w) for v in np.where(mask)[0] for w in range(len(labels[v]))]

    train_idx, held_idx = windows_of(~held), windows_of(held)
    model = build_matched(a.gate_at).to(a.device)
    print(f"gate_at={a.gate_at} params {model.n_params() / 1e6:.3f} M | train {len(train_idx):,} "
          f"windows | held out {len(held_idx):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    tl = torch.utils.data.DataLoader(WindowSet(store, labels, train_idx), batch_size=a.batch_size,
                                     shuffle=True, num_workers=a.workers, drop_last=True)
    hl = torch.utils.data.DataLoader(WindowSet(store, labels, held_idx), batch_size=4096,
                                     shuffle=False, num_workers=max(2, a.workers // 2))
    stacked = np.concatenate([labels[v] for v in np.where(~held)[0]])
    pos_weight = torch.tensor(((1 - stacked).sum(0) + 1) / (stacked.sum(0) + 1),
                              dtype=torch.float32, device=a.device)

    best, bad = -1.0, 0
    for ep in range(a.epochs):
        model.train(); t0, tot, n = time.time(), 0.0, 0
        for xa, xi, y in tl:
            xa, xi, y = xa.to(a.device), xi.to(a.device), y.to(a.device)
            loss = F.binary_cross_entropy_with_logits(model(xa, xi)[0], y, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += float(loss) * len(y); n += len(y)
        sched.step()

        model.eval(); P, Y = [], []
        with torch.no_grad():
            for xa, xi, y in hl:
                P.append(torch.sigmoid(model(xa.to(a.device), xi.to(a.device))[0]).cpu()); Y.append(y)
        pred, Y = torch.cat(P).numpy() > 0.5, torch.cat(Y).numpy() > 0
        bal = [0.5 * ((pred[:, c] & Y[:, c]).sum() / max(Y[:, c].sum(), 1)
                      + (~pred[:, c] & ~Y[:, c]).sum() / max((~Y[:, c]).sum(), 1)) for c in range(2)]
        row = {"epoch": ep, "loss": tot / max(n, 1), "bal_audio": float(bal[0]),
               "bal_image": float(bal[1]), "bal_mean": float(np.mean(bal)),
               "secs": round(time.time() - t0, 1)}
        with open(os.path.join(a.out, "metrics.jsonl"), "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"e{ep} loss {row['loss']:.4f} held-out balanced {row['bal_mean']:.4f}", flush=True)
        ck = {"model": model.state_dict(), "gate_at": a.gate_at, "epoch": ep, "seed": a.seed}
        torch.save(ck, os.path.join(a.out, "last.pt"))
        if row["bal_mean"] > best:
            best, bad = row["bal_mean"], 0
            torch.save(ck, os.path.join(a.out, "best.pt"))
        else:
            bad += 1
            if bad >= a.patience:
                break
    print(f"best held-out balanced accuracy {best:.4f} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
