"""Run the frozen preview encoder once over every window of every video (paper Sec. 2).

The preview is frozen, so the router trains on these cached 2048-d features, never on pixels.
Input is a window-agnostic **preview cache** of AV-Deepfake1M (``manifest_shard*.json`` plus
``preview/<key>.npz``: every other frame at 160x160 as JPEG and one full-video log-spectrogram);
windows are 1 s long at a 0.24 s stride and every window of a video is kept.

Output, per shard (read by ``mf-train-router`` and ``mf-evaluate-router``):

* ``shard<i>of<n>_feats.npy``: float16 features, ``(windows, 2048)``;
* ``shard<i>of<n>_meta.npz``: per-window ``labels`` (image forged, audio forged), ``offsets``
  per video, and the video ``keys``, ``modify_type``, ``video_file``, ``source``.

Example::

    mf-cache-features --preview-cache <avdf-preview-cache> --preview-weights \\
        checkpoints/preview/unbudgeted_gate.pt --out checkpoints/features --shard 0 --num-shards 8
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from ..router.preview import PreviewWindows, load_preview, load_records

#: longer than the longest video (132 windows), so nothing is truncated
MAX_WINDOWS = 160


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mf-cache-features", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preview-cache", required=True,
                   help="window-agnostic preview cache (manifest_shard*.json + preview/*.npz)")
    p.add_argument("--preview-weights", required=True,
                   help="frozen preview-encoder checkpoint (released file, or the unbudgeted "
                        "gate checkpoint it came from)")
    p.add_argument("--out", required=True, help="output directory for the feature shards")
    p.add_argument("--shard", type=int, default=0, help="index of this shard")
    p.add_argument("--num-shards", type=int, default=1,
                   help="number of shards the videos are split into (one process per shard)")
    p.add_argument("--only-sources", default="",
                   help="JSON list of source clips; cache only their videos (e.g. the benchmark "
                        "videos, for evaluation)")
    p.add_argument("--limit", type=int, default=0, help="stop after N videos (smoke tests)")
    p.add_argument("--window-chunk", type=int, default=32,
                   help="windows per forward pass; bounds GPU memory on long videos")
    p.add_argument("--workers", type=int, default=3, help="data-loader worker processes")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="torch device")
    p.add_argument("--overwrite", action="store_true",
                   help="replace this shard's files if they already exist")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    stem = os.path.join(args.out, f"shard{args.shard}of{args.num_shards}")
    if os.path.exists(stem + "_feats.npy") and not args.overwrite:
        raise SystemExit(f"{stem}_feats.npy exists; choose a new --out or pass --overwrite")
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(2)

    records = load_records(args.preview_cache)
    if args.only_sources:
        keep = set(json.load(open(args.only_sources)))
        records = [r for r in records if r["source"] in keep]
    mine = records[args.shard::args.num_shards]
    if args.limit:
        mine = mine[: args.limit]
    print(f"shard {args.shard}/{args.num_shards}: {len(mine):,} of {len(records):,} videos "
          f"-> {args.out}", flush=True)

    encoder = load_preview(args.preview_weights, args.device)
    ds = PreviewWindows(args.preview_cache, records=mine, max_windows=MAX_WINDOWS)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                         num_workers=args.workers, collate_fn=lambda b: b[0])
    meta_of = {r["key"]: (r["file"], r["source"]) for r in mine}
    feats, labels, offsets, keys, mtypes = [], [], [0], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, item in enumerate(loader):
            n = item["rgb"].shape[0]
            out = []
            for s in range(0, n, args.window_chunk):
                rgb = item["rgb"][s:s + args.window_chunk].to(args.device, non_blocking=True)
                spec = item["spec"][s:s + args.window_chunk].to(args.device, non_blocking=True)
                out.append(encoder.features([rgb, spec]).float().cpu())
            feats.append(torch.cat(out).to(torch.float16).numpy())
            labels.append(item["labels"].numpy().astype(np.uint8))
            offsets.append(offsets[-1] + n)
            keys.append(item["key"])
            mtypes.append(item["modify_type"])
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                print(f"{i + 1}/{len(mine)} videos, {el / (i + 1):.2f} s/video", flush=True)

    np.save(stem + "_feats.npy", np.concatenate(feats))
    np.savez(stem + "_meta.npz", labels=np.concatenate(labels),
             offsets=np.asarray(offsets, dtype=np.int64), keys=np.asarray(keys),
             modify_type=np.asarray(mtypes),
             video_file=np.asarray([meta_of[k][0] for k in keys]),
             source=np.asarray([meta_of[k][1] for k in keys]))
    print(f"{len(keys):,} videos, {offsets[-1]:,} windows in {(time.time() - t0) / 60:.1f} min "
          f"-> {stem}_*", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
