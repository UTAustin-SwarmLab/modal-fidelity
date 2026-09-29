"""Build the preview cache from AV-Deepfake1M videos (step 1 of re-creating the router's inputs).

    mf-cache-previews --dataset /path/to/AV-Deepfake1M --out previews/ --shard 0 --num-shards 32

Writes ``<out>/preview/<key>.npz`` per video and ``<out>/manifest_shard<k>.json`` with each
video's labels and per-stream forged segments. Videos already cached are skipped, so an
interrupted shard can be resumed. The paper's cache used 10,000 source clips (40,000 videos),
seed 0, 32 shards. Next step: ``mf-cache-features``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from ..paths import dataset_root


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=None,
                   help="AV-Deepfake1M root holding <split>/ and <split>_metadata.json "
                        "(default: $MODALFIDELITY_DATASET)")
    p.add_argument("--split", default="train", help="dataset split to read (default: train)")
    p.add_argument("--quadruples", type=int, default=10000,
                   help="number of source clips, each cached in all 4 variants (default: 10000)")
    p.add_argument("--seed", type=int, default=0, help="seed of the clip sample (default: 0)")
    p.add_argument("--shard", type=int, default=0, help="which shard to build (default: 0)")
    p.add_argument("--num-shards", type=int, default=1, help="total shards (default: 1)")
    p.add_argument("--only-sources", nargs="*", default=None,
                   help="cache only these source clips (e.g. id00018/iNlRMGqsmrQ/00118)")
    p.add_argument("--out", required=True, help="output directory of the preview cache")
    p.add_argument("--wav-dir", default=None,
                   help="where demuxed wav files go (default: <out>/_wav)")
    p.add_argument("--quality", type=int, default=90, help="JPEG quality (default: 90)")
    a = p.parse_args(argv)

    import torch
    from ..data.avdeepfake1m import VARIANTS
    from ..data.preview_cache import FPS, PREVIEW_SIZE, cache_video, find_quadruples
    torch.set_num_threads(1)
    root = a.dataset or dataset_root()
    wav_dir = a.wav_dir or os.path.join(a.out, "_wav")
    os.makedirs(os.path.join(a.out, "preview"), exist_ok=True)

    quads, meta, available = find_quadruples(root, a.split, a.quadruples, a.seed, a.shard,
                                             a.num_shards)
    if a.only_sources:
        wanted = set(a.only_sources)
        quads = [q for q in quads if q[0] in wanted]
    print(f"shard {a.shard}/{a.num_shards}: {len(quads)} source clips = {4 * len(quads)} videos "
          f"(from {available:,} complete quadruples)", flush=True)

    records, done, failed, start = [], 0, 0, time.time()
    for i, (src, files) in enumerate(quads):
        for fname, (modify_type, audio_fake, video_fake) in VARIANTS.items():
            rel = files[fname]
            key = rel.replace("/", "_").replace(".mp4", "")
            dst = os.path.join(a.out, "preview", key + ".npz")
            if os.path.exists(dst):
                done += 1
                continue
            try:
                total, stored, cols = cache_video(os.path.join(root, a.split, rel), dst,
                                                  wav_dir, key, a.quality)
            except Exception as exc:                                   # noqa: BLE001
                failed += 1
                print(f"  FAILED {rel}: {exc}", flush=True)
                continue
            entry = meta[rel]
            records.append({"key": key, "file": rel, "source": src, "modify_type": modify_type,
                            "video_frames": total, "stored_frames": stored, "spec_columns": cols,
                            "audio_forged": audio_fake, "video_forged": video_fake,
                            "visual_fake_segments": entry.visual_fake_segments,
                            "audio_fake_segments": entry.audio_fake_segments})
            done += 1
        if (i + 1) % 25 == 0:
            rate = (time.time() - start) / (i + 1)
            print(f"  {i + 1}/{len(quads)} clips, {rate:.2f} s/clip, {failed} failed", flush=True)

    manifest = {"cache_version": 2, "kind": "preview_window_agnostic", "split": a.split,
                "fps": FPS, "preview_size": PREVIEW_SIZE, "frame_stride_stored": 2,
                "quality": a.quality, "seed": a.seed, "shard": a.shard,
                "num_shards": a.num_shards, "quadruples": len(quads), "cached": done,
                "failed": failed, "metadata": "fixed (cross-modal segment leak repaired)",
                "records": records}
    path = os.path.join(a.out, f"manifest_shard{a.shard}.json")
    if os.path.exists(path):
        path = os.path.join(a.out, f"manifest_shard{a.shard}_{int(time.time())}.json")
    with open(path, "w") as handle:
        json.dump(manifest, handle)
    print(f"{done} videos cached, {failed} failed -> {a.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
