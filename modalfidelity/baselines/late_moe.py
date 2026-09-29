"""Late mixture-of-experts baselines of Table 1: the gate placed *after* both detectors.

Each late model reads frozen embeddings of both detectors for every window: the W2V2-AASIST
readout (160-d, averaged over the window) and the pooled GenD CLIP ViT-L feature (1024-d,
averaged over 4 frames). It outputs P(audio forged) and P(image forged) per window. Three gate
positions are compared:

    dense    no gate: concatenate both experts' features, one head
    feature  a router mixes the two experts' FEATURES, then one head
    output   each expert has its own head; a router mixes their LOGITS

All three run both detectors on every window, so each costs 2 calls (4,086.6 GFLOPs) per window
whatever it learns. Each is sized to the same 16.26 M parameters as the preview encoder plus the
router, so accuracy differences are not bought with capacity.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import FPS, WINDOW_FRAMES, WINDOW_STRIDE

TARGET_PARAMS = 16_260_000
AUDIO_DIM, IMAGE_DIM = 160, 1024
N_OUT = 2                              # (P(audio forged), P(image forged))
POSITIONS = ("dense", "feature", "output")


class Expert(nn.Module):
    """One modality's MLP tower over a detector embedding."""

    def __init__(self, in_dim: int, width: int, depth: int = 3, out_dim: int = 0):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(0.1)]
            d = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, out_dim) if out_dim else None
        self.out_width = width

    def forward(self, x):
        h = self.body(x)
        return h, (self.head(h) if self.head is not None else None)


class LateMoE(nn.Module):
    """Late fusion of the two detector embeddings with the gate at ``gate_at``."""

    def __init__(self, gate_at: str = "feature", width: int = 512, router_width: int = 256,
                 depth: int = 3):
        super().__init__()
        if gate_at not in POSITIONS:
            raise ValueError(f"gate_at must be one of {POSITIONS}, got {gate_at}")
        self.gate_at = gate_at
        per_expert_out = N_OUT if gate_at == "output" else 0
        self.audio = Expert(AUDIO_DIM, width, depth, per_expert_out)
        self.image = Expert(IMAGE_DIM, width, depth, per_expert_out)
        if gate_at == "dense":
            self.router = None
            self.head = nn.Linear(2 * width, N_OUT)
        else:
            # the router reads the raw detector vectors, so it decides before trusting either tower
            self.router = nn.Sequential(nn.Linear(AUDIO_DIM + IMAGE_DIM, router_width), nn.GELU(),
                                        nn.Linear(router_width, 2))
            self.head = nn.Linear(width, N_OUT) if gate_at == "feature" else None

    def forward(self, a: torch.Tensor, i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns per-window logits ``(N, 2)`` and the router's weights ``(N, 2)``."""
        ha, la = self.audio(a)
        hi, li = self.image(i)
        if self.gate_at == "dense":
            w = torch.full((a.shape[0], 2), 0.5, device=a.device, dtype=a.dtype)
            return self.head(torch.cat([ha, hi], -1)), w
        w = F.softmax(self.router(torch.cat([a, i], -1)), dim=-1)
        if self.gate_at == "feature":
            return self.head(w[:, :1] * ha + w[:, 1:] * hi), w
        return w[:, :1] * la + w[:, 1:] * li, w

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_matched(gate_at: str, target: int = TARGET_PARAMS, depth: int = 3,
                  router_width: int = 256) -> LateMoE:
    """The expert width whose total parameter count lands closest to ``target``."""
    best, best_err = None, None
    for width in range(64, 2049, 8):
        m = LateMoE(gate_at, width=width, router_width=router_width, depth=depth)
        err = abs(m.n_params() - target)
        if best_err is None or err < best_err:
            best, best_err = m, err
        elif m.n_params() > target and best_err < err:
            break
    return best


def load_late_moe(path: str, device: str = "cpu") -> LateMoE:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = build_matched(blob["gate_at"])
    model.load_state_dict(blob["model"])
    return model.to(device).eval()


# ---------------------------------------------------------------- embeddings and labels
def load_embeddings(emb_dir: str) -> Dict:
    """Memory-map detector-embedding shards: ``shard*_aud.npy`` (N, 160), ``shard*_img.npy``
    (N, 1024) and ``shard*_meta.npz`` with per-video ``offsets``, ``keys``, ``source`` and
    ``modify_type``."""
    stems = sorted(glob.glob(os.path.join(emb_dir, "shard*_aud.npy")))
    if not stems:
        raise FileNotFoundError(f"no embedding shards under {emb_dir}")
    aud, img, keys, srcs, mts, vid_shard, lo, hi = [], [], [], [], [], [], [], []
    for s, ap in enumerate(stems):
        aud.append(np.load(ap, mmap_mode="r"))
        img.append(np.load(ap.replace("_aud.npy", "_img.npy"), mmap_mode="r"))
        with np.load(ap.replace("_aud.npy", "_meta.npz"), allow_pickle=False) as m:
            off = m["offsets"]
            keys.append(m["keys"]); srcs.append(m["source"]); mts.append(m["modify_type"])
        vid_shard += [s] * (len(off) - 1)
        lo += off[:-1].tolist(); hi += off[1:].tolist()
    return {"aud": aud, "img": img, "keys": np.concatenate(keys).astype(str),
            "source": np.concatenate(srcs).astype(str),
            "modify_type": np.concatenate(mts).astype(str), "vid_shard": np.asarray(vid_shard),
            "lo": np.asarray(lo), "hi": np.asarray(hi)}


def video_embeddings(store: Dict, v: int) -> Tuple[np.ndarray, np.ndarray]:
    s, lo, hi = int(store["vid_shard"][v]), int(store["lo"][v]), int(store["hi"][v])
    return (np.asarray(store["aud"][s][lo:hi], np.float32),
            np.asarray(store["img"][s][lo:hi], np.float32))


def window_starts(n_frames: int, window: int = WINDOW_FRAMES, stride: int = WINDOW_STRIDE) -> List[int]:
    """Window start frames; the tail is covered by one extra window ending at the last frame."""
    if n_frames < window:
        return [0]
    starts = list(range(0, n_frames - window + 1, stride))
    if starts[-1] + window < n_frames:
        starts.append(n_frames - window)
    return starts


def overlaps(start: int, segments, window: int = WINDOW_FRAMES) -> float:
    """1.0 if the window [start, start + window) frames overlaps any (start_s, end_s) segment."""
    lo, hi = start / FPS, (start + window) / FPS
    return float(any(len(s) >= 2 and float(s[0]) < hi and float(s[1]) > lo for s in segments or []))


def window_labels(record: Dict) -> np.ndarray:
    """Per-window (audio forged, image forged) labels from an AV-Deepfake1M metadata record with
    ``video_frames``, ``audio_forged``, ``video_forged``, ``audio_fake_segments`` and
    ``visual_fake_segments``."""
    starts = window_starts(int(record["video_frames"]))
    lab = np.zeros((len(starts), 2), np.float32)
    for j, s in enumerate(starts):
        if record["audio_forged"]:
            lab[j, 0] = overlaps(s, record["audio_fake_segments"])
        if record["video_forged"]:
            lab[j, 1] = overlaps(s, record["visual_fake_segments"])
    return lab


def load_records(records_dir: str) -> List[Dict]:
    """Metadata records (``manifest_shard*.json`` with a ``records`` list), first copy of each key."""
    out, seen = [], set()
    for path in sorted(glob.glob(os.path.join(records_dir, "manifest_shard*.json"))):
        for r in json.load(open(path))["records"]:
            if r["key"] not in seen:
                seen.add(r["key"])
                out.append(r)
    if not out:
        raise FileNotFoundError(f"no manifest_shard*.json records under {records_dir}")
    return out


class WindowSet(torch.utils.data.Dataset):
    """One item = one window: its two detector embeddings and its two labels."""

    def __init__(self, store: Dict, labels: List[np.ndarray], index: List[Tuple[int, int]]):
        self.store, self.labels, self.index = store, labels, list(index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        v, w = self.index[k]
        s, base = int(self.store["vid_shard"][v]), int(self.store["lo"][v])
        a = np.asarray(self.store["aud"][s][base + w], np.float32)
        i = np.asarray(self.store["img"][s][base + w], np.float32)
        return torch.from_numpy(a), torch.from_numpy(i), torch.from_numpy(self.labels[v][w])


def identity_split(ids: np.ndarray, sources: np.ndarray, forced_heldout, seed: int,
                   fraction: float = 0.15) -> np.ndarray:
    """Hold out ``fraction`` of speaker identities, plus every identity of the benchmark videos
    (``forced_heldout`` sources). Returns a per-video held-out mask."""
    rng = np.random.default_rng(seed)
    uniq = np.asarray(sorted(set(ids.tolist())))
    held = set(uniq[rng.permutation(len(uniq))[: max(1, int(len(uniq) * fraction))]].tolist())
    held |= set(ids[np.isin(sources, list(forced_heldout))].tolist())
    return np.asarray([x in held for x in ids])
