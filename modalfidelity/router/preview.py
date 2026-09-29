"""The router's cheap preview of each window (paper Sec. 2, "Preview and policy").

A MobileNetV2 trunk per stream reads a few subsampled frames (4 frames at 160x160) and a
spectrogram of each 1 s window; the two trunk outputs are fused into one 2048-d feature. The
preview is inherited from an unbudgeted per-window gate and kept **frozen**, so it is run once
over the dataset (``mf-cache-features``) and the router trains on the cached features.

This module holds three things:

* the preview encoder (:class:`PreviewEncoder`) and :func:`load_preview`, which loads its frozen
  weights from a released checkpoint;
* :class:`PreviewWindows`, which slices the per-window preview inputs and per-stream labels out
  of a window-agnostic preview cache (every other frame of a video stored once as JPEG, plus one
  full-video spectrogram);
* the helpers that define the window grid (:func:`window_starts`) and the window labels.

The attribute names (``nets``, ``joint``, ``features``, ``conv``) match the released weights.
"""
from __future__ import annotations

import glob
import io
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

from ..constants import FEATURE_DIM, FPS, WINDOW_FRAMES, WINDOW_STRIDE

#: frames per window seen by the image trunk. It must be 4: the trunk halves the frame axis twice.
PREVIEW_FRAMES = 4
#: side of the square spectrogram crop fed to the audio trunk
SPEC_SIZE = 256
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# (expansion, output channels, repeats, stride): the standard MobileNetV2 schedule
_SCHEDULE = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2],
             [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]


# ----------------------------------------------------------------------------- encoder
def _make_divisible(v, divisor, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class TemporalPooling(nn.Module):
    """Pool along the frame axis only; frames are folded into the batch dimension."""

    def __init__(self, frames: int, kernel_size: int = 3, stride: int = 2):
        super().__init__()
        self.frames = frames
        self.pool = nn.MaxPool3d(kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1),
                                 padding=((kernel_size - 1) // stride, 0, 0))

    def forward(self, x):
        nt, c, h, w = x.shape
        x = x.view((-1, self.frames) + x.size()[1:]).transpose(1, 2)
        x = self.pool(x)
        return x.transpose(1, 2).contiguous().view(-1, c, h, w)


def _conv_bn(inp, oup, kernel, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, kernel, stride, kernel // 2, bias=False),
                         nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


class _InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual, optionally pooling the frame axis first."""

    def __init__(self, inp, oup, stride, expand_ratio, num_frames=None):
        super().__init__()
        self.temporal_pool = TemporalPooling(num_frames) if num_frames else None
        hidden = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
                nn.Conv2d(hidden, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
                nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
                nn.Conv2d(hidden, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.temporal_pool:
            x = self.temporal_pool(x)
        return x + self.conv(x) if self.identity else self.conv(x)


class PreviewTrunk(nn.Module):
    """One stream's MobileNetV2; frame pooling at the stages entering 64 and 160 channels."""

    def __init__(self, num_frames: int, input_channels: int):
        super().__init__()
        self.orig_num_frames = num_frames
        frames = num_frames
        channel = _make_divisible(32, 8)
        layers = [_conv_bn(input_channels, channel, 3, 2)]
        for t, c, n, s in _SCHEDULE:
            pools = c in (64, 160)
            out = _make_divisible(c, 8)
            for i in range(n):
                pool_frames = frames if i == 0 and pools and frames != 1 else None
                layers.append(_InvertedResidual(channel, out, s if i == 0 else 1, t, pool_frames))
                channel = out
            if pools:
                frames //= 2
        self.features = nn.Sequential(*layers)
        self.last_channel = 1280
        self.conv = _conv_bn(channel, self.last_channel, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def extract_features(self, x):
        bs, c_t, h, w = x.shape
        x = x.view(bs * self.orig_num_frames, c_t // self.orig_num_frames, h, w)
        x = self.conv(self.features(x))
        return self.avgpool(x).view(x.size(0), -1)


class PreviewEncoder(nn.Module):
    """Image trunk (4 RGB frames) + audio trunk (1 spectrogram) fused into a 2048-d feature."""

    def __init__(self):
        super().__init__()
        self.nets = nn.ModuleList([PreviewTrunk(PREVIEW_FRAMES, 3), PreviewTrunk(1, 1)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.last_channels = FEATURE_DIM
        width = sum(net.last_channel for net in self.nets)
        self.joint = nn.Sequential(nn.Linear(width, FEATURE_DIM), nn.ReLU(True),
                                   nn.Linear(FEATURE_DIM, FEATURE_DIM), nn.ReLU(True),
                                   nn.LayerNorm(FEATURE_DIM))

    def features(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """``inputs = [rgb (n, 12, 160, 160), spec (n, 1, 256, 256)]`` -> ``(n, 2048)``."""
        parts = [net.extract_features(x) for net, x in zip(self.nets, inputs)]
        joined = torch.cat(parts, dim=1)
        return self.joint(joined.view(joined.size(0), -1))

    forward = features


def load_preview(path: str, device: str = "cpu") -> PreviewEncoder:
    """Load the frozen preview weights.

    Accepts the released preview checkpoint or the unbudgeted-gate checkpoint it was taken from:
    either way the ``joint_net.*`` tensors (or an unprefixed state dict) are loaded strictly.
    """
    blob = torch.load(path, map_location="cpu", weights_only=True)
    state = blob.get("model", blob)
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    if any(k.startswith("joint_net.") for k in state):
        state = {k[len("joint_net."):]: v for k, v in state.items() if k.startswith("joint_net.")}
    model = PreviewEncoder()
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ----------------------------------------------------------------------------- windows
def window_starts(n_frames: int, window: int = WINDOW_FRAMES,
                  stride: int = WINDOW_STRIDE) -> List[int]:
    """Start frames of the 1 s windows at a 0.24 s stride; the tail is never dropped."""
    if n_frames < window:
        return [0]
    starts = list(range(0, n_frames - window + 1, stride))
    if starts[-1] + window < n_frames:
        starts.append(n_frames - window)
    return starts


def overlaps(start_frame: int, window: int, segments) -> float:
    """1.0 if the window overlaps any annotated forged segment ``[start_s, end_s]``."""
    lo, hi = start_frame / FPS, (start_frame + window) / FPS
    return float(any(len(s) >= 2 and float(s[0]) < hi and float(s[1]) > lo
                     for s in (segments or [])))


def load_records(cache_dir: str) -> List[Dict]:
    """Every video record of a preview cache (``manifest_shard*.json``), de-duplicated by key."""
    records, seen = [], set()
    for path in sorted(glob.glob(os.path.join(cache_dir, "manifest_shard*.json"))):
        with open(path) as fh:
            for r in json.load(fh)["records"]:
                if r["key"] not in seen:
                    seen.add(r["key"])
                    records.append(r)
    if not records:
        raise ValueError(f"no manifest_shard*.json records under {cache_dir}")
    return records


class PreviewWindows(data.Dataset):
    """One item = every window of one video: preview inputs, per-stream labels, a mask.

    Stored frame ``j`` of the preview cache is original frame ``2j``, so a 25-frame window at a
    6-frame stride covers 13 stored frames and advances exactly 3 stored frames per window.
    Labels are ``(image forged, audio forged)`` per window: a stream counts as forged in a window
    if the window overlaps any forged segment of that stream.
    """

    def __init__(self, cache_dir: str, records: Optional[Sequence[Dict]] = None,
                 max_windows: int = 160):
        self.cache_dir = cache_dir
        self.records = list(records) if records is not None else load_records(cache_dir)
        self.max_windows = int(max_windows)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict:
        from PIL import Image

        rec = self.records[index]
        with np.load(os.path.join(self.cache_dir, "preview", rec["key"] + ".npz"),
                     allow_pickle=True) as blob:
            rgb_blobs = blob["rgb"]
            spec_bytes = blob["spec"].tobytes() if blob["spec"].ndim == 0 else blob["spec"]
            lo, hi = [float(v) for v in blob["spec_range"]]
        spec_full = (np.asarray(Image.open(io.BytesIO(bytes(spec_bytes))), dtype=np.float32)
                     / 255.0 * (hi - lo) + lo)
        cols = spec_full.shape[1]
        seconds = rec["video_frames"] / FPS

        starts = window_starts(rec["video_frames"])[: self.max_windows]
        n_stored = len(rgb_blobs)
        per_window = []
        for s in starts:
            j0, j1 = (s + 1) // 2, (s + WINDOW_FRAMES - 1) // 2
            per_window.append([int(min(n_stored - 1, round(v)))
                               for v in np.linspace(j0, max(j0, j1), PREVIEW_FRAMES)])
        decoded = {}
        for j in {j for idx in per_window for j in idx}:
            img = Image.open(io.BytesIO(rgb_blobs[j].tobytes())).convert("RGB")
            t = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.0
            decoded[j] = (t - IMAGENET_MEAN) / IMAGENET_STD

        frames, specs, labels = [], [], []
        for s, idx in zip(starts, per_window):
            frames.append(torch.cat([decoded[j] for j in idx], dim=0))
            c0 = int(round(s / FPS / max(seconds, 1e-6) * cols))
            c1 = int(round((s + WINDOW_FRAMES) / FPS / max(seconds, 1e-6) * cols))
            chunk = spec_full[:, max(0, c0):max(c0 + 1, min(cols, c1))]
            chunk = torch.from_numpy(np.ascontiguousarray(chunk))[None, None]
            chunk = torch.nn.functional.interpolate(chunk, size=(SPEC_SIZE, SPEC_SIZE),
                                                    mode="bilinear", align_corners=False)[0]
            specs.append((chunk - chunk.mean()) / chunk.std().clamp_min(1e-5))
            labels.append([overlaps(s, WINDOW_FRAMES, rec["visual_fake_segments"]),
                           overlaps(s, WINDOW_FRAMES, rec["audio_fake_segments"])])
        return {"rgb": torch.stack(frames), "spec": torch.stack(specs),
                "labels": torch.tensor(labels, dtype=torch.float),
                "modify_type": rec["modify_type"], "key": rec["key"]}


__all__ = ["PreviewEncoder", "load_preview", "PreviewWindows", "load_records", "window_starts",
           "overlaps"]
