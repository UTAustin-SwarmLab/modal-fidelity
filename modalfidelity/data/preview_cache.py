"""The preview cache: what the router's preview encoder reads, stored once per video.

For each video the cache holds every other frame as a 160x160 JPEG and one log-magnitude STFT of
the whole audio track (quantised to 8 bits, stored as a JPEG with its value range). Windows are
slices of these arrays taken at load time (``modalfidelity.router.preview``), so the window
length and stride are not baked into the cache. No forensic detector is run here.

Source clips are drawn from those that exist in all four variants (real, audio-, image- and
both-forged), so speaker and utterance are held fixed across the four.
"""
from __future__ import annotations

import io
import os
import subprocess
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .avdeepfake1m import VARIANTS, Entry, load_metadata

PREVIEW_SIZE = 160
FPS = 25.0
AUDIO_RATE = 24000
JPEG_QUALITY = 90


def find_quadruples(root: str, split: str, wanted: int, seed: int, shard: int = 0,
                    num_shards: int = 1) -> Tuple[list, Dict[str, Entry], int]:
    """``wanted`` source clips with all four variants, sampled with ``seed`` and then sharded,
    so the union of the shards equals the unsharded selection."""
    groups: Dict[str, dict] = defaultdict(dict)
    meta: Dict[str, Entry] = {}
    for entry in load_metadata(root, split):
        parts = entry.file.split("/")
        groups["/".join(parts[:3])][parts[-1]] = entry.file
        meta[entry.file] = entry
    complete = sorted(s for s, v in groups.items() if set(VARIANTS) <= set(v))
    rng = np.random.default_rng(seed)
    chosen = [complete[i] for i in rng.permutation(len(complete))[:wanted]]
    return [(s, groups[s]) for s in chosen[shard::num_shards]], meta, len(complete)


def read_every_other_frame(path: str) -> Tuple[int, List[Image.Image]]:
    """Total frame count and frames 1, 3, 5, ... (1-based) of a video."""
    from torchvision.io import read_video
    frames, _audio, _info = read_video(path, pts_unit="sec")
    total = len(frames)
    keep = [min(max(i - 1, 0), total - 1) for i in range(1, total + 1, 2)]
    batch = frames[keep].numpy()
    return total, [Image.fromarray(f.astype(np.uint8)) for f in batch]


def extract_wav(video_path: str, target: str, rate: int = AUDIO_RATE) -> Optional[str]:
    """Demux the audio track to a mono 16-bit wav at ``rate`` with ffmpeg; None if it has none."""
    if os.path.isfile(target) and os.path.getsize(target) > 44:
        return target
    os.makedirs(os.path.dirname(target), exist_ok=True)
    result = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path, "-vn",
                             "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(int(rate)), target],
                            capture_output=True)
    return target if result.returncode == 0 and os.path.isfile(target) else None


def full_spectrogram(wav_path: str, rate: int = AUDIO_RATE, n_fft: int = 511,
                     hop_ms: float = 5.0, win_ms: float = 10.0) -> np.ndarray:
    """One log-magnitude STFT over the whole clip, shape (256, columns)."""
    import librosa
    y, _ = librosa.load(wav_path, sr=rate, mono=True)
    if y.size == 0:
        return np.zeros((256, 8), dtype=np.float32)
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=int(rate * hop_ms / 1000),
                               win_length=int(rate * win_ms / 1000)))
    return np.log(spec + 1e-6).astype(np.float32)


def quantise(array: np.ndarray) -> Tuple[np.ndarray, float, float]:
    lo, hi = float(array.min()), float(array.max())
    return ((array - lo) / (hi - lo + 1e-9) * 255.0).astype(np.uint8), lo, hi


def encode_jpeg(array: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def cache_video(video_path: str, dst: str, wav_dir: str, wav_key: str,
                quality: int = JPEG_QUALITY) -> Tuple[int, int, int]:
    """Write one video's preview to ``dst`` (.npz); return (frames, stored frames, spec columns)."""
    total, frames = read_every_other_frame(video_path)
    if not frames:
        raise RuntimeError("no frames decoded")
    blobs = [encode_jpeg(np.asarray(f.resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.BILINEAR)),
                         quality) for f in frames]
    wav = extract_wav(video_path, os.path.join(wav_dir, wav_key + ".wav"))
    spec = full_spectrogram(wav) if wav else np.zeros((256, 8), dtype=np.float32)
    q, lo, hi = quantise(spec)
    np.savez_compressed(dst,
                        rgb=np.array([np.frombuffer(b, dtype=np.uint8) for b in blobs],
                                     dtype=object),
                        spec=encode_jpeg(q, quality),
                        spec_range=np.array([lo, hi], dtype=np.float32),
                        spec_shape=np.array(q.shape, dtype=np.int32))
    return total, len(blobs), int(q.shape[1])
