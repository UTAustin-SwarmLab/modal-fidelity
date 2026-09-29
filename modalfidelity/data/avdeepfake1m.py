"""AV-Deepfake1M metadata, with the per-stream forgery segments repaired.

The released annotation copies a file's own last forged segment into the *other* stream's segment
list for 6,160 entries: ``audio_modified`` videos carry ``visual_fake_segments`` although their
video is authentic, and ``visual_modified`` videos carry ``audio_fake_segments`` although their
audio is authentic. Because the router learns per-window, per-stream labels, that field must be
empty. :func:`load_metadata` clears it (``repair=True``, the default) and asserts that
``fake_segments``, the union of both streams, is unchanged.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List

#: variant filename -> (modify_type, audio forged, image forged)
VARIANTS = {
    "real.mp4": ("real", False, False),
    "real_video_fake_audio.mp4": ("audio_modified", True, False),
    "fake_video_real_audio.mp4": ("visual_modified", False, True),
    "fake_video_fake_audio.mp4": ("both_modified", True, True),
}
#: the stream that cannot be forged for each single-stream variant
IMPOSSIBLE = {"audio_modified": "visual_fake_segments", "visual_modified": "audio_fake_segments"}


@dataclass
class Entry:
    file: str
    modify_type: str
    video_frames: int
    audio_fake_segments: List
    visual_fake_segments: List


def _segments(segs) -> set:
    return {tuple(s) for s in (segs or [])}


def repair(raw: list) -> int:
    """Clear the impossible per-stream field in place; return the number of entries changed."""
    changed = 0
    for x in raw:
        field = IMPOSSIBLE.get(x["modify_type"])
        if field and x.get(field):
            before = _segments(x.get("audio_fake_segments")) | _segments(x.get("visual_fake_segments"))
            x[field] = []
            after = _segments(x.get("audio_fake_segments")) | _segments(x.get("visual_fake_segments"))
            assert before == after, f"repair changed the forged union of {x['file']}"
            changed += 1
    return changed


def load_metadata(root: str, split: str = "train", repair_segments: bool = True) -> List[Entry]:
    """Read ``{root}/{split}_metadata.json`` and (by default) repair the segment leak."""
    path = os.path.join(root, f"{split}_metadata.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found; point --dataset at the AV-Deepfake1M root")
    with open(path) as handle:
        raw = json.load(handle)
    if repair_segments:
        repair(raw)
    return [Entry(file=x["file"], modify_type=x["modify_type"], video_frames=x["video_frames"],
                  audio_fake_segments=x.get("audio_fake_segments") or [],
                  visual_fake_segments=x.get("visual_fake_segments") or []) for x in raw]
