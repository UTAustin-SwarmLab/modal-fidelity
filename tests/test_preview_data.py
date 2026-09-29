"""AV-Deepfake1M metadata repair and preview-cache helpers."""
import json

import numpy as np

from modalfidelity.data.avdeepfake1m import load_metadata, repair
from modalfidelity.data.preview_cache import quantise


def test_repair_clears_only_the_impossible_stream(tmp_path):
    raw = [
        {"file": "a/b/c/real_video_fake_audio.mp4", "modify_type": "audio_modified",
         "video_frames": 100, "audio_frames": 1, "fake_segments": [[1.0, 1.5]],
         "audio_fake_segments": [[1.0, 1.5]], "visual_fake_segments": [[1.0, 1.5]]},
        {"file": "a/b/c/fake_video_real_audio.mp4", "modify_type": "visual_modified",
         "video_frames": 100, "audio_frames": 1, "fake_segments": [[2.0, 2.5]],
         "audio_fake_segments": [[2.0, 2.5]], "visual_fake_segments": [[2.0, 2.5]]},
        {"file": "a/b/c/fake_video_fake_audio.mp4", "modify_type": "both_modified",
         "video_frames": 100, "audio_frames": 1, "fake_segments": [[3.0, 3.5]],
         "audio_fake_segments": [[3.0, 3.5]], "visual_fake_segments": [[3.0, 3.5]]},
    ]
    json.dump(raw, open(tmp_path / "train_metadata.json", "w"))
    entries = {e.modify_type: e for e in load_metadata(str(tmp_path), "train")}
    assert entries["audio_modified"].visual_fake_segments == []
    assert entries["audio_modified"].audio_fake_segments == [[1.0, 1.5]]
    assert entries["visual_modified"].audio_fake_segments == []
    assert entries["both_modified"].audio_fake_segments == [[3.0, 3.5]]
    assert entries["both_modified"].visual_fake_segments == [[3.0, 3.5]]
    raw2 = json.load(open(tmp_path / "train_metadata.json"))
    assert repair(raw2) == 2 and repair(raw2) == 0


def test_quantise_round_trip():
    x = np.linspace(-13.8, 2.6, 256 * 40, dtype=np.float32).reshape(256, 40)
    q, lo, hi = quantise(x)
    assert q.dtype == np.uint8 and q.min() == 0 and q.max() >= 254
    assert np.abs(q / 255.0 * (hi - lo) + lo - x).max() < (hi - lo) / 255 + 1e-6
