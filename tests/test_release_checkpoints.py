"""With the released checkpoints and features downloaded, evaluation reproduces data/ exactly.

These runs take about 5 minutes each on a GPU, so they run only when MF_RUN_RELEASE=1 and
checkpoints/download.py has fetched the release (or MODALFIDELITY_DATA points at it).
"""
import json
import os
import subprocess
import sys

import pytest

from modalfidelity.paths import checkpoint, saved_data_dir

RUN = os.environ.get("MF_RUN_RELEASE") == "1"
pytestmark = pytest.mark.skipif(not RUN or not os.path.isdir(checkpoint("features")),
                                reason="set MF_RUN_RELEASE=1 and download the release first")

METHODS = ["random", "uniform", "image_only", "audio_only", "multimodal", "router", "oracle"]


@pytest.mark.parametrize("seed", ["seed0", "seed1", "seed2", "seed3"])
@pytest.mark.parametrize("which,expected", [("router", "budget_curves"),
                                            ("router_stage1", "budget_curves_stage1")])
def test_evaluation_reproduces_saved_curves(seed, which, expected, tmp_path):
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    # Fig. 3 seeds the random baseline with 0 for every router; the stage-1 file used the
    # router's own training seed (see data/MANIFEST.json)
    random_seed = "0" if which == "router" else seed[-1]
    subprocess.run([sys.executable, "-m", "modalfidelity.cli.evaluate_router",
                    "--checkpoint", checkpoint("router", seed, which + ".pt"), "--seed", random_seed,
                    "--device", device, "--out", str(tmp_path)], check=True)
    got = json.load(open(tmp_path / "detector_accuracy.json"))
    want = json.load(open(os.path.join(saved_data_dir(), expected, seed + ".json")))
    assert got["rhos"] == want["rhos"]
    for metric in ("accuracy", "balanced_accuracy", "recall", "fpr"):
        for m in METHODS:
            diff = max(abs(a - b) for a, b in zip(got[metric][m], want[metric][m]))
            assert diff <= 1e-9, (metric, m, diff)
