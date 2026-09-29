"""The saved results in data/ are exactly the files MANIFEST.json describes."""
import hashlib
import json
import os

import numpy as np

from modalfidelity.paths import saved_data_dir

DATA = saved_data_dir()


def test_manifest_checksums():
    manifest = json.load(open(os.path.join(DATA, "MANIFEST.json")))
    assert manifest["files"], "empty manifest"
    for name, entry in manifest["files"].items():
        digest = hashlib.sha256(open(os.path.join(DATA, name), "rb").read()).hexdigest()
        assert digest == entry["sha256"], name


def test_every_data_file_is_in_the_manifest():
    listed = set(json.load(open(os.path.join(DATA, "MANIFEST.json")))["files"])
    present = {os.path.relpath(os.path.join(d, f), DATA)
               for d, _, fs in os.walk(DATA) for f in fs if f != "MANIFEST.json"}
    assert present == listed


def test_benchmark_windows_shape():
    d = np.load(os.path.join(DATA, "benchmark_windows.npz"))
    n = len(d["source"])
    assert n == 72287
    for key in d.files:
        assert len(d[key]) == n, key
    # 500 source videos x 4 variants (real, audio-, image-, both-forged); the router evaluation
    # matches 1,999 of these 2,000 videos to the preview cache
    assert len(set(d["source"])) == 500
    assert len({(s, v) for s, v in zip(d["source"], d["variant"])}) == 2000


def test_budget_curves_cover_four_seeds():
    for sub in ("budget_curves", "budget_curves_stage1"):
        files = sorted(os.listdir(os.path.join(DATA, sub)))
        assert files == ["seed0.json", "seed1.json", "seed2.json", "seed3.json"]
        for f in files:
            q = json.load(open(os.path.join(DATA, sub, f)))
            assert q["rhos"] == [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
            assert q["videos"] == 1999
