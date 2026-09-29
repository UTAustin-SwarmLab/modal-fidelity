"""The reproduction notebook runs top to bottom, and the executed copy committed at the repository
root is clean and up to date (its saved figures and numbers are what a fresh run produces)."""
import json
import os
import re
import shutil
import subprocess

import pytest

from modalfidelity.paths import REPO_ROOT

NOTEBOOK = os.path.join(REPO_ROOT, "reproduce_paper.ipynb")


def outputs(path):
    return [c.get("outputs") for c in json.load(open(path))["cells"]]


def test_committed_notebook_is_executed_and_clean():
    raw = open(NOTEBOOK).read()
    nb = json.loads(raw)
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(c.get("outputs") for c in code), "commit the executed notebook"
    assert sum("image/png" in o.get("data", {}) for c in code for o in c["outputs"]) == 2
    assert "28/28 paper numbers reproduced" in raw
    # no machine paths, usernames or execution timestamps in what GitHub shows
    assert not re.search(r"/home/|/mnt/|/Users/|iopub\.|\d{4}-\d\d-\d\dT\d\d:", raw)


@pytest.mark.skipif(shutil.which("jupyter") is None, reason="jupyter not installed")
def test_notebook_executes_and_matches_the_committed_outputs(tmp_path):
    subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", NOTEBOOK,
                    "--output-dir", str(tmp_path), "--ExecutePreprocessor.timeout=600",
                    "--ExecutePreprocessor.record_timing=False"], check=True, cwd=REPO_ROOT)
    assert outputs(tmp_path / "reproduce_paper.ipynb") == outputs(NOTEBOOK), (
        "the committed notebook is stale; refresh it with the command in the README")
