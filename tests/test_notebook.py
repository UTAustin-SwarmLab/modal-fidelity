"""The reproduction notebook runs top to bottom."""
import os
import shutil
import subprocess

import pytest

from modalfidelity.paths import REPO_ROOT


@pytest.mark.skipif(shutil.which("jupyter") is None, reason="jupyter not installed")
def test_notebook_executes(tmp_path):
    nb = os.path.join(REPO_ROOT, "notebooks", "reproduce_paper.ipynb")
    subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", nb,
                    "--output-dir", str(tmp_path), "--ExecutePreprocessor.timeout=600"],
                   check=True, cwd=REPO_ROOT)
    assert (tmp_path / "reproduce_paper.ipynb").exists()
