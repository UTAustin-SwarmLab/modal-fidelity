"""Figs. 2-3 and Table 1 regenerate from data/ and match the paper."""
import os

import numpy as np
import pytest

from modalfidelity.paths import REPO_ROOT, saved_data_dir
from modalfidelity.plots import claims, fig2, fig3, table1

DATA = saved_data_dir()
ASSETS = os.path.join(REPO_ROOT, "assets")


@pytest.mark.parametrize("module,reference", [
    (fig2, "fig2_confidence_by_forgery.png"),
    (fig3, "fig3_budget_curves.png"),
])
def test_figure_is_pixel_identical_to_the_paper(module, reference, tmp_path):
    from PIL import Image
    png, pdf = module.plot(DATA, str(tmp_path / "fig.png"))
    new = np.asarray(Image.open(png).convert("RGB"))
    ref = np.asarray(Image.open(os.path.join(ASSETS, reference)).convert("RGB"))
    assert new.shape == ref.shape
    assert (new != ref).any(-1).mean() == 0.0


@pytest.mark.parametrize("module", [fig2, fig3])
def test_figure_pdf_has_no_type3_fonts(module, tmp_path):
    png, pdf = module.plot(DATA, str(tmp_path / "fig.png"))
    raw = open(pdf, "rb").read()
    assert b"/Type3" not in raw


def test_figures_refuse_to_overwrite(tmp_path):
    fig3.plot(DATA, str(tmp_path / "fig.png"))
    with pytest.raises(FileExistsError):
        fig3.plot(DATA, str(tmp_path / "fig.png"))


def test_table1_values():
    got = [(round(r[2], 3), round(r[3], 1), round(r[4], 4)) for r in table1.rows(table1.load(DATA))]
    assert got == [(2.0, 4086.6, 0.7264), (2.0, 4086.6, 0.7298), (2.0, 4086.6, 0.7322),
                   (0.13, 257.5, 0.7672), (0.161, 327.3, 0.8151), (0.183, 384.5, 0.8444)]


def test_every_paper_number_is_reproduced():
    rows = claims.check(DATA)
    failed = [r for r in rows if not r[3]]
    assert not failed, failed
    assert len(rows) == len(claims.PAPER)
