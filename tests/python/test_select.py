"""mgcv's `gam(select=TRUE)` through `Gam(select=True)`.

Fixtures from `scripts/r/gen_select_fixtures.R`; the Rust side of the parity
is `tests/select.rs`, which says why the bounds are what they are.
"""

from __future__ import annotations

import warnings

import gamrs
import numpy as np
import pytest
from conftest import load_fixture

CR = "select_cr_gaussian_n400"
TE = "select_te_gaussian_n400_k4x3"
TE_BY_NULL = "select_te_by_null_gaussian_n400_k4x3"

TERMS = {
    CR: [gamrs.CrTerm("x1", k=10), gamrs.CrTerm("x2", k=8), gamrs.CrTerm("x3", k=8)],
    TE: [gamrs.TeTerm(("x", "z"), k=(4, 3))],
    TE_BY_NULL: [gamrs.TeTerm(("x", "z"), k=(4, 3), by="w"), gamrs.CrTerm("v", k=8)],
}


def data(name):
    fx = load_fixture(name)
    cols = fx["inputs"]["columns"]
    return (
        fx["mgcv_output"],
        cols,
        np.asarray(fx["inputs"]["x_train"], dtype=float),
        np.asarray(fx["inputs"]["y_train"], dtype=float),
        np.asarray(fx["inputs"]["x_new"], dtype=float),
    )


@pytest.mark.parity
@pytest.mark.parametrize("name", [CR, TE, TE_BY_NULL])
def test_select_matches_mgcv(name):
    m, cols, X, y, X_new = data(name)
    g = gamrs.Gam(predictors=cols, terms=TERMS[name], select=True).fit(X, y)
    assert len(g.lambda_) == len(m["sp"])
    assert np.max(np.abs(g.predict(X) - m["fitted"])) < 1e-4
    assert np.max(np.abs(g.predict(X_new) - m["predictions_new"])) < 1e-4
    assert np.max(np.abs(g.predict(X, type="terms").T - m["term_train"])) < 1e-4
    assert np.max(np.abs(g.edf_ - m["edf_per_term"])) < 1e-3
    assert g.reml_value_ == pytest.approx(m["reml"], abs=2e-4)


def test_select_removes_a_term_with_no_signal():
    _, cols, X, y, _ = data(TE_BY_NULL)
    plain = gamrs.Gam(predictors=cols, terms=TERMS[TE_BY_NULL]).fit(X, y)
    selected = gamrs.Gam(predictors=cols, terms=TERMS[TE_BY_NULL], select=True).fit(X, y)
    assert plain.edf_[0] > 3.99
    assert selected.edf_[0] < 0.01
    assert np.max(np.abs(selected.predict(X, type="terms")[:, 0])) < 1e-3 * y.std()

    table = selected.summary().smooths
    assert list(table["predictor"]) == ["te(x,z):w", "v"]
    assert len(table["lambda"][0]) == 3 and len(table["lambda"][1]) == 2
    assert list(selected.get_edf_df()["predictor"]) == ["te(x,z):w", "v"]


def test_select_on_a_single_smooth():
    _, cols, X, y, _ = data(CR)
    one = gamrs.Gam(predictors=cols, terms=[gamrs.CrTerm("x3", k=8)], select=True).fit(X, y)
    assert len(one.lambda_) == 2
    assert one.edf_[0] < 0.05


def test_bam_refuses_select_and_freml_falls_back_to_reml():
    _, cols, X, y, _ = data(CR)
    with pytest.raises(ValueError, match="select=True"):
        gamrs.Gam(predictors=cols, terms=TERMS[CR], method="bam", select=True).fit(X, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = gamrs.Gam(predictors=cols, terms=TERMS[CR], method="fREML", select=True).fit(X, y)
    assert any("fitting by REML instead" in str(w.message) for w in caught)
    assert len(g.lambda_) == 6
