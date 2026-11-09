"""mgcv's numeric `te(x, z, by=w)` through the `Gam` surface:
`TeTerm(cols, k, by=)`.

The parity fixtures (`scripts/r/gen_te_by_fixtures.R`) fit
`te(x, z, k=c(4, 3), bs="cr", by=x)` with x >= 0, a surface pinned to 0
along x = 0, and `te(x, z, by=w) + s(v)` with w a covariate of its own.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamrs
from conftest import load_fixture

SELF = "te_by_self_gaussian_n400_k4x3_cr"
COVARIATE = "te_by_covariate_gaussian_n400_k4x3_cr"


def data(name):
    fx = load_fixture(name)
    return (
        fx,
        fx["inputs"]["columns"],
        np.asarray(fx["inputs"]["x_train"], dtype=float),
        np.asarray(fx["inputs"]["y_train"], dtype=float),
        np.asarray(fx["inputs"]["x_new"], dtype=float),
    )


def model(name, by):
    if name == SELF:
        return [gamrs.TeTerm(cols=("x", "z"), k=(4, 3), by=by)]
    return [gamrs.TeTerm(cols=("x", "z"), k=(4, 3), by=by), gamrs.CrTerm("v", k=8)]


def by_col(name):
    return "x" if name == SELF else "w"


@pytest.mark.parity
@pytest.mark.parametrize("name", [SELF, COVARIATE])
def test_te_by_matches_mgcv(name):
    fx, cols, X, y, X_new = data(name)
    m = fx["mgcv_output"]
    g = gamrs.Gam(predictors=cols, terms=model(name, by_col(name))).fit(X, y)
    assert [end - start + 1 for _, start, end in g.get_term_indices()] == m["n_coef_per_term"]
    # One margin's λ runs along a ridge towards infinity in both fits; the
    # bounds absorb where gamrs stops on it (tests/te_by_variable.rs).
    assert np.max(np.abs(g.predict(X) - m["fitted"])) < 5e-5
    assert np.max(np.abs(g.predict(X_new) - m["predictions_new"])) < 5e-5
    assert np.max(np.abs(g.predict(X, type="terms").T - m["term_train"])) < 5e-5
    assert np.max(np.abs(g.predict(X_new, type="terms").T - m["term_new"])) < 5e-5
    # edf_ has one entry per penalty and the tensor holds the first two.
    other = g.edf_[2:]
    edf = np.concatenate([[g.edf_total_ - 1 - other.sum()], other])
    assert np.max(np.abs(edf - m["edf_per_term"])) < 1e-3
    assert g.reml_value_ == pytest.approx(m["reml"], abs=1e-5)

    w = cols.index(by_col(name))
    for Xs in (X, X_new):
        terms = g.predict(Xs, type="terms")
        assert np.all(terms[Xs[:, w] == 0.0, 0] == 0.0)


@pytest.mark.smoke
def test_te_by_scales_with_w_and_survives_save_load(tmp_path):
    _, cols, X, y, X_new = data(COVARIATE)
    g = gamrs.Gam(predictors=cols, terms=model(COVARIATE, "w")).fit(X, y)
    base = g.predict(X_new, type="terms")[:, 0]
    doubled = X_new.copy()
    doubled[:, 2] *= 2.0
    np.testing.assert_allclose(g.predict(doubled, type="terms")[:, 0], 2 * base, atol=1e-12)
    assert np.max(np.abs(base)) > 0.1

    lp = g.evaluate_lpmatrix(X_new)
    _, start, end = g.get_term_indices()[0]
    np.testing.assert_allclose(lp[:, start : end + 1] @ g.coef_[start : end + 1], base, atol=1e-12)

    path = tmp_path / "te_by.gamrs"
    g.save(path)
    restored = gamrs.Gam.load(path)
    np.testing.assert_array_equal(restored.predict(X_new), g.predict(X_new))
    np.testing.assert_array_equal(restored.predict(X_new, type="terms"), g.predict(X_new, type="terms"))


@pytest.mark.smoke
def test_constant_by_falls_back_to_the_centred_tensor():
    fx = load_fixture("2d_gaussian_te_n300_k5x5")
    xz = np.asarray(fx["inputs"]["x_train"], dtype=float)
    X = np.column_stack([xz, np.full(len(xz), 2.5)])
    y = np.asarray(fx["inputs"]["y_train"], dtype=float)
    cols = ["x", "z", "w"]
    with_by = gamrs.Gam(predictors=cols, terms=[gamrs.TeTerm(("x", "z"), k=(5, 5), by="w")]).fit(X, y)
    plain = gamrs.Gam(predictors=cols, terms=[gamrs.TeTerm(("x", "z"), k=(5, 5))]).fit(X, y)
    assert len(with_by.coef_) == len(plain.coef_)
    np.testing.assert_allclose(with_by.predict(X), plain.predict(X), atol=1e-6)


@pytest.mark.smoke
def test_unknown_by_column_raises():
    _, cols, X, y, _ = data(COVARIATE)
    with pytest.raises(ValueError, match="'q' not found"):
        gamrs.Gam(predictors=cols, terms=model(COVARIATE, "q")).fit(X, y)


@pytest.mark.smoke
def test_by_mapping_with_terms_raises():
    _, cols, X, y, _ = data(COVARIATE)
    g = gamrs.Gam(predictors=cols, terms=model(COVARIATE, None), term_by_mapping={"x": "w"})
    with pytest.raises(ValueError, match="cannot be combined with terms="):
        g.fit(X, y)
