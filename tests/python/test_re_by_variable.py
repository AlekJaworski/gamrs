"""mgcv's numeric `s(g, by=w, bs="re")` through the `Gam` surface:
`predictor_basis_map={"g": "re"}` with `term_by_mapping={"g": "w"}` on the
predictors path and `ReTerm(by=)` on the typed-term path.

The parity fixtures (`scripts/r/gen_re_by_fixtures.R`) fit
`golf + s(course, by=golf, bs="re") + s(x)` with every non-golf row on a
placeholder course 0, and a random slope `w + s(g, by=w, bs="re") + s(x)`
with group 6 seen only where w = 0. Unlike a CR by-term, a random effect is
fully penalised, so the by column named as a parametric term is not aliased
with it: it carries the mean, the random effect the deviations from it.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamrs
from conftest import load_fixture

GOLF = "re_by_golf_gaussian_n400_cr8"
SLOPE = "re_by_slope_gaussian_n400_cr8"


def data(name):
    fx = load_fixture(name)
    return (
        fx,
        fx["inputs"]["columns"],
        np.asarray(fx["inputs"]["x_train"], dtype=float),
        np.asarray(fx["inputs"]["y_train"], dtype=float),
        np.asarray(fx["inputs"]["x_new"], dtype=float),
    )


def fit_mapped(cols, X, y, **kwargs):
    w, g, _ = cols
    kwargs.setdefault("term_by_mapping", {g: w})
    return gamrs.Gam(
        predictors=cols,
        predictor_basis_map={w: "parametric", g: "re"},
        k_default=8,
        **kwargs,
    ).fit(X, y)


def fit_typed(cols, X, y):
    w, g, x = cols
    terms = [gamrs.ParametricTerm(w), gamrs.ReTerm(g, by=w), gamrs.CrTerm(x, k=8)]
    return gamrs.Gam(predictors=cols, terms=terms).fit(X, y)


@pytest.mark.parity
@pytest.mark.parametrize("name", [GOLF, SLOPE])
def test_re_by_matches_mgcv(name):
    fx, cols, X, y, X_new = data(name)
    m = fx["mgcv_output"]
    for g in (fit_mapped(cols, X, y), fit_typed(cols, X, y)):
        assert [end - start + 1 for _, start, end in g.get_term_indices()][1:] == m[
            "n_coef_per_term"
        ]
        assert g.aliased_ == []
        assert np.max(np.abs(g.predict(X) - m["fitted"])) < 1e-5
        assert np.max(np.abs(g.predict(X_new) - m["predictions_new"])) < 1e-5
        assert np.max(np.abs(g.predict(X, type="terms")[:, 1:].T - m["term_train"])) < 1e-5
        assert np.max(np.abs(g.predict(X_new, type="terms")[:, 1:].T - m["term_new"])) < 1e-5
        _, start, end = g.get_term_indices()[1]
        assert np.max(np.abs(g.coef_[start : end + 1] - m["coef_re"])) < 1e-5
        assert np.max(np.abs(g.edf_ - m["edf_per_term"])) < 1e-4
        assert g.reml_value_ == pytest.approx(m["reml"], abs=1e-6)


@pytest.mark.smoke
def test_re_by_is_zero_where_w_is_zero_and_scales_with_w():
    _, cols, X, y, X_new = data(SLOPE)
    g = fit_mapped(cols, X, y)
    for Xs in (X, X_new):
        terms = g.predict(Xs, type="terms")
        assert np.all(terms[Xs[:, 0] == 0.0, 1] == 0.0)
    base = g.predict(X_new, type="terms")[:, 1]
    doubled = X_new.copy()
    doubled[:, 0] *= 2.0
    np.testing.assert_allclose(g.predict(doubled, type="terms")[:, 1], 2 * base, atol=1e-12)
    assert np.max(np.abs(base)) > 0.1


@pytest.mark.smoke
def test_placeholder_course_is_fitted_as_zero():
    _, cols, X, y, _ = data(GOLF)
    g = fit_typed(cols, X, y)
    _, start, _ = g.get_term_indices()[1]
    assert abs(g.coef_[start]) < 1e-12  # course 0, the lowest level


@pytest.mark.smoke
def test_by_column_is_data_unless_named():
    _, cols, X, y, _ = data(GOLF)
    w, grp, _ = cols
    g = gamrs.Gam(
        predictors=cols,
        predictor_basis_map={grp: "re"},
        term_by_mapping={grp: w},
        k_default=8,
    ).fit(X, y)
    assert [name for name, *_ in g.get_term_indices()] == [grp, "x"]


@pytest.mark.smoke
def test_re_by_survives_save_load(tmp_path):
    _, cols, X, y, X_new = data(GOLF)
    g = fit_mapped(cols, X, y)
    path = tmp_path / "re_by.gamrs"
    g.save(path)
    restored = gamrs.Gam.load(path)
    np.testing.assert_array_equal(restored.predict(X_new), g.predict(X_new))
    np.testing.assert_array_equal(
        restored.predict(X_new, type="terms"), g.predict(X_new, type="terms")
    )


@pytest.mark.smoke
def test_unknown_re_by_column_raises():
    _, cols, X, y, _ = data(GOLF)
    with pytest.raises(ValueError, match="'q' not found"):
        gamrs.Gam(predictors=cols, terms=[gamrs.ReTerm("course", by="q")]).fit(X, y)


@pytest.mark.smoke
def test_by_mapping_on_a_parametric_term_still_raises():
    _, cols, X, y, _ = data(GOLF)
    with pytest.raises(ValueError, match="needs a CR smooth or an 're' term"):
        gamrs.Gam(
            predictors=cols,
            predictor_basis_map={"course": "parametric"},
            term_by_mapping={"course": "golf"},
        ).fit(X, y)
