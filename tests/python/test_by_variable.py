"""mgcv's numeric `s(x, by=w)` through the `Gam` surface — `term_by_mapping=`
on the predictors path and `CrTerm(by=)` on the typed-term path.

The parity fixtures fit `y ~ s(x, by = w) + s(z)` in mgcv with w a 0/1
indicator and a distance-band step (1, 0.6, 0.25, 0)
(`scripts/r/gen_by_variable_fixtures.R`). The w column is data for the
by-term, not a term of its own, so `edf_` has two entries, not three.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamrs
from conftest import load_fixture

FIXTURES = ["by_indicator_gaussian_n400_k8_cr", "by_step_gaussian_n400_k8_cr"]
COLS = ["x", "z", "w"]


def data(name):
    fx = load_fixture(name)
    return (
        fx,
        np.asarray(fx["inputs"]["x_train"], dtype=float),
        np.asarray(fx["inputs"]["y_train"], dtype=float),
        np.asarray(fx["inputs"]["x_new"], dtype=float),
    )


def fit_by(X, y, **kwargs):
    kwargs.setdefault("term_by_mapping", {"x": "w"})
    return gamrs.Gam(predictors=COLS, k_default=8, **kwargs).fit(X, y)


@pytest.mark.parity
@pytest.mark.parametrize("name", FIXTURES)
def test_by_term_matches_mgcv(name):
    fx, X, y, X_new = data(name)
    m = fx["mgcv_output"]
    typed = gamrs.Gam(
        predictors=COLS,
        terms=[gamrs.CrTerm("x", k=8, by="w"), gamrs.CrTerm("z", k=8)],
    ).fit(X, y)
    for g in (fit_by(X, y), typed):
        assert len(g.edf_) == 2
        assert np.max(np.abs(g.predict(X) - m["fitted"])) < 1e-5
        assert np.max(np.abs(g.predict(X_new) - m["predictions_new"])) < 1e-5
        assert np.max(np.abs(g.predict(X_new, type="terms").T - m["term_new"])) < 1e-5
        assert np.max(np.abs(g.edf_ - m["edf_per_term"])) < 1e-4
        assert g.reml_value_ == pytest.approx(m["reml"], abs=1e-6)


@pytest.mark.smoke
def test_by_term_is_zero_where_w_is_zero_and_scales_with_w():
    _, X, y, X_new = data(FIXTURES[1])
    g = fit_by(X, y)
    assert [name for name, *_ in g.get_term_indices()] == ["x", "z"]

    train_terms = g.predict(X, type="terms")
    assert np.all(train_terms[X[:, 2] == 0.0, 0] == 0.0)

    base = g.predict(X_new, type="terms")
    doubled = X_new.copy()
    doubled[:, 2] *= 2.0
    zeroed = X_new.copy()
    zeroed[:, 2] = 0.0
    np.testing.assert_allclose(g.predict(doubled, type="terms")[:, 0], 2 * base[:, 0], atol=1e-12)
    np.testing.assert_allclose(g.predict(X_new) - g.predict(zeroed), base[:, 0], atol=1e-12)
    assert np.max(np.abs(base[:, 0])) > 0.1


@pytest.mark.smoke
def test_constant_by_falls_back_to_the_centred_smooth():
    _, X, y, _ = data(FIXTURES[1])
    X = X.copy()
    X[:, 2] = 2.5
    with_by = fit_by(X, y)
    plain = gamrs.Gam(predictors=["x", "z"], k_default=8).fit(X[:, :2], y)
    assert len(with_by.coef_) == len(plain.coef_)
    np.testing.assert_allclose(with_by.predict(X), plain.predict(X[:, :2]), atol=1e-5)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"term_by_mapping": {"q": "w"}}, "match no column"),
        ({"term_by_mapping": {"x": "q"}}, "match no column"),
        ({"term_by_mapping": {"x": "w"}, "term_pc_mapping": {"x": 0.5}}, "pc or by"),
        (
            {"term_by_mapping": {"x": "w"}, "predictor_basis_map": {"x": "parametric"}},
            "needs a CR smooth",
        ),
        (
            {"term_by_mapping": {"x": "w"}, "terms": [gamrs.CrTerm("x"), gamrs.CrTerm("z")]},
            "cannot be combined with terms=",
        ),
    ],
)
def test_a_by_mapping_that_cannot_apply_raises(kwargs, match):
    _, X, y, _ = data(FIXTURES[0])
    with pytest.raises(ValueError, match=match):
        fit_by(X, y, **kwargs)


@pytest.mark.smoke
def test_a_constant_smooth_column_with_a_by_raises_instead_of_being_dropped():
    _, X, y, _ = data(FIXTURES[0])
    X = X.copy()
    X[:, 0] = 0.5
    with pytest.raises(ValueError, match="needs a CR smooth"):
        fit_by(X, y)


@pytest.mark.smoke
def test_typed_by_term_with_too_few_values_raises_instead_of_demoting():
    _, X, y, _ = data(FIXTURES[0])
    X = X.copy()
    X[:, 0] = np.round(X[:, 0])
    g = gamrs.Gam(predictors=COLS, terms=[gamrs.CrTerm("x", by="w"), gamrs.CrTerm("z")])
    with pytest.raises(ValueError, match="no parametric fallback"):
        g.fit(X, y)


@pytest.mark.smoke
def test_by_column_becomes_a_term_only_when_asked():
    """And then it is aliased: an uncentred s(x):w already spans w * constant,
    which is why the by column is not added as a term by default."""
    _, X, y, _ = data(FIXTURES[1])
    with pytest.warns(UserWarning, match=r"w \(determined by s\(x\):w\)"):
        g = fit_by(X, y, predictor_basis_map={"w": "parametric"})
    assert [name for name, *_ in g.get_term_indices()] == ["x", "z", "w"]
    assert g.aliased_ == ["w"]


@pytest.mark.smoke
def test_by_survives_save_load_and_auto_k(tmp_path):
    _, X, y, X_new = data(FIXTURES[1])
    g = fit_by(X, y, auto_k=True)
    zeroed = X_new.copy()
    zeroed[:, 2] = 0.0
    assert np.max(np.abs(g.predict(X_new) - g.predict(zeroed))) > 0.1

    path = tmp_path / "by.gamrs"
    g.save(path)
    restored = gamrs.Gam.load(path)
    np.testing.assert_array_equal(restored.predict(X_new), g.predict(X_new))
    assert [name for name, *_ in restored.get_term_indices()] == ["x", "z"]
