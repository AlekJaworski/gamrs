"""A rank-deficient design fits with the aliased coefficient at 0 and says which.

Both shapes come from real hedonic fits: two land-use tags marking the same
homes, and two bath counts that always sum to 3 (so together they reproduce the
intercept). Each used to fail with "singular system: Cholesky failed".
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import gamrs

pytestmark = pytest.mark.smoke


def design(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 200
    X = pd.DataFrame({"gla": rng.uniform(800, 3000, n), "age": rng.uniform(0, 60, n)})
    X["water"] = (rng.uniform(size=n) < 0.03).astype(float)
    X["baths_full"] = rng.choice([1.0, 2.0], n, p=[0.1, 0.9])
    y = 12 + 3e-4 * X.gla - 0.01 * X.age + 0.1 * X.water + 0.05 * X.baths_full + rng.normal(0, 0.1, n)
    return X, y.to_numpy()


def fit_quiet(X, y, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = gamrs.Gam(**kw).fit(X, y)
    return g, [str(w.message) for w in caught if "rank deficient" in str(w.message)]


@pytest.mark.parametrize("family", ["gaussian", "scat"])
def test_duplicate_tag_is_zeroed_and_named(family):
    X, y = design()
    X["has_docks"] = X["water"]
    g, msgs = fit_quiet(X, y, family=family)
    ref, _ = fit_quiet(X.drop(columns="has_docks"), y, family=family)

    assert g.aliased_ == ["has_docks"]
    assert g.rank_ == len(g.coef_) - 1
    assert len(msgs) == 1 and "has_docks" in msgs[0]
    assert g.coef_[-1] == 0.0
    np.testing.assert_allclose(
        g.predict(X), ref.predict(X.drop(columns="has_docks")), rtol=0, atol=1e-6
    )


def test_complement_keeps_the_intercept():
    X, y = design()
    X["baths_half"] = 3.0 - X["baths_full"]
    g, msgs = fit_quiet(X, y)
    assert g.aliased_ == ["baths_half"]
    assert g.intercept_ != 0.0
    assert len(msgs) == 1


def test_well_posed_design_reports_full_rank():
    X, y = design()
    g, msgs = fit_quiet(X, y)
    assert g.aliased_ == [] and g.rank_ == len(g.coef_) and msgs == []


def test_aliased_survives_save_and_load(tmp_path):
    X, y = design()
    X["has_docks"] = X["water"]
    g, _ = fit_quiet(X, y)
    g.save(tmp_path / "m.gamrs")
    h = gamrs.Gam.load(tmp_path / "m.gamrs")
    assert h.aliased_ == ["has_docks"] and h.rank_ == g.rank_
    np.testing.assert_array_equal(h.predict(X), g.predict(X))


def test_shash_block_with_a_duplicate_tag_fits():
    X, y = design()
    X["has_docks"] = X["water"]
    full = [gamrs.CrTerm("gla", k=8), gamrs.ParametricTerm("water"), gamrs.ParametricTerm("has_docks")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = gamrs.fit_shash(X, y, mu_terms=full)
        b = gamrs.fit_shash(X, y, mu_terms=full[:2])
    np.testing.assert_allclose(
        a.predict_quantile(X, 0.9), b.predict_quantile(X, 0.9), rtol=0, atol=1e-6
    )
