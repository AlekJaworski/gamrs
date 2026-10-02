"""A rank-deficient design fits with the aliased coefficient at 0 and says which.

Shaped like the two real hedonic fits behind FEATURE REQUESTS §11, every value
drawn from the rng: two land-use tags marking the same homes, and two bath
counts that always sum to 2 (so together they reproduce the intercept). Each
used to fail with "singular system: Cholesky failed".
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import gamrs

pytestmark = pytest.mark.smoke

KW = dict(k_default=6, term_k_mapping={"quality": 12, "condition": 12})


def draw(rng, n, levels):
    return pd.DataFrame({c: rng.choice(np.sort(rng.uniform(0, 1, u)), n) for c, u in levels.items()})


def duplicate_tags(seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """300 sales, 10 house features + 12 tags; `has_docks` is `water`."""
    rng = np.random.default_rng(seed)
    X = draw(rng, 300, {"gla": 101, "lot_sqft": 202, "year_built": 15, "bedrooms": 4,
                        "baths_full": 2, "baths_half": 2, "garage_spaces": 3,
                        "quality": 14, "condition": 19})
    for c, k in {"water": 3, "open_space": 13, "commercial": 15, "parks": 12, "school": 6,
                 "school_fields": 8}.items():
        X[c] = 0.0
        X.loc[rng.choice(300, k, replace=False), c] = 1.0
    X["has_docks"] = X["water"]
    return X, rng.normal(0, 1, 300)


def complementary_baths(seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """100 sales; `baths_full + baths_half == 2` on every row."""
    rng = np.random.default_rng(seed)
    X = draw(rng, 100, {"gla": 41, "lot_sqft": 77, "year_built": 16, "bedrooms": 4,
                        "garage_spaces": 2, "quality": 13, "condition": 19})
    X["baths_full"] = 2.0
    X.loc[rng.choice(100, 3, replace=False), "baths_full"] = 1.0
    X["baths_half"] = 2.0 - X["baths_full"]
    return X, rng.normal(0, 1, 100)


def fit(X, y, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = gamrs.Gam(**{**KW, **kw}).fit(X, y)
    return g, [str(w.message) for w in caught if "rank deficient" in str(w.message)]


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(
    "make, aliased, partners",
    [
        (duplicate_tags, "has_docks", "water"),
        (complementary_baths, "baths_half", "(Intercept), baths_full"),
    ],
)
def test_aliased_coefficient_is_zero_and_the_fit_is_the_reduced_one(make, aliased, partners, seed):
    X, y = make(seed)
    g, msgs = fit(X, y)
    ref, ref_msgs = fit(X.drop(columns=aliased), y)

    assert g.aliased_ == [aliased]
    assert g.rank_ == len(g.coef_) - 1
    assert len(msgs) == 1 and f"{aliased} (determined by {partners})" in msgs[0]
    assert ref.aliased_ == [] and ref_msgs == []
    np.testing.assert_allclose(g.predict(X), ref.predict(X.drop(columns=aliased)), rtol=0, atol=1e-6)
    assert g.reml_value_ == pytest.approx(ref.reml_value_, abs=1e-6)


def test_the_intercept_is_kept():
    X, y = complementary_baths(0)
    g, _ = fit(X, y)
    assert g.intercept_ != 0.0


@pytest.mark.parametrize("family", ["binomial", "poisson", "scat"])
def test_other_families_fit(family):
    X, y = duplicate_tags(0)
    X = X[["gla", "water", "has_docks"]]
    y = (y > 0).astype(float) if family == "binomial" else np.exp(y) if family == "poisson" else y
    if family == "poisson":
        y = np.random.default_rng(0).poisson(y).astype(float)
    g, _ = fit(X, y, family=family)
    assert g.aliased_ == ["has_docks"]
    assert np.isfinite(g.predict(X)).all()


def test_well_posed_design_reports_full_rank():
    X, y = duplicate_tags(0)
    g, msgs = fit(X.drop(columns="has_docks"), y)
    assert g.aliased_ == [] and g.rank_ == len(g.coef_) and msgs == []


def test_aliased_survives_save_and_load(tmp_path):
    X, y = duplicate_tags(0)
    g, _ = fit(X, y)
    g.save(tmp_path / "m.gamrs")
    h = gamrs.Gam.load(tmp_path / "m.gamrs")
    assert h.aliased_ == ["has_docks"] and h.rank_ == g.rank_
    np.testing.assert_array_equal(h.predict(X), g.predict(X))


def test_shash_block_with_a_duplicate_tag_fits():
    X, y = duplicate_tags(0)
    full = [gamrs.CrTerm("gla", k=8), gamrs.ParametricTerm("water"), gamrs.ParametricTerm("has_docks")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = gamrs.fit_shash(X, y, mu_terms=full)
        b = gamrs.fit_shash(X, y, mu_terms=full[:2])
    np.testing.assert_array_equal(a.predict_quantile(X, 0.9), b.predict_quantile(X, 0.9))


@pytest.mark.parametrize("value, partner", [(0.0, "identically zero"), (3.5, "determined by (Intercept)")])
def test_constant_parametric_term_is_aliased_not_refused(value, partner):
    # A typed ParametricTerm on a constant column used to raise; it is the
    # intercept (or nothing) again, so it is zeroed and named like any alias.
    X, y = duplicate_tags(0)
    X = X[["gla", "water"]].assign(c=value)
    terms = [gamrs.CrTerm("gla", k=8), gamrs.ParametricTerm("water"), gamrs.ParametricTerm("c")]
    g, msgs = fit(X, y, terms=terms)
    ref, _ = fit(X, y, terms=terms[:2])
    assert g.aliased_ == ["c"]
    assert len(msgs) == 1 and f"c ({partner})" in msgs[0]
    np.testing.assert_allclose(g.predict(X), ref.predict(X), rtol=0, atol=1e-10)
