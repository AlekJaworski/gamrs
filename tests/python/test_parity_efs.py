"""Parity tier — ``method="fREML"`` against mgcv's Fellner-Schall optimiser.

gamrs's fREML is the Fellner-Schall update of mgcv's ``gam(optimizer="efs")``
(``efsudr``). Fixtures from ``scripts/r/gen_efs_fixtures.R``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gamrs

from conftest import load_fixture, max_rel_err, x_train_2d, y_train

pytestmark = pytest.mark.parity


@pytest.mark.parametrize(
    "stem,family",
    [
        ("2d_gamma_log_efs_n2000_k10_cr", "gamma"),
        ("2d_quasipoisson_log_efs_n2000_k10_cr", "quasipoisson"),
    ],
)
def test_freml_matches_mgcv_efs(stem, family):
    """Estimated-scale families: the update reads mgcv's scale estimate, so λ
    lands on efs's (it was off by 1/φ: log λ +1.04 for Gamma, −0.88 quasi)."""
    fx = load_fixture(stem)
    x = x_train_2d(fx)
    g = gamrs.Gam(
        terms=[gamrs.CrTerm(0, k=10), gamrs.CrTerm(1, k=10)], family=family, method="fREML"
    ).fit(x, y_train(fx))
    drho = np.max(np.abs(np.asarray(g.rho_) - np.log(fx["efs"]["sp"])))
    rel = max_rel_err(np.asarray(g.predict(x)), np.asarray(fx["efs"]["predictions_train"]))
    assert drho < 1e-3, f"max |Δlog λ| vs efs {drho:.3e}"
    assert rel < 1e-5, f"fitted rel {rel:.3e}"


def test_freml_takes_flat_terms_to_a_straight_line():
    """A term with no curvature to fit has REML's λ → ∞. fREML used to stop at
    λ = 1e7 (log λ 16.1), short of the straight line REML reaches."""
    rng = np.random.default_rng(0)
    n = 20000
    X = pd.DataFrame({k: rng.uniform(size=n) for k in "abcd"})
    y = rng.poisson(np.exp(0.3 + np.sin(6 * X.a) + X.b**2 - X.c)).astype(float)
    terms = [gamrs.CrTerm(k, k=10) for k in "abcd"]
    fs = gamrs.Gam(terms=terms, family="poisson", method="fREML").fit(X, y)
    reml = gamrs.Gam(terms=terms, family="poisson", method="REML").fit(X, y)
    assert fs.converged_
    assert np.atleast_1d(fs.rho_)[2] > 19.0, fs.rho_
    np.testing.assert_allclose(fs.edf_, reml.edf_, atol=0.01)
    assert abs(fs.reml_value_ - reml.reml_value_) < 0.01
