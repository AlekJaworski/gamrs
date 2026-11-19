"""Parity tier — prior-weighted fits of the Newton-weight families vs mgcv.

Fixtures from ``scripts/r/gen_weighted_fixtures.R``: 2-D additive, log link,
prior weights drawn from U(0.2, 3).
"""

from __future__ import annotations

import numpy as np
import pytest

import gamrs

from conftest import load_fixture, max_rel_err, x_train_2d, y_train

pytestmark = pytest.mark.parity


def _fit(stem, family, **kwargs):
    fx = load_fixture(stem)
    x = x_train_2d(fx)
    w = np.asarray(fx["inputs"]["weights"], dtype=np.float64)
    fitted = gamrs.fit_additive(
        family, x, y_train(fx), [gamrs.CrTerm(0, k=8), gamrs.CrTerm(1, k=8)], weights=w, **kwargs
    )
    mo = fx["mgcv_output"]
    rel = max_rel_err(np.exp(np.asarray(fitted.predict(x))), np.asarray(mo["predictions_train"]))
    drho = np.max(np.abs(np.asarray(fitted.rho) - np.log(mo["sp"])))
    return fx, fitted, rel, drho


@pytest.mark.parametrize(
    "stem,family",
    [
        ("2d_gamma_log_weighted_n400_k8_cr", "gamma"),
        ("2d_invgauss_log_weighted_n400_k8_cr", "inverse_gaussian"),
        ("2d_nb_log_weighted_n400_k8_cr", "nb"),
        ("2d_tw_log_weighted_n400_k8_cr", "tw"),
    ],
)
def test_weighted_parity(stem, family):
    fx, fitted, rel, drho = _fit(stem, family)
    assert rel < 5e-5, f"{stem}: μ rel {rel:.3e}"
    assert drho < 1e-3, f"{stem}: max |Δρ| {drho:.3e}"
    if family == "nb":
        theta = float(np.exp(fitted.shape_params[0]))
        assert abs(theta / fx["inputs"]["nb_theta_hat"] - 1) < 1e-4, theta
    assert abs(fitted.reml_value - fx["mgcv_output"]["reml"]) < 1e-5

