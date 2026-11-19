"""Parity tier — edf, scale and coefficient covariance of the non-canonical
link families and the estimated-scale families against mgcv.

mgcv computes edf and ``Vp`` from the Fisher weights even where the REML score
uses Newton weights, and reports the Fletcher scale estimate.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamrs

from conftest import load_fixture, y_train

pytestmark = pytest.mark.parity


@pytest.mark.parametrize(
    "stem,family",
    [
        ("1d_gamma_log_n300_k10_cr", "gamma"),
        ("1d_invgauss_log_n300_k10_cr", "inverse_gaussian"),
        ("1d_nb_log_n300_k10_cr", "nb"),
        ("1d_tweedie_log_n300_k10_cr", "tw"),
        ("1d_quasipoisson_log_n300_k10_cr", "quasipoisson"),
        ("1d_quasibinomial_logit_n300_k10_cr", "quasibinomial"),
    ],
)
def test_edf_scale_vcov_match_mgcv(stem, family):
    fx = load_fixture(stem)
    y = y_train(fx)
    x = np.asarray(fx["inputs"]["x_train"], dtype=np.float64).reshape(len(y), -1)
    f = gamrs.fit_additive(family, x, y, [gamrs.CrTerm(0, k=10)])
    mo = fx["mgcv_output"]
    rel = lambda a, b: abs(a / b - 1)
    assert rel(f.edf_total, mo["edf_total"]) < 1e-4, (f.edf_total, mo["edf_total"])
    if family != "nb":
        assert rel(f.scale, mo["scale"]) < 1e-4, (f.scale, mo["scale"])
    vp = np.asarray(mo["vcov"])
    vrel = np.max(np.abs(np.asarray(f.vcov()) - vp)) / np.max(np.abs(vp))
    assert vrel < 1e-3, vrel
