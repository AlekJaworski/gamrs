"""Parity tier — ``method="bam"`` against mgcv ``bam(method = "fREML")``.

Fixtures from ``scripts/r/gen_bam_freml_fixtures.R``; x2 is linear, so its λ
is where bam's Newton loop stops on a flat ridge.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamrs

from conftest import load_fixture, x_train_2d, y_train

pytestmark = pytest.mark.parity


@pytest.mark.parametrize(
    "stem",
    ["3d_gaussian_bam_freml_n2000_k10_cr", "3d_gaussian_weighted_bam_freml_n2000_k10_cr"],
)
def test_bam_freml_matches_mgcv_bam(stem):
    fx = load_fixture(stem)
    x = x_train_2d(fx)
    ref = fx["bam"]
    g = gamrs.Gam(terms=[gamrs.CrTerm(i, k=10) for i in range(3)], method="bam").fit(
        x, y_train(fx), sample_weight=fx["inputs"].get("weights")
    )
    fitted = np.max(np.abs(g.predict(x) / np.asarray(ref["predictions_train"]) - 1))
    sp = np.max(np.abs(np.asarray(g.lambda_) / np.asarray(ref["sp"]) - 1))
    assert fitted < 1e-9, f"fitted rel {fitted:.3e}"
    assert sp < 1e-7, f"sp rel {sp:.3e}"
    assert abs(g.edf_total_ / ref["edf_total"] - 1) < 1e-9
    assert abs(g.reml_value_ / ref["reml"] - 1) < 1e-10
    assert abs(g.scale_ / ref["scale"] - 1) < 1e-9


def test_bam_freml_refuses_other_families():
    fx = load_fixture("3d_nb_log_bam_freml_n2000_k10_cr")
    with pytest.raises(ValueError, match="gaussian"):
        gamrs.Gam(
            terms=[gamrs.CrTerm(i, k=10) for i in range(3)], family="negbin", method="bam"
        ).fit(x_train_2d(fx), y_train(fx))
