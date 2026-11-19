"""Parity tier — ``method="bam"`` against mgcv ``bam(method = "fREML")``.

Fixtures from ``scripts/r/gen_bam_freml_fixtures.R``; x2 is linear, so its λ
is where bam's Newton loop stops on a flat ridge. Non-gaussian families run
bam's performance iteration.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamrs

from conftest import load_fixture, x_train_2d, y_train

pytestmark = pytest.mark.parity


def mgcv_theta(family, g):
    sp = np.asarray(g.shape_params_)
    if family == "negbin":
        return [np.exp(sp[0])]
    if family == "scat":
        return [np.exp(sp[1]) + 3.0, np.exp(sp[0] / 2)]
    return []


@pytest.mark.parametrize(
    "stem,family,tol",
    [
        ("3d_gaussian_bam_freml_n2000_k10_cr", "gaussian", 1e-9),
        ("3d_gaussian_weighted_bam_freml_n2000_k10_cr", "gaussian", 1e-9),
        ("3d_nb_log_bam_freml_n2000_k10_cr", "negbin", 1e-8),
        ("3d_scat_identity_weighted_bam_freml_n1500_k10_cr", "scat", 1e-8),
        ("3d_gamma_log_bam_freml_n2000_k10_cr", "gamma", 1e-8),
    ],
)
def test_bam_matches_mgcv_bam(stem, family, tol):
    fx = load_fixture(stem)
    x = x_train_2d(fx)
    ref = fx["bam"]
    g = gamrs.Gam(
        terms=[gamrs.CrTerm(i, k=10) for i in range(3)], family=family, method="bam"
    ).fit(x, y_train(fx), sample_weight=fx["inputs"].get("weights"))
    fitted = np.max(np.abs(g.predict(x) / np.asarray(ref["predictions_train"]) - 1))
    sp = np.max(np.abs(np.asarray(g.lambda_) / np.asarray(ref["sp"]) - 1))
    assert fitted < tol, f"fitted rel {fitted:.3e}"
    assert sp < 100 * tol, f"sp rel {sp:.3e}"
    assert abs(g.edf_total_ / ref["edf_total"] - 1) < tol
    assert abs(g.reml_value_ / ref["reml"] - 1) < 1e-10
    for got, want in zip(mgcv_theta(family, g), ref.get("theta") or []):
        assert abs(got / want - 1) < tol


def test_bam_refuses_unported_families():
    fx = load_fixture("3d_gamma_log_bam_freml_n2000_k10_cr")
    with pytest.raises(ValueError, match="bam is implemented for"):
        gamrs.Gam(
            terms=[gamrs.CrTerm(i, k=10) for i in range(3)], family="tweedie", method="bam"
        ).fit(x_train_2d(fx), y_train(fx))


def lattice(n):
    """The integer lattice ``gen_bam_freml_fixtures.R`` fits above bam's chunk.size."""
    i = np.arange(1, n + 1, dtype=float)

    def u(a, m):
        return np.mod(i * a, m) / m

    x0, x1, x2 = u(7919, 10007), u(6037, 10009), u(4001, 10037)
    e1, e2, e3 = u(7907, 10039) - 0.5, u(5003, 10061) - 0.5, u(3001, 10067)
    s = 4 * x0 * (1 - x0) - 2 * (x1 - 0.5) * (x1 - 0.5) + 0.4 * x2
    return np.column_stack([x0, x1, x2]), {
        "gaussian": 2 + s + 0.8 * e1,
        "negbin": np.floor((1.5 + 3 * s) * 2 * (0.5 + e1) * (0.5 + e2) * 2),
        "scat": 10 + 3 * s + 0.3 * e1 / (0.05 + np.abs(e2)),
    }, 0.5 + 1.5 * e3


@pytest.mark.parametrize("n", [12000, 25000])
@pytest.mark.parametrize(
    "stem,family",
    [
        ("3d_gaussian_bam_freml_n{n}_lattice", "gaussian"),
        ("3d_nb_log_bam_freml_n{n}_lattice", "negbin"),
        ("3d_scat_identity_weighted_bam_freml_n{n}_lattice", "scat"),
    ],
)
def test_bam_above_chunk_size_matches_default_bam(stem, family, n):
    """Above 10,000 rows bam builds its bases from mgcv's seeded row sample."""
    ref = load_fixture(stem.format(n=n))["bam"]
    x, ys, w = lattice(n)
    g = gamrs.Gam(
        terms=[gamrs.CrTerm(i, k=10) for i in range(3)], family=family, method="bam"
    ).fit(x, ys[family], sample_weight=w if family == "scat" else None)
    rows = np.asarray(ref["rows"]) - 1
    fitted = np.max(np.abs(g.predict(x)[rows] / np.asarray(ref["predictions_rows"]) - 1))
    sp = np.max(np.abs(np.asarray(g.lambda_) / np.asarray(ref["sp"]) - 1))
    assert fitted < 1e-6, f"fitted rel {fitted:.3e}"
    assert sp < 1e-5, f"sp rel {sp:.3e}"
    assert abs(g.reml_value_ / ref["reml"] - 1) < 1e-9
