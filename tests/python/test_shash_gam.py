"""Parity tier — native sinh-arcsinh (`shash`) GAMLSS through the Python API.

The Python ``fit_shash`` reproduces the Rust phase-6a result (which matches mgcv
``gam(..., family=shash, bs="cr")`` to ~1e-6) at the FFI boundary the Rust
``fit_shash_matches_mgcv_two_smooth`` test doesn't touch.

Fixture ``shash_gam_mgcv.json`` (from ``scripts/r/gen_shash_gam_fixture.R``)
holds the RAW covariates x0/x1, the response y, mgcv's fitted η (n×4 row-major),
total EDF, and mgcv-derived quantiles at p ∈ {0.1, 0.5, 0.9}. The fixture fits
``bs="cr"`` smooths so the reference basis MATCHES gamrs's ``Cr`` term (mgcv's
default ``s(x,k)`` is a thin-plate spline — a different basis). gamrs builds its
OWN CR designs — s(x0,k=10) for μ, s(x1,k=10) for τ, intercept-only ε and φ —
and recovers mgcv's η, EDF and quantiles to ~1e-6. Tolerances mirror the Rust test.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamrs

from conftest import load_fixture

pytestmark = pytest.mark.parity


def test_shash_gam_matches_mgcv_two_smooth():
    fx = load_fixture("shash_gam_mgcv")
    n = int(fx["n"])
    x0 = np.asarray(fx["x0"], dtype=np.float64)
    x1 = np.asarray(fx["x1"], dtype=np.float64)
    y = np.asarray(fx["y"], dtype=np.float64)
    X = np.column_stack([x0, x1])  # (n, 2)
    # mgcv η is a flat n*4 row-major list: row i = (η_μ, η_τ, η_ε, η_φ).
    eta_mgcv = np.asarray(fx["eta"], dtype=np.float64).reshape(n, 4)

    # gamrs builds its OWN designs: s(x0,k=10) for μ, s(x1,k=10) for τ,
    # intercept-only ε and φ — matching the mgcv formula in the fixture.
    fit = gamrs.fit_shash(
        X,
        y,
        mu_terms=[gamrs.CrTerm(0, k=10)],
        tau_terms=[gamrs.CrTerm(1, k=10)],
        eps_terms=[],
        phi_terms=[],
    )
    assert fit.converged_, "native shash outer REML / inner solve did not converge"
    # Block widths: μ,τ = 10 (CR k=10), ε,φ = 1 (intercept-only).
    assert fit.block_p_ == (10, 10, 1, 1), f"unexpected block widths {fit.block_p_}"
    assert fit.b_ == fx["b"]

    # (1) Fitted η per block vs mgcv (gamrs rebuilds its own design on the same X).
    eta = fit.predict_eta(X)
    assert eta.shape == (n, 4)
    max_eta = float(np.abs(eta - eta_mgcv).max())
    per_block = np.abs(eta - eta_mgcv).max(axis=0)
    print(f"\n[shash-gam] max|η_gamrs − η_mgcv|={max_eta:.3e} per-block={per_block}")
    # gamrs Cr vs the fixture's mgcv bs="cr" fit (same basis) — agrees to ~1e-6.
    assert max_eta < 1e-4, f"fitted η: max|gamrs − mgcv|={max_eta:.3e} >= 1e-4"

    # (2) Total EDF vs mgcv.
    edf_diff = abs(fit.edf_ - float(fx["edf_total"]))
    print(f"[shash-gam] EDF gamrs={fit.edf_:.4f} mgcv={fx['edf_total']:.4f} diff={edf_diff:.3e}")
    assert edf_diff < 1e-3, f"EDF={fit.edf_} vs mgcv {fx['edf_total']} (diff {edf_diff})"

    # (3) Fitted quantiles at p ∈ {0.1, 0.5, 0.9} vs mgcv-derived quantiles.
    for p, key in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
        q = fit.predict_quantile(X, p)
        q_mgcv = np.asarray(fx[key], dtype=np.float64)
        max_q = float(np.abs(q - q_mgcv).max())
        print(f"[shash-gam] p={p} max|q_gamrs − q_mgcv|={max_q:.3e}")
        assert max_q < 1e-4, f"quantile p={p}: max|gamrs − mgcv|={max_q:.3e} >= 1e-4"


@pytest.mark.smoke
def test_shash_gam_predict_quantile_rejects_out_of_range_p():
    """p ∉ (0, 1) must raise ValueError (the native error mapped through PyO3)."""
    # A smooth signal with GENUINE Gaussian noise so the scale (and the near-
    # Gaussian shape) are identifiable — shash, like mgcv's, needs an
    # identifiable σ/shape; near-deterministic data can leave the penalised
    # Hessian singular.
    rng = np.random.default_rng(0)
    n = 200
    x = np.linspace(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x) + 0.4 * rng.standard_normal(n)
    fit = gamrs.fit_shash(
        x.reshape(-1, 1),
        y,
        mu_terms=[gamrs.CrTerm(0, k=6)],
    )
    assert fit.predict_quantile(x.reshape(-1, 1), 0.5).shape == (n,)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            fit.predict_quantile(x.reshape(-1, 1), bad)


def test_fit_shash_on_a_linear_smooth_is_not_round_off():
    """A μ-smooth that is exactly linear drives λ to the edge of the box. At the
    old fixed ρ = 30 the factorisation was past computable, so the answer was
    round-off: on this data scaling y by (1 + 1e-10) moved the 0.9-quantile by
    3.3e-7 (4.2e-5 with one more parametric term); bounded, 2.4e-9. The box now
    ends where `exp(ρ)·S0` stops being resolvable against `X'X`."""
    import warnings

    import pandas as pd

    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"gla": rng.uniform(800, 3000, n), "age": rng.uniform(0, 60, n)})
    X["water"] = (rng.uniform(size=n) < 0.03).astype(float)
    y = (12 + 3e-4 * X.gla - 0.01 * X.age + 0.1 * X.water + rng.normal(0, 0.1, n)).to_numpy()
    terms = [gamrs.CrTerm("gla", k=8), gamrs.ParametricTerm("water")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = gamrs.fit_shash(X, y, mu_terms=terms)
        b = gamrs.fit_shash(X, y * (1 + 1e-10), mu_terms=terms)
    assert np.all(np.asarray(a.rho_) < 30.0)
    np.testing.assert_allclose(a.predict_quantile(X, 0.9), b.predict_quantile(X, 0.9), rtol=0, atol=5e-8)
