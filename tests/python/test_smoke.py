"""Fast smoke tier — every public Python API path exercised once on tiny data.

Includes explicit regression guards for the two bugs that shipped in 0.3.0 and
were patched in 0.3.1 (see the 2026-05-29 checkpoint):

1. ``gamrs.GAM(...).fit(x, y)`` raised ``TypeError`` on 1-D ``x`` because the
   native ``fit`` was lifted to ``PyReadonlyArray2`` but ``_low_level`` still
   passed 1-D. ``test_fit_accepts_1d_x`` locks the fix.
2. ``TpsTerm`` was not exported / not dispatchable. ``test_tpsterm_*`` lock it.

Both classes of bug would have been caught by *any* Python-level test — this
file is the standing guard the checkpoint called for.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import gamrs

pytestmark = pytest.mark.smoke


# --------------------------------------------------------------------------- #
# Package surface                                                             #
# --------------------------------------------------------------------------- #


def test_public_exports_present():
    for name in (
        "GAM",
        "Gam",
        "fit_additive",
        "CrTerm",
        "CrStableTerm",
        "ReTerm",
        "TeTerm",
        "TpsTerm",  # regression: must be importable (0.3.1).
    ):
        assert hasattr(gamrs, name), f"gamrs.{name} missing from public API"


# --------------------------------------------------------------------------- #
# Low-level GAM facade                                                        #
# --------------------------------------------------------------------------- #


def test_fit_accepts_1d_x(toy_gaussian):
    """Regression for the 0.3.0 ``TypeError: 'ndarray' cannot be cast as
    'ndarray'`` on 1-D ``x``. Must fit cleanly and expose k coefficients."""
    x, y = toy_gaussian
    g = gamrs.GAM("gaussian", k=10).fit(x, y)  # x is 1-D — the bug case.
    assert g.converged_
    assert g.coef_.shape == (10,)


def test_fit_accepts_2d_column_x(toy_gaussian):
    x, y = toy_gaussian
    g = gamrs.GAM("gaussian", k=10).fit(x.reshape(-1, 1), y)
    assert g.coef_.shape == (10,)


def test_predict_link_and_response(toy_gaussian):
    x, y = toy_gaussian
    g = gamrs.GAM("gaussian", k=10).fit(x, y)
    link = g.predict(x, scale="link")
    resp = g.predict(x, scale="response")
    assert link.shape == y.shape == resp.shape
    # Gaussian identity link: link == response.
    assert np.allclose(link, resp)


def test_predict_response_inverse_link_poisson(rng):
    x = np.linspace(0, 1, 300)
    mu = np.exp(0.5 + np.sin(2 * np.pi * x))
    y = rng.poisson(mu).astype(float)
    g = gamrs.GAM("poisson", k=10).fit(x, y)
    link = g.predict(x, scale="link")
    resp = g.predict(x, scale="response")
    # log link: response == exp(link), and all positive.
    assert np.allclose(resp, np.exp(link))
    assert np.all(resp > 0)


def test_predict_ci_ordering(toy_gaussian):
    x, y = toy_gaussian
    g = gamrs.GAM("gaussian", k=10).fit(x, y)
    mean, lo, hi = g.predict_ci(x, level=0.95)
    assert mean.shape == lo.shape == hi.shape == y.shape
    assert np.all(lo <= mean + 1e-9)
    assert np.all(mean <= hi + 1e-9)


def test_predict_diff_shapes(toy_gaussian):
    x, y = toy_gaussian
    g = gamrs.GAM("gaussian", k=10).fit(x, y)
    a = np.array([0.25, 0.5])
    b = np.array([0.75, 0.5])
    diff, lo, hi = g.predict_diff(a, b)
    assert diff.shape == (2,)
    assert np.all(lo <= diff + 1e-9) and np.all(diff <= hi + 1e-9)


def test_vcov_symmetric_psd(toy_gaussian):
    x, y = toy_gaussian
    g = gamrs.GAM("gaussian", k=10).fit(x, y)
    v = g.vcov()
    assert v.shape == (g.coef_.size, g.coef_.size)
    assert np.allclose(v, v.T, atol=1e-8)
    # PSD: smallest eigenvalue not meaningfully negative.
    assert np.linalg.eigvalsh(v).min() > -1e-8


def test_getters(toy_gaussian):
    x, y = toy_gaussian
    g = gamrs.GAM("gaussian", k=10).fit(x, y)
    assert isinstance(g.scale_, float) and g.scale_ > 0
    assert isinstance(g.edf_total_, float) and g.edf_total_ > 1.0
    assert isinstance(g.rho_, float)  # single smooth -> scalar.
    assert g.lambda_.shape == (1,)
    assert isinstance(g.n_iters_, int) and g.n_iters_ >= 1
    assert isinstance(g.converged_, bool)


# --------------------------------------------------------------------------- #
# Error paths                                                                 #
# --------------------------------------------------------------------------- #


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        gamrs.GAM("gaussian").predict(np.zeros(3))


def test_length_mismatch_raises(toy_gaussian):
    x, y = toy_gaussian
    with pytest.raises(ValueError):
        gamrs.GAM("gaussian").fit(x, y[:-1])


def test_bad_scale_raises(toy_gaussian):
    x, y = toy_gaussian
    g = gamrs.GAM("gaussian", k=10).fit(x, y)
    with pytest.raises(ValueError):
        g.predict(x, scale="nonsense")


def test_degenerate_input_never_aborts_the_interpreter(rng):
    """Contract guard for the wheel's `panic = "unwind"` strategy.

    A degenerate / non-finite fit must surface as a *catchable* Python
    exception, never a hard process abort. PyO3 converts a Rust panic into a
    `PanicException` (a ``BaseException``) only when the wheel is built with
    `panic = "unwind"` — under `panic = "abort"` the same panic ``SIGABRT``s
    the interpreter and this test's process would die before the final
    assertion (pytest would report the worker as crashed). Reaching the end
    means every degenerate input was handled without killing the process.
    """
    n = 60
    x = rng.standard_normal((n, 1))
    degenerate_responses = [
        np.full(n, np.nan),          # non-finite response
        np.full(n, np.inf),          # non-finite response
        np.full(n, 3.0),             # zero-variance response
        np.zeros(n),                 # all-zero response
    ]
    for y in degenerate_responses:
        try:
            gamrs.GAM("gaussian", k=10).fit(x, y)
        except BaseException:  # noqa: BLE001 — catchable is the whole point
            pass
    assert True  # survived every case ⇒ no input aborted the interpreter


# --------------------------------------------------------------------------- #
# Multi-smooth fit_additive + typed terms                                     #
# --------------------------------------------------------------------------- #


def test_fit_additive_two_cr_terms(rng):
    n = 300
    x = rng.uniform(0, 1, size=(n, 2))
    y = np.sin(2 * np.pi * x[:, 0]) + 0.5 * x[:, 1] ** 2 + rng.normal(0, 0.1, n)
    fitted = gamrs.fit_additive(
        "gaussian", x, y, [gamrs.CrTerm(0, k=8), gamrs.CrTerm(1, k=8)]
    )
    assert fitted.converged
    assert len(fitted.rho) == 2  # one smoothing param per term.
    pred = np.asarray(fitted.predict(x))
    assert pred.shape == (n,)


def test_fit_additive_requires_2d_x(rng):
    with pytest.raises(ValueError):
        gamrs.fit_additive("gaussian", np.linspace(0, 1, 10), np.zeros(10), [gamrs.CrTerm(0)])


def test_fit_additive_empty_terms_raises(rng):
    x = rng.uniform(0, 1, size=(10, 1))
    with pytest.raises(ValueError):
        gamrs.fit_additive("gaussian", x, np.zeros(10), [])


def test_tpsterm_default_k():
    # Regression: TpsTerm must construct and default k to 10 * len(cols).
    t = gamrs.TpsTerm(cols=(0, 1))
    assert t.k is None  # defaulted lazily at the FFI boundary.


def test_tpsterm_fit_dispatch(rng):
    """Regression: the ``"tp"`` basis branch must dispatch (0.3.1 fix)."""
    n = 200
    x = rng.uniform(0, 1, size=(n, 2))
    y = np.sin(3 * x[:, 0]) + np.cos(3 * x[:, 1]) + rng.normal(0, 0.1, n)
    fitted = gamrs.fit_additive("gaussian", x, y, [gamrs.TpsTerm(cols=(0, 1), k=15)])
    assert fitted.converged
    pred = np.asarray(fitted.predict(x))
    assert pred.shape == (n,)
    assert np.all(np.isfinite(pred))


# --------------------------------------------------------------------------- #
# High-level Gam wrapper                                                       #
# --------------------------------------------------------------------------- #


def test_high_level_gam_smoke(toy_gaussian):
    x, y = toy_gaussian
    g = gamrs.Gam(predictors=["x0"], target="y", family="gaussian").fit(
        x.reshape(-1, 1), y
    )
    pred = np.asarray(g.predict(x.reshape(-1, 1)))
    assert pred.shape == y.shape
    assert np.all(np.isfinite(pred))
    assert g.edf_total_ > 1.0


def test_discrete_true_emits_warning(rng):
    """discrete=True is a no-op for API compat; warns to keep users from
    silently mismatching their mgcv expectations."""
    x = rng.uniform(0, 1, 100)
    y = x + rng.normal(0, 0.1, 100)
    with pytest.warns(UserWarning, match="discrete=True is accepted"):
        gamrs.Gam(family="gaussian", discrete=True).fit(x, y)


def test_nthreads_emits_warning(rng):
    """nthreads is accepted for source compat but routes through the BLAS
    env vars in gamrs; warn so users aren't surprised it's a no-op."""
    x = rng.uniform(0, 1, 100)
    y = x + rng.normal(0, 0.1, 100)
    with pytest.warns(UserWarning, match="nthreads=4"):
        gamrs.Gam(family="gaussian", nthreads=4).fit(x, y)


def test_constant_column_auto_dropped(rng):
    """A constant predictor column (n_unique=1) is silently dropped on the
    predictors= path, exposed on dropped_predictors_, and a UserWarning is
    emitted. Predicts from a DataFrame containing the dropped column still
    work (it gets subselected away). Matches mgcv_rust 0.23.0 + mgcv R."""
    pd = pytest.importorskip("pandas")
    n = 200
    df = pd.DataFrame({
        "x0": rng.uniform(0, 10, n),
        "stories": np.full(n, 1.0),
        "x2": rng.uniform(-5, 5, n),
    })
    df["y"] = np.sin(df.x0) + 0.3 * df.x2 + rng.normal(0, 0.3, n)

    with pytest.warns(UserWarning, match="'stories' is constant"):
        g = gamrs.Gam(family="gaussian").fit(df[["x0", "stories", "x2"]], df.y)

    assert g.dropped_predictors_ == {"stories": 1.0}
    assert g._effective_predictors == ["x0", "x2"]
    # Predict from the original (3-col) DataFrame — the constant col gets subselected
    mu = g.predict(df[["x0", "stories", "x2"]])
    assert mu.shape == (n,)


def test_typed_terms_constant_col_raises_clearly(rng):
    """When the user is explicit via `terms=`, a constant column on a
    CrTerm raises a clear `column is constant` ValueError BEFORE any
    auto-promotion — gives a friendlier error than the natural downstream
    singular-matrix failure. (Auto-drop only happens on the predictors=
    path; the typed-term path is fully explicit, so we surface the issue
    rather than silently mangling the design.)"""
    pd = pytest.importorskip("pandas")
    n = 200
    df = pd.DataFrame({
        "x0": rng.uniform(0, 10, n),
        "stories": np.full(n, 1.0),
    })
    df["y"] = np.sin(df.x0) + rng.normal(0, 0.3, n)
    with pytest.raises(ValueError, match="column is constant"):
        gamrs.Gam(
            terms=[gamrs.CrTerm("x0", k=10), gamrs.CrTerm("stories", k=10)]
        ).fit(df[["x0", "stories"]], df.y)


def test_all_constant_predictors_raises(rng):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": np.ones(100), "b": np.full(100, 2.0)})
    df["y"] = np.zeros(100)
    with pytest.raises(ValueError, match="all predictor columns are constant"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            gamrs.Gam(family="gaussian").fit(df[["a", "b"]], df.y)


def test_default_gam_emits_no_warning(rng):
    """Sanity: a default Gam() does NOT warn — the warnings are only for
    the API-compat knobs that don't take effect."""
    x = rng.uniform(0, 1, 100)
    y = x + rng.normal(0, 0.1, 100)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        gamrs.Gam(family="gaussian").fit(x, y)


def test_parametric_term_recovers_slope(rng):
    """ParametricTerm fits the raw coefficient (no smoothing penalty).
    On `y = sin(x) + 2 * is_promo`, the parametric coefficient should
    land within a couple percent of the truth."""
    pd = pytest.importorskip("pandas")
    n = 1500
    x_smooth = rng.uniform(0, 10, n)
    is_promo = rng.integers(0, 2, n).astype(float)
    y = np.sin(x_smooth) + 2.0 * is_promo + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"x": x_smooth, "is_promo": is_promo})
    X = df[["x", "is_promo"]]

    g = gamrs.Gam(terms=[
        gamrs.CrTerm("x", k=10),
        gamrs.ParametricTerm("is_promo"),
    ]).fit(X, y)

    # Parametric coef is the LAST one (terms concatenated in order).
    # Only the smooth has a smoothing parameter — lambda len == 1.
    assert len(g.lambda_) == 1
    assert abs(g.coef_[-1] - 2.0) < 0.05  # within 2.5% of truth
    assert g.converged_


def test_parametric_via_predictor_basis_map_matches_typed_term(rng):
    """Two ways to spell a parametric column: typed `ParametricTerm("x")`
    and `predictor_basis_map={"x": "parametric"}` produce identical fits."""
    pd = pytest.importorskip("pandas")
    n = 500
    df = pd.DataFrame({
        "x": rng.uniform(0, 10, n),
        "is_promo": rng.integers(0, 2, n).astype(float),
    })
    df["y"] = np.sin(df.x) + 1.5 * df.is_promo + rng.normal(0, 0.3, n)
    X = df[["x", "is_promo"]]

    g_typed = gamrs.Gam(terms=[
        gamrs.CrTerm("x", k=10),
        gamrs.ParametricTerm("is_promo"),
    ]).fit(X, df.y)
    g_map = gamrs.Gam(
        family="gaussian",
        predictor_basis_map={"is_promo": "parametric"},
    ).fit(X, df.y)
    np.testing.assert_allclose(g_typed.coef_, g_map.coef_, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(g_typed.lambda_, g_map.lambda_, rtol=1e-10)


def test_typed_crterm_k2_auto_promotes_to_parametric(rng):
    """A typed `CrTerm("x", k=2)` auto-promotes to `ParametricTerm` with
    a UserWarning instead of failing with the cryptic 'needs k ≥ 3'
    error. (mgcv R bumps silently; we warn explicitly.)"""
    pd = pytest.importorskip("pandas")
    n = 200
    df = pd.DataFrame({
        "x_smooth": rng.uniform(0, 10, n),
        "z": rng.uniform(-5, 5, n),
    })
    df["y"] = np.sin(df.x_smooth) + 0.3 * df.z + rng.normal(0, 0.3, n)
    with pytest.warns(UserWarning, match=r"CrTerm\('z', k=2\).*auto-promoted"):
        g = gamrs.Gam(terms=[
            gamrs.CrTerm("x_smooth", k=10),
            gamrs.CrTerm("z", k=2),
        ]).fit(df[["x_smooth", "z"]], df.y)
    assert g.converged_
    # `z`'s coefficient is now a single parametric slope, not a 2-DoF smooth.
    # Total coefs = 1 (intercept) + 9 (x_smooth centred CR) + 1 (z param) = 11.
    assert len(g.coef_) == 11


def test_typed_n_obs_lt_k_raises_clear_error(rng):
    """`n_obs < k` for a typed CrTerm fails fast with an explicit
    ValueError instead of a native panic."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"x": rng.uniform(0, 10, 5)})
    df["y"] = df.x + rng.normal(0, 0.3, 5)
    with pytest.raises(ValueError, match="n_obs=5 < k=10"):
        gamrs.Gam(terms=[gamrs.CrTerm("x", k=10)]).fit(df[["x"]], df.y)


def test_predictors_path_warns_on_k_bump_and_cap(rng):
    """`term_k_mapping={"x": 2}` bumps k to the min_k floor and warns;
    `k_default=50` with n_unique=10 caps k and warns.

    The cap lands at n_unique, which is where mgcv's own `smooth.construct.cr`
    puts it; `k_cap_offset=1` is the older, one-stricter cap. Both are asserted
    on the resulting basis dimension, not just on the warning text.
    """
    pd = pytest.importorskip("pandas")
    n = 200
    df = pd.DataFrame({
        "x": rng.uniform(0, 10, n),
        "z": rng.choice(np.linspace(0, 5, 10), n),  # only 10 unique values
    })
    df["y"] = np.sin(df.x) + df.z + rng.normal(0, 0.3, n)

    with pytest.warns(UserWarning, match="bumped from 2 to 3"):
        gamrs.Gam(term_k_mapping={"x": 2}).fit(df[["x"]], df.y)

    with pytest.warns(UserWarning, match="capped from 50 to 10"):
        capped = gamrs.Gam(k_default=50).fit(df[["z"]], df.y)
    assert capped._k_used == [10]
    # k basis functions, one absorbed by the intercept — same width mgcv reports
    # as k' for s(z, k=10).
    assert [b - a + 1 for _, a, b in capped.get_term_indices()] == [9]

    with pytest.warns(UserWarning, match="capped from 50 to 9"):
        stricter = gamrs.Gam(k_default=50, k_cap_offset=1).fit(df[["z"]], df.y)
    assert stricter._k_used == [9]


def test_all_parametric_design_fits_from_a_dataframe(rng):
    """A design of only parametric terms is fitted, through the DataFrame path too.

    This asserted a refusal until 0.12.2. It fits now: with zero smoothing parameters there is
    nothing to optimise, so the answer is plain least squares. Kept alongside the array-input
    test because the column-name path resolves terms differently.
    """
    pd = pytest.importorskip("pandas")
    n = 100
    df = pd.DataFrame({"is_promo": rng.integers(0, 2, n).astype(float)})
    df["y"] = 2.0 * df.is_promo + rng.normal(0, 0.3, n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        gam = gamrs.Gam().fit(df[["is_promo"]], df.y)

    design = np.column_stack([np.ones(n), df.is_promo.to_numpy()])
    ols, *_ = np.linalg.lstsq(design, df.y.to_numpy(), rcond=None)
    assert np.allclose(np.asarray(gam.coef_), ols, atol=1e-6)
    assert len(np.asarray(gam.get_lambdas())) == 0


def test_n_unique_2_auto_promotes_to_parametric(rng):
    """A 0/1 indicator column on the implicit predictors= path is
    auto-promoted to ParametricTerm with a UserWarning (matches
    mgcv_rust + mgcv R smooth.r:1460 'reduce k' semantics)."""
    pd = pytest.importorskip("pandas")
    n = 600
    df = pd.DataFrame({
        "x": rng.uniform(0, 10, n),
        "is_promo": rng.integers(0, 2, n).astype(float),
    })
    df["y"] = np.sin(df.x) + 1.7 * df.is_promo + rng.normal(0, 0.3, n)
    X = df[["x", "is_promo"]]
    with pytest.warns(UserWarning, match="'is_promo'.*auto-promoted"):
        g = gamrs.Gam(family="gaussian").fit(X, df.y)
    # Only the smooth has a smoothing parameter
    assert len(g.lambda_) == 1
    assert abs(g.coef_[-1] - 1.7) < 0.05


def test_auto_k_works_with_parametric_terms(rng):
    """auto_k=True grows k on smooth terms but skips ParametricTerm (k=0
    placeholder; never grown). The parametric coefficient still recovers."""
    pd = pytest.importorskip("pandas")
    n = 1200
    df = pd.DataFrame({
        "x": rng.uniform(0, 10, n),
        "is_promo": rng.integers(0, 2, n).astype(float),
    })
    df["y"] = (
        np.sin(df.x * 1.5) + np.cos(df.x * 2.5)
        + 1.7 * df.is_promo + rng.normal(0, 0.2, n)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        g = gamrs.Gam(
            family="gaussian", auto_k=True, k_default=4, max_k_auto=20
        ).fit(df[["x", "is_promo"]], df.y)
    # x's k grew from 4 → something larger; is_promo stayed parametric (k=0)
    assert g._k_used[0] > 4
    assert g._k_used[1] == 0
    assert abs(g.coef_[-1] - 1.7) < 0.05


def test_ocat_single_smooth_usable(rng):
    """Single-smooth ocat: regression guard on `predict_proba` quality
    and well-formedness. With the `(-3, 3)` log-gap bounds + projected-
    gradient KKT check, single-smooth ocat now converges cleanly on
    typical fixtures (the bound is hit and the projected gradient on
    the ρ axis is small). Pin both converged_=True and the proba
    quality so a regression in either is visible."""
    n = 1200
    x = rng.uniform(0, 10, n)
    eta_true = np.sin(x)
    qs = np.quantile(eta_true, [0.25, 0.5, 0.75])
    y = (np.digitize(eta_true, qs) + 1).astype(float)
    g = gamrs.Gam(family="ocat", r=4).fit(x, y)
    assert g.converged_, (
        f"single-smooth ocat regressed: converged_={g.converged_}, "
        f"n_iters={g.n_iters_}, λ={g.lambda_}"
    )
    proba = g.predict_proba(x)
    assert proba.shape == (n, 4)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
    acc = float((np.argmax(proba, axis=1) + 1 == y).mean())
    assert acc > 0.95


def test_ocat_multi_smooth_predict_proba_usable(rng):
    """Multi-smooth ocat on NEAR-SEPARABLE data (noiseless quantile-cut
    categories): converged_ may be False because the latent scale wants to
    blow up (θ pins at the ±3 bound). This is the exact regime where mgcv
    itself converges to a degenerate θ≈181 solution or aborts with "inner
    loop 1; can't correct step size" — gamrs's θ-bound keeps it stable and
    the conservative flag is correct. `predict_proba` is scale-invariant
    under the ridge and stays accurate, so pin that it's still usable.

    The well-posed (noisy-latent) regime — where gamrs converges and matches
    mgcv — is covered by test_parity_multismooth.py::test_additive_ocat_parity.
    """
    n = 1500
    X = np.column_stack([rng.uniform(0, 10, n), rng.uniform(0, 10, n)])
    eta = np.sin(X[:, 0]) + 0.5 * np.sin(X[:, 1] * 0.5)
    qs = np.quantile(eta, [0.25, 0.5, 0.75])
    y = (np.digitize(eta, qs) + 1).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        g = gamrs.Gam(
            family="ocat", r=4,
            terms=[gamrs.CrTerm(0, k=8), gamrs.CrTerm(1, k=8)],
        ).fit(X, y)
    # Probabilities must be well-formed regardless of convergence flag.
    proba = g.predict_proba(X)
    assert proba.shape == (n, 4)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
    # Classification accuracy ≥ 95% — captures "the fit is usable" even
    # if the optimisation didn't formally converge.
    acc = float((np.argmax(proba, axis=1) + 1 == y).mean())
    assert acc > 0.95


def test_ocat_predict_proba_shape_and_row_sums(rng):
    """predict_proba for ocat should return (n, R) with rows summing
    to 1 (probabilities). Both single-smooth and multi-smooth ocat fits
    use the same Python-side `P(Y=k) = F(α_k - η) - F(α_{k-1} - η)`
    derivation from `shape_params_` + η."""
    n = 800
    x = rng.uniform(0, 10, n)
    eta_true = np.sin(x)
    qs = np.quantile(eta_true, [0.25, 0.5, 0.75])
    y = (np.digitize(eta_true, qs) + 1).astype(float)

    # Single-smooth ocat
    g1 = gamrs.Gam(family="ocat", r=4).fit(x, y)
    p1 = g1.predict_proba(x)
    assert p1.shape == (n, 4)
    np.testing.assert_allclose(p1.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(p1 >= -1e-12) and np.all(p1 <= 1.0 + 1e-12)
    acc1 = float((np.argmax(p1, axis=1) + 1 == y).mean())
    assert acc1 > 0.9  # synthetic data: should be very accurate

    # Multi-smooth ocat — note converged_ may be False, but probabilities
    # are still well-defined because P(Y=k) is invariant under joint
    # (η, θ) scale shift.
    X = np.column_stack([rng.uniform(0, 10, n), rng.uniform(0, 10, n)])
    eta_m = np.sin(X[:, 0]) + 0.5 * np.sin(X[:, 1] * 0.5)
    qs_m = np.quantile(eta_m, [0.25, 0.5, 0.75])
    y_m = (np.digitize(eta_m, qs_m) + 1).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        g2 = gamrs.Gam(
            family="ocat", r=4,
            terms=[gamrs.CrTerm(0, k=8), gamrs.CrTerm(1, k=8)],
        ).fit(X, y_m)
    p2 = g2.predict_proba(X)
    assert p2.shape == (n, 4)
    np.testing.assert_allclose(p2.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(p2 >= -1e-12) and np.all(p2 <= 1.0 + 1e-12)


def test_parametric_subset_view_substitutes_training_mean(rng):
    """On a subset view (link scale), the masked-out parametric column
    is filled with its training mean — NOT zero. This keeps the
    `β_param · mean(x_param_train)` contribution in the prediction so
    the subset prediction isn't biased by that amount. mgcv_rust
    _fitter.py:451 has the same logic."""
    pd = pytest.importorskip("pandas")
    n = 800
    df = pd.DataFrame({
        "x_smooth": rng.uniform(0, 10, n),
        "is_promo": rng.integers(0, 2, n).astype(float),
    })
    df["y"] = np.sin(df.x_smooth) + 2.0 * df.is_promo + rng.normal(0, 0.2, n)
    X = df[["x_smooth", "is_promo"]]
    g = gamrs.Gam(terms=[
        gamrs.CrTerm("x_smooth", k=10),
        gamrs.ParametricTerm("is_promo"),
    ]).fit(X, df.y)

    beta_param = g.coef_[-1]
    mean_param = float(df.is_promo.mean())

    # Full-model link prediction vs subset-with-intercept on link scale.
    mu_full = g.predict(X, scale="link")
    mu_sub = g[["x_smooth", gamrs.Gam.INTERCEPT]].predict(X, scale="link")
    # Difference should equal β_param * (x_param_i - mean) exactly.
    expected_diff = beta_param * (df.is_promo.values - mean_param)
    np.testing.assert_allclose(mu_full - mu_sub, expected_diff, atol=1e-12)

    # scale='deviation' should NOT include the baseline (pure marginal):
    # the smooth's deviation contribution is roughly mean-zero across X.
    mu_dev = g[["x_smooth"]].predict(X, scale="deviation")
    assert abs(float(mu_dev.mean())) < 0.1


def test_parametric_linear_alias(rng):
    """`predictor_basis_map={"x": "linear"}` is the mgcv-user-friendly
    alias for `"parametric"` — same fit either way."""
    pd = pytest.importorskip("pandas")
    n = 300
    df = pd.DataFrame({
        "x": rng.uniform(0, 10, n),
        "z": rng.integers(0, 2, n).astype(float),
    })
    df["y"] = np.sin(df.x) + 0.7 * df.z + rng.normal(0, 0.3, n)
    X = df[["x", "z"]]
    g_p = gamrs.Gam(predictor_basis_map={"z": "parametric"}).fit(X, df.y)
    g_l = gamrs.Gam(predictor_basis_map={"z": "linear"}).fit(X, df.y)
    np.testing.assert_allclose(g_p.coef_, g_l.coef_, rtol=1e-12)


def test_term_string_cols_resolve_against_dataframe(rng):
    """CrTerm/TeTerm/TpsTerm/ReTerm accept string column names and resolve
    them against the DataFrame at fit time."""
    pd = pytest.importorskip("pandas")
    n = 200
    df = pd.DataFrame({
        "a": rng.uniform(0, 1, n),
        "b": rng.uniform(0, 1, n),
        "g": rng.integers(0, 5, n).astype(float),
    })
    df["y"] = np.sin(3 * df.a) + 0.5 * df.b**2 + rng.normal(0, 0.1, n)
    X = df[["a", "b", "g"]]

    # CrTerm with string column name
    g_str = gamrs.Gam(terms=[gamrs.CrTerm("a", k=10), gamrs.CrTerm("b", k=10)]).fit(
        X[["a", "b"]], df.y
    )
    # Equivalent fit with int indices
    g_int = gamrs.Gam(terms=[gamrs.CrTerm(0, k=10), gamrs.CrTerm(1, k=10)]).fit(
        X[["a", "b"]].values, df.y.values
    )
    np.testing.assert_allclose(
        np.asarray(g_str.predict(X[["a", "b"]])),
        np.asarray(g_int.predict(X[["a", "b"]].values)),
        rtol=1e-10,
        atol=1e-12,
    )

    # TeTerm + ReTerm with string names
    g_mix = gamrs.Gam(terms=[
        gamrs.TeTerm(cols=("a", "b"), k=(5, 5)),
        gamrs.ReTerm("g"),
    ]).fit(X, df.y)
    assert g_mix.converged_ is None or g_mix.converged_  # may be None for FS path
    assert g_mix.edf_total_ > 1.0


def test_term_string_col_unknown_raises(rng):
    """Unknown column names produce a clear error pointing at the term."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": rng.uniform(0, 1, 100)})
    df["y"] = df.a + rng.normal(0, 0.1, 100)
    with pytest.raises(ValueError, match="CrTerm.*'does_not_exist'"):
        gamrs.Gam(terms=[gamrs.CrTerm("does_not_exist")]).fit(df[["a"]], df.y)


def test_json_roundtrip_preserves_predictions(rng):
    """Gam.to_json / Gam.from_json round-trip yields machine-epsilon
    identical predictions to the original fit, just like serialize/
    deserialize but in a human-debuggable plain-text form."""
    pd = pytest.importorskip("pandas")
    import json as stdlib_json
    n = 200
    df = pd.DataFrame({
        "x0": rng.uniform(0, 10, n),
        "x1": rng.uniform(-5, 5, n),
    })
    df["y"] = np.sin(df.x0) + 0.3 * df.x1**2 + rng.normal(0, 0.3, n)
    X = df[["x0", "x1"]]
    g = gamrs.Gam(
        terms=[gamrs.CrTerm("x0", k=10), gamrs.CrTerm("x1", k=12)]
    ).fit(X, df.y)

    payload = g.to_json()
    assert isinstance(payload, str)
    # Valid JSON
    parsed = stdlib_json.loads(payload)
    assert "beta" in parsed
    assert "edf_total" in parsed

    g_roundtrip = gamrs.Gam.from_json(payload)
    mu_orig = g.predict(X)
    mu_back = g_roundtrip.predict(X)
    # serde_json round-trips f64 through decimal — accept machine epsilon.
    np.testing.assert_allclose(mu_orig, mu_back, rtol=1e-12, atol=1e-14)


def test_subset_view_predict_ci(rng):
    """``gam[["x0"]].predict_ci(..., scale="deviation")`` returns the
    Wald CI on the masked η contribution. Same numbers as partial_effect."""
    pd = pytest.importorskip("pandas")
    n = 300
    df = pd.DataFrame({
        "x0": rng.uniform(0, 10, n),
        "x1": rng.uniform(-5, 5, n),
    })
    df["y"] = np.sin(df.x0) + 0.3 * df.x1**2 + rng.normal(0, 0.3, n)
    X = df[["x0", "x1"]]
    g = gamrs.Gam(
        terms=[gamrs.CrTerm("x0", k=10), gamrs.CrTerm("x1", k=15)]
    ).fit(X, df.y)

    # scale='deviation' is supported on subset views
    mean, lo, hi = g[["x0"]].predict_ci(X, level=0.95, scale="deviation")
    assert mean.shape == lo.shape == hi.shape == (n,)
    assert np.all(lo <= mean) and np.all(mean <= hi)

    # scale='link' includes the intercept when requested explicitly
    mean_int, _, _ = g[["x0", gamrs.Gam.INTERCEPT]].predict_ci(
        X, level=0.95, scale="link"
    )
    # difference between link-with-intercept and deviation should be a
    # constant (the intercept)
    diffs = mean_int - mean
    assert np.std(diffs) < 1e-8

    # scale='response' on a subset view is rejected with a clear message
    with pytest.raises(ValueError, match="only defined for full-model"):
        g[["x0"]].predict_ci(X, level=0.95, scale="response")

    # scale='deviation' without a subset view is also rejected
    with pytest.raises(ValueError, match="only meaningful on subset views"):
        g.predict_ci(X, level=0.95, scale="deviation")


def test_all_parametric_design_fits_as_plain_least_squares():
    """A design with no smooths is fitted, not refused.

    Low-cardinality predictors (2-3 distinct values) are demoted to parametric, so refusing an
    all-parametric design meant refusing the feature — mgcv fits these without complaint, and a
    caller fitting one submodel per feature hits it on any binary column. With no smoothing
    parameters there is nothing to optimise, so the answer is exactly weighted least squares.
    """
    rng = np.random.default_rng(4)
    n = 300
    binary = rng.integers(0, 2, n).astype(float)  # 2 distinct -> parametric
    # A second 2-distinct column, not a 3-distinct one: at the mgcv cap (k <= n_unique)
    # three distinct values support a k=3 smooth, which is what mgcv fits for them, so a
    # 3-value column no longer lands in an all-parametric design. Asserted below.
    other = rng.integers(0, 2, n).astype(float)
    y = 5000.0 + 20000.0 * binary - 7000.0 * other + rng.normal(0, 3000, n)
    X = np.column_stack([binary, other])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the demotion warnings are the point, not a failure
        gam = gamrs.Gam(predictors=["pool", "stories"], family="gaussian", k_default=6, min_k=2)
        gam.fit(X, y)

    design = np.column_stack([np.ones(n), binary, other])
    ols, *_ = np.linalg.lstsq(design, y, rcond=None)

    assert np.allclose(np.asarray(gam.coef_), ols, atol=1e-6)
    assert np.allclose(np.asarray(gam.predict(X)), design @ ols, atol=1e-6)
    # No penalties, so no smoothing parameters and edf is exactly the coefficient count.
    assert len(np.asarray(gam.get_lambdas())) == 0
    assert gam.edf_total_ == pytest.approx(design.shape[1], abs=1e-6)
    # sigma-hat is the residual variance on n - p, as an unpenalised gaussian fit's must be.
    resid = y - design @ ols
    assert gam.scale_ == pytest.approx(float(resid @ resid) / (n - design.shape[1]), rel=1e-8)
    assert np.asarray(gam.get_vcov()).shape == (design.shape[1], design.shape[1])

    # The 3-distinct case, for contrast: k resolves to n_unique = 3 and stays a smooth even
    # at min_k=2, which is what `s(bathrooms, k=3, bs='cr')` does in mgcv on a 3-value column.
    triple = rng.integers(1, 4, n).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        smooth = gamrs.Gam(predictors=["baths"], k_default=6, min_k=2).fit(
            triple.reshape(-1, 1), y
        )
    assert smooth._k_used == [3]
    assert len(np.asarray(smooth.get_lambdas())) == 1


def test_all_parametric_design_fits_for_a_non_gaussian_family():
    """Every family fits an all-parametric design now, not just gaussian: `combined_s` takes the
    design width, so an empty penalty list assembles a zero penalty of the right shape and the
    penalised PIRLS path becomes plain unpenalised IRLS.

    One binary predictor makes the answer exact rather than approximate: the logistic MLE for a
    saturated 2x2 table IS the log-odds, so this pins the whole path — solver, weights, and
    convergence — against closed form instead of asserting that nothing blew up.
    """
    rng = np.random.default_rng(5)
    n = 400
    flag = rng.integers(0, 2, n).astype(float)
    y = (rng.random(n) < np.where(flag > 0.5, 0.75, 0.30)).astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gam = gamrs.Gam(predictors=["flag"], family="bernoulli", k_default=6, min_k=2)
        gam.fit(flag.reshape(-1, 1), y)

    rate_off, rate_on = y[flag < 0.5].mean(), y[flag > 0.5].mean()
    log_odds = lambda q: np.log(q / (1 - q))  # noqa: E731
    assert np.allclose(np.asarray(gam.coef_), [log_odds(rate_off), log_odds(rate_on) - log_odds(rate_off)], atol=1e-8)
    # Fitted probabilities are the two group rates, which is what a saturated logistic fit gives.
    assert np.allclose(np.unique(np.asarray(gam.predict(flag.reshape(-1, 1)))), sorted([rate_off, rate_on]), atol=1e-8)
    # No penalties, so no smoothing parameters and edf is the coefficient count.
    assert len(np.asarray(gam.get_lambdas())) == 0
    assert gam.edf_total_ == pytest.approx(2.0, abs=1e-5)
    assert np.asarray(gam.get_vcov()).shape == (2, 2)


# --------------------------------------------------------------------------- #
# method= is honoured, or substituted out loud — never silently dropped       #
# --------------------------------------------------------------------------- #
def test_freml_falls_back_to_reml_where_fellner_schall_was_never_ported():
    """`method="fREML"` used to be a silent no-op on most fit paths.

    Fellner-Schall reached the GLM envelope driver only. The gaussian
    closed-form path, the quantile path and the shape-parameter families
    (scat / negbin / tweedie / ocat) ran damped Newton regardless, so REML and
    fREML came out bit-identical and nothing told the caller. They now warn and
    fit on REML — same criterion, and Newton is the stronger optimiser here, so
    the substitution costs nothing but has to be visible.

    scat additionally used to *default* to `method="fREML"`, so every scat fit
    ever made declared an optimiser it did not run.
    """
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 300)
    y = np.sin(2 * np.pi * x) + rng.normal(scale=0.2, size=x.size)

    assert gamrs.Gam(family="t-dist").method == "REML", "scat must declare what it runs"

    for family in ("gaussian", "t-dist", "nb"):
        y_fam = np.abs(y) * 3.0 if family == "nb" else y
        reml = gamrs.Gam(family=family, method="REML", k_default=6).fit(x, y_fam)
        with pytest.warns(UserWarning, match="not available for this fit path"):
            fell = gamrs.Gam(family=family, method="fREML", k_default=6).fit(x, y_fam)
        # Fell back, so it must be the REML answer itself, not a near miss.
        assert np.allclose(fell.predict(x), reml.predict(x), rtol=0, atol=1e-12), family


def test_freml_still_runs_fellner_schall_where_it_was_ported():
    """The fallback must not have swallowed the solver it falls back from.

    Poisson goes through the GLM envelope driver, the one path Fellner-Schall
    was ported to. Asserting "does not raise" would still pass if `fREML` had
    become an alias for REML, so this asserts the two optimisers leave
    different tracks: FS takes a different iteration count to reach the same
    criterion, and warns about nothing.
    """
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 300)
    y = rng.poisson(np.exp(1.0 + np.sin(2 * np.pi * x))).astype(float)

    reml = gamrs.Gam(family="poisson", method="REML", k_default=10).fit(x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fell = gamrs.Gam(family="poisson", method="fREML", k_default=10).fit(x, y)

    assert fell.n_iters_ != reml.n_iters_, (
        f"fREML took the same {reml.n_iters_} iterations as REML — "
        "Fellner-Schall looks aliased away, not run"
    )
    # Different routes, different answers — each matching its mgcv counterpart
    # on this data (mgcv 1.9.3, s(x, k=10, bs="cr")): gam(method="REML") sp
    # 91.558444, bam(method="fREML") sp 91.982778. Fellner-Schall's update holds
    # the working weights fixed, so it lands where REML would if log|H|'s
    # β-chain were dropped — which is exactly where REML landed until 0.14.6,
    # and why these two used to "agree".
    assert np.allclose(reml.get_lambdas(), [91.558444], rtol=1e-4)
    assert np.allclose(fell.get_lambdas(), [91.982778], rtol=1e-4)


def test_non_convergence_warns_instead_of_being_silent():
    """A fit the outer optimiser could not fully converge must SAY so.

    The Rust side stopped raising on iteration-budget exhaustion (mgcv warns
    and returns the estimate — `gam.fit3.r:1656`), so `converged_ = False`
    became reachable through the normal path. Reachable and silent is worse
    than the old exception, hence the warning; this pins that it fires.

    Driven through a stub because the real regime is hard to provoke on
    demand: every draw in `tests/outer_indefinite_axis.rs` now converges, and
    a test that needs a non-converging fixture would be pinned to whichever
    seed happens to be pathological this month.
    """
    import warnings

    import gamrs

    class NotConverged:
        converged = False
        n_iters = 200

    g = gamrs.Gam(predictors=["x"], target="y")
    g._fitted = NotConverged()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g._warn_if_not_converged()
    assert len(caught) == 1, "expected exactly one warning"
    msg = str(caught[0].message)
    assert "did not reach its gradient tolerance" in msg
    assert "200 iterations" in msg
    assert issubclass(caught[0].category, UserWarning)

    class Converged:
        converged = True
        n_iters = 12

    g._fitted = Converged()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g._warn_if_not_converged()
    assert caught == [], "a converged fit must stay quiet"
