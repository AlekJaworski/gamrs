"""qgam-style σ-calibration helpers for the ELF / quantile family.

Why this lives in Python: the inner GAM fit is fast Rust (the ELF family
runs through gamrs's standard PIRLS + REML); only the outer σ search is
cheap Python loops. This module ports v0.x's
:func:`mgcv_rust.fit_quantile` and :func:`mgcv_rust.tune_quantile_sigma`
to gamrs.

Two helpers:

- :func:`tune_quantile_sigma`: K-fold CV with Brent on log σ. Returns σ̂
  and a small info dict (loss curve etc.). Pinball loss is the only
  criterion supported here — the bootstrap-KL ("cal_kl") variant from
  v0.x is deferrable to a follow-up; the location-scale "lss" pipeline
  is also deferrable.

- :func:`fit_quantile`: one-shot helper that auto-calibrates σ then fits
  a :class:`gamrs.Gam` at the chosen σ. Returns the fitted ``Gam`` with
  a ``sigma_`` attribute attached.

Why CV-on-σ and not LAML / REML: Fasiolo et al. 2021 show ELF's
likelihood is a Gibbs posterior, not a true likelihood — the MLE σ is
structurally degenerate (the REML score collapses to σ→0 because
in-sample pinball shrinks faster than complexity penalties grow). CV is
the only in-sample-free criterion that breaks the tie cleanly.

Deferred (TODO follow-up):

- ``fit_quantile_lss`` / ``QuantileLSSFit`` (heteroskedastic location-
  scale port of qgam ≥1.3).
- ``loss="cal_kl"`` bootstrap-KL criterion (qgam's ``tuneLearnFast``
  ``loss="cal"`` path).
- v0.x's auto-σ via ``compute_err_param`` (qgam's ``.getErrParam``)
  initial heuristic (depends on the SHASH helper, also unported).
"""

from __future__ import annotations

import dataclasses
import statistics
import warnings
from typing import Any, Optional, Sequence

import numpy as np

from . import _gamrs_native
from ._coerce import to_1d_array, to_2d_with_columns
from ._fitter import Gam, _resolve_term_cols
from ._low_level import CrStableTerm, CrTerm, TeTerm


def _pinball(y: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Mean pinball loss at quantile ``tau``."""
    r = y - y_pred
    return float(np.maximum(tau * r, (tau - 1.0) * r).mean())


def _build_folds(
    n: int, n_folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """K equal-size folds via a permuted index split.

    Returns a list of ``(test_idx, train_idx)`` pairs.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_size = n // n_folds
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_folds):
        test_idx = perm[i * fold_size : (i + 1) * fold_size]
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        folds.append((test_idx, np.where(train_mask)[0]))
    return folds


def _fit_elf_native(
    x_2d: np.ndarray,
    y: np.ndarray,
    tau: float,
    sigma: float,
    k: int,
    design: str,
    terms: Optional[Sequence[Any]] = None,
) -> Any:
    """Run a single ELF fit at the given (τ, σ). Returns the ``FittedGam``.

    The σ argument is plumbed as the native binding's ``elf_sigma`` kwarg
    (gamrs's name for qgam's σ scale parameter). ``sigma=0.0`` defers to
    the Rust-side heuristic, matching v0.x semantics.

    Single-smooth (``terms is None``): hits the native ``fit`` directly with
    ``(k, design)``. Multi-smooth (``terms`` given): routes through
    ``Gam(family="quantile", terms=...)`` so the additive design + the same
    (τ, σ) plumbing apply; ``k`` / ``design`` are ignored. ``terms`` must be
    integer-column (resolved upfront by the caller) so it survives the CV
    row-slicing that drops the column-name context.
    """
    if terms is not None:
        g = Gam(family="quantile", terms=list(terms))
        g._elf_tau = float(tau)  # type: ignore[attr-defined]
        g._elf_sigma = float(sigma)  # type: ignore[attr-defined]
        g.fit(x_2d, y)
        return g._fitted
    return _gamrs_native.fit(
        "elf",
        x_2d,
        y,
        k=int(k),
        design=design,
        tau=float(tau),
        elf_sigma=float(sigma),
    )


def _shash_co(
    x_2d: np.ndarray,
    y: np.ndarray,
    tau: float,
    k: int,
    design: str,
    terms: Optional[Sequence[Any]] = None,
) -> float:
    """qgam-faithful ELF scale ``co`` from a SHASH ``err``-param fit.

    Ports mgcv_rust's ``fast_oos`` σ source (qgam ``.getErrParam``):

    1. Gaussian pilot GAM (gamrs's own API) → μ₀ and per-smooth EDF.
    2. ``var_hat = mean((y − μ₀)²)``; ``r_std = (y − μ₀)/sqrt(var_hat)``.
    3. ``d_eff = 1 + Σ edf_smooths``.
    4. ``err = compute_err_param(r_std, d_eff, [tau])`` (SHASH BFGS; the
       documented ``err = 0.05`` fallback on divergence).
    5. ``co = err · sqrt(2π·var_hat) / (2·ln 2)``.

    Returns ``co`` (> 0); the native ELF fit then uses λ = σ = co (pass
    ``elf_sigma = co``, ``elf_lambda`` unset — see ``src/fit/quantile.rs``).
    Returns ``0.0`` to defer to the Rust σ heuristic when the pilot is
    degenerate OR scipy/SHASH is unavailable (keeps scipy an optional boost,
    not a hard requirement of the fast path).
    """
    # The pilot runs in units of sd(y) so that co scales exactly with y.
    y = np.asarray(y, dtype=np.float64).ravel()
    y_scale = float(np.std(y))
    if not np.isfinite(y_scale) or y_scale <= 0.0:
        return 0.0
    y = y / y_scale
    if terms is not None:
        g_pilot = Gam(family="gaussian", terms=list(terms))
    else:
        g_pilot = Gam(family="gaussian", k_default=int(k), design=design)
    g_pilot.fit(x_2d, y)
    mu0 = np.asarray(g_pilot.predict(x_2d, scale="response"), dtype=np.float64).ravel()
    resid = y - mu0
    var_hat = float((resid**2).mean())
    if not np.isfinite(var_hat) or var_hat <= 1e-12:
        return 0.0
    r_std = resid / np.sqrt(var_hat)
    d_eff = 1.0 + float(np.sum(np.asarray(g_pilot.edf_, dtype=np.float64).ravel()))
    try:
        from ._shash import compute_err_param
    except ImportError:
        return 0.0  # scipy unavailable -> defer to the native Rust heuristic
    try:
        err = float(compute_err_param(r_std, d_eff, [float(tau)])[0])
        if not np.isfinite(err) or err <= 0.0:
            err = 0.05
    except Exception:
        err = 0.05  # SHASH BFGS divergence — qgam's documented default
    return float(y_scale * err * np.sqrt(2.0 * np.pi * var_hat) / (2.0 * np.log(2.0)))


def _calibrate_quantile_intercept(
    fitted: Any, x_2d: np.ndarray, y: np.ndarray, tau: float
) -> float:
    """qgam-style coverage calibration (policy; ported from mgcv_rust).

    After the ELF fit, shift the intercept so the empirical training coverage
    matches τ: ``shift = τ-th order statistic of (y − μ̂)`` (floor-index, the
    exact convention of mgcv_rust's native ``calibrate_quantile_intercept``),
    then apply it via the core's generic :meth:`shift_intercept` primitive.
    Cheap (one predict + one quantile) and family-agnostic mechanism — the
    quantile-specific policy lives here, not in the Rust core.

    Returns the applied shift.
    """
    # ELF uses an identity link, so the link-scale prediction IS μ̂.
    mu = np.asarray(fitted.predict(x_2d), dtype=np.float64).ravel()
    resid = np.asarray(y, dtype=np.float64).ravel() - mu
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        raise ValueError("cannot calibrate quantile intercept from non-finite residuals")
    resid.sort()
    idx = int(np.clip(np.floor((resid.size - 1) * tau), 0, resid.size - 1))
    shift = float(resid[idx])
    fitted.shift_intercept(shift)
    return shift


def _cv_loss_at_sigma(
    sigma: float,
    x_2d: np.ndarray,
    y: np.ndarray,
    tau: float,
    k: int,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    design: str,
    terms: Optional[Sequence[Any]] = None,
) -> float:
    """K-fold mean pinball loss at the given σ. Folds that fail to fit
    or produce NaN predictions contribute ``+inf``.
    """
    losses: list[float] = []
    for test_idx, train_idx in folds:
        try:
            x_tr = np.ascontiguousarray(x_2d[train_idx], dtype=np.float64)
            y_tr = np.ascontiguousarray(y[train_idx], dtype=np.float64)
            x_te = np.ascontiguousarray(x_2d[test_idx], dtype=np.float64)
            f = _fit_elf_native(
                x_tr, y_tr, tau=tau, sigma=sigma, k=k, design=design, terms=terms
            )
            y_pred = np.asarray(f.predict(x_te))
            if not np.all(np.isfinite(y_pred)):
                losses.append(np.inf)
                continue
            losses.append(_pinball(y[test_idx], y_pred, tau))
        except Exception:
            losses.append(np.inf)
    if not losses:
        return float("inf")
    return float(np.mean(losses))


def tune_quantile_sigma(
    X: Any,
    y: Any,
    tau: float,
    k: int = 10,
    K_folds: int = 5,
    brent_bracket: Optional[tuple[float, float]] = None,
    design: str = "cr",
    seed: int = 0,
    xatol: float = 0.05,
    terms: Optional[Sequence[Any]] = None,
) -> tuple[float, dict[str, Any]]:
    """Pick σ for the ELF / quantile family via K-fold pinball CV.

    Runs SciPy's bounded Brent on ``log σ`` and minimises mean held-out
    pinball loss across ``K_folds`` folds. Returns ``(σ̂, info)`` where
    ``info`` contains the loss at the optimum, the Brent eval count, and
    the (log) bracket.

    Args:
      X: ``(n, d)`` design matrix (DataFrame / ndarray / 1-D vector
        accepted).
      y: ``(n,)`` response.
      tau: target quantile, in ``(0, 1)``.
      k: basis dimension for the single ELF smooth (matches the inner
        ``Gam(family='elf')`` fit).
      K_folds: number of CV folds (default 5).
      brent_bracket: ``(log_lo, log_hi)`` bracket on ``log σ``. Defaults
        to ``(log(0.05 · sd(y)), log(5 · sd(y)))`` — a 100× span centred
        on the response scale, matching v0.x's default.
      design: spline basis for the smooth (``'cr'`` default).
      seed: RNG seed for fold construction.
      xatol: Brent absolute tolerance in ``log σ`` space (default 0.05 ≈
        5% in σ — empirically enough for stable σ̂ on n ≥ 200).
      terms: optional typed-term list (``CrTerm`` / ``TeTerm`` / …) for a
        multi-smooth additive quantile. When given, ``k`` / ``design`` are
        ignored and every fold fit uses this additive design. σ stays a
        single family-level scale (one CV search), not per-term.
    """
    try:
        from scipy.optimize import minimize_scalar
    except ImportError as exc:
        raise ImportError(
            "tune_quantile_sigma requires scipy. Install with `pip install scipy`."
        ) from exc

    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1); got tau={tau}")

    x_2d, cols_for_terms = to_2d_with_columns(X, None)
    y_arr = to_1d_array(y, name="y")
    if x_2d.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X has {x_2d.shape[0]} rows but y has {y_arr.shape[0]} elements"
        )
    n = y_arr.shape[0]

    folds = _build_folds(n, K_folds, seed)

    if brent_bracket is None:
        y_scale = float(np.std(y_arr))
        if y_scale <= 0.0:
            y_scale = 1.0
        brent_bracket = (np.log(0.05 * y_scale), np.log(5.0 * y_scale))
    lo, hi = float(brent_bracket[0]), float(brent_bracket[1])
    if not lo < hi:
        raise ValueError(
            f"brent_bracket must satisfy lo < hi; got ({lo}, {hi})"
        )

    # Resolve any string-column terms to integer columns now — CV slices
    # x_2d to a bare ndarray per fold, losing the column-name context.
    resolved_terms = (
        [_resolve_term_cols(t, list(cols_for_terms)) for t in terms]
        if terms is not None
        else None
    )

    def objective(log_sigma: float) -> float:
        return _cv_loss_at_sigma(
            float(np.exp(log_sigma)), x_2d, y_arr, tau, k, folds, design,
            terms=resolved_terms,
        )

    result = minimize_scalar(
        objective,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": float(xatol)},
    )
    sigma_hat = float(np.exp(result.x))
    info: dict[str, Any] = {
        "sigma": sigma_hat,
        "log_sigma": float(result.x),
        "cv_loss": float(result.fun),
        "n_brent_evals": int(result.nfev),
        "n_folds": int(K_folds),
        "bracket_log_sigma": (lo, hi),
        "tau": float(tau),
        "k": int(k),
    }
    return sigma_hat, info


def fit_quantile(
    X: Any,
    y: Any,
    tau: float,
    k: int = 10,
    K_folds: int = 5,
    brent_bracket: Optional[tuple[float, float]] = None,
    design: str = "cr",
    seed: int = 0,
    xatol: float = 0.05,
    sigma: Optional[float] = None,
    coverage_calibrate: bool = False,
    preset: Optional[str] = None,
    terms: Optional[Sequence[Any]] = None,
) -> Gam:
    """Fit a quantile (ELF) GAM with σ chosen by K-fold pinball CV.

    This is the v0.x ``mgcv_rust.fit_quantile`` convenience wrapper,
    ported to gamrs. The inner ELF fit goes through
    :class:`gamrs.Gam` with ``family='quantile'`` (which dispatches to
    the Rust ELF family — see :data:`gamrs._coerce.FAMILY_TO_GAMRS`); the
    outer σ search is :func:`tune_quantile_sigma`.

    Args:
      X: ``(n, d)`` design (DataFrame / ndarray / 1-D vector accepted).
      y: ``(n,)`` response.
      tau: target quantile in ``(0, 1)``.
      k: basis dim for the single ELF smooth (ignored when ``terms`` is set).
      K_folds: CV folds for σ tuning.
      brent_bracket: ``(log_lo, log_hi)`` bracket on ``log σ``.
      design: basis kind (default ``'cr'``).
      seed: RNG seed for fold construction.
      xatol: Brent tolerance on ``log σ`` (default 0.05).
      sigma: if given, skip CV and fit at this σ directly (escape hatch
        for callers that already have a tuned σ̂).
      coverage_calibrate: if True, after the fit shift the intercept so the
        empirical training coverage matches τ (qgam-style coverage
        calibration, ported from mgcv_rust). Cheap post-fit step.
      preset: convenience bundles ported from mgcv_rust's OOS quantile paths:

        - ``"fast_oos"``: qgam-faithful SHASH err-param σ (NO CV) + coverage
          calibration. The speed/quality balance — one ELF pass; matches
          mgcv_rust OOS pinball into the extreme tail (τ≳0.95) at a fraction
          of the CV cost. Falls back to the native σ heuristic if scipy is
          absent.
        - ``"quality_oos"``: CV-tuned σ + coverage calibration.
      terms: optional typed-term list (``CrTerm`` / ``TeTerm`` / …) for a
        multi-smooth additive quantile (``y ~ s(x0) + s(x1) + …``). When
        given, ``k`` / ``design`` are ignored and the SHASH pilot, every CV
        fold, and the final fit all use this additive design. σ remains a
        single family-level scale (one CV / SHASH search across all terms).

    Returns:
      A fitted :class:`gamrs.Gam` with extra attributes attached:

      - ``sigma_``: the σ used in the final fit (``0.0`` = native heuristic).
      - ``tune_info_``: the :func:`tune_quantile_sigma` info dict
        (``None`` when σ was not CV-tuned).
      - ``coverage_shift_``: the applied coverage-calibration shift, or
        ``None`` when ``coverage_calibrate`` was off.
    """
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be in (0, 1); got tau={tau}")

    # Preset resolution mirrors mgcv_rust._quantile.fit_quantile.
    use_shash = False
    if preset is not None:
        if preset == "fast_oos":
            coverage_calibrate = True
            use_shash = sigma is None  # SHASH σ unless the caller pinned σ
        elif preset == "quality_oos":
            coverage_calibrate = True  # σ stays CV-tuned (sigma is None)
        else:
            raise ValueError(
                f"unknown quantile preset {preset!r}; expected 'fast_oos' or 'quality_oos'"
            )

    # Coerce once — needed for the SHASH pilot and/or coverage calibration.
    x_2d_full, cols_for_terms = to_2d_with_columns(X, None)
    y_full = to_1d_array(y, name="y")
    x_2d_contig = np.ascontiguousarray(x_2d_full, dtype=np.float64)
    y_contig = np.ascontiguousarray(y_full, dtype=np.float64)

    # Multi-smooth: resolve string-column terms to integer columns up front
    # so the SHASH pilot and the CV folds (which slice X to a bare ndarray,
    # dropping the column-name context) all see the same integer-column terms.
    resolved_terms = (
        [_resolve_term_cols(t, list(cols_for_terms)) for t in terms]
        if terms is not None
        else None
    )

    # fast_oos σ: qgam-faithful SHASH err-param (closes the extreme-tail gap
    # the bare Rust σ heuristic leaves at τ≳0.95). co=0.0 → defer to the Rust
    # heuristic (degenerate pilot or scipy absent).
    co_val: Optional[float] = None
    if use_shash:
        co_val = _shash_co(
            x_2d_contig, y_contig, float(tau), int(k), design, terms=resolved_terms
        )
        sigma = co_val if co_val > 0.0 else 0.0  # σ = co; 0.0 = native heuristic

    info: Optional[dict[str, Any]] = None
    if sigma is None:
        sigma, info = tune_quantile_sigma(
            X,
            y,
            tau=tau,
            k=k,
            K_folds=K_folds,
            brent_bracket=brent_bracket,
            design=design,
            seed=seed,
            xatol=xatol,
            terms=resolved_terms,
        )

    # Build + fit the Gam ONCE at the target (τ, σ). We plumb τ/σ through the
    # Gam's ELF config so `g.fit()` lands at the right fit directly — no
    # fit-then-replace double pass (matches mgcv_rust's single fit). The
    # `Gam.fit` path sets `_gamrs_family='elf'` for family='quantile'.
    # Multi-smooth → typed-term Gam; single-smooth → the (k, design) Gam.
    if resolved_terms is not None:
        g = Gam(family="quantile", terms=resolved_terms)
    else:
        g = Gam(family="quantile", k_default=int(k), design=design)
    g._elf_tau = float(tau)  # type: ignore[attr-defined]
    g._elf_sigma = float(sigma)  # type: ignore[attr-defined]
    g.fit(X, y)

    # qgam-style coverage calibration (policy lives in this module; the
    # intercept shift is applied via the core's generic shift_intercept).
    coverage_shift: Optional[float] = None
    if coverage_calibrate:
        coverage_shift = _calibrate_quantile_intercept(g._fitted, x_2d_contig, y_contig, tau)

    # v0.x parity: expose σ̂ + the calibration trace on the fitted Gam.
    g.sigma_ = float(sigma)  # type: ignore[attr-defined]
    g.tau_ = float(tau)  # type: ignore[attr-defined]
    g.tune_info_ = info  # type: ignore[attr-defined]
    g.coverage_shift_ = coverage_shift  # type: ignore[attr-defined]
    g.co_ = co_val  # type: ignore[attr-defined]  # qgam ELF scale (=σ̂); None if not fast_oos
    return g


# ─────────────────────────────────────────────────────────────────────────
# Distributional (location-scale) quantile — the `gaulss`/`shash` view.
# ─────────────────────────────────────────────────────────────────────────

# E[log|Z|] for Z ~ N(0, 1) = -γ/2 - (log 2)/2 ≈ -0.6351. Subtracting it from
# log|y - μ̂| makes that an unbiased estimator of log σ (the residual is σ·Z).
_E_LOG_ABS_NORMAL = -0.6351814227307388


def _halve_term_k(term: Any) -> Any:
    """Return a copy of a smooth term with its basis dim ~halved (floor 3).

    The scale model σ(x) is usually a flatter function than the location, and
    a lower k guards against overfitting `log|residuals|`. CR/CR-stable carry a
    scalar k; tensor terms carry a per-margin tuple; `re`/parametric have no k.
    """
    if isinstance(term, (CrTerm, CrStableTerm)):
        return dataclasses.replace(term, k=max(3, int(term.k) // 2))
    if isinstance(term, TeTerm):
        ka, kb = term.k
        return dataclasses.replace(term, k=(max(3, ka // 2), max(3, kb // 2)))
    k = getattr(term, "k", None)
    if isinstance(k, tuple):
        return dataclasses.replace(term, k=tuple(max(3, ki // 2) for ki in k))
    return term  # ReTerm / ParametricTerm — nothing to halve


class QuantileLSSFit:
    """Distributional location-scale quantile fit — ONE fit, ALL τ.

    Models the conditional distribution of `y | x` by a smooth location μ(x)
    and a smooth scale σ(x), then derives every quantile as

        q_τ(x) = μ(x) + σ(x) · z_τ

    where `z_τ` is the τ-quantile of the standardised residual distribution —
    `Φ⁻¹(τ)` for `shape="gaussian"`, or a fitted SHASH quantile (skew/kurtosis)
    for `shape="shash"`. Because `z_τ` is monotone in τ and σ(x) > 0, the
    quantiles **never cross**, and a single fit yields *every* τ — the mgcv
    `gaulss`/`shash` distributional view, in contrast to the per-τ pinball fit
    of :func:`fit_quantile`.

    Attributes:
      shape: ``"gaussian"`` or ``"shash"``.
      shash_params_: ``[mu, tau, eps, phi]`` SHASH MLE on the standardised
        residuals (``None`` for the Gaussian shape).
    """

    def __init__(self, g_loc: Gam, g_scale: Gam, shape: str, shash_params: Optional[np.ndarray]):
        self._g_loc = g_loc
        self._g_scale = g_scale
        self.shape = shape
        self.shash_params_ = shash_params

    def predict_loc(self, X: Any) -> np.ndarray:
        """Conditional location μ̂(x) (the median for symmetric `z_τ`)."""
        return np.asarray(self._g_loc.predict(X), dtype=float).ravel()

    def predict_sigma(self, X: Any) -> np.ndarray:
        """Conditional scale σ̂(x) = exp(η̂_scale(x)) — always positive."""
        return np.exp(np.asarray(self._g_scale.predict(X), dtype=float).ravel())

    def _z(self, tau: float) -> float:
        if self.shape == "gaussian":
            return statistics.NormalDist().inv_cdf(tau)
        from ._shash import shash_qf

        return shash_qf(tau, self.shash_params_)

    def predict_quantile(self, X: Any, tau: Any) -> np.ndarray:
        """`q_τ(x)` for one τ (returns ``(n,)``) or many (returns ``(n, n_τ)``).

        Monotone in τ by construction, so the returned bands never cross.
        """
        mu = self.predict_loc(X)
        sigma = self.predict_sigma(X)
        scalar = np.ndim(tau) == 0
        taus = np.atleast_1d(np.asarray(tau, dtype=float))
        if np.any((taus <= 0.0) | (taus >= 1.0)):
            raise ValueError("all tau must be in the open interval (0, 1)")
        z = np.array([self._z(float(t)) for t in taus])
        q = mu[:, None] + sigma[:, None] * z[None, :]
        return q[:, 0] if scalar else q

    def predict(self, X: Any, tau: float = 0.5) -> np.ndarray:
        """Alias for :meth:`predict_quantile`; defaults to the median."""
        return self.predict_quantile(X, tau)


def fit_quantile_lss(
    X: Any,
    y: Any,
    terms: Optional[Sequence[Any]] = None,
    k: int = 10,
    design: str = "cr",
    k_scale: Optional[int] = None,
    scale_terms: Optional[Sequence[Any]] = None,
    shape: str = "gaussian",
    method: Optional[str] = None,
) -> QuantileLSSFit:
    """Fit a distributional location-scale quantile model — one fit, all τ.

    Unlike :func:`fit_quantile` (a per-τ smoothed-pinball fit), this models the
    *whole* conditional distribution and derives every quantile from it, so the
    bands never cross and a single fit serves all τ. It's the mgcv
    `gaulss`/`shash` view, implemented as a two-stage estimator:

      1. **Location** μ(x): a Gaussian GAM of `y` on `x`.
      2. **Scale** σ(x): a Gaussian GAM of `log|y − μ̂(x)| − E[log|N(0,1)|]`
         on `x` (the Euler–Mascheroni correction makes this an unbiased `log σ`
         estimator), then σ̂(x) = exp(·).
      3. **Shape** (`shape="shash"` only): a SHASH MLE on the standardised
         residuals `(y − μ̂)/σ̂` captures skew/kurtosis; `z_τ` becomes the SHASH
         quantile instead of `Φ⁻¹(τ)`.

    There is no `tau` argument — τ is chosen at predict time
    (:meth:`QuantileLSSFit.predict_quantile`), since one fit yields all τ.

    Args:
      X: ``(n, d)`` design (DataFrame / ndarray / 1-D vector).
      y: ``(n,)`` response.
      terms: optional typed-term list for a multi-smooth location
        (`CrTerm` / `TeTerm` / …). When given, `k` / `design` are ignored.
      k: location basis dim when `terms` is None (default 10).
      design: location basis kind when `terms` is None (default ``"cr"``).
      k_scale: scale basis dim when `terms`/`scale_terms` are None. Defaults to
        ``max(3, k // 2)`` — σ(x) is usually flatter than μ(x).
      scale_terms: optional explicit typed terms for the scale model. When
        None and `terms` is given, the scale reuses the location terms with
        each basis dim ~halved.
      shape: ``"gaussian"`` (default; `z_τ = Φ⁻¹(τ)`, no scipy needed) or
        ``"shash"`` (fit skew/kurtosis; needs the `[quantile]` extra). A SHASH
        fit that diverges falls back to Gaussian with a warning.
      method: outer optimiser for the two GAMs (``"REML"`` default / ``"fREML"``).

    Returns:
      A :class:`QuantileLSSFit`. Use `.predict_quantile(X, tau)` for any/all τ,
      `.predict_loc(X)` for μ̂, `.predict_sigma(X)` for σ̂.
    """
    if shape not in ("gaussian", "shash"):
        raise ValueError(f"shape must be 'gaussian' or 'shash'; got {shape!r}")

    y_arr = to_1d_array(y, name="y")
    x_2d, _cols = to_2d_with_columns(X, None)
    if x_2d.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"X has {x_2d.shape[0]} rows but y has {y_arr.shape[0]} elements"
        )

    # ── Stage 1: location μ(x) via a Gaussian GAM. ──
    if terms is not None:
        g_loc = Gam(family="gaussian", terms=list(terms), method=method)
    else:
        g_loc = Gam(family="gaussian", k_default=int(k), design=design, method=method)
    g_loc.fit(X, y)
    mu = np.asarray(g_loc.predict(X), dtype=float).ravel()

    # ── Stage 2: scale σ(x) via a Gaussian GAM on the Euler-corrected
    #            log|residual|. ──
    floor = 1e-3 * (float(np.std(y_arr)) or 1.0)
    log_abs_r = np.log(np.maximum(np.abs(y_arr - mu), floor)) - _E_LOG_ABS_NORMAL
    if scale_terms is not None:
        g_scale = Gam(family="gaussian", terms=list(scale_terms), method=method)
    elif terms is not None:
        g_scale = Gam(family="gaussian", terms=[_halve_term_k(t) for t in terms], method=method)
    else:
        ks = int(k_scale) if k_scale is not None else max(3, int(k) // 2)
        g_scale = Gam(family="gaussian", k_default=ks, design=design, method=method)
    g_scale.fit(X, log_abs_r)
    sigma = np.exp(np.asarray(g_scale.predict(X), dtype=float).ravel())

    # ── Stage 3 (shape="shash"): SHASH MLE on the standardised residuals. ──
    shash_params: Optional[np.ndarray] = None
    if shape == "shash":
        from ._shash import fit_shash

        z_std = (y_arr - mu) / np.maximum(sigma, 1e-12)
        try:
            shash_params = fit_shash(z_std)
        except Exception:
            warnings.warn(
                "SHASH fit on standardised residuals diverged; falling back to "
                "Gaussian z_τ (Φ⁻¹).",
                UserWarning,
                stacklevel=2,
            )
            shape = "gaussian"

    return QuantileLSSFit(g_loc, g_scale, shape, shash_params)
