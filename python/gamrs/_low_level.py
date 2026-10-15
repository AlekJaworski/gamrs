"""Minimal coercion facade over :func:`gamrs._gamrs_native.fit`.

The native binding is strict (it wants 1-D float64 contiguous arrays
for x and y). This wrapper handles the common 2-D ``(n, 1)`` /
non-float64 / non-contiguous inputs by coercing them once before
delegating, and exposes the returned :class:`FittedGam` directly.

Mirrors the role of :mod:`mgcv_rust._low_level` for v0.x users.

Multi-smooth additive fits (epic 94b/94c) are surfaced via the typed
:class:`CrTerm` / :class:`ReTerm` / :class:`TeTerm` term classes plus
the :func:`fit_additive` helper. The FFI boundary still uses tuples
internally, but the Python surface stays typed (no string-keyed knobs
leak to user code).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from . import _gamrs_native


def _ensure_1d_float64(arr: Any, *, name: str) -> np.ndarray:
    """Coerce ``arr`` to a 1-D ``float64`` contiguous ``ndarray``.

    2-D inputs of shape ``(n, 1)`` are reshaped to ``(n,)`` — the
    common notebook case where a DataFrame column was selected.
    """
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    if a.ndim == 2 and a.shape[1] == 1:
        a = a.reshape(-1)
    if a.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D (or a 2-D column vector); got ndim={a.ndim} (shape={a.shape})"
        )
    return a


def _ensure_2d_float64(arr: Any, *, name: str) -> np.ndarray:
    """Coerce ``arr`` to a 2-D ``(n, d)`` ``float64`` contiguous ``ndarray``.

    1-D inputs of shape ``(n,)`` are reshaped to ``(n, 1)``. The Rust
    native ``fit`` expects 2-D x since the cram::fit lift to
    ``Array2<f64>`` (task #96) — but pass-through helpers retain the
    1-D-friendly notebook ergonomics by reshaping here.
    """
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(
            f"{name} must be 1-D or 2-D; got ndim={a.ndim} (shape={a.shape})"
        )
    return a


class GAM:
    """Coerce-and-forward wrapper around :func:`gamrs._gamrs_native.fit`.

    Use this when you want a single-predictor 1-D fit without the
    DataFrame plumbing of :class:`gamrs.Gam`. Method signatures match
    the native ``fit`` function plus light input coercion.
    """

    __slots__ = ("_fitted", "family_name", "design", "k")

    def __init__(
        self,
        family: str = "gaussian",
        *,
        k: int = 10,
        design: str = "cr",
    ) -> None:
        self.family_name = family
        self.k = int(k)
        self.design = design
        self._fitted: Optional[Any] = None

    def fit(
        self,
        x: Any,
        y: Any,
        *,
        weights: Any = None,
        k: Optional[int] = None,
        design: Optional[str] = None,
        **family_kwargs: Any,
    ) -> "GAM":
        x_arr = _ensure_2d_float64(x, name="x")
        y_arr = _ensure_1d_float64(y, name="y")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"x and y must have the same length; got x={x_arr.shape[0]}, y={y_arr.shape[0]}"
            )
        w_arr = (
            _ensure_1d_float64(weights, name="weights") if weights is not None else None
        )
        self._fitted = _gamrs_native.fit(
            self.family_name,
            x_arr,
            y_arr,
            weights=w_arr,
            k=int(k if k is not None else self.k),
            design=design if design is not None else self.design,
            **family_kwargs,
        )
        return self

    def _require_fitted(self) -> Any:
        if self._fitted is None:
            raise RuntimeError("GAM has not been fitted yet — call .fit() first.")
        return self._fitted

    def predict(self, x: Any, *, scale: str = "link") -> np.ndarray:
        f = self._require_fitted()
        x_arr = _ensure_2d_float64(x, name="x")
        if scale == "link":
            return np.asarray(f.predict(x_arr))
        if scale == "response":
            return np.asarray(f.predict_response(x_arr))
        raise ValueError(f"scale must be 'link' or 'response', got {scale!r}")

    def predict_ci(
        self, x: Any, *, level: float = 0.95, scale: str = "response"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = self._require_fitted()
        x_arr = _ensure_2d_float64(x, name="x")
        mean, lo, hi = f.predict_ci(x_arr, float(level), scale)
        return np.asarray(mean), np.asarray(lo), np.asarray(hi)

    def predict_diff(
        self, x_a: Any, x_b: Any, *, level: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = self._require_fitted()
        a = _ensure_2d_float64(x_a, name="x_a")
        b = _ensure_2d_float64(x_b, name="x_b")
        diff, lo, hi = f.predict_diff(a, b, float(level))
        return np.asarray(diff), np.asarray(lo), np.asarray(hi)

    def vcov(self) -> np.ndarray:
        return np.asarray(self._require_fitted().vcov())

    # --- Forward common getters as attributes for sklearn-ish ergonomics. -----

    @property
    def coef_(self) -> np.ndarray:
        return np.asarray(self._require_fitted().beta)

    @property
    def scale_(self) -> float:
        return float(self._require_fitted().scale)

    @property
    def edf_total_(self) -> float:
        return float(self._require_fitted().edf_total)

    @property
    def rho_(self) -> Union[float, np.ndarray]:
        """Fitted log smoothing parameters. Returns a float for single-
        smooth fits (v0.x-compatible scalar) and an ndarray for multi-
        smooth `Additive` fits."""
        rho = np.asarray(self._require_fitted().rho)
        if rho.shape == (1,):
            return float(rho[0])
        return rho

    @property
    def lambda_(self) -> np.ndarray:
        # `lambda` is a Python keyword — fetch via getattr to access the
        # native PyO3 getter of the same name.
        return np.asarray(getattr(self._require_fitted(), "lambda"))

    @property
    def n_iters_(self) -> int:
        return int(self._require_fitted().n_iters)

    @property
    def converged_(self) -> bool:
        return bool(self._require_fitted().converged)


# =============================================================================
# Typed term dataclasses for multi-smooth `fit_additive` (94b/94c).
# =============================================================================


# `Col` — a column reference. Either an integer index into the design `x`
# or a string column name. Strings are resolved against the DataFrame /
# `predictors=` column-name list at :meth:`Gam.fit` time; ints pass through.
Col = Union[int, str]


@dataclass(frozen=True)
class CrTerm:
    """Univariate CR-spline term `s(x_col, bs='cr', k=k)`.

    ``col`` is either an integer column index into the design ``x`` or a
    string column name (resolved against the DataFrame / ``predictors=``
    list at fit time). ``k`` is the spline basis dim (defaults to 10,
    matching mgcv).

    ``pc`` is mgcv's ``s(x, pc=v)`` point constraint: the smooth passes
    through zero at ``x = v`` instead of being centred over the fit rows,
    and the intercept takes the level. Fitted values, λ̂ and edf are
    unchanged — only the intercept-versus-smooth split moves, which is
    what a caller reading a *partial* curve sees.

    ``by`` is mgcv's numeric ``s(x, by=w)``: a column (index or name) whose
    value multiplies the smooth row by row, so the term contributes
    ``w_i * f(x_i)`` and is exactly 0 where ``w = 0``. Unless ``w`` is
    constant the term is not centred (mgcv's rule) and keeps all ``k``
    coefficients. Exclusive with ``pc``.
    """
    col: Col
    k: int = 10
    pc: Optional[float] = None
    by: Optional[Col] = None


@dataclass(frozen=True)
class CrStableTerm:
    """CR + sum-to-zero + StableReparam rotation term. ``col`` accepts an
    int index or a string column name. ``pc`` and ``by`` as for
    :class:`CrTerm`."""
    col: Col
    k: int = 10
    pc: Optional[float] = None
    by: Optional[Col] = None


@dataclass(frozen=True)
class ReTerm:
    """Random-effect term `s(x_col, bs='re')` — one-hot encoded grouping.
    ``col`` accepts an int index or a string column name."""
    col: Col


@dataclass(frozen=True)
class ParametricTerm:
    """Parametric (linear, unsmoothed) term — `+ x_col` in the formula.

    Adds one raw column to the linear predictor with no spline expansion
    and no smoothing penalty; the coefficient retains its full degree of
    freedom. Equivalent to mgcv R's "pterms" block and mgcv_rust's
    ``bs="parametric"`` / ``"linear"``.

    Use this for 0/1 indicators, count covariates, or any linear effect
    you don't want to smooth. ``col`` accepts an int index or a string
    column name.
    """
    col: Col


@dataclass(frozen=True)
class TeTerm:
    """Anisotropic 2-margin tensor product term `te(x_col_a, x_col_b)` —
    two smoothing parameters per term (one per margin). CR margins.

    ``cols`` is the pair ``(col_a, col_b)``; each entry accepts an int
    index or a string column name. ``k`` is the pair ``(k_a, k_b)`` of
    marginal basis dims (defaults to ``(10, 10)``).
    """
    cols: Tuple[Col, Col]
    k: Tuple[int, int] = (10, 10)


@dataclass(frozen=True)
class TeMultiTerm:
    """Anisotropic n-margin tensor product term `te(x_c0, ..., x_c{D-1})`
    — one smoothing parameter per margin (D >= 2). CR margins, uncentred
    marginals (main effects retained, matching mgcv ``te``).

    ``cols`` is the tuple of column references (length D >= 2; ints or
    strings, mixed allowed). ``k`` is the matching tuple of marginal basis
    dims (defaults to ``(5, ...)``).
    """
    cols: Tuple[Col, ...]
    k: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True)
class TiTerm:
    """N-margin tensor interaction term `ti(x_c0, ..., x_c{D-1})` — pure
    interaction with each margin's main effect excluded (per-margin
    sum-to-zero, matching mgcv ``ti``). One smoothing parameter per
    margin (D >= 2). CR margins.

    ``cols`` is the tuple of column references (length D >= 2; ints or
    strings, mixed allowed). ``k`` is the matching tuple of marginal basis
    dims (defaults to ``(5, ...)``).
    """
    cols: Tuple[Col, ...]
    k: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True)
class TpsTerm:
    """Isotropic thin-plate regression spline `s(x_cols, bs='tp')` —
    one smoothing parameter per term, arbitrary number of input
    margins. Read from ``cols`` columns of ``x``.

    ``cols`` is a tuple of column references (length ≥ 2; ints or strings,
    mixed allowed). ``k`` is the truncated-eigenbasis dimension (defaults
    to ``10 * len(cols)``).
    """
    cols: Tuple[Col, ...]
    k: Optional[int] = None


# Sum type at the Python boundary. Type-checked closed set — adding a
# new term kind extends this union (a library-controlled change).
Term = Union[
    CrTerm,
    CrStableTerm,
    ReTerm,
    ParametricTerm,
    TeTerm,
    TeMultiTerm,
    TiTerm,
    TpsTerm,
]


def pc_or_none(pc: Optional[float]) -> Optional[float]:
    """Coerce a point-constraint value for the FFI boundary. numpy scalars
    reach here from DataFrame-driven callers and the native extractor only
    accepts a Python float or None."""
    return None if pc is None else float(pc)


def by_or_none(by: Optional[Col]) -> Optional[int]:
    """A resolved by-column index for the FFI boundary. A name that reaches
    here was never resolved against the fit's columns."""
    if isinstance(by, str):
        raise TypeError(f"by={by!r} must be resolved to a column index before the fit")
    return None if by is None else int(by)


def _term_to_tuple(term: Term) -> tuple:
    """Convert a typed `Term` to the tuple form `_gamrs_native.fit_additive`
    expects at the FFI boundary. Strings live ONLY here, between the
    Python typed surface and the Rust enum — they never leak in either
    direction."""
    if isinstance(term, CrTerm):
        return (int(term.col), "cr", int(term.k), pc_or_none(term.pc), by_or_none(term.by))
    if isinstance(term, CrStableTerm):
        return (
            int(term.col), "cr_stable", int(term.k), pc_or_none(term.pc), by_or_none(term.by)
        )
    if isinstance(term, ReTerm):
        return (int(term.col), "re")
    if isinstance(term, ParametricTerm):
        return (int(term.col), "parametric")
    if isinstance(term, TeTerm):
        return (
            (int(term.cols[0]), int(term.cols[1])),
            "te",
            (int(term.k[0]), int(term.k[1])),
        )
    if isinstance(term, TeMultiTerm):
        cols_tup = tuple(int(c) for c in term.cols)
        k_tup = (
            tuple(int(ki) for ki in term.k)
            if term.k is not None
            else tuple(5 for _ in cols_tup)
        )
        return (cols_tup, "te_multi", k_tup)
    if isinstance(term, TiTerm):
        cols_tup = tuple(int(c) for c in term.cols)
        k_tup = (
            tuple(int(ki) for ki in term.k)
            if term.k is not None
            else tuple(5 for _ in cols_tup)
        )
        return (cols_tup, "ti", k_tup)
    if isinstance(term, TpsTerm):
        cols_tup = tuple(int(c) for c in term.cols)
        k_val = int(term.k) if term.k is not None else 10 * len(cols_tup)
        return (cols_tup, "tp", k_val)
    raise TypeError(
        f"unknown term type {type(term).__name__}; expected one of "
        "CrTerm / CrStableTerm / ReTerm / ParametricTerm / TeTerm / "
        "TeMultiTerm / TiTerm / TpsTerm"
    )


def fit_additive(
    family: str,
    x: Any,
    y: Any,
    terms: Sequence[Term],
    *,
    weights: Any = None,
    **family_kwargs: Any,
) -> Any:
    """Fit a multi-smooth additive GAM `y ~ s(x_{c_0}) + s(x_{c_1}) + ...`.

    `x` must be a 2-D array of shape ``(n_obs, n_input_dims)``. `y` is
    1-D of length `n_obs`. `terms` is a sequence of typed term objects
    — :class:`CrTerm`, :class:`CrStableTerm`, :class:`ReTerm`,
    :class:`TeTerm`, :class:`TeMultiTerm`, :class:`TiTerm`, or
    :class:`TpsTerm`. Tensor terms provide one smoothing parameter per
    margin and read several columns of `x`.

    Returns the native :class:`FittedGam`. Use
    :attr:`FittedGam.rho` / :attr:`FittedGam.lambda` / :attr:`FittedGam.edf_per_term`
    (all length = number of smoothing params, i.e. 1 per univariate term,
    one per tensor margin) for per-term diagnostics.

    `family` accepts the same strings as :func:`_gamrs_native.fit`,
    including the shape-managed families ``negbin`` / ``nb`` (profile-θ)
    and ``tweedie`` / ``tw``. Family shape kwargs forward via
    ``**family_kwargs`` — e.g. ``theta=`` for negbin; for Tweedie omit
    ``tweedie_p`` for profile-p (p estimated, mgcv ``tw()``) or pass
    ``tweedie_p=val`` to hold p fixed (mgcv ``Tweedie(p=val)``).
    """
    x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if x_arr.ndim != 2:
        raise ValueError(
            f"x must be 2-D (n_obs, n_input_dims); got ndim={x_arr.ndim} (shape={x_arr.shape})"
        )
    y_arr = _ensure_1d_float64(y, name="y")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"x and y must have the same number of rows; got x.shape[0]={x_arr.shape[0]}, "
            f"y.shape[0]={y_arr.shape[0]}"
        )
    w_arr = (
        _ensure_1d_float64(weights, name="weights") if weights is not None else None
    )
    term_tuples = [_term_to_tuple(t) for t in terms]
    if not term_tuples:
        raise ValueError("terms must be non-empty")
    return _gamrs_native.fit_additive(
        family, x_arr, y_arr, term_tuples, weights=w_arr, **family_kwargs
    )
