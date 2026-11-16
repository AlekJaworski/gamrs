"""gamrs — Rust GAM core with a Python API that mirrors
:mod:`mgcv_rust` 1-1 for the basic single-smooth use case.

Drop-in goal: replacing ``from mgcv_rust import Gam`` with
``from gamrs import Gam`` should work for code that fits a single-smooth
Gaussian / Bernoulli / Poisson / Gamma / NegBin / Tweedie / TDist GAM
and uses the standard sklearn-style API (``fit``, ``predict``,
``predict_ci``, ``predict_diff``, ``vcov_``, ``coef_``, ``lambda_``,
…). Calls that exercise features gamrs doesn't yet wire (multi-smooth
fits, subset views, ``predict(type='terms')``, ``partial_effect``,
``plot``, ``serialize``, ``predict_proba`` for ocat, posterior
sampling, ``evaluate_lpmatrix``) raise ``NotImplementedError`` with a
pointer back at :mod:`mgcv_rust`.

Layers (mirrors v0.x's mgcv_rust shape exactly):

1. **High-level wrapper** (:class:`gamrs.Gam`) — sklearn-style class.
   See :mod:`gamrs._fitter` for the full constructor / method surface.
2. **Low-level coercion facade** (:class:`gamrs.GAM`) — minimal
   shape / dtype coercion over the native :func:`gamrs._gamrs_native.fit`
   function. See :mod:`gamrs._low_level`.

Plus 1-1 names matching v0.x's exports:
- :class:`gamrs.GAMFitter` — deprecated alias of :class:`Gam` (emits
  ``DeprecationWarning``).
- :class:`gamrs.TermContributions`, :class:`gamrs.GamSummary`,
  :class:`gamrs.GamPredictor` — shape-matching stubs that error with a
  clear message when invoked, so consumer code that imports them by
  name doesn't blow up at import time.

The native PyO3 module is :mod:`gamrs._gamrs_native` with:
- ``fit(family_name, x, y, weights=None, k=10, design='cr', ...)`` →
  :class:`gamrs._gamrs_native.FittedGam`.
- :class:`gamrs._gamrs_native.FittedGam` exposing ``beta``, ``rho``,
  ``scale``, ``edf_total``, ``n_iters``, ``converged``, ``reml_value``
  getters and ``predict``, ``predict_response``, ``predict_ci``,
  ``predict_diff``, ``vcov`` methods.
"""

from ._low_level import (
    GAM,
    CrStableTerm,
    CrTerm,
    ParametricTerm,
    ReTerm,
    ScTerm,
    TeMultiTerm,
    TeScTerm,
    TeTerm,
    Term,
    TiTerm,
    TpsTerm,
    fit_additive,
)
from ._fitter import Gam, GAMFitter
from ._predictor import GamPredictor
from ._gaulss import GaulssFit, fit_gaulss
from ._shash_gam import ShashGamFit, fit_shash
from ._quantile import (
    QuantileLSSFit,
    fit_quantile,
    fit_quantile_lss,
    tune_quantile_sigma,
)
from ._stubs import GamSummary, TermContributions

__all__ = [
    "Gam",
    "GAMFitter",
    "GAM",
    "GamPredictor",
    "GamSummary",
    "TermContributions",
    # Multi-smooth typed terms + helper (94b/94c).
    "CrTerm",
    "CrStableTerm",
    "ReTerm",
    "ParametricTerm",
    "TeTerm",
    "ScTerm",
    "TeScTerm",
    "TeMultiTerm",
    "TiTerm",
    "TpsTerm",
    "Term",
    "fit_additive",
    # qgam-style σ-calibration (task #111).
    "fit_quantile",
    "tune_quantile_sigma",
    # Distributional location-scale quantile (gaulss/shash view).
    "fit_quantile_lss",
    "QuantileLSSFit",
    # GAMLSS: Gaussian location-scale (joint fit).
    "fit_gaulss",
    "GaulssFit",
    # GAMLSS: native sinh-arcsinh (joint fit) — distinct from the two-stage
    # fit_quantile_lss(shape="shash").
    "fit_shash",
    "ShashGamFit",
]
