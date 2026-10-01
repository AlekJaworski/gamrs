//! PyO3 bindings — `gamrs._gamrs_native` Python extension module.
//!
//! Wraps gamrs's canonical typed Rust API (`gamrs::fit{,_with_design}` +
//! `FittedGam`) behind a single `fit(family, ...)` function that takes
//! string family / design names at the Python boundary and dispatches to
//! the typed `gamrs::fit_with_design(...)` call internally. This is the
//! only place strings cross into the type layer — the Rust core stays
//! fully typed (project standard: zero-string config in Rust).
//!
//! Surface (mirrors v0.x's `mgcv_rust.GAM` shape where it makes sense):
//!
//! - `fit(family_name, x, y, weights=None, k=10, design="cr")` →
//!   `PyFittedGam`.
//! - `PyFittedGam` exposes `beta`, `rho`, `scale`, `edf_total`,
//!   `n_iters`, `converged`, `reml_value` as getters; `predict(x)`,
//!   `predict_ci(x, level, scale)`, `predict_diff(x_a, x_b, level)`,
//!   `vcov()` as methods.
//!
//! Errors map gamrs's `GamrsError::InvalidParameter` (which already carries
//! row-aware guidance — "Gamma requires y > 0; got y=-0.3 at row 42") to
//! Python's `ValueError`, so callers see actionable messages.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyType};
use pyo3::wrap_pyfunction;
use pyo3::Bound;

use crate::design::{Additive, Cr, CrStable, DesignStrategy, MarginKind, Predictor, Re, TermSpec};
use crate::error::GamrsError;
use crate::family::{
    bernoulli_logit, elf_identity, gamma_inverse, gamma_log, gaussian_identity,
    inverse_gaussian_log, negbin_log, ocat_identity, poisson_log, quasibinomial_logit,
    quasipoisson_log, tdist_identity, tweedie_log, tweedie_log_fixed_p,
};
use crate::fit::{scat_response_scale, FamilyFit, FittedGam, PredictScale};

// =============================================================================
// Error mapping — GamrsError → PyValueError / PyRuntimeError.
// =============================================================================

fn map_err(e: GamrsError) -> PyErr {
    match e {
        GamrsError::InvalidParameter(msg) => PyValueError::new_err(msg),
        GamrsError::SingularSystem(msg) => {
            PyRuntimeError::new_err(format!("singular system: {msg}"))
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

// =============================================================================
// PyFittedGam — Python-visible wrapper around the Rust `FittedGam`.
// =============================================================================

#[pyclass(name = "FittedGam", module = "gamrs._gamrs_native")]
pub struct PyFittedGam {
    inner: FittedGam,
}

#[pymethods]
impl PyFittedGam {
    #[getter]
    fn beta<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.beta.clone().into_pyarray(py)
    }

    /// Indices into `beta` of coefficients the design could not identify;
    /// they are fitted as exactly 0 with zero variance.
    #[getter]
    fn aliased_coefficients(&self) -> Vec<usize> {
        self.inner.aliased_coefficients()
    }

    /// Composable post-fit primitive: add `delta` to the model intercept
    /// (β₀), shifting every prediction by `delta`. Family-agnostic mechanism
    /// — the *policy* for choosing `delta` lives in the caller. The quantile
    /// module (`gamrs._quantile`) uses it for qgam-style coverage calibration
    /// (set β₀ so empirical training coverage matches τ), keeping that policy
    /// out of the core fit. No-op on an (impossible) empty coefficient vector.
    fn shift_intercept(&mut self, delta: f64) {
        if !self.inner.beta.is_empty() {
            self.inner.beta[0] += delta;
        }
    }

    /// Fitted log smoothing parameters (one per term). Returns a 1-D
    /// float64 ndarray of length `n_terms` — `len 1` for single-smooth
    /// fits (`Cr` / `Re` / `CrStable`), `len T` for `Additive { terms }`.
    #[getter]
    fn rho<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.rho.clone().into_pyarray(py)
    }

    /// Fitted smoothing parameters `λ_j = exp(ρ_j)` per term. 1-D float64
    /// ndarray of length `n_terms`.
    #[getter]
    fn lambda<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.lambda.clone().into_pyarray(py)
    }

    /// Per-term effective degrees of freedom. 1-D float64 ndarray of
    /// length `n_terms`. Sums to `edf_total - 1` (excluding intercept).
    #[getter]
    fn edf_per_term<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.edf_per_term.clone().into_pyarray(py)
    }

    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale
    }

    #[getter]
    fn edf_total(&self) -> f64 {
        self.inner.edf_total
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }

    #[getter]
    fn n_iters(&self) -> usize {
        self.inner.n_iters
    }

    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    #[getter]
    fn reml_value(&self) -> f64 {
        self.inner.reml_value
    }

    /// Posterior covariance of β̂ — `σ̂² · A⁻¹`. Returns a `(p, p)`
    /// float64 ndarray.
    fn vcov<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.vcov.clone().into_pyarray(py)
    }

    /// Family-specific fitted shape parameters. Empty for families without
    /// any (Gaussian, Bernoulli, Poisson, …). For Ocat: the `R − 2` log-gap
    /// thresholds `θ_j` that map to the `R + 1` category boundaries (mgcv
    /// convention). For TDist: `[log_nu, log_sigma2]`.
    #[getter]
    fn shape_params<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.shape_params.clone().into_pyarray(py)
    }

    /// Diagnostic counters captured during the fit. Always present;
    /// dict has keys: outer_iterations, line_search_trials,
    /// no_refresh_attempts, no_refresh_hits, inner_pirls_calls,
    /// inner_pirls_iterations_total, plus derived:
    /// pirls_iters_per_call, no_refresh_hit_rate.
    #[getter]
    fn fit_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let s = &self.inner.stats;
        let d = pyo3::types::PyDict::new(py);
        d.set_item("outer_iterations", s.outer_iterations)?;
        d.set_item("line_search_trials", s.line_search_trials)?;
        d.set_item("no_refresh_attempts", s.no_refresh_attempts)?;
        d.set_item("no_refresh_hits", s.no_refresh_hits)?;
        d.set_item("inner_pirls_calls", s.inner_pirls_calls)?;
        d.set_item(
            "inner_pirls_iterations_total",
            s.inner_pirls_iterations_total,
        )?;
        d.set_item("pirls_iters_per_call", s.pirls_iters_per_call())?;
        d.set_item("no_refresh_hit_rate", s.no_refresh_hit_rate())?;
        Ok(d)
    }

    /// Diagnostic: evaluate the ocat REML score at an arbitrary
    /// `theta = [ρ_1, …, ρ_T, θ_ocat_1, …]` without re-fitting the
    /// outer Newton. Re-runs the inner PIRLS at the requested θ and
    /// returns the score value. Used for parity diagnostics against
    /// v0.x's `evaluate_reml_ocat_proper_at`.
    ///
    /// Requires the fit to be an ocat fit (the caller's responsibility);
    /// otherwise returns an error.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_reml_at_ocat<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        x: PyReadonlyArray2<'py, f64>,
        theta: PyReadonlyArray1<'py, f64>,
        n_cats: usize,
        k_per_term: Vec<usize>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use crate::design::{Additive, Cr, DesignStrategy, TermSpec};
        use crate::family::ocat_identity;
        use crate::inner::{CholeskySolver, OcatInner, PirlsOpts};
        use crate::traits::InnerSolver;
        use ndarray::Array1;
        use std::marker::PhantomData;

        let _ = Cr { k: 0 };
        let y_arr: Array1<f64> = y.as_array().to_owned();
        let x_view: ArrayView2<f64> = x.as_array();
        let theta_arr: Array1<f64> = theta.as_array().to_owned();

        let terms: Vec<TermSpec> = k_per_term
            .iter()
            .enumerate()
            .map(|(i, &k)| TermSpec::Cr {
                col: i,
                k,
                pc: None,
            })
            .collect();
        let prep = Additive { terms }.prepare(x_view).map_err(map_err)?;
        let n_terms = prep.s_list.len();
        let rho_slice: Array1<f64> = theta_arr.slice(ndarray::s![..n_terms]).to_owned();
        let thresholds: Array1<f64> = if theta_arr.len() > n_terms {
            theta_arr.slice(ndarray::s![n_terms..]).to_owned()
        } else {
            Array1::zeros(n_cats.saturating_sub(2))
        };
        let family = ocat_identity(thresholds, n_cats);
        let inner = OcatInner::<CholeskySolver> {
            x_design: prep.x_design.clone(),
            y: y_arr.clone(),
            prior_weights: None,
            s_list: prep.s_list.clone(),
            family,
            opts: PirlsOpts::default(),
            _solver: PhantomData,
        };
        let fit = inner.fit(&rho_slice).map_err(map_err)?;

        // Decompose the score components per
        // `reml/ocat_joint.rs::reml_criterion_ocat_proper`. Apply the
        // family's mgcv-style rank adjustment (ocat: -1) at the
        // `Σ rank·log λ` term so the diagnostic matches what the
        // outer-Newton score path actually uses.
        let mut bsb_total = 0.0_f64;
        let mut bsb_per_term: Vec<f64> = Vec::with_capacity(n_terms);
        let mut log_det_lambda_s = 0.0_f64;
        let rank_adj = {
            use crate::traits::Loss;
            crate::family::OcatLoss::new(Array1::zeros(n_cats.saturating_sub(2)), n_cats)
                .score_rank_adjustment()
        };
        for j in 0..n_terms {
            let s_beta = prep.s_list[j].dot(&fit.beta);
            let bsb_j: f64 = fit.beta.iter().zip(s_beta.iter()).map(|(a, b)| a * b).sum();
            bsb_per_term.push(bsb_j);
            let lambda_j = rho_slice[j].exp();
            bsb_total += lambda_j * bsb_j;
            let adj_rank_j = ((prep.rank_s_list[j] as i32 + rank_adj).max(1)) as f64;
            log_det_lambda_s += adj_rank_j * rho_slice[j] + prep.log_pseudo_det_s_list[j];
        }
        let dp = fit.deviance + bsb_total;
        let log_det_h = fit.log_det_a();
        let mp = prep.mp;
        let two_pi_ln = (2.0_f64 * std::f64::consts::PI).ln();
        let score =
            dp / 2.0 + 0.5 * log_det_h - 0.5 * log_det_lambda_s - 0.5 * (mp as f64) * two_pi_ln;

        // Reuse the same envelope ρ-gradient helper the outer Newton uses,
        // so the diagnostic and the optimiser stay byte-equivalent (DRY).
        use crate::score::{FixedAtOneProfile, OcatInnerBuilder, ShapeAwareEnvelopeScore};
        use crate::traits::CoordsKind;
        // Need per-term tr(H⁻¹S_j) too, which trace_a_inv computes for us.
        let mut tr_hinv_s_per_term: Vec<f64> = Vec::with_capacity(n_terms);
        for j in 0..n_terms {
            tr_hinv_s_per_term.push(fit.trace_a_inv(prep.s_list[j].view()));
        }
        let score_holder = ShapeAwareEnvelopeScore::<
            crate::family::OcatLoss,
            crate::family::IdentityLink,
            crate::family::OcatVariance,
            OcatInnerBuilder,
            FixedAtOneProfile,
            CholeskySolver,
        > {
            x_design: prep.x_design.clone(),
            y: y_arr.clone(),
            prior_weights: None,
            s_list: prep.s_list.clone(),
            family_base: crate::family::ocat_identity(
                Array1::zeros(n_cats.saturating_sub(2)),
                n_cats,
            ),
            rank_s_list: prep.rank_s_list.clone(),
            mp: prep.mp,
            log_pseudo_det_s_list: prep.log_pseudo_det_s_list.clone(),
            coords: CoordsKind::Identity,
            pirls_opts: PirlsOpts::default(),
            inner_builder: OcatInnerBuilder,
            profile: FixedAtOneProfile,
            _solver: PhantomData,
            accepted_state: std::cell::RefCell::new(None),
            last_eta: std::cell::RefCell::new(None),
            stats: crate::stats::FitStats::new(),
        };
        let family_ocat = crate::family::ocat_identity(
            {
                let mut a = Array1::<f64>::zeros(n_cats.saturating_sub(2));
                for k in 0..n_cats.saturating_sub(2) {
                    a[k] = theta_arr[n_terms + k];
                }
                a
            },
            n_cats,
        );
        let rho_vec = rho_slice.to_vec();
        let grad_rho = score_holder.compute_rho_envelope_gradient(
            &fit,
            &family_ocat,
            &rho_vec,
            &bsb_per_term,
            &tr_hinv_s_per_term,
            1.0,
        );

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("score", score)?;
        dict.set_item("beta", fit.beta.into_pyarray(py))?;
        dict.set_item("eta", fit.eta.into_pyarray(py))?;
        dict.set_item("deviance", fit.deviance)?;
        dict.set_item("bsb_total", bsb_total)?;
        dict.set_item("bsb_per_term", bsb_per_term)?;
        dict.set_item("log_det_h", log_det_h)?;
        dict.set_item("log_det_lambda_s", log_det_lambda_s)?;
        dict.set_item("mp", mp)?;
        dict.set_item("iters", fit.iterations)?;
        dict.set_item("converged", fit.converged)?;
        dict.set_item("grad_rho", grad_rho)?;
        Ok(dict)
    }

    /// Diagnostic: evaluate the Tweedie REML score at an arbitrary
    /// `theta = [ρ_1, …, ρ_T, log_φ, p_transform]` without re-fitting the
    /// outer Newton. Re-runs the inner PIRLS at the requested θ and
    /// returns a dict with every additive component of mgcv's score
    /// formula `Dp/(2φ) - ls + log|H|/2 - log|λS|+/2 - Mp/2·log(2πφ)`.
    /// Used for parity diagnostics against v0.x's
    /// `evaluate_reml_tweedie_components_at`.
    ///
    /// Requires the family to be Tweedie. The shape part of `theta` is
    /// the standard gamrs Tweedie shape vector — `[log φ, p_transform]`
    /// where `p = 1 + sigmoid(p_transform)` clamped to `[1.05, 1.95]`
    /// (`tweedie.rs::set_shape_params`).
    #[allow(clippy::too_many_arguments)]
    fn evaluate_reml_at_tweedie<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        x: PyReadonlyArray2<'py, f64>,
        theta: PyReadonlyArray1<'py, f64>,
        k_per_term: Vec<usize>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use crate::design::{Additive, DesignStrategy, TermSpec};
        use crate::family::tweedie_log;
        use crate::inner::{CholeskySolver, PirlsInner, PirlsOpts};
        use crate::traits::{InnerSolver, Loss};
        use ndarray::Array1;
        use std::marker::PhantomData;

        let y_arr: Array1<f64> = y.as_array().to_owned();
        let x_view: ArrayView2<f64> = x.as_array();
        let theta_arr: Array1<f64> = theta.as_array().to_owned();

        let terms: Vec<TermSpec> = k_per_term
            .iter()
            .enumerate()
            .map(|(i, &k)| TermSpec::Cr {
                col: i,
                k,
                pc: None,
            })
            .collect();
        let prep = Additive { terms }.prepare(x_view).map_err(map_err)?;
        let n_terms = prep.s_list.len();
        if theta_arr.len() != n_terms + 2 {
            return Err(PyValueError::new_err(format!(
                "evaluate_reml_at_tweedie expects theta=[ρ_1..ρ_T, log_φ, p_transform] \
                 (length {}); got {}",
                n_terms + 2,
                theta_arr.len()
            )));
        }
        let rho_slice: Array1<f64> = theta_arr.slice(ndarray::s![..n_terms]).to_owned();
        let shape_slice: Vec<f64> = theta_arr.iter().skip(n_terms).copied().collect();

        // Build the Tweedie family with the supplied shape params. Use any
        // valid init (p=1.5, φ=1.0); set_shape_params rewrites both.
        let mut family = tweedie_log(1.5, 1.0);
        family.set_shape_params(&shape_slice);
        let p_value = family.loss.p;
        let phi_value = family.loss.phi;

        // Per-family inner-PIRLS tolerance via the typed hook (Tweedie
        // currently uses the default 1e-9; the override pathway is in
        // place if v0.x parity ever needs tighter).
        let opts = PirlsOpts {
            dev_rel_tol: family.loss.pirls_dev_rel_tol(),
            ..PirlsOpts::default()
        };

        let inner = PirlsInner::<_, _, _, CholeskySolver> {
            x_design: prep.x_design.clone(),
            y: y_arr.clone(),
            prior_weights: None,
            s_list: prep.s_list.clone(),
            family: family.clone(),
            opts,
            _solver: PhantomData,
        };
        let fit = inner.fit(&rho_slice).map_err(map_err)?;

        // Score decomposition — mirrors `score/shape_aware.rs::score_value`
        // for Tweedie/OwnedByLossProfile. Apply the family's mgcv-style
        // rank adjustment so the diagnostic matches the outer-Newton
        // path. Tweedie's default `score_rank_adjustment() = 0`; the
        // typed hook lets future families override.
        let rank_adj = family.loss.score_rank_adjustment();
        let mut bsb_total = 0.0_f64;
        let mut bsb_per_term: Vec<f64> = Vec::with_capacity(n_terms);
        let mut log_det_lambda_s = 0.0_f64;
        for j in 0..n_terms {
            let s_beta = prep.s_list[j].dot(&fit.beta);
            let bsb_j: f64 = fit.beta.iter().zip(s_beta.iter()).map(|(a, b)| a * b).sum();
            bsb_per_term.push(bsb_j);
            let lambda_j = rho_slice[j].exp();
            bsb_total += lambda_j * bsb_j;
            let adj_rank_j = ((prep.rank_s_list[j] as i32 + rank_adj).max(1)) as f64;
            log_det_lambda_s += adj_rank_j * rho_slice[j] + prep.log_pseudo_det_s_list[j];
        }
        let dp = fit.deviance + bsb_total;
        let log_det_h = fit.log_det_a();
        let mp = prep.mp;

        // OwnedByLossProfile reads φ live off the family — same as the
        // shape-aware score body.
        let phi = phi_value.max(1e-12);

        // Saturated log-lik: Σ_i ls_i(y_i; φ, p).
        let ls_sum: f64 = y_arr
            .iter()
            .map(|&yi| family.loss.saturated_log_lik(yi, phi))
            .sum();

        // One definition of the score, shared with the fitter — a diagnostic
        // that re-derives it is not a diagnostic of it.
        let score =
            crate::score::reml_score_from_parts(dp, phi, mp, log_det_h, log_det_lambda_s, ls_sum);

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("score", score)?;
        dict.set_item("beta", fit.beta.clone().into_pyarray(py))?;
        dict.set_item("eta", fit.eta.clone().into_pyarray(py))?;
        dict.set_item("mu", fit.mu.clone().into_pyarray(py))?;
        dict.set_item("deviance", fit.deviance)?;
        dict.set_item("bsb_total", bsb_total)?;
        dict.set_item("bsb_per_term", bsb_per_term)?;
        dict.set_item("dp", dp)?;
        dict.set_item("log_det_h", log_det_h)?;
        dict.set_item("log_det_lambda_s", log_det_lambda_s)?;
        dict.set_item("ls", ls_sum)?;
        dict.set_item("phi", phi)?;
        dict.set_item("p", p_value)?;
        dict.set_item("mp", mp)?;
        dict.set_item("iters", fit.iterations)?;
        dict.set_item("converged", fit.converged)?;
        Ok(dict)
    }

    /// Diagnostic: evaluate the scat (TDist) LAML score at an arbitrary
    /// `theta = [ρ_1, …, ρ_T, log_σ², log(ν - 2)]` without re-fitting the
    /// outer Newton. Re-runs the inner PIRLS at the requested θ and
    /// returns a dict with every additive component of mgcv's GamFit5
    /// formula `Dp/2 - ls + log|H|/2 - log|λS|+/2 - Mp/2·log(2π)`.
    /// Used for parity diagnostics against v0.x's
    /// `evaluate_reml_scat_components_at`.
    ///
    /// Requires the family to be scat/TDist. The shape part of `theta` is
    /// the standard gamrs TDist shape vector — `[log σ², log(ν - 2)]`
    /// (`tdist.rs::set_shape_params`).
    #[allow(clippy::too_many_arguments)]
    fn evaluate_reml_at_scat<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        x: PyReadonlyArray2<'py, f64>,
        theta: PyReadonlyArray1<'py, f64>,
        k_per_term: Vec<usize>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use crate::design::{Additive, DesignStrategy, TermSpec};
        use crate::family::tdist_identity;
        use crate::inner::{CholeskySolver, PirlsInner, PirlsOpts};
        use crate::traits::{InnerSolver, Loss};
        use ndarray::Array1;
        use std::marker::PhantomData;

        let y_arr: Array1<f64> = y.as_array().to_owned();
        let x_view: ArrayView2<f64> = x.as_array();
        let theta_arr: Array1<f64> = theta.as_array().to_owned();

        let terms: Vec<TermSpec> = k_per_term
            .iter()
            .enumerate()
            .map(|(i, &k)| TermSpec::Cr {
                col: i,
                k,
                pc: None,
            })
            .collect();
        let prep = Additive { terms }.prepare(x_view).map_err(map_err)?;
        let n_terms = prep.s_list.len();
        if theta_arr.len() != n_terms + 2 {
            return Err(PyValueError::new_err(format!(
                "evaluate_reml_at_scat expects theta=[ρ_1..ρ_T, log_σ², log(ν-2)] \
                 (length {}); got {}",
                n_terms + 2,
                theta_arr.len()
            )));
        }
        let rho_slice: Array1<f64> = theta_arr.slice(ndarray::s![..n_terms]).to_owned();
        let shape_slice: Vec<f64> = theta_arr.iter().skip(n_terms).copied().collect();

        // Build the TDist family with the supplied shape params. Use any
        // valid init (ν=5, σ²=1); set_shape_params rewrites both.
        let mut family = tdist_identity(5.0, 1.0);
        family.set_shape_params(&shape_slice);
        let nu_value = family.loss.nu;
        let sigma2_value = family.loss.sigma2;

        // Per-family inner-PIRLS tolerance via the typed hook.
        let opts = PirlsOpts {
            dev_rel_tol: family.loss.pirls_dev_rel_tol(),
            ..PirlsOpts::default()
        };

        let inner = PirlsInner::<_, _, _, CholeskySolver> {
            x_design: prep.x_design.clone(),
            y: y_arr.clone(),
            prior_weights: None,
            s_list: prep.s_list.clone(),
            family: family.clone(),
            opts,
            _solver: PhantomData,
        };
        let fit = inner.fit(&rho_slice).map_err(map_err)?;

        // Score decomposition — mirrors `score/shape_aware.rs::score_value`
        // for TDist/FixedAtOneProfile. Apply the family's mgcv-style
        // rank adjustment so the diagnostic matches the outer-Newton path.
        // TDist's default `score_rank_adjustment() = 0`; the typed hook
        // lets future families override.
        let rank_adj = family.loss.score_rank_adjustment();
        let mut bsb_total = 0.0_f64;
        let mut bsb_per_term: Vec<f64> = Vec::with_capacity(n_terms);
        let mut log_det_lambda_s = 0.0_f64;
        for j in 0..n_terms {
            let s_beta = prep.s_list[j].dot(&fit.beta);
            let bsb_j: f64 = fit.beta.iter().zip(s_beta.iter()).map(|(a, b)| a * b).sum();
            bsb_per_term.push(bsb_j);
            let lambda_j = rho_slice[j].exp();
            bsb_total += lambda_j * bsb_j;
            let adj_rank_j = ((prep.rank_s_list[j] as i32 + rank_adj).max(1)) as f64;
            log_det_lambda_s += adj_rank_j * rho_slice[j] + prep.log_pseudo_det_s_list[j];
        }
        let dp = fit.deviance + bsb_total;
        // Same `log|H|` precedence as `ShapeAwareEnvelopeScore` — this
        // diagnostic has to evaluate the SAME criterion the fitter optimises,
        // or a profile taken with it is a profile of a different function.
        let prior_w_ones = Array1::<f64>::ones(fit.n);
        let s_total_eval = crate::design::combined_s(
            &prep.s_list,
            &Array1::from(rho_slice.to_vec()),
            prep.x_design.ncols(),
        );
        let log_det_h = crate::inner::pirls::score_log_det_h(
            &family,
            &y_arr,
            &fit.eta,
            &fit.mu,
            &prior_w_ones,
            &prep.x_design,
            &s_total_eval,
            &fit,
        );
        let mp = prep.mp;

        // FixedAtOneProfile sets φ=1 for scat — σ² lives inside the loss.
        let phi: f64 = 1.0;

        // Saturated log-lik: Σ_i ls_i(y_i; sigma2, nu). For scat this is
        // location-scale, independent of y — same formula v0.x uses
        // (`pirls/mod.rs:521-528`).
        let ls_sum: f64 = y_arr
            .iter()
            .map(|&yi| family.loss.saturated_log_lik(yi, phi))
            .sum();

        // One definition of the score, shared with the fitter — a diagnostic
        // that re-derives it is not a diagnostic of it.
        let score =
            crate::score::reml_score_from_parts(dp, phi, mp, log_det_h, log_det_lambda_s, ls_sum);

        // Analytic gradient from the SAME score type the outer Newton drives,
        // so a diagnostic FD compares against what the optimiser is actually
        // handed rather than a re-derivation. Needed because the shape axes are
        // a catastrophic cancellation: on the real sale_date term the log-sigma2
        // gradient is 310.000 - 306.534 - 3.467 = -8.7e-4, a residual 3e-6 the
        // size of its own terms, so nothing but the real gradient will do.
        let grad_vec: crate::error::Result<Vec<f64>> = {
            use crate::score::{FixedAtOneProfile, PirlsInnerBuilder, ShapeAwareEnvelopeScore};
            use crate::traits::{CoordsKind, ScoreDerivatives};
            let holder = ShapeAwareEnvelopeScore::<
                crate::family::TDist,
                crate::family::IdentityLink,
                crate::family::TVariance,
                PirlsInnerBuilder,
                FixedAtOneProfile,
                CholeskySolver,
            > {
                x_design: prep.x_design.clone(),
                y: y_arr.clone(),
                prior_weights: None,
                s_list: prep.s_list.clone(),
                family_base: tdist_identity(5.0, 1.0),
                rank_s_list: prep.rank_s_list.clone(),
                mp: prep.mp,
                log_pseudo_det_s_list: prep.log_pseudo_det_s_list.clone(),
                coords: CoordsKind::Identity,
                pirls_opts: PirlsOpts {
                    dev_rel_tol: family.loss.pirls_dev_rel_tol(),
                    ..PirlsOpts::default()
                },
                inner_builder: PirlsInnerBuilder,
                profile: FixedAtOneProfile,
                _solver: PhantomData,
                accepted_state: std::cell::RefCell::new(None),
                last_eta: std::cell::RefCell::new(None),
                stats: crate::stats::FitStats::new(),
            };
            holder.value_and_grad(&theta_arr).map(|(_v, g)| g.to_vec())
        };

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("score", score)?;
        // Report the failure instead of omitting the key: a caller defaulting a
        // missing `grad` to NaN cannot tell "not computed" from "computed NaN".
        match grad_vec {
            Ok(g) => dict.set_item("grad", g)?,
            Err(e) => dict.set_item("grad_error", format!("{e}"))?,
        }
        dict.set_item("beta", fit.beta.clone().into_pyarray(py))?;
        dict.set_item("eta", fit.eta.clone().into_pyarray(py))?;
        dict.set_item("mu", fit.mu.clone().into_pyarray(py))?;
        dict.set_item("deviance", fit.deviance)?;
        dict.set_item("bsb_total", bsb_total)?;
        dict.set_item("bsb_per_term", bsb_per_term)?;
        dict.set_item("dp", dp)?;
        dict.set_item("log_det_h", log_det_h)?;
        dict.set_item("log_det_lambda_s", log_det_lambda_s)?;
        dict.set_item("ls", ls_sum)?;
        dict.set_item("nu", nu_value)?;
        dict.set_item("sigma2", sigma2_value)?;
        dict.set_item("mp", mp)?;
        dict.set_item("iters", fit.iterations)?;
        dict.set_item("converged", fit.converged)?;
        Ok(dict)
    }

    /// Diagnostic: evaluate the NegBin REML score at an arbitrary
    /// `theta = [ρ_1, …, ρ_T, log_θ]` without re-fitting the outer Newton.
    /// Re-runs the inner PIRLS at the requested θ and returns a dict with
    /// every additive component of mgcv's score formula
    /// `Dp/(2φ) - ls - 0.5·Mp·log(2πφ) + 0.5·log|H| - 0.5·log|λS|+` with
    /// φ=1 fixed for NegBin. Used for parity diagnostics against v0.x's
    /// `evaluate_reml_negbin_components_at`.
    ///
    /// Requires the family to be NegBin. The shape part of `theta` is the
    /// standard gamrs NegBin shape vector — `[log θ]` (`negbin.rs::set_shape_params`).
    #[allow(clippy::too_many_arguments)]
    fn evaluate_reml_at_negbin<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        x: PyReadonlyArray2<'py, f64>,
        theta: PyReadonlyArray1<'py, f64>,
        k_per_term: Vec<usize>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use crate::design::{Additive, DesignStrategy, TermSpec};
        use crate::family::negbin_log;
        use crate::inner::{CholeskySolver, PirlsInner, PirlsOpts};
        use crate::traits::{InnerSolver, Loss};
        use ndarray::Array1;
        use std::marker::PhantomData;

        let y_arr: Array1<f64> = y.as_array().to_owned();
        let x_view: ArrayView2<f64> = x.as_array();
        let theta_arr: Array1<f64> = theta.as_array().to_owned();

        let terms: Vec<TermSpec> = k_per_term
            .iter()
            .enumerate()
            .map(|(i, &k)| TermSpec::Cr {
                col: i,
                k,
                pc: None,
            })
            .collect();
        let prep = Additive { terms }.prepare(x_view).map_err(map_err)?;
        let n_terms = prep.s_list.len();
        if theta_arr.len() != n_terms + 1 {
            return Err(PyValueError::new_err(format!(
                "evaluate_reml_at_negbin expects theta=[ρ_1..ρ_T, log_θ] \
                 (length {}); got {}",
                n_terms + 1,
                theta_arr.len()
            )));
        }
        let rho_slice: Array1<f64> = theta_arr.slice(ndarray::s![..n_terms]).to_owned();
        let shape_slice: Vec<f64> = theta_arr.iter().skip(n_terms).copied().collect();

        // Build the NegBin family with the supplied shape param. Use any
        // valid init (θ=2.0); set_shape_params rewrites it.
        let mut family = negbin_log(2.0);
        family.set_shape_params(&shape_slice);
        let theta_value = family.loss.theta;

        // Per-family inner-PIRLS tolerance via the typed hook (uses
        // `NegBin::pirls_dev_rel_tol` — overridable per family without
        // touching the diagnostic).
        let opts = PirlsOpts {
            dev_rel_tol: family.loss.pirls_dev_rel_tol(),
            ..PirlsOpts::default()
        };

        let inner = PirlsInner::<_, _, _, CholeskySolver> {
            x_design: prep.x_design.clone(),
            y: y_arr.clone(),
            prior_weights: None,
            s_list: prep.s_list.clone(),
            family: family.clone(),
            opts,
            _solver: PhantomData,
        };
        let fit = inner.fit(&rho_slice).map_err(map_err)?;

        // Score decomposition — mirrors `score/shape_aware.rs::score_value`
        // for NegBin/FixedAtOneProfile. Apply the family's mgcv-style rank
        // adjustment so the diagnostic matches the outer-Newton path.
        // NegBin's `score_rank_adjustment()` is family-overridable; the
        // typed hook lets the value flow through both the score body and
        // this diagnostic by construction.
        let rank_adj = family.loss.score_rank_adjustment();
        let mut bsb_total = 0.0_f64;
        let mut bsb_per_term: Vec<f64> = Vec::with_capacity(n_terms);
        let mut log_det_lambda_s = 0.0_f64;
        for j in 0..n_terms {
            let s_beta = prep.s_list[j].dot(&fit.beta);
            let bsb_j: f64 = fit.beta.iter().zip(s_beta.iter()).map(|(a, b)| a * b).sum();
            bsb_per_term.push(bsb_j);
            let lambda_j = rho_slice[j].exp();
            bsb_total += lambda_j * bsb_j;
            let adj_rank_j = ((prep.rank_s_list[j] as i32 + rank_adj).max(1)) as f64;
            log_det_lambda_s += adj_rank_j * rho_slice[j] + prep.log_pseudo_det_s_list[j];
        }
        let dp = fit.deviance + bsb_total;
        // log|H| — `fit.log_det_a()` returns the Fisher-W A's log|det|
        // via the stored factorisation. When `use_newton_irls() = true`
        // we lazily compute the Newton-W path via the trait method
        // (mgcv_rust port: Newton-A is built at score time, not PIRLS
        // time — see `src/reml/mod.rs:460-483`). Mirrors the score body
        // in `envelope.rs`.
        let log_det_h = inner
            .lazy_newton_log_det_h(&fit, &rho_slice)
            .unwrap_or_else(|| fit.log_det_a());
        let mp = prep.mp;

        // FixedAtOneProfile: φ=1 for NegBin (dispersion lives in θ).
        let phi: f64 = 1.0;

        // Saturated NB log-lik per `negbin.rs::saturated_log_lik`. v0.x's
        // closed form keeps the lgamma(y+θ)-lgamma(θ) θ-dependent block
        // and the y·log(y/(y+θ)) + θ·log(θ/(y+θ)) terms; the lgamma(y+1)
        // term is dropped (constant in θ). The two-engine score comparison
        // already accounts for the dropped constant — both sides drop it.
        let ls_sum: f64 = y_arr
            .iter()
            .map(|&yi| family.loss.saturated_log_lik(yi, phi))
            .sum();

        // One definition of the score, shared with the fitter — a diagnostic
        // that re-derives it is not a diagnostic of it.
        let score =
            crate::score::reml_score_from_parts(dp, phi, mp, log_det_h, log_det_lambda_s, ls_sum);

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("score", score)?;
        dict.set_item("beta", fit.beta.clone().into_pyarray(py))?;
        dict.set_item("eta", fit.eta.clone().into_pyarray(py))?;
        dict.set_item("mu", fit.mu.clone().into_pyarray(py))?;
        dict.set_item("deviance", fit.deviance)?;
        dict.set_item("bsb_total", bsb_total)?;
        dict.set_item("bsb_per_term", bsb_per_term)?;
        dict.set_item("dp", dp)?;
        dict.set_item("log_det_h", log_det_h)?;
        dict.set_item("log_det_lambda_s", log_det_lambda_s)?;
        dict.set_item("ls", ls_sum)?;
        dict.set_item("theta", theta_value)?;
        dict.set_item("mp", mp)?;
        dict.set_item("iters", fit.iterations)?;
        dict.set_item("converged", fit.converged)?;
        Ok(dict)
    }

    /// Per-term column ranges into the lpmatrix `[1 | C_1 | C_2 | …]`.
    /// Returns a list of `(first, last_exclusive)` tuples — one per term
    /// for an Additive fit, or a single `[(1, p)]` for a single-smooth fit.
    /// The intercept always sits at column 0, outside any term range.
    fn term_col_ranges(&self) -> Vec<(usize, usize)> {
        let p = self.inner.beta.len();
        match &self.inner.predictor {
            Predictor::Additive(ap) => ap.term_col_ranges.clone(),
            _ => vec![(1, p)],
        }
    }

    /// Rebuild the design matrix (lpmatrix) at new `x_new`. Shape
    /// `(n_new, p)` with column 0 = intercept and columns 1..p = the
    /// per-term blocks. Use with `coef_` for partial / subset predictions
    /// or with `vcov` for posterior sampling.
    fn evaluate_lpmatrix<'py>(
        &self,
        py: Python<'py>,
        x_new: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_view: ArrayView2<f64> = x_new.as_array();
        let lp = self.inner.predictor.design(x_view).map_err(map_err)?;
        Ok(lp.into_pyarray(py))
    }

    /// Predict η (linear predictor) on new x. `x_new` is a 2-D float64
    /// ndarray of shape `(n_new, n_input_dims)`. Returns a 1-D float64
    /// ndarray of length `n_new`.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x_new: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_view: ArrayView2<f64> = x_new.as_array();
        let eta = self.inner.predict(x_view).map_err(map_err)?;
        Ok(eta.into_pyarray(py))
    }

    /// Predict μ (response scale) by inverse-linking η elementwise.
    fn predict_response<'py>(
        &self,
        py: Python<'py>,
        x_new: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_view: ArrayView2<f64> = x_new.as_array();
        let eta = self.inner.predict(x_view).map_err(map_err)?;
        let link = self.inner.link_kind;
        let mu: Array1<f64> = eta.mapv(|e| link.inverse(e));
        Ok(mu.into_pyarray(py))
    }

    /// Wald-style CI for predictions at `x_new`. Returns `(mean, lo, hi)`
    /// as three 1-D float64 arrays on the requested scale.
    ///
    /// `scale` accepts `"link"` (η) or `"response"` (μ).
    #[pyo3(signature = (x_new, level=0.95, scale="response"))]
    fn predict_ci<'py>(
        &self,
        py: Python<'py>,
        x_new: PyReadonlyArray2<'py, f64>,
        level: f64,
        scale: &str,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let x_view: ArrayView2<f64> = x_new.as_array();
        let s = match scale {
            "link" => PredictScale::Link,
            "response" => PredictScale::Response,
            other => {
                return Err(PyValueError::new_err(format!(
                    "scale must be 'link' or 'response', got {other:?}"
                )))
            }
        };
        let (mean, lo, hi) = self.inner.predict_ci(x_view, level, s).map_err(map_err)?;
        Ok((
            mean.into_pyarray(py),
            lo.into_pyarray(py),
            hi.into_pyarray(py),
        ))
    }

    /// Serialize the fit to a length-framed binary buffer (magic +
    /// version + JSON body). Round-trips bit-for-bit through
    /// :meth:`deserialize` — predictions are FP-identical after a
    /// reload. See `crates/gamrs/src/fit/persistence.rs` for the wire
    /// format.
    fn serialize<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.inner.serialize().map_err(map_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Classmethod: rebuild a :class:`FittedGam` from the bytes produced
    /// by :meth:`serialize`. Raises ``ValueError`` (mapped from
    /// ``GamrsError::InvalidParameter``) on a bad magic / version /
    /// truncated body / unparseable JSON.
    #[classmethod]
    fn deserialize(_cls: &Bound<'_, PyType>, bytes: &[u8]) -> PyResult<Self> {
        let inner = FittedGam::deserialize(bytes).map_err(map_err)?;
        Ok(Self { inner })
    }

    /// Serialize the fit to a plain UTF-8 JSON string (unframed,
    /// human-debuggable). Carries the same fitted state as
    /// :meth:`serialize` — β, vcov, knots, centring, reparam — so
    /// predictions after a JSON round-trip are numerically identical
    /// (mod f64 round-tripping through decimal representation, which
    /// `serde_json` performs losslessly).
    fn serialize_json(&self) -> PyResult<String> {
        self.inner.serialize_json().map_err(map_err)
    }

    /// Classmethod: rebuild a :class:`FittedGam` from a JSON string
    /// produced by :meth:`serialize_json`. Raises ``ValueError`` on
    /// unparseable input.
    #[classmethod]
    fn deserialize_json(_cls: &Bound<'_, PyType>, s: &str) -> PyResult<Self> {
        let inner = FittedGam::deserialize_json(s).map_err(map_err)?;
        Ok(Self { inner })
    }

    /// Python pickle protocol — `pickle.dumps(fit)` / `pickle.loads(...)`
    /// work transparently by routing through :meth:`serialize` /
    /// :meth:`deserialize`. Returns ``(_reconstruct, (bytes,))`` so the
    /// pickle stream is the same compact binary frame.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Py<PyAny>, (Bound<'py, PyBytes>,))> {
        let bytes = self.inner.serialize().map_err(map_err)?;
        let cls = py.get_type::<PyFittedGam>();
        let reconstruct = cls.getattr("deserialize")?.unbind();
        Ok((reconstruct, (PyBytes::new(py, &bytes),)))
    }

    /// Wald CI for the contrast `Δ = predict(x_a) - predict(x_b)` on the
    /// η scale. Returns `(diff, lo, hi)` as three 1-D float64 arrays.
    /// Broadcasts when one of the arrays has a single element.
    #[pyo3(signature = (x_a, x_b, level=0.95))]
    fn predict_diff<'py>(
        &self,
        py: Python<'py>,
        x_a: PyReadonlyArray2<'py, f64>,
        x_b: PyReadonlyArray2<'py, f64>,
        level: f64,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let a: ArrayView2<f64> = x_a.as_array();
        let b: ArrayView2<f64> = x_b.as_array();
        let (diff, lo, hi) = self.inner.predict_diff(a, b, level).map_err(map_err)?;
        Ok((
            diff.into_pyarray(py),
            lo.into_pyarray(py),
            hi.into_pyarray(py),
        ))
    }
}

// =============================================================================
// Internal helpers — design-strategy dispatch and a per-family fit macro.
// =============================================================================

/// Run `gamrs::fit_with_design` for a typed family using one of the
/// canonical design strategies, with the string `design` keyword
/// mediated at this single boundary.
/// Design choice resolved from the FFI string under the GIL, so the
/// GIL-released closure below never captures a Python-borrowed `&str`.
#[derive(Clone, Copy)]
enum DesignChoice {
    Cr,
    Re,
    CrStable,
}

fn fit_dispatch_design<'py, L, K, V>(
    py: Python<'py>,
    family: crate::family::Family<L, K, V>,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: Option<ArrayView1<f64>>,
    design_name: &str,
    k: usize,
) -> PyResult<FittedGam>
where
    // `+ Send` so the GIL-released closure satisfies pyo3's `Ungil`
    // (`unsafe impl<T: Send> Ungil`). The family/link/variance types are plain
    // Rust structs, so Send holds.
    L: FamilyFit<K, V> + Send,
    K: crate::traits::Link + Clone + Send,
    V: crate::traits::VarianceFn + Clone + Send,
{
    // Resolve the design string under the GIL (cheap) — keeps the
    // GIL-released section below free of any Python-borrowed data.
    let choice = match design_name {
        "cr" => DesignChoice::Cr,
        "re" => DesignChoice::Re,
        "cr_stable" => DesignChoice::CrStable,
        other => {
            return Err(PyValueError::new_err(format!(
                "design must be 'cr', 're', or 'cr_stable'; got {other:?}"
            )))
        }
    };
    // Release the GIL for the heavy basis build + PIRLS/REML solve so Python
    // threads can fit GAMs in parallel (e.g. a parallel DAG). `x` / `y` /
    // `weights` are views into caller-OWNED Rust arrays (copied off the numpy
    // buffers before this call), so nothing in the closure aliases a live
    // Python object while the GIL is dropped.
    py.detach(move || {
        let prep = match choice {
            DesignChoice::Cr => Cr { k }.prepare(x)?,
            DesignChoice::Re => Re.prepare(x)?,
            DesignChoice::CrStable => CrStable { k }.prepare(x)?,
        };
        L::fit_from_prep(family, prep, x, y, weights)
    })
    .map_err(map_err)
}

// =============================================================================
// The single string→type dispatch boundary — `fit(family_name, ...)`.
// =============================================================================

/// Fit a gamrs GAM. The only string-keyed entry into the typed core.
///
/// `family_name` accepts the v0.x-compatible names:
/// - `"gaussian"` → `gaussian_identity()`
/// - `"bernoulli"` / `"binomial"` → `bernoulli_logit()`
/// - `"poisson"` → `poisson_log()`
/// - `"quasipoisson"` → `quasipoisson_log()`
/// - `"quasibinomial"` → `quasibinomial_logit()`
/// - `"Gamma"` → `gamma_inverse()` (mgcv's canonical default for `Gamma()`)
/// - `"gamma"` → `gamma_log()` (backwards-compatible log-link alias)
/// - `"inverse_gaussian"` / `"inverse.gaussian"` → `inverse_gaussian_log()`
/// - `"negbin"` / `"nb"` → `negbin_log(theta=2.0)` (or user-passed theta)
/// - `"tdist"` / `"scat"` → `tdist_identity(nu=5, sigma2=1)`
/// - `"tweedie"` / `"tw"` → Tweedie + log link. `tweedie_p` toggles the
///   mode (mgcv_rust convention): `tweedie_p=None` → **profile-p** (mgcv
///   `tw()`): `p` is estimated jointly with `φ` and the smoothing params,
///   initialised at 1.5. `tweedie_p=Some(val)` → **fixed-p** (mgcv
///   `Tweedie(p=val)`): `p` is held CONSTANT at `val` (must be in `(1, 2)`)
///   and only `φ` + smoothing params are estimated.
/// - `"ocat"` → `ocat_identity(n_cats=r)` — requires `r`.
/// - `"elf"` / `"quantile"` → `elf_identity(tau=0.5, sigma=0, lambda=0)`
///   (auto-tuned warm start).
///
/// `design` accepts `"cr"` (default), `"re"`, or `"cr_stable"`.
#[pyfunction]
#[pyo3(signature = (
    family_name,
    x,
    y,
    weights=None,
    k=10,
    design="cr",
    theta=None,
    nu=None,
    sigma2=None,
    tweedie_p=None,
    tweedie_phi=None,
    r=None,
    tau=None,
    elf_sigma=None,
    elf_lambda=None,
))]
fn fit<'py>(
    py: Python<'py>,
    family_name: &str,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    k: usize,
    design: &str,
    theta: Option<f64>,
    nu: Option<f64>,
    sigma2: Option<f64>,
    tweedie_p: Option<f64>,
    tweedie_phi: Option<f64>,
    r: Option<usize>,
    tau: Option<f64>,
    elf_sigma: Option<f64>,
    elf_lambda: Option<f64>,
) -> PyResult<PyFittedGam> {
    // Copy the numpy inputs into owned Rust arrays so the per-family dispatch
    // can release the GIL during the solve (see `fit_dispatch_design`) without
    // aliasing the live numpy buffers. The copy is O(n·p), negligible next to
    // the fit.
    let x_owned: Array2<f64> = x.as_array().to_owned();
    let y_owned: Array1<f64> = y.as_array().to_owned();
    let w_owned: Option<Array1<f64>> = weights.map(|w| w.as_array().to_owned());
    let x_view: ArrayView2<f64> = x_owned.view();
    let y_view: ArrayView1<f64> = y_owned.view();
    let w_view: Option<ArrayView1<f64>> = w_owned.as_ref().map(|a| a.view());

    let fitted: FittedGam = match family_name {
        "gaussian" => {
            fit_dispatch_design(py, gaussian_identity(), x_view, y_view, w_view, design, k)?
        }
        "bernoulli" | "binomial" => {
            fit_dispatch_design(py, bernoulli_logit(), x_view, y_view, w_view, design, k)?
        }
        "poisson" => fit_dispatch_design(py, poisson_log(), x_view, y_view, w_view, design, k)?,
        "quasipoisson" => {
            fit_dispatch_design(py, quasipoisson_log(), x_view, y_view, w_view, design, k)?
        }
        "quasibinomial" => {
            fit_dispatch_design(py, quasibinomial_logit(), x_view, y_view, w_view, design, k)?
        }
        "Gamma" => fit_dispatch_design(py, gamma_inverse(), x_view, y_view, w_view, design, k)?,
        "gamma" => fit_dispatch_design(py, gamma_log(), x_view, y_view, w_view, design, k)?,
        "inverse_gaussian" | "inverse.gaussian" => fit_dispatch_design(
            py,
            inverse_gaussian_log(),
            x_view,
            y_view,
            w_view,
            design,
            k,
        )?,
        "negbin" | "nb" => {
            let theta_val = theta.unwrap_or(2.0);
            if theta_val <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "negbin theta must be > 0; got theta={theta_val}"
                )));
            }
            fit_dispatch_design(py, negbin_log(theta_val), x_view, y_view, w_view, design, k)?
        }
        "tdist" | "scat" => {
            let nu_val = nu.unwrap_or(5.0);
            // scat (TDist + identity) standardizes the response internally for
            // raw-scale conditioning — see `TDist::fit_from_prep_canonical`. We
            // pass y and σ² in original units; σ² defaults to var(y) (the same
            // value the previous standardized-1.0 default mapped to, and mgcv's
            // data-scale dispersion init).
            let sigma2_init = sigma2.unwrap_or_else(|| scat_response_scale(y_view).powi(2));
            fit_dispatch_design(
                py,
                tdist_identity(nu_val, sigma2_init),
                x_view,
                y_view,
                w_view,
                design,
                k,
            )?
        }
        "tweedie" | "tw" => {
            let phi_val = tweedie_phi.unwrap_or(1.0);
            // tweedie_p semantics (mgcv_rust convention):
            //   None      → profile-p (mgcv `tw()`): p estimated jointly.
            //   Some(val) → fixed-p   (mgcv `Tweedie(p=val)`): p held = val.
            match tweedie_p {
                None => fit_dispatch_design(
                    py,
                    tweedie_log(1.5, phi_val),
                    x_view,
                    y_view,
                    w_view,
                    design,
                    k,
                )?,
                Some(p_val) => {
                    if !(1.0 < p_val && p_val < 2.0) {
                        return Err(PyValueError::new_err(format!(
                            "tweedie fixed p must be in (1, 2); got tweedie_p={p_val}"
                        )));
                    }
                    fit_dispatch_design(
                        py,
                        tweedie_log_fixed_p(p_val, phi_val),
                        x_view,
                        y_view,
                        w_view,
                        design,
                        k,
                    )?
                }
            }
        }
        "ocat" => {
            let n_cats = r.ok_or_else(|| {
                PyValueError::new_err(
                    "family='ocat' requires r=K (number of ordered categories, K >= 3)",
                )
            })?;
            if n_cats < 3 {
                return Err(PyValueError::new_err(format!(
                    "ocat requires r >= 3, got r={n_cats}"
                )));
            }
            let thresholds = Array1::<f64>::zeros(n_cats - 2);
            fit_dispatch_design(
                py,
                ocat_identity(thresholds, n_cats),
                x_view,
                y_view,
                w_view,
                design,
                k,
            )?
        }
        "elf" | "quantile" => {
            let tau_val = tau.unwrap_or(0.5);
            if !(0.0 < tau_val && tau_val < 1.0) {
                return Err(PyValueError::new_err(format!(
                    "elf/quantile tau must be in (0, 1); got tau={tau_val}"
                )));
            }
            let sigma_val = elf_sigma.unwrap_or(0.0);
            let lambda_val = elf_lambda.unwrap_or(0.0);
            fit_dispatch_design(
                py,
                elf_identity(tau_val, sigma_val, lambda_val),
                x_view,
                y_view,
                w_view,
                design,
                k,
            )?
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown family {other:?}; supported: gaussian, bernoulli, poisson, \
                 quasipoisson, quasibinomial, gamma, Gamma, inverse_gaussian, negbin, tdist, \
                 tweedie, ocat, elf"
            )))
        }
    };

    Ok(PyFittedGam { inner: fitted })
}

// =============================================================================
// Module entry point. The name must match the `[lib].name = "gamrs"` in
// Cargo.toml, prefixed with an underscore so Python imports it as
// `gamrs._gamrs_native` after the `python/gamrs/` package layer re-exports.
// =============================================================================

// =============================================================================
// fit_additive — multi-smooth `y ~ s(x_{c_0}) + s(x_{c_1}) + …` entry point.
//
// Python boundary: `terms` is a list of `(col, basis_name, k, pc)` tuples
// (k ignored for `bs="re"`; `pc` optional, univariate CR bases only). The string `basis_name` is converted to the
// typed `TermSpec` enum at this FFI boundary and never leaks into the Rust
// core — matches the project rule "strings ok at the Python FFI only".
// =============================================================================

/// mgcv's `s(x, pc=)` value, the optional 4th element of a univariate
/// term tuple. Absent (or `None`) leaves the smooth centred.
fn point_constraint(tup: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Option<f64>> {
    if tup.len() < 4 {
        return Ok(None);
    }
    tup.get_item(3)?.extract()
}

fn build_term_specs(terms: &Bound<'_, pyo3::types::PyList>) -> PyResult<Vec<TermSpec>> {
    let mut out: Vec<TermSpec> = Vec::with_capacity(terms.len());
    for (j, item) in terms.iter().enumerate() {
        let tup: &Bound<'_, pyo3::types::PyTuple> =
            item.cast::<pyo3::types::PyTuple>().map_err(|_| {
                PyValueError::new_err(format!(
                    "fit_additive: term {j} must be a tuple; got {item:?}"
                ))
            })?;
        if tup.len() < 2 {
            return Err(PyValueError::new_err(format!(
                "fit_additive: term {j} tuple must have at least 2 elements"
            )));
        }
        // Tensor terms use a sentinel basis name "te" with a 2-tuple of
        // columns and (optionally) a 2-tuple of k values:
        //   (cols_tuple, "te", k_tuple)
        // where cols_tuple = (col_a, col_b) and k_tuple = (k_a, k_b).
        let first = tup.get_item(0)?;
        let basis: String = tup.get_item(1)?.extract()?;
        let term = if basis == "tp" {
            // Tps tuple: (cols_tuple, "tp", k). `cols_tuple` may be of
            // arbitrary length ≥ 2 (the smooth is isotropic over its
            // input dims). `k` is a single int.
            let cols_tup: &Bound<'_, pyo3::types::PyTuple> =
                first.cast::<pyo3::types::PyTuple>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "fit_additive: tps term {j} first element must be a tuple of column indices"
                    ))
                })?;
            if cols_tup.len() < 2 {
                return Err(PyValueError::new_err(format!(
                    "fit_additive: tps term {j} cols tuple must have at least 2 elements"
                )));
            }
            let mut cols: Vec<usize> = Vec::with_capacity(cols_tup.len());
            for ci in 0..cols_tup.len() {
                cols.push(cols_tup.get_item(ci)?.extract()?);
            }
            let k: usize = if tup.len() >= 3 {
                tup.get_item(2)?.extract()?
            } else {
                10 * cols.len()
            };
            TermSpec::Tps { cols, k }
        } else if basis == "te_multi" || basis == "ti" {
            // N-margin te(...) / ti(...): (cols_tuple, "te_multi"|"ti", k_tuple)
            // where cols_tuple has length D >= 2 and k_tuple has the same
            // length (one marginal basis dim per margin).
            let cols_tup: &Bound<'_, pyo3::types::PyTuple> =
                first.cast::<pyo3::types::PyTuple>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "fit_additive: term {j} first element must be a tuple of column indices"
                    ))
                })?;
            if cols_tup.len() < 2 {
                return Err(PyValueError::new_err(format!(
                    "fit_additive: term {j} cols tuple must have at least 2 elements"
                )));
            }
            let mut cols: Vec<usize> = Vec::with_capacity(cols_tup.len());
            for ci in 0..cols_tup.len() {
                cols.push(cols_tup.get_item(ci)?.extract()?);
            }
            let k: Vec<usize> = if tup.len() >= 3 {
                let k_item = tup.get_item(2)?;
                let k_tup = k_item.cast::<pyo3::types::PyTuple>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "fit_additive: term {j} k must be a tuple of marginal basis dims"
                    ))
                })?;
                if k_tup.len() != cols.len() {
                    return Err(PyValueError::new_err(format!(
                        "fit_additive: term {j} k tuple length ({}) must match cols length ({})",
                        k_tup.len(),
                        cols.len()
                    )));
                }
                let mut kv = Vec::with_capacity(k_tup.len());
                for ki in 0..k_tup.len() {
                    kv.push(k_tup.get_item(ki)?.extract()?);
                }
                kv
            } else {
                vec![5; cols.len()]
            };
            let bs = vec![MarginKind::Cr; cols.len()];
            if basis == "ti" {
                TermSpec::Ti { cols, k, bs }
            } else {
                TermSpec::TeMulti { cols, k, bs }
            }
        } else if basis == "te" {
            let cols_tup: &Bound<'_, pyo3::types::PyTuple> =
                first.cast::<pyo3::types::PyTuple>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "fit_additive: tensor term {j} first element must be a (col_a, col_b) tuple"
                    ))
                })?;
            if cols_tup.len() != 2 {
                return Err(PyValueError::new_err(format!(
                    "fit_additive: tensor term {j} cols tuple must have exactly 2 elements"
                )));
            }
            let col_a: usize = cols_tup.get_item(0)?.extract()?;
            let col_b: usize = cols_tup.get_item(1)?.extract()?;
            let (k_a, k_b): (usize, usize) = if tup.len() >= 3 {
                let k_item = tup.get_item(2)?;
                let k_tup = k_item.cast::<pyo3::types::PyTuple>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "fit_additive: tensor term {j} k must be a (k_a, k_b) tuple"
                    ))
                })?;
                if k_tup.len() != 2 {
                    return Err(PyValueError::new_err(format!(
                        "fit_additive: tensor term {j} k tuple must have exactly 2 elements"
                    )));
                }
                (k_tup.get_item(0)?.extract()?, k_tup.get_item(1)?.extract()?)
            } else {
                (10, 10)
            };
            TermSpec::Tensor {
                col_a,
                col_b,
                k_a,
                k_b,
                bs_a: MarginKind::Cr,
                bs_b: MarginKind::Cr,
            }
        } else {
            // Univariate term — first element is a single column index.
            let col: usize = first.extract().map_err(|_| {
                PyValueError::new_err(format!(
                    "fit_additive: univariate term {j} first element must be an integer column index"
                ))
            })?;
            match basis.as_str() {
                "cr" => {
                    let k: usize = if tup.len() >= 3 {
                        tup.get_item(2)?.extract()?
                    } else {
                        10
                    };
                    TermSpec::Cr {
                        col,
                        k,
                        pc: point_constraint(tup)?,
                    }
                }
                "cr_stable" => {
                    let k: usize = if tup.len() >= 3 {
                        tup.get_item(2)?.extract()?
                    } else {
                        10
                    };
                    TermSpec::CrStable {
                        col,
                        k,
                        pc: point_constraint(tup)?,
                    }
                }
                "re" => TermSpec::Re { col },
                "parametric" | "linear" => TermSpec::Parametric { col },
                other => {
                    return Err(PyValueError::new_err(format!(
                        "fit_additive: term {j} basis must be 'cr', 'cr_stable', 're', \
                         'parametric' (alias 'linear'), 'te', 'te_multi', 'ti', or 'tp'; \
                         got {other:?}"
                    )))
                }
            }
        };
        out.push(term);
    }
    if out.is_empty() {
        return Err(PyValueError::new_err(
            "fit_additive: terms list must be non-empty",
        ));
    }
    Ok(out)
}

/// Run `gamrs::fit_with_design(..., Additive { terms })` for a typed family.
/// String dispatch on `family_name` happens here at the FFI boundary.
fn fit_additive_dispatch<'py, L, K, V>(
    py: Python<'py>,
    family: crate::family::Family<L, K, V>,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: Option<ArrayView1<f64>>,
    terms: Vec<TermSpec>,
) -> PyResult<FittedGam>
where
    // `+ Send` so the GIL-released closure satisfies pyo3's `Ungil` — see
    // `fit_dispatch_design`.
    L: FamilyFit<K, V> + Send,
    K: crate::traits::Link + Clone + Send,
    V: crate::traits::VarianceFn + Clone + Send,
{
    // GIL released for the basis build + solve (the heavy part); `x` / `y` /
    // `weights` view caller-owned arrays and `terms` is owned, so the closure
    // never touches a live Python object. See `fit_dispatch_design`.
    py.detach(move || {
        let prep = Additive { terms }.prepare(x)?;
        L::fit_from_prep(family, prep, x, y, weights)
    })
    .map_err(map_err)
}

/// Fit a multi-smooth additive gamrs GAM: `y ~ s(x_{c_0}) + s(x_{c_1}) + …`.
///
/// `terms` is a Python list of `(col, basis_name, k)` tuples — one tuple
/// per smoothing term. `basis_name` is one of `"cr"`, `"cr_stable"`, or
/// `"re"`. The `k` element is required for `"cr"`/`"cr_stable"` and ignored
/// for `"re"`. Strings live ONLY at this FFI boundary — the typed
/// `TermSpec` enum flows into the Rust core.
///
/// `family_name` accepts the same set as `fit(...)`. Shape-aware
/// families (tdist/scat, negbin, tweedie, ocat, elf) take the same
/// shape kwargs as `fit(...)` and run the multi-smooth outer Newton
/// over `[ρ_1, …, ρ_T, shape_params]`.
#[pyfunction]
#[pyo3(signature = (
    family_name,
    x,
    y,
    terms,
    weights=None,
    theta=None,
    nu=None,
    sigma2=None,
    tweedie_p=None,
    tweedie_phi=None,
    r=None,
    tau=None,
    elf_sigma=None,
    elf_lambda=None,
))]
fn fit_additive<'py>(
    py: Python<'py>,
    family_name: &str,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    terms: Bound<'py, pyo3::types::PyList>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    theta: Option<f64>,
    nu: Option<f64>,
    sigma2: Option<f64>,
    tweedie_p: Option<f64>,
    tweedie_phi: Option<f64>,
    r: Option<usize>,
    tau: Option<f64>,
    elf_sigma: Option<f64>,
    elf_lambda: Option<f64>,
) -> PyResult<PyFittedGam> {
    // Own the numpy inputs so the dispatch can release the GIL during the solve
    // (see `fit_dispatch_design`). `term_specs` is built under the GIL (it reads
    // the Python list) and is already owned.
    let x_owned: Array2<f64> = x.as_array().to_owned();
    let y_owned: Array1<f64> = y.as_array().to_owned();
    let w_owned: Option<Array1<f64>> = weights.map(|w| w.as_array().to_owned());
    let x_view: ArrayView2<f64> = x_owned.view();
    let y_view: ArrayView1<f64> = y_owned.view();
    let w_view: Option<ArrayView1<f64>> = w_owned.as_ref().map(|a| a.view());
    let term_specs = build_term_specs(&terms)?;

    let fitted: FittedGam = match family_name {
        "gaussian" => {
            fit_additive_dispatch(py, gaussian_identity(), x_view, y_view, w_view, term_specs)?
        }
        "bernoulli" | "binomial" => {
            fit_additive_dispatch(py, bernoulli_logit(), x_view, y_view, w_view, term_specs)?
        }
        "poisson" => fit_additive_dispatch(py, poisson_log(), x_view, y_view, w_view, term_specs)?,
        "quasipoisson" => {
            fit_additive_dispatch(py, quasipoisson_log(), x_view, y_view, w_view, term_specs)?
        }
        "quasibinomial" => fit_additive_dispatch(
            py,
            quasibinomial_logit(),
            x_view,
            y_view,
            w_view,
            term_specs,
        )?,
        "Gamma" => fit_additive_dispatch(py, gamma_inverse(), x_view, y_view, w_view, term_specs)?,
        "gamma" => fit_additive_dispatch(py, gamma_log(), x_view, y_view, w_view, term_specs)?,
        "inverse_gaussian" | "inverse.gaussian" => fit_additive_dispatch(
            py,
            inverse_gaussian_log(),
            x_view,
            y_view,
            w_view,
            term_specs,
        )?,
        "negbin" | "nb" => {
            let theta_val = theta.unwrap_or(2.0);
            if theta_val <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "negbin theta must be > 0; got theta={theta_val}"
                )));
            }
            fit_additive_dispatch(
                py,
                negbin_log(theta_val),
                x_view,
                y_view,
                w_view,
                term_specs,
            )?
        }
        "tdist" | "scat" => {
            let nu_val = nu.unwrap_or(5.0);
            // scat standardizes the response in-core — see the single-smooth
            // arm above. Pass y / σ² in original units.
            let sigma2_init = sigma2.unwrap_or_else(|| scat_response_scale(y_view).powi(2));
            fit_additive_dispatch(
                py,
                tdist_identity(nu_val, sigma2_init),
                x_view,
                y_view,
                w_view,
                term_specs,
            )?
        }
        "tweedie" | "tw" => {
            let phi_val = tweedie_phi.unwrap_or(1.0);
            // tweedie_p semantics (mgcv_rust convention):
            //   None      → profile-p (mgcv `tw()`): p estimated jointly.
            //   Some(val) → fixed-p   (mgcv `Tweedie(p=val)`): p held = val.
            match tweedie_p {
                None => fit_additive_dispatch(
                    py,
                    tweedie_log(1.5, phi_val),
                    x_view,
                    y_view,
                    w_view,
                    term_specs,
                )?,
                Some(p_val) => {
                    if !(1.0 < p_val && p_val < 2.0) {
                        return Err(PyValueError::new_err(format!(
                            "tweedie fixed p must be in (1, 2); got tweedie_p={p_val}"
                        )));
                    }
                    fit_additive_dispatch(
                        py,
                        tweedie_log_fixed_p(p_val, phi_val),
                        x_view,
                        y_view,
                        w_view,
                        term_specs,
                    )?
                }
            }
        }
        "ocat" => {
            let n_cats = r.ok_or_else(|| {
                PyValueError::new_err(
                    "family='ocat' requires r=K (number of ordered categories, K >= 3)",
                )
            })?;
            if n_cats < 3 {
                return Err(PyValueError::new_err(format!(
                    "ocat requires r >= 3, got r={n_cats}"
                )));
            }
            let thresholds = Array1::<f64>::zeros(n_cats - 2);
            fit_additive_dispatch(
                py,
                ocat_identity(thresholds, n_cats),
                x_view,
                y_view,
                w_view,
                term_specs,
            )?
        }
        "elf" | "quantile" => {
            let tau_val = tau.unwrap_or(0.5);
            if !(0.0 < tau_val && tau_val < 1.0) {
                return Err(PyValueError::new_err(format!(
                    "elf/quantile tau must be in (0, 1); got tau={tau_val}"
                )));
            }
            let sigma_val = elf_sigma.unwrap_or(0.0);
            let lambda_val = elf_lambda.unwrap_or(0.0);
            fit_additive_dispatch(
                py,
                elf_identity(tau_val, sigma_val, lambda_val),
                x_view,
                y_view,
                w_view,
                term_specs,
            )?
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown family {other:?}; supported: gaussian, bernoulli, poisson, \
                 quasipoisson, quasibinomial, gamma, Gamma, inverse_gaussian, negbin, tdist, \
                 tweedie, ocat, elf"
            )))
        }
    };
    Ok(PyFittedGam { inner: fitted })
}

/// Diagnostic: override outer-Newton tolerances for all subsequent fits
/// on this thread. Pass `grad_tol=None, reml_tol=None` to clear. Intended
/// for the tolerance-sweep script in `scripts/sweep_tolerances.py` —
/// production callers should leave this alone.
#[pyfunction]
#[pyo3(signature = (grad_tol=None, reml_tol=None))]
fn set_outer_tuning_override(grad_tol: Option<f64>, reml_tol: Option<f64>) {
    match (grad_tol, reml_tol) {
        (None, None) => crate::outer::clear_tuning_override(),
        _ => {
            let mut t = crate::outer::OuterTuning::mgcv_default();
            if let Some(g) = grad_tol {
                t.grad_tol = g;
            }
            if let Some(r) = reml_tol {
                t.reml_tol = r;
            }
            crate::outer::set_tuning_override(t);
        }
    }
}

/// Override the outer-Newton algorithm. Pass `"REML"` for the default
/// damped Newton (mgcv R `gam()` equivalent) or `"fREML"` for Wood &
/// Fasiolo (2017) Fellner-Schall multiplicative updates (mgcv R `bam()`
/// equivalent — cheaper per-iter, wins on GLM families at large n).
/// Pass `None` to clear the override and revert to Newton.
#[pyfunction]
#[pyo3(signature = (algorithm=None))]
fn set_outer_algorithm(algorithm: Option<&str>) -> PyResult<()> {
    match algorithm {
        None => {
            crate::outer::clear_algorithm_override();
            Ok(())
        }
        Some("REML") | Some("Newton") => {
            crate::outer::set_algorithm_override(crate::outer::OuterAlgorithm::Newton);
            Ok(())
        }
        Some("fREML") | Some("FellnerSchall") | Some("fellner-schall") => {
            crate::outer::set_algorithm_override(crate::outer::OuterAlgorithm::FellnerSchall);
            Ok(())
        }
        Some(other) => Err(PyValueError::new_err(format!(
            "algorithm must be 'REML' or 'fREML' (got {other:?})"
        ))),
    }
}

/// Fit a Gaussian location-scale (`gaulss`) GAMLSS model: two linear
/// predictors (μ via `mu_terms`, log σ via `sigma_terms`), fit jointly by
/// orthogonal alternating Fisher scoring. Returns the two block fits (`μ` and
/// `log σ`, each a `FittedGam`) plus `(n_iters, converged)`. The Python
/// `gamrs.fit_gaulss` wrapper composes them into quantiles.
#[pyfunction]
#[pyo3(signature = (x, y, mu_terms, sigma_terms, max_iter=50, tol=1e-6))]
fn fit_gaulss<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    mu_terms: Bound<'py, pyo3::types::PyList>,
    sigma_terms: Bound<'py, pyo3::types::PyList>,
    max_iter: usize,
    tol: f64,
) -> PyResult<(PyFittedGam, PyFittedGam, usize, bool)> {
    let x_owned: Array2<f64> = x.as_array().to_owned();
    let y_owned: Array1<f64> = y.as_array().to_owned();
    let mu_specs = build_term_specs(&mu_terms)?;
    let sigma_specs = build_term_specs(&sigma_terms)?;
    let opts = crate::fit::GaulssOpts { max_iter, tol };
    let fit = py
        .detach(move || {
            crate::fit::fit_gaulss(mu_specs, sigma_specs, x_owned.view(), y_owned.view(), opts)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        PyFittedGam { inner: fit.loc },
        PyFittedGam { inner: fit.scale },
        fit.n_iters,
        fit.converged,
    ))
}

// =============================================================================
// PyShashGamFit — Python-visible wrapper around the Rust `ShashGamFit`.
// =============================================================================

/// A fitted sinh-arcsinh (`shash`) GAMLSS model: the NATIVE joint fit over four
/// linear predictors (μ, log σ via the `logeb` link, skewness ε, log-kurtosis
/// φ), fitted by the dense block-Newton inner solve under outer REML. Exposes
/// `predict_eta` / `predict_params` / `predict_quantile` plus REML diagnostics.
///
/// Distinct from the two-stage per-residual `fit_quantile_lss(shape="shash")`
/// helper (`gamrs._shash`): this is the genuine joint GAMLSS MLE.
#[pyclass(name = "ShashGamFit", module = "gamrs._gamrs_native")]
pub struct PyShashGamFit {
    inner: crate::fit::ShashGamFit,
}

#[pymethods]
impl PyShashGamFit {
    /// The four fitted linear predictors on new `x`: an `(n, 4)` float64 ndarray
    /// with columns `(η_μ, η_τ, η_ε, η_φ)`.
    fn predict_eta<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let eta = self.inner.predict_eta(x.as_array()).map_err(map_err)?;
        Ok(eta.into_pyarray(py))
    }

    /// The four fitted parameters on new `x` as a tuple of four 1-D float64
    /// ndarrays `(μ, σ, ε, δ)` (σ = exp(τ) via the `logeb` link, δ = exp(φ)).
    fn predict_params<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let (mu, sigma, eps, del) = self.inner.predict_params(x.as_array()).map_err(map_err)?;
        Ok((
            mu.into_pyarray(py),
            sigma.into_pyarray(py),
            eps.into_pyarray(py),
            del.into_pyarray(py),
        ))
    }

    /// The fitted `p`-quantile per observation on new `x` (1-D float64 ndarray).
    /// `p` must lie in the open interval `(0, 1)`; otherwise raises `ValueError`.
    fn predict_quantile<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        p: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let q = self
            .inner
            .predict_quantile(x.as_array(), p)
            .map_err(map_err)?;
        Ok(q.into_pyarray(py))
    }

    /// Selected log-smoothing-parameters ρ̂ (one per penalised block, in block
    /// order μ, τ, ε, φ). 1-D float64 ndarray; empty when no block is penalised.
    #[getter]
    fn rho<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.rho.clone().into_pyarray(py)
    }

    /// Per-block coefficient counts `[p_μ, p_τ, p_ε, p_φ]` as a length-4 tuple.
    #[getter]
    fn block_p(&self) -> (usize, usize, usize, usize) {
        let bp = self.inner.block_p;
        (bp[0], bp[1], bp[2], bp[3])
    }

    /// Total effective degrees of freedom `total_p − tr(Hp⁻¹ S_ρ)`.
    #[getter]
    fn edf(&self) -> f64 {
        self.inner.edf
    }

    /// The LAML / REML criterion `V(ρ̂)` at the optimum.
    #[getter]
    fn laml(&self) -> f64 {
        self.inner.laml
    }

    /// The `logeb` bound `b` used for the τ (log-scale) link.
    #[getter]
    fn b(&self) -> f64 {
        self.inner.b
    }

    /// Whether the outer REML (smoothing-parameter) ascent converged.
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    /// Whether the inner penalised Newton converged at ρ̂.
    #[getter]
    fn inner_converged(&self) -> bool {
        self.inner.inner_converged
    }
}

/// Fit a sinh-arcsinh (`shash`) GAMLSS model end-to-end: four predictor term
/// lists (μ, τ, ε, φ), a covariate matrix `x` (`n × d`) and response `y`. Each
/// term list uses the SAME tuple form as `fit_additive` (parsed by
/// `build_term_specs`); an EMPTY list is an intercept-only predictor (handled
/// by `crate::fit::fit_shash`). `b` is the τ-link `logeb` bound (default 1e-2).
/// Returns a `ShashGamFit`.
#[pyfunction]
#[pyo3(signature = (mu_terms, tau_terms, eps_terms, phi_terms, x, y, b=None))]
fn fit_shash<'py>(
    py: Python<'py>,
    mu_terms: Bound<'py, pyo3::types::PyList>,
    tau_terms: Bound<'py, pyo3::types::PyList>,
    eps_terms: Bound<'py, pyo3::types::PyList>,
    phi_terms: Bound<'py, pyo3::types::PyList>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    b: Option<f64>,
) -> PyResult<PyShashGamFit> {
    // Own the numpy inputs so the solve can release the GIL (see
    // `fit_additive`). The four term lists are parsed under the GIL first (they
    // read live Python objects); an empty list → an intercept-only predictor, so
    // — unlike `build_term_specs`, which rejects empty lists — we map empty to an
    // empty `Vec<TermSpec>` here.
    let x_owned: Array2<f64> = x.as_array().to_owned();
    let y_owned: Array1<f64> = y.as_array().to_owned();
    let parse = |terms: &Bound<'py, pyo3::types::PyList>| -> PyResult<Vec<TermSpec>> {
        if terms.is_empty() {
            Ok(Vec::new())
        } else {
            build_term_specs(terms)
        }
    };
    let mu_specs = parse(&mu_terms)?;
    let tau_specs = parse(&tau_terms)?;
    let eps_specs = parse(&eps_terms)?;
    let phi_specs = parse(&phi_terms)?;

    let opts = crate::fit::ShashGamOpts {
        b: b.unwrap_or(1e-2),
        ..crate::fit::ShashGamOpts::default()
    };
    let fit = py
        .detach(move || {
            crate::fit::fit_shash(
                mu_specs,
                tau_specs,
                eps_specs,
                phi_specs,
                x_owned.view(),
                y_owned.view(),
                opts,
            )
        })
        .map_err(map_err)?;
    Ok(PyShashGamFit { inner: fit })
}

#[pymodule]
fn _gamrs_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFittedGam>()?;
    m.add_class::<PyShashGamFit>()?;
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_function(wrap_pyfunction!(fit_additive, m)?)?;
    m.add_function(wrap_pyfunction!(fit_gaulss, m)?)?;
    m.add_function(wrap_pyfunction!(fit_shash, m)?)?;
    m.add_function(wrap_pyfunction!(set_outer_tuning_override, m)?)?;
    m.add_function(wrap_pyfunction!(set_outer_algorithm, m)?)?;
    Ok(())
}
