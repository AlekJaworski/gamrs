//! Sinh-arcsinh (`shash`) GAMLSS — the **non-orthogonal** end-to-end driver.
//!
//! This is the user-facing counterpart to [`fit_gaulss`](crate::fit::gaulss):
//! given four lists of additive smooth term specs (one per linear predictor
//! `μ, τ, ε, φ`), a covariate matrix `x`, and a response `y`, it builds its
//! OWN per-predictor designs + penalties, runs the joint penalised inner Newton
//! under outer REML smoothing-parameter selection, and predicts params and
//! quantiles on new x.
//!
//! ## Why this driver is structured differently from `gaulss`
//!
//! `gaulss` is the *orthogonal* GAMLSS case: the Gaussian location-scale Fisher
//! information is block-diagonal in `(μ, log σ)`, so the joint fit decomposes
//! into an alternation of two independent single-predictor Gaussian REML fits,
//! each of which reuses [`fit_with_design`](crate::fit::fit_with_design)
//! verbatim. `shash` has NO such orthogonality — μ, τ, ε, φ are coupled through
//! the sinh-arcsinh density's cross-derivatives, so a single dense block-Newton
//! over all four predictors is mandatory. We therefore do NOT reuse
//! `fit_with_design`; instead we feed raw per-block designs + penalties into the
//! frozen phase 1-5 machinery:
//!   - [`shash_init`](crate::gamlss::shash_init::shash_init) warm-starts β;
//!   - [`ShashProblem`] / [`fit_reml`] runs the dense joint inner Newton
//!     ([`crate::gamlss::shash_inner`]) under the outer LAML/REML ascent
//!     ([`crate::gamlss::shash_reml`]).
//!
//! ## Design construction (gamrs builds its own basis)
//!
//! For each of the four predictors:
//!   - a NON-empty term list `t` is prepared via
//!     [`Additive { terms: t }`](crate::design::Additive)`.prepare(x)`, whose
//!     `x_design` is `[intercept | constrained-smooth-cols…]` — exactly mgcv's
//!     per-linear-predictor block layout (a shared intercept plus the
//!     sum-to-zero-centred CR columns). For v1 we REQUIRE at most one smooth per
//!     predictor (`s_list.len() <= 1`): a single penalty maps cleanly onto the
//!     `ShashProblem`'s one-penalty-per-block contract. Multi-smooth per
//!     predictor is a documented follow-up.
//!   - an EMPTY term list is an intercept-only block: `X_b = ones(n, 1)`, no
//!     penalty. (`Additive` rejects an empty term list, so we build the ones
//!     column directly — this matches mgcv's `~ 1` intercept-only predictor.)
//!
//! The four `X_b` and their optional penalties become a [`ShashProblem`]; we
//! initialise β with [`shash_init`](crate::gamlss::shash_init::shash_init) at
//! the [`ShashBlocks`]-style block offsets, set `ρ₀ = 0`, and call `fit_reml`.
//!
//! ## Prediction
//!
//! Each block stores its predict-time rebuilder. Smooth blocks keep the
//! [`Predictor`] returned by `Additive.prepare` (it rebuilds the design on new x
//! from the training knots + centring); intercept-only blocks store `None` and
//! predict the constant `β₀`. [`ShashGamFit::predict_eta`] rebuilds the four
//! designs and dots them with the per-block β̂. [`ShashGamFit::predict_params`]
//! maps η → `(μ, σ, ε, δ)` via [`ShashDensity::linkinv`] (`σ = exp(τ)`,
//! `δ = exp(φ)`). [`ShashGamFit::predict_quantile`] applies the shash quantile
//! function (mirrors R qgam's `.shashQf` / `python/gamrs/_shash.py`):
//! ```text
//!   q(p) = μ + (δ·σ)·sinh( (1/δ)·asinh(Φ⁻¹(p)) + ε/δ )
//! ```
//!
//! Validated end-to-end against an mgcv `shash` gam (gamrs constructs its own
//! designs, NOT fed mgcv's) — see `fit_shash_matches_mgcv_two_smooth`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::design::identifiability::{aliased_columns, with_zero_coefficients, without_columns};
use crate::design::{Additive, DesignStrategy, Predictor, TermSpec};
use crate::error::{GamrsError, Result};
use crate::gamlss::shash::ShashDensity;
use crate::gamlss::shash_init::shash_init;
use crate::gamlss::shash_reml::{
    fit_reml, ShashPenalty, ShashProblem, ShashRemlEval, ShashRemlOpts,
};

/// One fitted shash linear-predictor block: its coefficient slice plus the
/// predict-time design rebuilder. `predictor == None` marks an intercept-only
/// block (predict the constant `beta[0]`); otherwise the [`Predictor`] rebuilds
/// the additive design on new x from the training knots/centring.
struct ShashBlockFit {
    /// Coefficient slice for this block (length = block width `p_b`).
    beta: Array1<f64>,
    /// Predict-time design rebuilder; `None` for an intercept-only block.
    predictor: Option<Predictor>,
}

impl ShashBlockFit {
    /// Rebuild this block's linear predictor `η_b = X_b(x_new)·β̂_b`.
    fn predict(&self, x_new: ArrayView2<f64>) -> Result<Array1<f64>> {
        match &self.predictor {
            Some(p) => Ok(p.design(x_new)?.dot(&self.beta)),
            // Intercept-only: every row gets the single constant coefficient.
            None => Ok(Array1::from_elem(x_new.nrows(), self.beta[0])),
        }
    }
}

/// Options for the end-to-end [`fit_shash`] driver.
#[derive(Clone, Copy, Debug)]
pub struct ShashGamOpts {
    /// `logeb` bound `b` for the τ link (`σ = exp(log(exp(η₂) + b)) ≥ b`). mgcv
    /// default 1e-2.
    pub b: f64,
    /// Outer REML (smoothing-parameter) ascent controls.
    pub reml: ShashRemlOpts,
}

impl Default for ShashGamOpts {
    fn default() -> Self {
        Self {
            b: 1e-2,
            reml: ShashRemlOpts::default(),
        }
    }
}

/// A fitted shash GAMLSS model: the flat penalised MLE β̂, the per-block fits
/// (coefficients + predict-time rebuilders), the selected log-smoothing-params
/// ρ̂, and the REML diagnostics (total EDF, LAML, convergence).
pub struct ShashGamFit {
    /// Flat penalised MLE β̂ (block order μ, τ, ε, φ at the [`ShashBlocks`]
    /// offsets).
    pub beta: Array1<f64>,
    /// Per-block coefficient counts `[p_μ, p_τ, p_ε, p_φ]`.
    pub block_p: [usize; 4],
    /// Selected log-smoothing-parameters ρ̂ (one per penalised block, in block
    /// order). Empty when no block is penalised.
    pub rho: Array1<f64>,
    /// Total effective degrees of freedom `total_p − tr(Hp⁻¹ S_ρ)`.
    pub edf: f64,
    /// The LAML / REML criterion `V(ρ̂)` at the optimum (up to the omitted
    /// ρ-independent constant — see [`crate::gamlss::shash_reml`]).
    pub laml: f64,
    /// `logeb` bound `b` used for the τ link.
    pub b: f64,
    /// Whether the outer REML ascent converged.
    pub converged: bool,
    /// Whether the inner penalised Newton converged at ρ̂.
    pub inner_converged: bool,
    /// Per-block fits (coefficient slices + predict-time rebuilders).
    blocks: [ShashBlockFit; 4],
}

impl ShashGamFit {
    /// Total coefficient count `Σ p_b`.
    pub fn total_p(&self) -> usize {
        self.block_p.iter().sum()
    }

    /// The four linear predictors on new x: `n_new × 4` with columns
    /// `(η_μ, η_τ, η_ε, η_φ)`. Each column is its block's rebuilt design dotted
    /// with the block β̂.
    pub fn predict_eta(&self, x_new: ArrayView2<f64>) -> Result<Array2<f64>> {
        let n = x_new.nrows();
        let mut eta = Array2::<f64>::zeros((n, 4));
        for b in 0..4 {
            let eta_b = self.blocks[b].predict(x_new)?;
            eta.column_mut(b).assign(&eta_b);
        }
        Ok(eta)
    }

    /// The four fitted params on new x as `(μ, σ, ε, δ)` per observation.
    /// `linkinv` maps η → `(μ, τ, ε, φ)` (τ via the `logeb` link), then
    /// `σ = exp(τ)` and `δ = exp(φ)`.
    pub fn predict_params(
        &self,
        x_new: ArrayView2<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>)> {
        let eta = self.predict_eta(x_new)?;
        let n = eta.nrows();
        let mut mu = Array1::<f64>::zeros(n);
        let mut sigma = Array1::<f64>::zeros(n);
        let mut eps = Array1::<f64>::zeros(n);
        let mut del = Array1::<f64>::zeros(n);
        for i in 0..n {
            let [m, tau, e, phi] =
                ShashDensity::linkinv([eta[[i, 0]], eta[[i, 1]], eta[[i, 2]], eta[[i, 3]]], self.b);
            mu[i] = m;
            sigma[i] = tau.exp();
            eps[i] = e;
            del[i] = phi.exp();
        }
        Ok((mu, sigma, eps, del))
    }

    /// The fitted `p`-quantile per observation on new x.
    ///
    /// Applies the sinh-arcsinh quantile function (mirrors R qgam's `.shashQf`
    /// and `python/gamrs/_shash.py`):
    /// ```text
    ///   q(p) = μ + (δ·σ)·sinh( (1/δ)·asinh(Φ⁻¹(p)) + ε/δ )
    /// ```
    /// with `Φ⁻¹` the standard-normal inverse CDF ([`norm_ppf`]). `0 < p < 1`.
    pub fn predict_quantile(&self, x_new: ArrayView2<f64>, p: f64) -> Result<Array1<f64>> {
        if !(p > 0.0 && p < 1.0) {
            return Err(GamrsError::InvalidParameter(format!(
                "shash quantile p must be in (0, 1); got {p}"
            )));
        }
        let (mu, sigma, eps, del) = self.predict_params(x_new)?;
        let zp = norm_ppf(p);
        let asinh_zp = zp.asinh();
        let n = mu.len();
        let mut q = Array1::<f64>::zeros(n);
        for i in 0..n {
            q[i] = mu[i] + del[i] * sigma[i] * (asinh_zp / del[i] + eps[i] / del[i]).sinh();
        }
        Ok(q)
    }
}

/// Build one predictor's design `X_b` and its optional penalty from a term list.
///
/// - NON-empty terms → `Additive.prepare(x)`; require `s_list.len() <= 1` (one
///   smooth per predictor in v1). Returns `(x_design, penalty, Some(predictor))`
///   where `penalty` is `Some` iff the predictor carries exactly one smooth.
/// - EMPTY terms → intercept-only block `ones(n, 1)`, no penalty, no predictor.
///
/// Columns the block cannot identify are removed (see
/// [`crate::design::identifiability`]) and returned so the fit can put them
/// back as zeros.
fn build_block(
    terms: &[TermSpec],
    x: ArrayView2<f64>,
    block_name: &str,
) -> Result<(Array2<f64>, Option<ShashPenalty>, Option<Predictor>, Vec<usize>)> {
    if terms.is_empty() {
        // Intercept-only predictor — a column of ones (mgcv's `~ 1` block).
        let n = x.nrows();
        return Ok((Array2::<f64>::ones((n, 1)), None, None, Vec::new()));
    }
    let prepared = Additive {
        terms: terms.to_vec(),
    }
    .prepare(x)?;
    let dropped = aliased_columns(&prepared.x_design, &prepared.s_list, None)?;
    let prepared = if dropped.is_empty() {
        prepared
    } else {
        without_columns(prepared, &dropped)?
    };
    if prepared.s_list.len() > 1 {
        return Err(GamrsError::InvalidParameter(format!(
            "fit_shash: the {block_name} predictor has {} smooths; v1 supports at most one \
             smooth per predictor (multi-smooth-per-predictor is a documented follow-up)",
            prepared.s_list.len()
        )));
    }
    let penalty = if prepared.s_list.len() == 1 {
        Some(ShashPenalty {
            s0: prepared.s_list[0].clone(),
            rank: prepared.rank_s_list[0],
        })
    } else {
        // A purely parametric predictor (e.g. a single `Parametric` term):
        // unpenalised, but still a real design (intercept + raw column).
        None
    };
    Ok((prepared.x_design, penalty, Some(prepared.predictor), dropped))
}

/// Fit a sinh-arcsinh (`shash`) GAMLSS end-to-end: build per-predictor designs
/// from the four term lists, run the joint penalised inner Newton under outer
/// REML, and return a [`ShashGamFit`] that predicts params and quantiles.
///
/// `mu_terms` / `tau_terms` / `eps_terms` / `phi_terms` are the additive smooth
/// specs for the location, log-scale, skewness and log-kurtosis predictors. An
/// empty list is an intercept-only predictor. For v1 each predictor supports at
/// most ONE smooth term (see [`build_block`]). `x` is `n × n_dims`; `y` is the
/// length-`n` response.
pub fn fit_shash(
    mu_terms: Vec<TermSpec>,
    tau_terms: Vec<TermSpec>,
    eps_terms: Vec<TermSpec>,
    phi_terms: Vec<TermSpec>,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    opts: ShashGamOpts,
) -> Result<ShashGamFit> {
    let n = y.len();
    if n == 0 {
        return Err(GamrsError::InvalidParameter(
            "fit_shash: empty response".into(),
        ));
    }
    if x.nrows() != n {
        return Err(GamrsError::InvalidParameter(format!(
            "fit_shash: x has {} rows but y has {} entries",
            x.nrows(),
            n
        )));
    }

    // ── Build the four per-predictor designs + optional penalties. ──
    let (x_mu, pen_mu, pred_mu, drop_mu) = build_block(&mu_terms, x, "mu")?;
    let (x_tau, pen_tau, pred_tau, drop_tau) = build_block(&tau_terms, x, "tau")?;
    let (x_eps, pen_eps, pred_eps, drop_eps) = build_block(&eps_terms, x, "eps")?;
    let (x_phi, pen_phi, pred_phi, drop_phi) = build_block(&phi_terms, x, "phi")?;
    let dropped = [drop_mu, drop_tau, drop_eps, drop_phi];

    let block_p = [x_mu.ncols(), x_tau.ncols(), x_eps.ncols(), x_phi.ncols()];

    // ── Warm-start β via the phase-3 initialiser (ridge 0.0), packed at the
    //    ShashBlocks block offsets (μ at 0, then τ, ε, φ). ──
    let init = shash_init(
        x_mu.view(),
        x_tau.view(),
        x_eps.view(),
        x_phi.view(),
        y,
        0.0,
    )?;
    let total_p: usize = block_p.iter().sum();
    let off = [
        0usize,
        block_p[0],
        block_p[0] + block_p[1],
        block_p[0] + block_p[1] + block_p[2],
    ];
    let mut beta0 = Array1::<f64>::zeros(total_p);
    beta0
        .slice_mut(ndarray::s![off[0]..off[0] + block_p[0]])
        .assign(&init.beta_mu);
    beta0
        .slice_mut(ndarray::s![off[1]..off[1] + block_p[1]])
        .assign(&init.beta_tau);
    beta0
        .slice_mut(ndarray::s![off[2]..off[2] + block_p[2]])
        .assign(&init.beta_eps);
    beta0
        .slice_mut(ndarray::s![off[3]..off[3] + block_p[3]])
        .assign(&init.beta_phi);

    // ── Assemble the REML problem and run the outer ascent from ρ₀ = 0. ──
    let problem = ShashProblem {
        x: [x_mu, x_tau, x_eps, x_phi],
        y: y.to_owned(),
        penalties: [pen_mu, pen_tau, pen_eps, pen_phi],
        b: opts.b,
    };
    let density = ShashDensity::default();
    let rho0 = vec![0.0_f64; problem.n_sp()];
    let fit = fit_reml(&density, &problem, &rho0, beta0.view(), opts.reml)?;

    // ── Slice β̂ per block; pair each block with its predict-time rebuilder. ──
    let ShashRemlEval {
        beta, edf, laml, ..
    } = fit.eval;
    let predictors = [pred_mu, pred_tau, pred_eps, pred_phi];
    let make_block = |b: usize, predictor: Option<Predictor>| ShashBlockFit {
        beta: with_zero_coefficients(
            beta.slice(ndarray::s![off[b]..off[b] + block_p[b]]),
            &dropped[b],
        ),
        predictor,
    };
    let [p0, p1, p2, p3] = predictors;
    let blocks = [
        make_block(0, p0),
        make_block(1, p1),
        make_block(2, p2),
        make_block(3, p3),
    ];

    Ok(ShashGamFit {
        beta: ndarray::concatenate(
            ndarray::Axis(0),
            &[blocks[0].beta.view(), blocks[1].beta.view(), blocks[2].beta.view(), blocks[3].beta.view()],
        )
        .expect("blocks are 1-D"),
        block_p: std::array::from_fn(|b| blocks[b].beta.len()),
        rho: fit.rho,
        edf,
        laml,
        b: opts.b,
        converged: fit.converged,
        inner_converged: fit.eval.inner_converged,
        blocks,
    })
}

/// Standard-normal inverse CDF via Acklam's rational approximation (abs error
/// < 1.15e-9 across `(0, 1)`) — the same dependency-free `Φ⁻¹` used by
/// [`crate::fit::gaulss`], replicated here so the shash quantile path has no
/// cross-module private dependency.
// Verbatim Acklam coefficients, mirrored from `gaulss` (same lints there).
#[allow(clippy::unreadable_literal, clippy::excessive_precision)]
fn norm_ppf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    // --- the end-to-end mgcv confrontation (gamrs builds its OWN designs) ----
    //
    // Loads scripts/r/gen_shash_gam_fixture.R's output: the RAW covariates
    // x0/x1, the response y, mgcv's fitted η (n×4), total EDF, and mgcv-derived
    // quantiles. gamrs constructs its own CR designs from x0/x1 (Cr{k=10} per
    // predictor) and must recover mgcv's η, EDF and quantiles. This is the real
    // phase-6a parity proof — it exercises gamrs's design construction, not just
    // the REML criterion (which the phase-5 fixture already validates on mgcv's
    // own designs).

    #[derive(serde::Deserialize)]
    struct GamFixture {
        n: usize,
        b: f64,
        edf_total: f64,
        x0: Vec<f64>,
        x1: Vec<f64>,
        y: Vec<f64>,
        eta: Vec<f64>,
        q10: Vec<f64>,
        q50: Vec<f64>,
        q90: Vec<f64>,
    }

    fn load_gam_fixture() -> GamFixture {
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests/fixtures/shash_gam_mgcv.json");
        let txt = std::fs::read_to_string(&p)
            .unwrap_or_else(|e| panic!("missing fixture {}: {e}", p.display()));
        serde_json::from_str(&txt).expect("malformed shash_gam fixture")
    }

    #[test]
    fn norm_ppf_matches_known_quantiles() {
        // Sanity: the replicated Φ⁻¹ agrees with standard values.
        assert!((norm_ppf(0.5)).abs() < 1e-9);
        assert!((norm_ppf(0.975) - 1.959963984540054).abs() < 1e-6);
        assert!((norm_ppf(0.1) + 1.2815515594465).abs() < 1e-6);
    }

    #[test]
    fn predict_quantile_rejects_out_of_range_p() {
        // Only the p-validation path matters, but shash (like mgcv's) needs an
        // identifiable scale/shape: near-deterministic data can leave the
        // penalised Hessian non-SPD. So use a smooth signal + GENUINE Gaussian
        // noise (golden-ratio Box-Muller, deterministic) so the fit converges.
        let n = 200usize;
        let frac = |v: f64| v - v.floor();
        let pnormal = |u1: f64, u2: f64| {
            (-2.0 * u1.max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let mut x = Array2::<f64>::zeros((n, 1));
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = (i as f64 + 0.5) / n as f64;
            x[[i, 0]] = t;
            let u1 = frac((i as f64 + 0.5) * 0.7548776662466927 + 0.31);
            let u2 = frac((i as f64 + 0.5) * 0.5698402909980532 + 0.59);
            y[i] = (2.0 * std::f64::consts::PI * t).sin() + 0.4 * pnormal(u1, u2);
        }
        let fit = fit_shash(
            vec![TermSpec::Cr {
                col: 0,
                k: 6,
                pc: None,
            }],
            vec![],
            vec![],
            vec![],
            x.view(),
            y.view(),
            ShashGamOpts::default(),
        )
        .expect("fit_shash on tiny data");
        assert!(fit.predict_quantile(x.view(), 0.0).is_err());
        assert!(fit.predict_quantile(x.view(), 1.0).is_err());
        assert!(fit.predict_quantile(x.view(), 0.5).is_ok());
    }

    // --- robustness across data SHAPES ------------------------------------
    //
    // Regression guard for the v0.12.0 fragility: `fit_shash` errored
    // ("Hp not SPD") on near-GAUSSIAN data (ε≈φ≈0) because the outer REML probed
    // points with a non-SPD observed Hessian and bailed instead of perturbing.
    // shash *can* represent a Gaussian (ε=0, δ=1), so a fit on Gaussian — and
    // skewed, and heavy-tailed — data must CONVERGE and recover the location.
    // (The earlier mgcv parity test only used shash-distributed data, so it
    // never exercised the Gaussian case; this battery does.)

    fn frac(v: f64) -> f64 {
        v - v.floor()
    }
    fn pnormal(u1: f64, u2: f64) -> f64 {
        (-2.0 * u1.max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Fit `fit_shash` with μ = s(x0), constant scale/shape, on `y` generated by
    /// `noise(z1, z2)` (two decorrelated N(0,1) draws) around μ(x)=sin(2πx0);
    /// assert convergence + that μ̂(x) tracks the truth (corr > 0.9) + finite,
    /// monotone quantiles.
    fn assert_shape_ok(label: &str, noise: impl Fn(f64, f64) -> f64) {
        let n = 400usize;
        let mut x = Array2::<f64>::zeros((n, 1));
        let mut y = Array1::<f64>::zeros(n);
        let mut mu_true = Array1::<f64>::zeros(n);
        for i in 0..n {
            let t = frac((i as f64 + 0.5) * 0.6180339887498949);
            x[[i, 0]] = t;
            let z1 = pnormal(
                frac((i as f64 + 0.5) * 0.7548776662466927 + 0.31),
                frac((i as f64 + 0.5) * 0.5698402909980532 + 0.59),
            );
            let z2 = pnormal(
                frac((i as f64 + 0.5) * 0.4359911239867 + 0.13),
                frac((i as f64 + 0.5) * 0.9171523356672 + 0.77),
            );
            mu_true[i] = (2.0 * std::f64::consts::PI * t).sin();
            y[i] = mu_true[i] + 0.2 * noise(z1, z2);
        }
        let fit = fit_shash(
            vec![TermSpec::Cr {
                col: 0,
                k: 10,
                pc: None,
            }],
            vec![],
            vec![],
            vec![],
            x.view(),
            y.view(),
            ShashGamOpts::default(),
        )
        .unwrap_or_else(|e| panic!("{label}: fit_shash errored: {e}"));
        assert!(fit.converged, "{label}: did not converge");
        // μ̂(x) tracks the truth.
        let (mu, _s, _e, _d) = fit.predict_params(x.view()).expect("predict_params");
        let mm = mu.mean().unwrap();
        let tm = mu_true.mean().unwrap();
        let (mut cov, mut vm, mut vt) = (0.0, 0.0, 0.0);
        for i in 0..n {
            cov += (mu[i] - mm) * (mu_true[i] - tm);
            vm += (mu[i] - mm).powi(2);
            vt += (mu_true[i] - tm).powi(2);
        }
        let corr = cov / (vm.sqrt() * vt.sqrt());
        assert!(
            corr > 0.9,
            "{label}: μ̂ vs truth corr {corr:.3} too low (edf {:.2})",
            fit.edf
        );
        // Quantiles finite + monotone in p.
        let q10 = fit.predict_quantile(x.view(), 0.1).expect("q10");
        let q50 = fit.predict_quantile(x.view(), 0.5).expect("q50");
        let q90 = fit.predict_quantile(x.view(), 0.9).expect("q90");
        for i in 0..n {
            assert!(
                q10[i].is_finite() && q10[i] <= q50[i] && q50[i] <= q90[i],
                "{label}: quantiles not finite/monotone at i={i}"
            );
        }
    }

    #[test]
    fn fit_shash_converges_on_gaussian_data() {
        // The exact v0.12.0 failure case: symmetric, mesokurtic (ε≈φ≈0).
        assert_shape_ok("gaussian", |z1, _z2| z1);
    }

    #[test]
    fn fit_shash_converges_on_skewed_data() {
        // Right-skewed (exponentiated normal, centred).
        assert_shape_ok("right-skew", |z1, _z2| (0.6 * z1).exp() - (0.18f64).exp());
    }

    #[test]
    fn fit_shash_converges_on_heavy_tailed_data() {
        // Heavy-tailed (scale-mixture of normals → leptokurtic, symmetric).
        assert_shape_ok("heavy-tail", |z1, z2| z1 * (1.0 + 0.8 * z2.abs()));
    }

    #[test]
    fn fit_shash_matches_mgcv_two_smooth() {
        let fx = load_gam_fixture();
        let n = fx.n;
        // x = [x0, x1] as an n×2 design.
        let mut x = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            x[[i, 0]] = fx.x0[i];
            x[[i, 1]] = fx.x1[i];
        }
        let y = Array1::from(fx.y.clone());

        // gamrs builds its OWN CR designs: s(x0,k=10) for μ, s(x1,k=10) for τ,
        // intercept-only ε and φ — matching the mgcv formula in the fixture.
        let fit = fit_shash(
            vec![TermSpec::Cr {
                col: 0,
                k: 10,
                pc: None,
            }],
            vec![TermSpec::Cr {
                col: 1,
                k: 10,
                pc: None,
            }],
            vec![],
            vec![],
            x.view(),
            y.view(),
            ShashGamOpts::default(),
        )
        .expect("fit_shash");
        assert!(fit.converged, "outer REML did not converge");
        assert!(fit.inner_converged, "inner solve did not converge at ρ̂");
        // NB: gamrs's ρ̂ is NOT comparable to mgcv's log(sp) in absolute terms —
        // the smoothing parameter scales a basis-specific penalty S0, and gamrs
        // fits in its OWN CR basis (different centring/penalty scale than mgcv's
        // reparametrised one). What IS comparable is the basis-invariant fitted
        // function (η) and EDF, confronted below.
        eprintln!("fit_shash: gamrs ρ̂ = {:?}", fit.rho.as_slice().unwrap());
        // Block widths: μ,τ = 10 (CR k=10), ε,φ = 1 (intercept-only).
        assert_eq!(fit.block_p, [10, 10, 1, 1]);
        assert_eq!(fit.b, fx.b);

        // (1) Fitted η per block vs mgcv. gamrs rebuilds its own design on the
        //     SAME x and dots with β̂ — directly comparable to mgcv's η.
        let eta_mgcv = Array2::from_shape_vec((n, 4), fx.eta.clone()).expect("eta shape");
        let eta = fit.predict_eta(x.view()).expect("predict_eta");
        let mut max_eta = 0.0_f64;
        let mut per_block = [0.0_f64; 4];
        for i in 0..n {
            for b in 0..4 {
                let d = (eta[[i, b]] - eta_mgcv[[i, b]]).abs();
                max_eta = max_eta.max(d);
                per_block[b] = per_block[b].max(d);
            }
        }
        eprintln!("fit_shash: max |η_gamrs − η_mgcv| = {max_eta:.3e}  per-block {per_block:?}");
        // gamrs `Cr` vs the fixture's mgcv `bs="cr"` fit (SAME basis) — agrees to
        // ~1e-6; 1e-4 leaves margin for cross-platform FP/BLAS + FD-Newton. (Note:
        // mgcv's DEFAULT `s(x,k)` is a thin-plate spline — a different basis — so
        // the fixture MUST request bs="cr" to match gamrs's Cr term.)
        assert!(
            max_eta < 1e-4,
            "fitted η: max|gamrs − mgcv| = {max_eta} (>= 1e-4 — basis-type mismatch \
             (cr vs thin-plate?) or a real regression, NOT a tolerance issue)"
        );

        // (2) Total EDF vs mgcv.
        let edf_diff = (fit.edf - fx.edf_total).abs();
        eprintln!(
            "fit_shash: EDF gamrs {} vs mgcv {} (diff {edf_diff:.3e})",
            fit.edf, fx.edf_total
        );
        assert!(
            edf_diff < 1e-3,
            "EDF = {} vs mgcv {} (diff {edf_diff})",
            fit.edf,
            fx.edf_total
        );

        // (3) Fitted quantiles at p ∈ {0.1, 0.5, 0.9} vs mgcv-derived quantiles.
        for (p, q_mgcv) in [(0.1, &fx.q10), (0.5, &fx.q50), (0.9, &fx.q90)] {
            let q = fit.predict_quantile(x.view(), p).expect("predict_quantile");
            let mut max_q = 0.0_f64;
            for i in 0..n {
                max_q = max_q.max((q[i] - q_mgcv[i]).abs());
            }
            eprintln!("fit_shash: p={p} max |q_gamrs − q_mgcv| = {max_q:.3e}");
            assert!(
                max_q < 1e-4,
                "quantile p={p}: max|gamrs − mgcv| = {max_q} (>= 1e-4)"
            );
        }
    }
}
