//! Per-family `FamilyFitWithSolver` impls.
//!
//! Each Loss impls `FamilyFitWithSolver<K, V, S>` exactly once for its
//! (Link, Variance) pair; the impl body is the validate-wire-drive
//! sequence that used to live in the deleted per-family
//! `fit_*_cr_with_solver` wrappers. The trait declaration and the
//! `gamrs::fit{,_with,_with_design,_with_solver}` public functions live in
//! `canonical.rs`.
//!
//! Impls consume a `PreparedDesign` built upstream by a
//! `DesignStrategy::prepare(x)` — basis choice is decoupled from family
//! dispatch.

use std::marker::PhantomData;

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::design::PreparedDesign;
use crate::error::{GamrsError, Result};
use crate::family::{
    bernoulli_logit, gamma_inverse, gamma_log, inverse_gaussian_log, negbin_log, ocat_identity,
    poisson_log, quasibinomial_logit, quasipoisson_log, tdist_identity, tweedie_log,
    tweedie_log_fixed_p, Bernoulli, BinomialVariance, ConstantVariance, ElfLoss, ElfVariance,
    Family, Gamma, GammaVariance, Gaussian, IdentityLink, InverseGaussian, InverseGaussianVariance,
    InverseLink, LogLink, LogitLink, NegBin, NegBinVariance, OcatLoss, OcatVariance, Poisson,
    PoissonVariance, QuasiBinomial, QuasiPoisson, TDist, TVariance, Tweedie, TweedieVariance,
    TDIST_MIN_DF,
};
use crate::inner::{GaussianInnerFit, LinearSolver, PirlsOpts};
use crate::outer::NewtonWithHalving;
use crate::score::{
    FixedAtOneProfile, MgcvTwoSigmaProfile, OcatInnerBuilder, OwnedByLossProfile,
    PirlsInnerBuilder, ShapeAwareEnvelopeScore, ShapeInnerBuilder,
};
use crate::traits::{CoordsKind, InnerSolver, Loss, OuterSolver};

use super::canonical::FamilyFitWithSolver;
use super::driver::{fit_pirls_envelope, fit_shape_aware, make_pearson_scale_fn};
use super::driver::{LambdaInit, MgcvInit, SmartInit};
use super::gaussian::fit_gaussian_from_prep;
use super::profile_shape::fit_shape_aware_profile;
use super::quantile::fit_quantile_from_prep;
use super::{
    check_lengths, check_y_in_unit, check_y_nonneg, check_y_positive, compute_edf, compute_vcov,
    FittedGam, LinkKind,
};

// --- Gaussian: identity link + constant variance, closed-form inner ---
impl<S: LinearSolver> FamilyFitWithSolver<IdentityLink, ConstantVariance, S> for Gaussian {
    fn fit_from_prep_canonical(
        _family: Family<Self, IdentityLink, ConstantVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        check_lengths(x, y, prior_weights)?;
        fit_gaussian_from_prep::<S>(prep, x, y, prior_weights)
    }
}

// --- Bernoulli: logit + binomial variance, σ²≡1 ---
impl<S: LinearSolver> FamilyFitWithSolver<LogitLink, BinomialVariance, S> for Bernoulli {
    fn fit_from_prep_canonical(
        _family: Family<Self, LogitLink, BinomialVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        check_lengths(x, y, prior_weights)?;
        check_y_in_unit(y, "Bernoulli")?;
        fit_pirls_envelope::<_, _, _, _, _, S, _>(
            prep,
            x,
            y,
            prior_weights,
            bernoulli_logit(),
            Bernoulli,
            FixedAtOneProfile,
            |_, _| 1.0,
            LinkKind::Logit,
        )
    }
}

// --- Poisson: log link + Poisson variance, σ²≡1 ---
impl<S: LinearSolver> FamilyFitWithSolver<LogLink, PoissonVariance, S> for Poisson {
    fn fit_from_prep_canonical(
        _family: Family<Self, LogLink, PoissonVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        check_lengths(x, y, prior_weights)?;
        check_y_nonneg(y, "Poisson")?;
        fit_pirls_envelope::<_, _, _, _, _, S, _>(
            prep,
            x,
            y,
            prior_weights,
            poisson_log(),
            Poisson,
            FixedAtOneProfile,
            |_, _| 1.0,
            LinkKind::Log,
        )
    }
}

// --- QuasiPoisson: log link + Poisson variance, profiled φ ---
impl<S: LinearSolver> FamilyFitWithSolver<LogLink, PoissonVariance, S> for QuasiPoisson {
    fn fit_from_prep_canonical(
        _family: Family<Self, LogLink, PoissonVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        check_lengths(x, y, prior_weights)?;
        check_y_nonneg(y, "QuasiPoisson")?;
        let n = x.nrows();
        let scale_fn = make_pearson_scale_fn::<S, _>(
            y.to_owned(),
            prior_weights.map(|w| w.to_owned()),
            n,
            1e-300,
            |mu_i| mu_i,
        );
        fit_pirls_envelope::<_, _, _, _, _, S, _>(
            prep,
            x,
            y,
            prior_weights,
            quasipoisson_log(),
            QuasiPoisson,
            MgcvTwoSigmaProfile,
            scale_fn,
            LinkKind::Log,
        )
    }
}

// --- QuasiBinomial: logit + binomial variance, profiled φ ---
impl<S: LinearSolver> FamilyFitWithSolver<LogitLink, BinomialVariance, S> for QuasiBinomial {
    fn fit_from_prep_canonical(
        _family: Family<Self, LogitLink, BinomialVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        check_lengths(x, y, prior_weights)?;
        check_y_in_unit(y, "QuasiBinomial")?;
        let n = x.nrows();
        let scale_fn = make_pearson_scale_fn::<S, _>(
            y.to_owned(),
            prior_weights.map(|w| w.to_owned()),
            n,
            1e-15,
            |mu_i| {
                let mu = mu_i.min(1.0 - 1e-15);
                mu * (1.0 - mu)
            },
        );
        fit_pirls_envelope::<_, _, _, _, _, S, _>(
            prep,
            x,
            y,
            prior_weights,
            quasibinomial_logit(),
            QuasiBinomial,
            MgcvTwoSigmaProfile,
            scale_fn,
            LinkKind::Logit,
        )
    }
}

// --- Gamma: log link + μ² variance, profiled φ ---
impl<S: LinearSolver> FamilyFitWithSolver<LogLink, GammaVariance, S> for Gamma {
    fn fit_from_prep_canonical(
        _family: Family<Self, LogLink, GammaVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        check_lengths(x, y, prior_weights)?;
        check_y_positive(y, "Gamma")?;
        let n = x.nrows();
        let scale_fn = make_pearson_scale_fn::<S, _>(
            y.to_owned(),
            prior_weights.map(|w| w.to_owned()),
            n,
            1e-300,
            |mu_i| mu_i * mu_i,
        );
        fit_pirls_envelope::<_, _, _, _, _, S, _>(
            prep,
            x,
            y,
            prior_weights,
            gamma_log(),
            Gamma,
            MgcvTwoSigmaProfile,
            scale_fn,
            LinkKind::Log,
        )
    }
}

// --- Gamma: inverse (canonical) link + μ² variance, profiled φ ---
// mgcv's default for `family = Gamma()`. Reuses the link-free `Gamma`
// Loss + `μ²` `GammaVariance` from the log-link impl; only the Link
// type swaps. Canonical pair → Fisher == Newton in PIRLS.
impl<S: LinearSolver> FamilyFitWithSolver<InverseLink, GammaVariance, S> for Gamma {
    fn fit_from_prep_canonical(
        _family: Family<Self, InverseLink, GammaVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        check_lengths(x, y, prior_weights)?;
        check_y_positive(y, "Gamma")?;
        let n = x.nrows();
        let scale_fn = make_pearson_scale_fn::<S, _>(
            y.to_owned(),
            prior_weights.map(|w| w.to_owned()),
            n,
            1e-300,
            |mu_i| mu_i * mu_i,
        );
        fit_pirls_envelope::<_, _, _, _, _, S, _>(
            prep,
            x,
            y,
            prior_weights,
            gamma_inverse(),
            Gamma,
            MgcvTwoSigmaProfile,
            scale_fn,
            LinkKind::Inverse,
        )
    }
}

// --- InverseGaussian: log link + μ³ variance, profiled φ ---
impl<S: LinearSolver> FamilyFitWithSolver<LogLink, InverseGaussianVariance, S> for InverseGaussian {
    fn fit_from_prep_canonical(
        _family: Family<Self, LogLink, InverseGaussianVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        check_lengths(x, y, prior_weights)?;
        check_y_positive(y, "InverseGaussian")?;
        let n = x.nrows();
        let scale_fn = make_pearson_scale_fn::<S, _>(
            y.to_owned(),
            prior_weights.map(|w| w.to_owned()),
            n,
            1e-300,
            |mu_i| mu_i * mu_i * mu_i,
        );
        fit_pirls_envelope::<_, _, _, _, _, S, _>(
            prep,
            x,
            y,
            prior_weights,
            inverse_gaussian_log(),
            InverseGaussian,
            MgcvTwoSigmaProfile,
            scale_fn,
            LinkKind::Log,
        )
    }
}

// --- NegBin: log link + NegBin variance, shape-managed θ ---
// Initial θ is read off the family (`negbin_log(init_theta)` stamped it
// into `family.loss.theta`).
impl<S: LinearSolver> FamilyFitWithSolver<LogLink, NegBinVariance, S> for NegBin {
    fn fit_from_prep_canonical(
        family: Family<Self, LogLink, NegBinVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        let init_theta = family.loss.theta;
        check_lengths(x, y, prior_weights)?;
        if !(init_theta > 0.0 && init_theta.is_finite()) {
            return Err(GamrsError::InvalidParameter(format!(
                "NegBin overdispersion init_theta must be a positive finite scalar (small θ → high variance, large θ → Poisson limit); got init_theta={init_theta}"
            )));
        }
        check_y_nonneg(y, "NegBin")?;

        let n_terms = prep.s_list.len();
        let rho_init = SmartInit.init(y, &prep.x_design, &prep.s_list);
        let mut theta0_vec: Vec<f64> = rho_init.to_vec();
        theta0_vec.push(init_theta.ln());
        let theta0 = Array1::from_vec(theta0_vec);

        // NegBin uses the profile-θ driver — port of mgcv_rust's NegBin
        // outer-Newton pattern at `src/smooth.rs:1866-1869` + `3562-3637`:
        // M-dim ρ-Newton + sequential 1-D log(θ) profile Newton each
        // outer iter. PIRLS economy matches mgcv_rust's (~4 PIRLS/iter vs
        // the joint Newton's ~9), closing the 11× perf gap vs mgcv_rust
        // on `1d_nb_log_n300`.
        fit_shape_aware_profile::<_, _, _, _, S, _, _>(
            prep,
            x,
            y,
            prior_weights,
            negbin_log(init_theta),
            theta0,
            PirlsInnerBuilder,
            move |theta| {
                let mut f = negbin_log(init_theta);
                f.set_shape_params(&[theta[n_terms]]);
                f
            },
            |family, _fit, _theta| family.loss.theta,
            LinkKind::Log,
        )
    }
}

/// Response scale for scat's internal standardization — the sample standard
/// deviation of `y` (ddof=1); 1.0 only when that is zero or not finite.
///
/// It used to be floored at 1.0 as well, so a response with sd < 1 (prices
/// in millions, sd ≈ 0.1) was never standardized: the default σ² start of
/// sd² then sat far from the O(1) scale everything else is tuned for, and the
/// outer loop walked log σ² down one capped step per iteration — 40–200 outer
/// iterations where mgcv takes 8–16 (docs/scat_start_basin_bug.md).
///
/// scat's observed IRLS weight is `W = ½·Dμμ ~ 1/σ²`, and the inner solve
/// forms `X'WX + λS` and Cholesky-factorizes it. A raw response with
/// `var(y) ≫ 1` (e.g. prices, `var ~ 1e11`) drives `X'WX ~ 1e-12·X'X` — far
/// below `λS` — so the unpenalised directions go numerically unidentified and
/// the outer Newton stalls (`NotConverged`). scat with identity link is
/// location-scale equivariant, so we fit the standardized response `ỹ = y/s`
/// (giving `σ̃² ~ O(1)`) and rescale the fit back via [`rescale_scat_fit`].
/// This is the same conditioning fix mgcv gets from its data-scale dispersion
/// init (`efam.r` preinitialize, `σ = 0.8·sd(y)`).
pub(crate) fn scat_response_scale(y: ArrayView1<f64>) -> f64 {
    let n = y.len();
    if n < 2 {
        return 1.0;
    }
    let mean = y.sum() / n as f64;
    let var = y.iter().map(|&yi| (yi - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
    let sd = var.sqrt();
    if sd.is_finite() && sd > 0.0 {
        sd
    } else {
        1.0
    }
}

/// Undo scat's internal standardization on a fit of `ỹ = y/s`. With the
/// identity link `μ = Xβ`, every fitted quantity is homogeneous in `s`:
/// `β, μ ∝ s`, `Var(β̂), σ² ∝ s²`. Rescaling these makes `predict`, `vcov`,
/// `scale`, and the reported `σ²` come out in the original `y` units (predict
/// goes through `β` so beta rescaling suffices). `reml_value` stays on the
/// standardized scale — it shifts by a `y`-independent constant and is only
/// ever compared at fixed `s`. No-op when `s == 1.0`.
fn rescale_scat_fit(mut fit: FittedGam, s: f64) -> FittedGam {
    if s == 1.0 {
        return fit;
    }
    let s2 = s * s;
    fit.beta.mapv_inplace(|b| b * s);
    fit.vcov.mapv_inplace(|v| v * s2);
    fit.scale *= s2;
    // shape_params[0] = log σ̃²  →  log(s²·σ̃²) = log σ̃² + 2·log s.
    if !fit.shape_params.is_empty() {
        fit.shape_params[0] += 2.0 * s.ln();
    }
    fit
}

/// The working weights mgcv's `initial.spg` puts on `scat`'s design:
/// `½·Dmu2` at `mustart = y + 0.1·[y == 0]`, i.e.
/// `(ν+1)(νσ² − r²)/(νσ² + r²)²` with `r = y − mustart`; if any row comes
/// out negative, mgcv switches every row to the expected `½·EDmu2 =
/// (ν+1)/((ν+3)σ²)`.
fn scat_initial_weights(y: ArrayView1<f64>, nu: f64, sigma2: f64) -> Array1<f64> {
    let ns2 = nu * sigma2;
    let w: Array1<f64> = y.mapv(|yi| {
        let r2 = if yi == 0.0 { 0.01 } else { 0.0 };
        (nu + 1.0) * (ns2 - r2) / ((ns2 + r2) * (ns2 + r2))
    });
    if w.iter().any(|&wi| wi < 0.0) {
        Array1::from_elem(y.len(), (nu + 1.0) / ((nu + 3.0) * sigma2))
    } else {
        w
    }
}

// --- TDist (scat): identity link + T variance, shape-managed σ², ν ---
impl<S: LinearSolver> FamilyFitWithSolver<IdentityLink, TVariance, S> for TDist {
    fn fit_from_prep_canonical(
        family: Family<Self, IdentityLink, TVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        let init_nu = family.loss.nu;
        let init_sigma2 = family.loss.sigma2;
        check_lengths(x, y, prior_weights)?;
        // ν must exceed MIN_DF: the shape transform is θ₁ = log(ν − MIN_DF),
        // so ν ≤ MIN_DF would make the initial log(ν − MIN_DF) non-finite.
        if !(init_nu > TDIST_MIN_DF && init_nu.is_finite()) {
            return Err(GamrsError::InvalidParameter(format!(
                "scat/TDist degrees-of-freedom init_nu must be > {TDIST_MIN_DF} (mgcv min.df); got init_nu={init_nu}"
            )));
        }
        if !(init_sigma2 > 0.0 && init_sigma2.is_finite()) {
            return Err(GamrsError::InvalidParameter(format!(
                "scat/TDist scale init_sigma2 must be a positive finite scalar; got init_sigma2={init_sigma2}"
            )));
        }

        // scat is location-scale equivariant: fit the standardized response
        // `ỹ = y/s` and rescale the result back (EXACT). This keeps the inner
        // solve's normal-equations Cholesky `X'WX + λS` conditioned at any data
        // magnitude — see [`scat_response_scale`]. Previously applied only in
        // the Python wrapper; doing it in-core makes the Rust API robust on
        // raw-scale data too. `s == 1.0` (already ~unit scale) → exact no-op.
        let s = scat_response_scale(y);
        let ys: Array1<f64> = if s == 1.0 {
            y.to_owned()
        } else {
            y.mapv(|v| v / s)
        };
        let init_sigma2_std = init_sigma2 / (s * s);

        let n_terms = prep.s_list.len();
        let rho_init = MgcvInit {
            weights: Some(scat_initial_weights(ys.view(), init_nu, init_sigma2_std)),
        }
        .init(ys.view(), &prep.x_design, &prep.s_list);
        let mut theta0_vec: Vec<f64> = rho_init.to_vec();
        theta0_vec.push(init_sigma2_std.ln());
        theta0_vec.push((init_nu - TDIST_MIN_DF).ln());
        let theta0 = Array1::from_vec(theta0_vec);

        let fit = fit_shape_aware::<_, _, _, _, _, S, _, _>(
            prep,
            x,
            ys.view(),
            prior_weights,
            tdist_identity(init_nu, init_sigma2_std),
            theta0,
            PirlsInnerBuilder,
            FixedAtOneProfile,
            move |theta| {
                let mut f = tdist_identity(init_nu, init_sigma2_std);
                f.set_shape_params(&[theta[n_terms], theta[n_terms + 1]]);
                f
            },
            move |_family, _fit, theta| theta[n_terms].exp(),
            LinkKind::Identity,
        )?;
        Ok(rescale_scat_fit(fit, s))
    }
}

// --- Tweedie: log link + μ^p variance, shape-managed p, φ ---
impl<S: LinearSolver> FamilyFitWithSolver<LogLink, TweedieVariance, S> for Tweedie {
    fn fit_from_prep_canonical(
        family: Family<Self, LogLink, TweedieVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        let init_p = family.loss.p;
        let init_phi = family.loss.phi;
        let profile_p = family.loss.profile_p;
        check_lengths(x, y, prior_weights)?;
        if !(1.0 < init_p && init_p < 2.0) {
            return Err(GamrsError::InvalidParameter(format!(
                "Tweedie variance power must be in (1, 2) — strict interior \
                 (p=1 → Poisson, p=2 → Gamma — use those families directly); \
                 got init_p={init_p}"
            )));
        }
        if !(init_phi > 0.0 && init_phi.is_finite()) {
            return Err(GamrsError::InvalidParameter(format!(
                "Tweedie dispersion init_phi must be a positive finite scalar; got init_phi={init_phi}"
            )));
        }
        check_y_nonneg(y, "Tweedie")?;

        let n_terms = prep.s_list.len();
        let rho_init = SmartInit.init(y, &prep.x_design, &prep.s_list);
        let mut theta0_vec: Vec<f64> = rho_init.to_vec();
        theta0_vec.push(init_phi.ln());
        if profile_p {
            // Only present when p is profiled.
            theta0_vec.push(crate::family::tweedie::tweedie_theta_from_p(init_p));
        }
        let theta0 = Array1::from_vec(theta0_vec);

        // Profile-p has 2 shape params [log φ, p_t]; fixed-p has 1 [log φ].
        let make_family = move || {
            if profile_p {
                tweedie_log(init_p, init_phi)
            } else {
                tweedie_log_fixed_p(init_p, init_phi)
            }
        };

        fit_shape_aware::<_, _, _, _, _, S, _, _>(
            prep,
            x,
            y,
            prior_weights,
            make_family(),
            theta0,
            PirlsInnerBuilder,
            OwnedByLossProfile,
            move |theta| {
                let mut f = make_family();
                let n_shape = f.n_shape_params();
                let shape_slice: Vec<f64> = (0..n_shape).map(|i| theta[n_terms + i]).collect();
                f.set_shape_params(&shape_slice);
                f
            },
            // scale = φ̂ from the converged family.
            |family, _fit, _theta| family.loss.phi,
            LinkKind::Log,
        )
    }
}

/// Smart θ initialisation for ocat from empirical category frequencies.
/// Port of mgcv R `~/gitlab/mgcv/R/efam.r:464-476` (`ocat.ini`).
///
/// Computes:
/// ```text
///   p_k       = (Σ_{j ≤ k} count_j + 1) / (N + R)   (with Laplace smoothing)
///   η         = −1 − logit(p_1)                     (latent mean s.t. F(α_1 − η) = p_1)
///   α_k       = logit(p_k) + η  for k = 2..R−1      (interior cut points)
///   gap_k     = max(α_{k+1} − α_k, 0.01)            (positive gaps, floor)
///   θ_k       = log(gap_k)                          (log-gap parameterisation)
/// ```
///
/// With this initialiser the joint Newton starts near the optimum
/// instead of `θ = 0`, dramatically reducing the walk through the
/// near-flat (η-scale, θ-magnitude) ridge that otherwise traps it.
fn ocat_smart_init(y: ArrayView1<f64>, n_cats: usize) -> Array1<f64> {
    if n_cats < 3 {
        return Array1::<f64>::zeros(0);
    }
    // Laplace-smoothed empirical counts: add one observation to every
    // category so `p[0] > 0` (mgcv R does `y <- c(1:R, y)` for the same
    // effect — avoids divide-by-zero on the logit).
    let mut counts = vec![1usize; n_cats];
    for &yi in y.iter() {
        if !yi.is_finite() {
            continue;
        }
        let yi_c = yi.round() as i64;
        if (1..=n_cats as i64).contains(&yi_c) {
            counts[yi_c as usize - 1] += 1;
        }
    }
    let total: f64 = counts.iter().sum::<usize>() as f64;
    let mut p: Vec<f64> = Vec::with_capacity(n_cats);
    let mut acc = 0.0;
    for &c in counts.iter() {
        acc += c as f64 / total;
        p.push(acc.clamp(1e-9, 1.0 - 1e-9));
    }
    let logit = |q: f64| (q / (1.0 - q)).ln();
    let eta = -1.0 - logit(p[0]);
    // alpha[0] = -1 (fixed); alpha[k] = logit(p[k]) + eta for k = 1..R-2
    let mut alpha = vec![-1.0_f64; n_cats - 1];
    for k in 1..(n_cats - 1) {
        alpha[k] = logit(p[k]) + eta;
    }
    // gaps + log
    let mut theta = Array1::<f64>::zeros(n_cats - 2);
    for k in 0..(n_cats - 2) {
        let gap = (alpha[k + 1] - alpha[k]).max(0.01);
        theta[k] = gap.ln();
    }
    theta
}

// --- Ocat: identity link + Ocat variance, n_cats + thresholds on family ---
impl<S: LinearSolver> FamilyFitWithSolver<IdentityLink, OcatVariance, S> for OcatLoss {
    fn fit_from_prep_canonical(
        family: Family<Self, IdentityLink, OcatVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        let n_cats = family.loss.n_cats;
        let init_thresholds: Array1<f64> = family.loss.thresholds.clone();

        check_lengths(x, y, prior_weights)?;
        if n_cats < 3 {
            return Err(GamrsError::InvalidParameter(format!(
                "Ocat requires n_cats ≥ 3 (≤ 2 levels collapses to Bernoulli — use bernoulli_logit instead); got n_cats={n_cats}"
            )));
        }
        for (i, &yi) in y.iter().enumerate() {
            let r = yi.round() as i64;
            if r < 1 || (r as usize) > n_cats {
                return Err(GamrsError::InvalidParameter(format!(
                    "Ocat requires y to be an integer in 1..={n_cats}; got y={yi} at row {i}"
                )));
            }
        }

        let prior = prior_weights.map(|w| w.to_owned());
        let n = x.nrows();

        // Smart θ init from empirical category frequencies (port of
        // mgcv R `ocat.ini`). Replaces the family's default zero-init.
        // If the caller supplied non-default thresholds via the family
        // constructor, honour them; otherwise compute from data.
        let theta0_shape: Array1<f64> = if init_thresholds.iter().all(|&t| t == 0.0) {
            ocat_smart_init(y, n_cats)
        } else {
            init_thresholds
        };
        if theta0_shape.len() != n_cats - 2 {
            return Err(GamrsError::InvalidParameter(format!(
                "Ocat init_theta length must equal n_cats - 2 = {} (log-gap thresholds between adjacent categories above the first); got {}",
                n_cats - 2,
                theta0_shape.len()
            )));
        }

        // 0.2: multi-smooth — θ = [ρ_1, …, ρ_T, θ₁, …, θ_{R-2}].
        let n_terms = prep.s_list.len();

        let family_base = ocat_identity(theta0_shape.clone(), n_cats);
        let score = ShapeAwareEnvelopeScore::<
            OcatLoss,
            IdentityLink,
            OcatVariance,
            OcatInnerBuilder,
            FixedAtOneProfile,
            S,
        > {
            x_design: prep.x_design.clone(),
            y: y.to_owned(),
            prior_weights: prior.clone(),
            s_list: prep.s_list.clone(),
            family_base,
            rank_s_list: prep.rank_s_list.clone(),
            mp: prep.mp,
            log_pseudo_det_s_list: prep.log_pseudo_det_s_list.clone(),
            coords: CoordsKind::Identity,
            pirls_opts: PirlsOpts {
                dev_rel_tol: family.loss.pirls_dev_rel_tol(),
                ..Default::default()
            },
            inner_builder: OcatInnerBuilder,
            profile: FixedAtOneProfile,
            _solver: PhantomData,
            accepted_state: std::cell::RefCell::new(None),
            last_eta: std::cell::RefCell::new(None),
            stats: crate::stats::FitStats::new(),
        };

        // θ₀ = [SmartInit ρ_1, …, SmartInit ρ_T, θ₁, …, θ_{R-2}]
        let rho_init = SmartInit.init(y, &prep.x_design, &prep.s_list);
        let mut theta0 = Array1::<f64>::zeros(n_terms + theta0_shape.len());
        for i in 0..n_terms {
            theta0[i] = rho_init[i];
        }
        for (i, &t) in theta0_shape.iter().enumerate() {
            theta0[n_terms + i] = t;
        }
        // Ocat sticks with the joint (ρ, θ) Newton via `NewtonWithHalving`.
        // The R-2 ordered thresholds are tightly coupled (α_1 < α_2 < …),
        // and splitting them across the per-axis 1-D Newton loop in
        // `ProfileShapeNewton` loses critical off-diagonal Hessian
        // information — empirically that turns single-smooth ocat from a
        // 12-iter convergence into a 200-iter non-convergence. NegBin's
        // single bounded log θ is the right shape for the profile pattern;
        // ocat is not.
        //
        // Convergence flag caveat at multi-smooth: the joint Newton walks
        // a near-flat coordinated-shift ridge until step-halving exhausts,
        // so `converged_=False` is common. Probabilities (via predict_proba)
        // are scale-invariant under the ridge, so the model is still
        // usable. See README.
        crate::outer::reject_unsupported_algorithm("ocat")?;
        let outer_solver = NewtonWithHalving::new(
            crate::outer::resolve_tuning(&score.family_base.loss).to_newton_opts(),
        );
        let outer = outer_solver.minimize(&score, theta0)?;

        let rho_hat: Array1<f64> = outer.theta.slice(ndarray::s![..n_terms]).to_owned();
        let theta_hat: Array1<f64> = outer.theta.slice(ndarray::s![n_terms..]).to_owned();
        let family_final: Family<OcatLoss, IdentityLink, OcatVariance> =
            ocat_identity(theta_hat.clone(), n_cats);
        let final_inner = ShapeInnerBuilder::<OcatLoss, IdentityLink, OcatVariance, S>::build(
            &score.inner_builder,
            family_final.clone(),
            prep.x_design.clone(),
            y.to_owned(),
            prior,
            prep.s_list.clone(),
            // edf and vcov read the Fisher pair, and this is the one fit they
            // read — so ask for it here and nowhere else.
            PirlsOpts {
                want_fisher: true,
                ..PirlsOpts::default()
            },
        );
        let final_fit: GaussianInnerFit<S> = final_inner.fit(&rho_hat)?;

        let edf = compute_edf(&prep.x_design, &final_fit.a_weights, &final_fit);
        let vcov = compute_vcov(&final_fit, 1.0);
        let lambda_vec: Array1<f64> = rho_hat.iter().map(|&r| r.exp()).collect();
        let edf_per_term =
            super::compute_edf_per_term(&prep.s_list, &rho_hat, prep.x_design.ncols(), &final_fit);

        Ok(FittedGam {
            beta: final_fit.beta,
            rho: rho_hat,
            lambda: lambda_vec,
            scale: 1.0,
            edf_total: edf,
            edf_per_term,
            n,
            n_iters: outer.iterations,
            converged: outer.converged && final_fit.converged,
            reml_value: outer.value,
            predictor: prep.predictor,
            vcov,
            link_kind: LinkKind::Identity,
            shape_params: theta_hat,
            stats: score.stats.snapshot(),
        })
    }
}

// --- ELF (quantile): identity link + ElfVariance ---
// τ, σ, λ_elf all live on the Loss; warm-start picks defaults if
// init_sigma/init_lambda are <= 0.
impl<S: LinearSolver> FamilyFitWithSolver<IdentityLink, ElfVariance, S> for ElfLoss {
    fn fit_from_prep_canonical(
        family: Family<Self, IdentityLink, ElfVariance>,
        prep: PreparedDesign,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        prior_weights: Option<ArrayView1<f64>>,
    ) -> Result<FittedGam> {
        check_lengths(x, y, prior_weights)?;
        // tau validation lives in fit_quantile_from_prep (so the
        // helper carries actionable error messages for the row).
        fit_quantile_from_prep::<S>(
            prep,
            x,
            y,
            prior_weights,
            family.loss.tau,
            family.loss.sigma,
            family.loss.lambda,
        )
    }
}
