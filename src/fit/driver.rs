//! Generic fit drivers — collapse the per-family boilerplate.
//!
//! Each driver is generic over `S: LinearSolver` (default `CholeskySolver`
//! at the public-entry-point level). The S parameter flows from the
//! caller's choice of `gamrs::fit::<_, _, _, LuSolver>(...)` through to
//! the inner solver and ultimately the emitted `GaussianInnerFit<S>`.

use std::marker::PhantomData;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::design::PreparedDesign;
use crate::error::Result;
use crate::family::Family;
use crate::inner::{CholeskySolver, GaussianInnerFit, LinearSolver, PirlsInner, PirlsOpts};
use crate::outer::NewtonWithHalving;
use crate::score::{EnvelopeScore, Profile, ShapeAwareEnvelopeScore, ShapeInnerBuilder};
use crate::traits::{
    CoordsKind, InnerSolver, Link, Loss, OuterSolver, ScoreDerivatives, VarianceFn,
};

use super::{compute_edf, compute_edf_per_term, compute_vcov, FittedGam, LinkKind};

/// Strategy for picking the outer Newton's initial ρ vector.
///
/// Concrete strategies are zero-cost stateless markers (`ZeroInit`,
/// `SmartInit`) plus a stateful `ExplicitInit { rho }` for warm-starts.
/// Data flows in per call — keeps the trait lifetime-free and means
/// the same instance can re-initialize across refits (e.g. auto-k).
pub trait LambdaInit {
    /// Materialise the initial ρ vector for the given data + penalty
    /// list. Returns length `s_list.len()`.
    fn init(
        &self,
        y: ArrayView1<f64>,
        x_design: &Array2<f64>,
        s_list: &[Array2<f64>],
    ) -> Array1<f64>;
}

/// `ρ_j = 0` (`λ_j = 1`) for all terms. Cheap and basis-agnostic;
/// matches the gamrs ≤ 0.1 default. Robust for Gaussian / Bernoulli /
/// Poisson where the basin around λ=1 is wide.
#[derive(Clone, Copy, Default)]
pub struct ZeroInit;
impl LambdaInit for ZeroInit {
    fn init(
        &self,
        _y: ArrayView1<f64>,
        _x_design: &Array2<f64>,
        s_list: &[Array2<f64>],
    ) -> Array1<f64> {
        Array1::<f64>::zeros(s_list.len())
    }
}

/// `λ_init_j = y_var · ‖S_j‖_F · n / (‖X‖_F² + ε)`, clamped to
/// `[1e-6, 1e6]`. Ported from mgcv_rust v0.x's
/// `initialize_lambda_smart` (`gam_optimized.rs:479-502`). Default for
/// shape-aware multi-smooth fits — puts the outer Newton in v0.x's
/// convergence basin (parity report 2026-05-27).
#[derive(Clone, Copy, Default)]
pub struct SmartInit;
impl LambdaInit for SmartInit {
    fn init(
        &self,
        y: ArrayView1<f64>,
        x_design: &Array2<f64>,
        s_list: &[Array2<f64>],
    ) -> Array1<f64> {
        let n = y.len() as f64;
        let y_mean = y.sum() / n.max(1.0);
        let y_var = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>() / n.max(1.0);
        let x_norm_sq: f64 = x_design.iter().map(|&v| v * v).sum();
        let mut rho = Array1::<f64>::zeros(s_list.len());
        for (j, s_j) in s_list.iter().enumerate() {
            let s_norm: f64 = s_j.iter().map(|&v| v * v).sum::<f64>().sqrt();
            let lambda = (y_var * s_norm * n) / (x_norm_sq + 1e-10);
            rho[j] = lambda.clamp(1e-6, 1e6).ln();
        }
        rho
    }
}

/// mgcv's `initial.sp` (mgcv 1.9-3) on the weighted design `√w·X` — what
/// `initial.spg` does for an extended family such as `scat`. Each term's
/// λ is set from its OWN block: the mean of `diag(X'WX)` over the columns
/// its penalty acts on, over the mean of `diag(S_j)` there. All λ are then
/// scaled together by powers of 10 until the penalised columns are about
/// 40% data-dominated (`mean(xx/(xx+ss)) ≥ 0.4`).
///
/// [`SmartInit`] instead divides every `‖S_j‖_F` by one global `‖X‖_F²`,
/// which can start a term on the wrong side of a ridge in the criterion:
/// on a 1,000-sale `scat` fit it started `lot_sqft` at log λ 6.48 against
/// mgcv's −0.11, past a basin boundary at ~5.1, and the fit ended with the
/// term a straight line and REML 15 units worse
/// (`docs/scat_start_basin_bug.md`).
#[derive(Clone, Default)]
pub struct MgcvInit {
    /// Working weights `w_i` (not their square roots); `None` = all 1.
    pub weights: Option<Array1<f64>>,
}
impl LambdaInit for MgcvInit {
    fn init(
        &self,
        _y: ArrayView1<f64>,
        x_design: &Array2<f64>,
        s_list: &[Array2<f64>],
    ) -> Array1<f64> {
        let p = x_design.ncols();
        let ldxx: Array1<f64> = (0..p)
            .map(|c| match &self.weights {
                Some(w) => x_design
                    .column(c)
                    .iter()
                    .zip(w.iter())
                    .map(|(v, wi)| wi * v * v)
                    .sum(),
                None => x_design.column(c).iter().map(|v| v * v).sum(),
            })
            .collect();
        let mut ldss = Array1::<f64>::zeros(p);
        let mut pen = vec![false; p];
        let mut sp = Array1::<f64>::zeros(s_list.len());
        for (j, s_j) in s_list.iter().enumerate() {
            // mgcv's S[[i]] is the term's own block; ours is embedded in p×p, so
            // its row/column means are taken over the block's width, not p.
            let nonzero: Vec<usize> = (0..p)
                .filter(|&r| s_j.row(r).iter().any(|&v| v != 0.0))
                .collect();
            let (Some(&first), Some(&last)) = (nonzero.first(), nonzero.last()) else {
                continue; // an all-zero penalty penalises nothing; λ stays 1
            };
            let width = (last - first + 1) as f64;
            let max_abs = s_j.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
            let thresh = f64::EPSILON.powf(0.8) * max_abs;
            let penalised: Vec<usize> = (first..=last)
                .filter(|&r| {
                    s_j.row(r).iter().map(|v| v.abs()).sum::<f64>() / width > thresh
                        && s_j.column(r).iter().map(|v| v.abs()).sum::<f64>() / width > thresh
                        && s_j[[r, r]].abs() > thresh
                })
                .collect();
            let m = penalised.len() as f64;
            let size_xx = penalised.iter().map(|&r| ldxx[r]).sum::<f64>() / m;
            let size_s = penalised.iter().map(|&r| s_j[[r, r]]).sum::<f64>() / m;
            if !(size_s > 0.0 && size_xx.is_finite()) {
                continue; // mgcv stops here ("S not +ve definite"); λ stays 1
            }
            sp[j] = size_xx / size_s;
            for &r in &penalised {
                pen[r] = true;
            }
            for r in first..=last {
                ldss[r] += sp[j] * s_j[[r, r]];
            }
        }
        let cols: Vec<usize> = (0..p)
            .filter(|&c| ldss[c] > 0.0 && pen[c] && ldxx[c] > 0.0)
            .collect();
        if !cols.is_empty() {
            let data_share = |scale: f64| {
                cols.iter()
                    .map(|&c| ldxx[c] / (ldxx[c] + scale * ldss[c]))
                    .sum::<f64>()
                    / cols.len() as f64
            };
            let mut scale = 1.0_f64;
            while data_share(scale) > 0.4 {
                scale *= 10.0;
            }
            while data_share(scale) < 0.4 {
                scale /= 10.0;
            }
            sp.mapv_inplace(|v| v * scale);
        }
        sp.mapv(|v| if v > 0.0 { v.ln() } else { 0.0 })
    }
}

/// Caller-supplied ρ vector (e.g. from a pickled warm-restart or a
/// hand-tuned starting point). The vector's length must equal the
/// `s_list.len()` of the surrounding fit (debug-asserted).
#[derive(Clone)]
pub struct ExplicitInit {
    pub rho: Array1<f64>,
}
impl LambdaInit for ExplicitInit {
    fn init(
        &self,
        _y: ArrayView1<f64>,
        _x_design: &Array2<f64>,
        s_list: &[Array2<f64>],
    ) -> Array1<f64> {
        debug_assert_eq!(
            self.rho.len(),
            s_list.len(),
            "ExplicitInit ρ vector length must match s_list"
        );
        self.rho.clone()
    }
}

/// Build a Pearson-φ̂ `scale_fn` closure for `fit_pirls_envelope`. The
/// returned closure captures `(y, prior_weights, n, mu_floor, V(μ))` and
/// computes `Σ wᵢ·(yᵢ-μᵢ)² / V(μᵢ)  /  (n - edf)`. Returns `NaN` if
/// `n - edf ≤ 0`. Eliminates the boilerplate copy-paste in the four
/// profiled-φ GLM families (QuasiPoisson, QuasiBinomial, Gamma, IG).
///
/// Generic over `S: LinearSolver` so the closure signature lines up
/// with the `S`-parameterised `GaussianInnerFit<S>` it consumes.
pub(crate) fn make_pearson_scale_fn<'a, S, F>(
    y_owned: Array1<f64>,
    prior_owned: Option<Array1<f64>>,
    n: usize,
    mu_floor: f64,
    variance_fn: F,
) -> impl FnOnce(&GaussianInnerFit<S>, f64) -> f64 + 'a
where
    S: LinearSolver + 'a,
    F: Fn(f64) -> f64 + 'a,
{
    move |fit, edf| {
        let n_minus_edf = (n as f64) - edf;
        if n_minus_edf <= 0.0 {
            return f64::NAN;
        }
        let sum: f64 = (0..n)
            .map(|i| {
                let mu_i = fit.mu[i].max(mu_floor);
                let r = y_owned[i] - mu_i;
                let w_i = prior_owned.as_ref().map(|w| w[i]).unwrap_or(1.0);
                w_i * r * r / variance_fn(mu_i)
            })
            .sum();
        sum / n_minus_edf
    }
}

/// Outer-Newton + final-fit + EDF for the "PIRLS + EnvelopeScore" stack.
/// Used by every family that fits this pattern (Bernoulli, Poisson,
/// QuasiPoisson, QuasiBinomial, Gamma, InverseGaussian). The 4 family-
/// specific bits stay in the public entry point: y-validation, the
/// `Family<L,K,V>` constructor, the `(LossMarker, Profile)` pair, and the
/// `scale_fn` (Pearson φ̂ or fixed 1.0).
///
/// Returns the assembled `FittedGam` directly so call sites are 3-4 lines.
///
/// `S: LinearSolver` defaults to `CholeskySolver` — pass an explicit
/// turbofish (`fit_pirls_envelope::<_, _, _, _, _, LuSolver, _>(...)`) to
/// switch to LU.
pub(crate) fn fit_pirls_envelope<L, K, V, M, P, S, SF>(
    prep: PreparedDesign,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    prior_weights: Option<ArrayView1<f64>>,
    family: Family<L, K, V>,
    loss_marker: M,
    profile: P,
    scale_fn: SF,
    link_kind: LinkKind,
) -> Result<FittedGam>
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    M: Loss,
    P: Profile<M>,
    S: LinearSolver,
    SF: FnOnce(&GaussianInnerFit<S>, f64) -> f64,
{
    let prior = prior_weights.map(|w| w.to_owned());
    let n = x.nrows();
    let n_terms = prep.s_list.len();
    let pirls: PirlsInner<L, K, V, S> = PirlsInner {
        x_design: prep.x_design.clone(),
        y: y.to_owned(),
        prior_weights: prior,
        s_list: prep.s_list.clone(),
        family,
        opts: PirlsOpts::default(),
        _solver: PhantomData,
    };
    let score = EnvelopeScore::<M, PirlsInner<L, K, V, S>, P, S>::with_inner(
        pirls,
        loss_marker,
        profile,
        y.to_owned(),
        prep.s_list.clone(),
        prep.rank_s_list.clone(),
        prep.mp,
        prep.log_pseudo_det_s_list.clone(),
    );
    let outer = match crate::outer::resolved_algorithm() {
        crate::outer::OuterAlgorithm::Newton => {
            let solver =
                NewtonWithHalving::new(crate::outer::resolve_tuning(&score.loss).to_newton_opts());
            solver.minimize(&score, Array1::zeros(n_terms))?
        }
        crate::outer::OuterAlgorithm::FellnerSchall => {
            crate::outer::fellner_schall_minimize(
                &score.inner,
                &score.s_list,
                &score.rank_s_list,
                Array1::zeros(n_terms),
                // FS dispersion: 1.0 for fixed-φ families (Bernoulli,
                // Poisson, NegBin), Pearson φ̂ for profiled families.
                // mgcv R defaults φ=1 for fREML on GLMs (matches mgcv_rust
                // smooth.rs:3894 caller).
                |_fit| 1.0,
                crate::outer::FellnerSchallOpts::default(),
                score.stats(),
            )?
        }
    };

    // Reuse the score's inner solver for the final fit (closes audit §B4).
    let final_fit: GaussianInnerFit<S> = score.inner.fit(&outer.theta)?;
    let edf = compute_edf(&prep.x_design, &final_fit.a_weights, &final_fit);
    let scale = scale_fn(&final_fit, edf);
    let vcov = compute_vcov(&final_fit, scale);
    let rho_hat = outer.theta.clone();
    let lambda_hat: Array1<f64> = rho_hat.iter().map(|&r| r.exp()).collect();
    let edf_per_term =
        compute_edf_per_term(&prep.s_list, &rho_hat, prep.x_design.ncols(), &final_fit);

    Ok(FittedGam {
        beta: final_fit.beta,
        rho: rho_hat,
        lambda: lambda_hat,
        scale,
        edf_total: edf,
        edf_per_term,
        n,
        n_iters: outer.iterations,
        converged: outer.converged && final_fit.converged,
        reml_value: outer.value,
        predictor: prep.predictor,
        vcov,
        link_kind,
        shape_params: Array1::zeros(0),
        stats: score.stats.snapshot(),
    })
}

/// Default-Cholesky-backend shorthand for `fit_pirls_envelope`. 99% of
/// callers go through this; the explicit-S version exists for the
/// `gamrs::fit::<_, _, _, LuSolver>(...)` opt-in.
#[allow(dead_code)]
pub(crate) fn fit_pirls_envelope_cholesky<L, K, V, M, P, SF>(
    prep: PreparedDesign,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    prior_weights: Option<ArrayView1<f64>>,
    family: Family<L, K, V>,
    loss_marker: M,
    profile: P,
    scale_fn: SF,
    link_kind: LinkKind,
) -> Result<FittedGam>
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    M: Loss,
    P: Profile<M>,
    SF: FnOnce(&GaussianInnerFit<CholeskySolver>, f64) -> f64,
{
    fit_pirls_envelope::<L, K, V, M, P, CholeskySolver, SF>(
        prep,
        x,
        y,
        prior_weights,
        family,
        loss_marker,
        profile,
        scale_fn,
        link_kind,
    )
}

/// Outer-Newton + shape-aware final fit + EDF for `ShapeAwareEnvelopeScore`.
/// Handles families with extra shape params jointly optimised with ρ
/// (TDist/scat, Tweedie, NegBin). Caller supplies:
/// - `family_base`: starting family (cloned per probe)
/// - `theta0`: initial θ = [ρ₀, shape₀…]
/// - `inner_builder`: how to instantiate the inner solver (PIRLS vs ocat).
/// - `profile`: dispersion convention (`FixedAtOneProfile` for scat /
///   NegBin / ocat, `OwnedByLossProfile` for Tweedie).
/// - `rebuild_final_family`: produces the final family with shape params
///   pulled from `outer.theta[1..]` (caller-specific because each family's
///   constructor takes different args).
/// - `scale_fn`: reports `FittedGam::scale` from the final fit + family.
///
/// Final fit reuses the score's `inner_builder.build(...)` (closes audit
/// §B4 / §79 — the final PIRLS must come from the same builder the score
/// used per probe).
pub(crate) fn fit_shape_aware<L, K, V, B, P, S, RF, SF>(
    prep: PreparedDesign,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    prior_weights: Option<ArrayView1<f64>>,
    family_base: Family<L, K, V>,
    theta0: Array1<f64>,
    inner_builder: B,
    profile: P,
    rebuild_final_family: RF,
    scale_fn: SF,
    link_kind: LinkKind,
) -> Result<FittedGam>
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    B: ShapeInnerBuilder<L, K, V, S> + Clone,
    P: Profile<L>,
    S: LinearSolver,
    RF: FnOnce(&Array1<f64>) -> Family<L, K, V>,
    SF: FnOnce(&Family<L, K, V>, &GaussianInnerFit<S>, &Array1<f64>) -> f64,
{
    // 0.2: multi-smooth — outer Newton joint-optimises [ρ_1, …, ρ_T, shape_params]
    // via ShapeAwareEnvelopeScore. The inner solvers (PirlsInner / OcatInner)
    // already accept arbitrary-length rho.
    let prior = prior_weights.map(|w| w.to_owned());
    let n = x.nrows();

    let pirls_opts = PirlsOpts {
        dev_rel_tol: family_base.loss.pirls_dev_rel_tol(),
        ..Default::default()
    };
    let score = ShapeAwareEnvelopeScore::<L, K, V, B, P, S> {
        x_design: prep.x_design.clone(),
        y: y.to_owned(),
        prior_weights: prior.clone(),
        s_list: prep.s_list.clone(),
        family_base,
        rank_s_list: prep.rank_s_list.clone(),
        mp: prep.mp,
        log_pseudo_det_s_list: prep.log_pseudo_det_s_list.clone(),
        coords: CoordsKind::Identity,
        pirls_opts,
        inner_builder: inner_builder.clone(),
        profile,
        _solver: PhantomData,
        accepted_state: std::cell::RefCell::new(None),
        last_eta: std::cell::RefCell::new(None),
        stats: crate::stats::FitStats::new(),
    };

    crate::outer::reject_unsupported_algorithm("shape-parameter (joint Newton over [rho, theta])")?;
    let outer_solver = NewtonWithHalving::new(
        crate::outer::resolve_tuning(&score.family_base.loss).to_newton_opts(),
    );
    let outer = outer_solver.minimize(&score, theta0)?;

    // Final fit: reuse the score's inner_builder so the final PIRLS sees
    // exactly the same solver settings as every outer probe (closes audit
    // §79 / §B4). Each family's constructor signature differs, so the
    // caller still supplies `rebuild_final_family`.
    let n_terms = prep.s_list.len();
    let rho_hat: Array1<f64> = outer.theta.slice(ndarray::s![..n_terms]).to_owned();
    let family_final = rebuild_final_family(&outer.theta);
    let final_inner = score.inner_builder.build(
        family_final.clone(),
        prep.x_design.clone(),
        y.to_owned(),
        prior,
        prep.s_list.clone(),
        // edf and vcov read the Fisher pair, and this is the one fit they read
        // — so ask for it here and nowhere else.
        PirlsOpts {
            want_fisher: true,
            ..PirlsOpts::default()
        },
    );
    let final_fit: GaussianInnerFit<S> = final_inner.fit(&rho_hat)?;

    let edf = compute_edf(&prep.x_design, &final_fit.a_weights, &final_fit);
    let scale = scale_fn(&family_final, &final_fit, &outer.theta);
    let vcov = compute_vcov(&final_fit, scale);
    let lambda_vec: Array1<f64> = rho_hat.iter().map(|&r| r.exp()).collect();
    let edf_per_term =
        compute_edf_per_term(&prep.s_list, &rho_hat, prep.x_design.ncols(), &final_fit);

    // outer.theta layout for shape-aware families (0.2 multi-smooth):
    // `[ρ_1, …, ρ_T, shape_1, shape_2, …]`. The trailing shape slice
    // gives us the fitted family-specific shape params (ocat thresholds,
    // t-dist ν / σ², tweedie p / phi, …).
    let shape_params = if outer.theta.len() > n_terms {
        outer.theta.slice(ndarray::s![n_terms..]).to_owned()
    } else {
        Array1::<f64>::zeros(0)
    };

    Ok(FittedGam {
        beta: final_fit.beta,
        rho: rho_hat,
        lambda: lambda_vec,
        scale,
        edf_total: edf,
        edf_per_term,
        n,
        n_iters: outer.iterations,
        converged: outer.converged && final_fit.converged,
        reml_value: outer.value,
        predictor: prep.predictor,
        vcov,
        link_kind,
        shape_params,
        stats: score.stats.snapshot(),
    })
}
