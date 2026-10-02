//! `shash` GAMLSS outer REML / LAML criterion and numerical smoothing-parameter
//! selection (TDD phase 5a) — choosing the per-block log-smoothing-parameters ρ
//! that maximise the (Laplace-approximate) marginal likelihood.
//!
//! Above the joint inner solve ([`crate::gamlss::shash_inner::fit_inner`], which
//! returns the penalised MLE β̂ at *fixed* penalties) sits the outer problem of
//! selecting the smoothing parameters themselves. mgcv does this by maximising
//! the LAML / REML criterion `V(ρ)` over the log-smoothing-parameters; here we
//! port that criterion verbatim and drive it with a compact finite-difference
//! damped-Newton ascent.
//!
//! ## The criterion (mgcv `gam.fit4.r:1408-1414`, gamma = 1)
//!
//! ```text
//!   V(ρ) = ℓ(β̂)
//!          − ½ β̂ᵀ S_ρ β̂            (penalty quadratic at β̂)
//!          + ½ log|S_ρ|₊            (generalised log-det of the penalty)
//!          − ½ log|Hp|              (Hp = −∇²(penalised loglik) at β̂)
//!          + (Mp/2)·log(2π)
//! ```
//! where:
//!   - β̂ = β̂(ρ) is the penalised MLE from `fit_inner` at the penalties `S_ρ`;
//!   - `ℓ(β̂) = Σᵢ ℓ₀(...)` is the UNPENALISED log-likelihood, recovered as
//!     `penalized_loglik(...) + ½ β̂ᵀ S_ρ β̂` (the inner objective adds the
//!     `−½ βᵀSβ` penalty back in);
//!   - `S_ρ` is block-diagonal over the four predictors. For a penalised block
//!     `b` with one penalty, `S_ρ block = exp(ρ_b)·S0_b`; unpenalised blocks
//!     (e.g. intercept-only) contribute a zero penalty and carry NO ρ entry;
//!   - `Hp = −hess` (the penalised Hessian from `penalized_grad_hess`, which is
//!     negative-definite near the maximum so `Hp` is SPD), and
//!     `log|Hp| = CholeskySolver::logdet(&factorize(Hp))`;
//!   - `|S_ρ|₊` is the *generalised* determinant (product of the POSITIVE
//!     eigenvalues). Its ρ-DEPENDENT part is exactly `Σ_b rank(S0_b)·ρ_b`
//!     (scaling `S0_b` by `exp(ρ_b)` scales each of its `rank` nonzero
//!     eigenvalues by `exp(ρ_b)`). **We OMIT the ρ-independent constant
//!     `Σ_b log|S0_b|₊`.** That constant shifts the absolute value of `V` by a
//!     fixed amount but affects neither `argmax_ρ V` nor the fit β̂(ρ̂); our
//!     `laml` therefore differs from mgcv's reported REML by this constant.
//!   - `Mp = total_p − Σ_b rank(S0_b)` is the penalty null-space dimension.
//!
//! The effective degrees of freedom is
//!   `EDF = total_p − tr(Hp⁻¹ S_ρ) = total_p − trace_a_inv(Hp_fact, S_ρ)`.
//!
//! ## The analytic gradient `∂V/∂ρ` (phase 5b)
//!
//! [`reml_grad`] computes the REML gradient analytically (mgcv
//! `gam.fit4.r:1346-1365,1408-1413`), replacing the central-difference gradient
//! the outer optimiser used in phase 5a. Per smoothing parameter `i` (penalised
//! block `b(i)`, `λ_i = exp(ρ_i)`):
//! ```text
//!   ∂V/∂ρ_i = −½·λ_i·β̂_b(i)ᵀ S0_b(i) β̂_b(i)     (penalty quadratic)
//!             + ½·rank_i                            (from ½ Σ rank ρ)
//!             − ½·d log|Hp|/dρ_i
//!
//!   d log|Hp|/dρ_i = tr(Hp⁻¹ dS_i) + tr(Hp⁻¹ M_i)
//! ```
//! where `dS_i` is the dense (tp×tp) matrix carrying `λ_i S0_i` in block `b(i)`
//! (zero elsewhere) and `M_i = d(−∇²ℓ_data)/dρ_i`. Since `−∇²ℓ_data` depends on
//! ρ only through β̂, implicit differentiation of the inner stationarity
//! condition gives `dβ̂/dρ_i = −Hp⁻¹·(dS_i·β̂)`, hence
//! `dη_c[obs]/dρ_i = X_c[obs]·(dβ̂/dρ_i)_block_c`, and
//! ```text
//!   M_i[block a, block b] = −Σ_obs X_a[obs]ᵀ X_b[obs]·(Σ_c l3_eta[a,b,c]·dη_c/dρ_i)
//! ```
//! i.e. the same block assembly as `penalized_grad_hess`'s `−∇²ℓ_data`, but
//! weighting each observation's 4×4 by `Σ_c l3_eta[a,b,c]·dη_c/dρ_i` instead of
//! `l2_eta[a,b]`. The traces are taken with `CholeskySolver::trace_a_inv`
//! against the `Hp` factorisation. [`reml_grad`] is FD-gated against
//! `reml_eval(...).laml` (`reml_grad_matches_finite_difference`) and drives the
//! [`fit_reml`] outer optimiser (the FD Hessian from `fd_grad_hess` is retained
//! only for the Newton step).

use ndarray::{Array1, Array2};

use super::shash::{ShashDensity, L3_INDEX};
use super::shash_inner::{
    fit_inner, penalized_grad_hess, penalized_loglik, ShashBlocks, ShashInnerOpts,
};
use crate::error::{GamrsError, Result};
use crate::inner::{CholeskySolver, LinearSolver};

/// One unscaled penalty `S0` plus its (caller-supplied) rank.
///
/// `rank` is the rank of the *unscaled* penalty `s0`; it is a plain input —
/// we deliberately do NOT recompute it via an eigendecomposition here. For a
/// 2nd-difference penalty on a length-`k` block this is `k − 2`; for a full-
/// rank ridge it is `k`; for a single penalised direction it is `1`.
#[derive(Clone, Debug)]
pub struct ShashPenalty {
    /// Unscaled penalty matrix `S0` (`p_k × p_k` for block `k`).
    pub s0: Array2<f64>,
    /// Rank of `s0` (number of strictly-positive eigenvalues). Caller-supplied.
    pub rank: usize,
}

/// The outer REML problem: the four per-predictor designs, an optional single
/// penalty per block (`None` = unpenalised, e.g. an intercept-only block), and
/// the `logeb` bound `b` for the τ link.
///
/// The smoothing-parameter vector ρ has one entry per `Some` penalty, in block
/// order `0..4`; [`ShashProblem::combined_penalties`] maps a ρ slice back to the
/// four `S_ρ` blocks that [`ShashBlocks`] consumes.
#[derive(Clone, Debug)]
pub struct ShashProblem {
    /// Per-block design matrices `Xᵦ`, each `n×pᵦ`.
    pub x: [Array2<f64>; 4],
    /// The response vector `y` (length `n`).
    pub y: Array1<f64>,
    /// Per-block penalty (`None` for an unpenalised block).
    pub penalties: [Option<ShashPenalty>; 4],
    /// `logeb` bound `b` for the τ link (mgcv default 1e-2).
    pub b: f64,
}

impl ShashProblem {
    /// Per-block coefficient counts `[p₁,p₂,p₃,p₄]`.
    pub fn p(&self) -> [usize; 4] {
        [
            self.x[0].ncols(),
            self.x[1].ncols(),
            self.x[2].ncols(),
            self.x[3].ncols(),
        ]
    }

    /// Total coefficient count `Σ pᵦ`.
    pub fn total_p(&self) -> usize {
        self.p().iter().sum()
    }

    /// Number of smoothing parameters = number of penalised blocks. ρ has this
    /// length and is laid out in block order `0..4` over the `Some` penalties.
    pub fn n_sp(&self) -> usize {
        self.penalties.iter().filter(|p| p.is_some()).count()
    }

    /// Per-smoothing-parameter upper bound on ρ, in ρ order: where `exp(ρ)·S0`
    /// starts swamping `X'X` in the factorisation. Past
    /// `ln((grad_tol / eps) · max diag X'X / max diag S0)` the smooth's null-space
    /// directions come out of a cancellation of order `eps·λ·max S0 / max X'X`,
    /// and the criterion is round-off: a fit pinned at ρ = 30 moved its laml by
    /// 2.8e-3 when y was scaled by (1 + 1e-10). Same bound as the scat outer
    /// (`ShapeAwareEnvelopeScore::axis_bounds`), capped at the old ±30.
    pub fn rho_upper_bounds(&self, rho_abs_max: f64) -> Vec<f64> {
        let cond_limit = crate::outer::NewtonOpts::default().grad_tol / f64::EPSILON;
        (0..4)
            .filter_map(|k| {
                let pen = self.penalties[k].as_ref()?;
                let x = &self.x[k];
                let xtx = (0..x.ncols())
                    .map(|c| x.column(c).mapv(|v| v * v).sum())
                    .fold(f64::MIN_POSITIVE, f64::max);
                let s = pen.s0.diag().iter().cloned().fold(0.0_f64, f64::max);
                Some(if s > 0.0 {
                    (cond_limit * xtx / s).ln().min(rho_abs_max)
                } else {
                    rho_abs_max
                })
            })
            .collect()
    }

    /// Build the four λ-combined penalty blocks `S_ρ[k]` from a ρ slice:
    /// `exp(ρ_i)·s0` for penalised block `k` (consuming ρ in block order), a
    /// `p_k×p_k` zero matrix for an unpenalised block. Returns them owned so the
    /// caller can build the borrowed [`ShashBlocks`] views.
    ///
    /// Panics if `rho.len()` differs from [`Self::n_sp`].
    pub fn combined_penalties(&self, rho: &[f64]) -> [Array2<f64>; 4] {
        assert_eq!(
            rho.len(),
            self.n_sp(),
            "rho length {} must equal the number of penalised blocks {}",
            rho.len(),
            self.n_sp()
        );
        let p = self.p();
        let mut out: [Array2<f64>; 4] = [
            Array2::zeros((p[0], p[0])),
            Array2::zeros((p[1], p[1])),
            Array2::zeros((p[2], p[2])),
            Array2::zeros((p[3], p[3])),
        ];
        let mut ri = 0usize;
        for k in 0..4 {
            if let Some(pen) = &self.penalties[k] {
                let lam = rho[ri].exp();
                out[k] = pen.s0.mapv(|v| v * lam);
                ri += 1;
            }
        }
        out
    }

    /// Build the borrowed [`ShashBlocks`] for given owned penalty blocks `s`.
    fn blocks<'a>(&'a self, s: &'a [Array2<f64>; 4]) -> ShashBlocks<'a> {
        ShashBlocks {
            x: [
                self.x[0].view(),
                self.x[1].view(),
                self.x[2].view(),
                self.x[3].view(),
            ],
            s: [s[0].view(), s[1].view(), s[2].view(), s[3].view()],
            b: self.b,
        }
    }
}

/// The result of one REML evaluation at a fixed ρ.
#[derive(Clone, Debug)]
pub struct ShashRemlEval {
    /// Penalised MLE β̂(ρ) (flat, block `b` at the [`ShashBlocks`] offset).
    pub beta: Array1<f64>,
    /// The LAML / REML criterion `V(ρ)` (up to the omitted ρ-independent
    /// `½ Σ_b log|S0_b|₊` constant — see the module doc).
    pub laml: f64,
    /// Effective degrees of freedom `total_p − tr(Hp⁻¹ S_ρ)`.
    pub edf: f64,
    /// Unpenalised log-likelihood `ℓ(β̂) = Σᵢ ℓ₀(...)`.
    pub loglik: f64,
    /// `log|Hp|` at β̂ (`Hp = −∇²(penalised loglik)`).
    pub log_det_hp: f64,
    /// Whether the inner penalised Newton converged at this ρ.
    pub inner_converged: bool,
}

/// Cholesky-factorise the penalised Hessian `Hp = −∇²(penalised loglik)`,
/// inflating its diagonal by a growing `τ` until it is SPD — mgcv's "ensure
/// negative-definiteness" step for the Laplace approximation. Returns the
/// factor and the `τ` used (0 when `Hp` was already SPD).
///
/// `Hp` is SPD at a clean penalised maximum, but the outer REML search probes
/// ρ values (and warm-starts) where the *observed* Hessian can be indefinite or
/// merely positive-semidefinite — most notably on near-Gaussian data, where the
/// skewness/kurtosis directions are weakly identified and carry ~0 curvature.
/// Erroring there would abort an otherwise-fine fit; a minimal perturbation
/// (tiny `τ`, negligible effect on `log|Hp|`/EDF) lets the criterion stay finite
/// so the outer optimiser can step away from / settle near such points.
fn factorize_hp_spd(hp: Array2<f64>, tp: usize) -> Result<(Array2<f64>, f64)> {
    let diag_max = (0..tp).map(|i| hp[[i, i]].abs()).fold(1.0_f64, f64::max);
    let mut tau = 0.0_f64;
    for _ in 0..40 {
        let mut a = hp.clone();
        if tau > 0.0 {
            for i in 0..tp {
                a[[i, i]] += tau;
            }
        }
        if let Ok(fact) = CholeskySolver::factorize(a) {
            return Ok((fact, tau));
        }
        tau = if tau == 0.0 {
            1e-8 * diag_max
        } else {
            tau * 10.0
        };
    }
    Err(GamrsError::SingularSystem(
        "shash REML: −Hess (Hp) could not be stabilised to SPD even with a large \
         diagonal perturbation"
            .into(),
    ))
}

/// Evaluate the REML / LAML criterion `V(ρ)` at a fixed ρ.
///
/// Steps (mgcv `gam.fit4.r:1408-1414`):
///   1. build the four `S_ρ` blocks from ρ → [`ShashBlocks`];
///   2. `fit_inner` → β̂ (warm-started from `beta0`);
///   3. recompute `(grad, hess) = penalized_grad_hess` at β̂; form `Hp = −hess`;
///   4. factorise `Hp`, perturbing to SPD if needed ([`factorize_hp_spd`]);
///   5. `pen_quad = Σ_b β_bᵀ S_ρ_b β_b`,
///      `loglik = penalized_loglik(β̂) + ½ pen_quad`,
///      `ldetS_dep = Σ_b rank_b·ρ_b`, `Mp = total_p − Σ rank_b`,
///      `laml = loglik − ½ pen_quad + ½ ldetS_dep − ½ log|Hp| + (Mp/2)·ln(2π)`,
///      `edf = total_p − tr(Hp⁻¹ S_ρ)`.
pub fn reml_eval(
    density: &ShashDensity,
    problem: &ShashProblem,
    rho: &[f64],
    beta0: ndarray::ArrayView1<f64>,
    inner_opts: ShashInnerOpts,
) -> Result<ShashRemlEval> {
    let y = problem.y.view();
    let s = problem.combined_penalties(rho);
    let blocks = problem.blocks(&s);
    let tp = blocks.total_p();

    // Inner penalised MLE at this ρ.
    let fit = fit_inner(density, &blocks, beta0, y, inner_opts)?;
    let beta = fit.beta;

    // Penalised loglik and its Hessian at β̂; Hp = −∇²(penalised loglik).
    let pen_loglik = penalized_loglik(density, &blocks, beta.view(), y);
    let (_grad, hess) = penalized_grad_hess(density, &blocks, beta.view(), y);
    let hp = hess.mapv(|v| -v);
    let (hp_fact, _tau) = factorize_hp_spd(hp, tp)?;
    let log_det_hp = CholeskySolver::logdet(&hp_fact);

    // Penalty quadratic ½ β̂ᵀ S_ρ β̂ → recover the unpenalised loglik.
    let mut pen_quad = 0.0;
    let off = [
        blocks.offset(0),
        blocks.offset(1),
        blocks.offset(2),
        blocks.offset(3),
    ];
    let pblk = blocks.p();
    for b in 0..4 {
        let beta_b = beta.slice(ndarray::s![off[b]..off[b] + pblk[b]]);
        let sbeta = s[b].dot(&beta_b);
        pen_quad += beta_b.dot(&sbeta);
    }
    let loglik = pen_loglik + 0.5 * pen_quad;

    // ρ-dependent part of ½ log|S_ρ|₊ and the null-space dimension.
    // (The ρ-independent ½ Σ_b log|S0_b|₊ constant is omitted — see module doc.)
    let mut ldet_s_dep = 0.0;
    let mut rank_sum = 0usize;
    let mut ri = 0usize;
    for k in 0..4 {
        if let Some(pen) = &problem.penalties[k] {
            ldet_s_dep += pen.rank as f64 * rho[ri];
            rank_sum += pen.rank;
            ri += 1;
        }
    }
    let mp = tp - rank_sum;

    let laml = loglik - 0.5 * pen_quad + 0.5 * ldet_s_dep - 0.5 * log_det_hp
        + 0.5 * mp as f64 * (2.0 * std::f64::consts::PI).ln();

    // Build the flat S_ρ matrix once for the EDF trace.
    let mut s_rho = Array2::<f64>::zeros((tp, tp));
    for b in 0..4 {
        for r in 0..pblk[b] {
            for c in 0..pblk[b] {
                s_rho[[off[b] + r, off[b] + c]] = s[b][[r, c]];
            }
        }
    }
    let edf = tp as f64 - CholeskySolver::trace_a_inv(&hp_fact, s_rho.view());

    Ok(ShashRemlEval {
        beta,
        laml,
        edf,
        loglik,
        log_det_hp,
        inner_converged: fit.converged,
    })
}

/// Analytic REML / LAML gradient `∂V/∂ρ` (length [`ShashProblem::n_sp`]) at a
/// converged β̂.
///
/// `beta_hat` MUST be the penalised MLE returned by [`reml_eval`] at this same
/// ρ — then `Hp = −∇²(penalised loglik)` is SPD and the implicit-function
/// derivative `dβ̂/dρ_i = −Hp⁻¹(dS_i β̂)` is well defined. See the module doc
/// (and mgcv `gam.fit4.r:1346-1365,1408-1413`) for the full derivation; in
/// brief, per smoothing parameter `i` (penalised block `b(i)`):
/// ```text
///   ∂V/∂ρ_i = −½ λ_i β̂_b(i)ᵀ S0_b(i) β̂_b(i) + ½ rank_i − ½ d log|Hp|/dρ_i
///   d log|Hp|/dρ_i = tr(Hp⁻¹ dS_i) + tr(Hp⁻¹ M_i)
/// ```
/// with `dS_i = λ_i S0_i` embedded in block `b(i)` and `M_i` the third-
/// derivative-weighted data-Hessian sensitivity (assembled exactly like
/// `penalized_grad_hess`'s `−∇²ℓ_data`, but per-obs weight
/// `Σ_c l3_eta[a,b,c]·dη_c/dρ_i`).
pub fn reml_grad(
    density: &ShashDensity,
    problem: &ShashProblem,
    rho: &[f64],
    beta_hat: ndarray::ArrayView1<f64>,
) -> Result<Array1<f64>> {
    let n_sp = problem.n_sp();
    assert_eq!(
        rho.len(),
        n_sp,
        "rho length {} must equal the number of penalised blocks {}",
        rho.len(),
        n_sp
    );
    let s = problem.combined_penalties(rho);
    let blocks = problem.blocks(&s);
    let y = problem.y.view();
    let tp = blocks.total_p();
    let p = blocks.p();
    let off = [
        blocks.offset(0),
        blocks.offset(1),
        blocks.offset(2),
        blocks.offset(3),
    ];

    // Hp = −∇²(penalised loglik) at β̂; SPD near the inner optimum, perturbed to
    // SPD otherwise (matches `reml_eval` — the gradient must stay defined wherever
    // the criterion is, e.g. near-Gaussian data with weakly-identified ε/φ).
    let (_grad, hess) = penalized_grad_hess(density, &blocks, beta_hat, y);
    let hp = hess.mapv(|v| -v);
    let (hp_fact, _tau) = factorize_hp_spd(hp, tp)?;

    // Linear predictors at β̂ (shared across all smoothing parameters).
    let eta = blocks.eta(beta_hat);

    // Map each smoothing-parameter index i → its penalised block k and the
    // corresponding unscaled penalty S0 and rank.
    let mut grad = Array1::<f64>::zeros(n_sp);
    let mut i = 0usize; // smoothing-parameter index
    for k in 0..4 {
        let Some(pen) = &problem.penalties[k] else {
            continue;
        };
        let lam = rho[i].exp();
        let pk = p[k];
        let offk = off[k];
        let beta_k = beta_hat.slice(ndarray::s![offk..offk + pk]);

        // (1) Penalty-quadratic term: −½ λ_i β̂_kᵀ S0_k β̂_k.
        let s0beta = pen.s0.dot(&beta_k);
        let pen_quad = lam * beta_k.dot(&s0beta);
        let mut gi = -0.5 * pen_quad;

        // (2) Rank term: +½ rank_i.
        gi += 0.5 * pen.rank as f64;

        // (3) −½ d log|Hp|/dρ_i = −½ [tr(Hp⁻¹ dS_i) + tr(Hp⁻¹ M_i)].
        //
        // dS_i: the dense (tp×tp) penalty derivative, λ_i S0_k in block k.
        let mut ds_i = Array2::<f64>::zeros((tp, tp));
        for r in 0..pk {
            for c in 0..pk {
                ds_i[[offk + r, offk + c]] = lam * pen.s0[[r, c]];
            }
        }
        let tr_ds = CholeskySolver::trace_a_inv(&hp_fact, ds_i.view());

        // dβ̂/dρ_i = −Hp⁻¹ (dS_i · β̂). (rhs is the full vector dS_i·β̂; only
        // block k of β̂ contributes since dS_i is block-k only.)
        let mut rhs = Array1::<f64>::zeros(tp);
        for r in 0..pk {
            rhs[offk + r] = lam * s0beta[r];
        }
        let dbeta = CholeskySolver::solve(&hp_fact, rhs.view()).mapv(|v| -v);

        // dη_c[obs]/dρ_i = X_c[obs] · (dβ̂/dρ_i)_block_c   →  n×4.
        let n = blocks.n();
        let mut deta = Array2::<f64>::zeros((n, 4));
        for (c, &pc) in p.iter().enumerate() {
            let dbeta_c = dbeta.slice(ndarray::s![off[c]..off[c] + pc]);
            let de_c = blocks.x[c].dot(&dbeta_c);
            deta.column_mut(c).assign(&de_c);
        }

        // M_i[block a, block b] = −Σ_obs X_a[obs]ᵀ X_b[obs] · W_i[obs][a,b],
        // with W_i[obs][a,b] = Σ_c l3_eta[obs][a,b,c]·dη_c[obs]/dρ_i.
        let mut m_i = Array2::<f64>::zeros((tp, tp));
        for obs in 0..n {
            let eta_obs = [eta[[obs, 0]], eta[[obs, 1]], eta[[obs, 2]], eta[[obs, 3]]];
            let l3_packed = density.l3_eta(y[obs], eta_obs, blocks.b);
            // Unpack l3_eta → dense symmetric 4×4×4.
            let mut l3 = [[[0.0_f64; 4]; 4]; 4];
            for (idx, &(a, bb, cc)) in L3_INDEX.iter().enumerate() {
                let v = l3_packed[idx];
                // 6 permutations (duplicates collapse on assignment).
                l3[a][bb][cc] = v;
                l3[a][cc][bb] = v;
                l3[bb][a][cc] = v;
                l3[bb][cc][a] = v;
                l3[cc][a][bb] = v;
                l3[cc][bb][a] = v;
            }
            let deta_obs = [
                deta[[obs, 0]],
                deta[[obs, 1]],
                deta[[obs, 2]],
                deta[[obs, 3]],
            ];
            // W[a][b] = Σ_c l3[a][b][c]·dη_c, then M block += −X_aᵀ X_b · W[a][b].
            for a in 0..4 {
                let xa_row = blocks.x[a].row(obs);
                for b in 0..4 {
                    let mut w = 0.0;
                    for c in 0..4 {
                        w += l3[a][b][c] * deta_obs[c];
                    }
                    if w == 0.0 {
                        continue;
                    }
                    let w = -w; // the leading minus of −∇²ℓ_data.
                    let xb_row = blocks.x[b].row(obs);
                    for r in 0..p[a] {
                        let xar_w = xa_row[r] * w;
                        let mrow = off[a] + r;
                        for sc in 0..p[b] {
                            m_i[[mrow, off[b] + sc]] += xar_w * xb_row[sc];
                        }
                    }
                }
            }
        }
        let tr_m = CholeskySolver::trace_a_inv(&hp_fact, m_i.view());

        gi -= 0.5 * (tr_ds + tr_m);
        grad[i] = gi;
        i += 1;
    }

    Ok(grad)
}

/// Options for the outer (smoothing-parameter) REML ascent.
#[derive(Clone, Copy, Debug)]
pub struct ShashRemlOpts {
    /// Maximum outer (ρ-space) iterations.
    pub max_iter: usize,
    /// Converge when `‖g_fd‖∞ < grad_tol`.
    pub grad_tol: f64,
    /// Central-difference step in ρ for the FD gradient/Hessian.
    pub fd_h: f64,
    /// Also converge when an accepted ρ-step has `‖Δρ‖∞ < step_tol`.
    pub step_tol: f64,
    /// Maximum step-halvings per outer iteration.
    pub max_halvings: usize,
    /// Inner-solve options used at every ρ probe.
    pub inner_opts: ShashInnerOpts,
}

impl Default for ShashRemlOpts {
    fn default() -> Self {
        Self {
            max_iter: 50,
            grad_tol: 1e-4,
            fd_h: 1e-3,
            step_tol: 1e-6,
            max_halvings: 30,
            inner_opts: ShashInnerOpts::default(),
        }
    }
}

/// The result of the outer REML ascent.
#[derive(Clone, Debug)]
pub struct ShashRemlFit {
    /// Selected log-smoothing-parameters ρ̂.
    pub rho: Array1<f64>,
    /// The REML evaluation (β̂, laml, edf, …) at ρ̂.
    pub eval: ShashRemlEval,
    /// Outer iterations taken.
    pub n_iter: usize,
    /// Whether an outer convergence criterion was met (vs hitting `max_iter`).
    pub converged: bool,
}

/// Central finite-difference gradient and Hessian of `laml(ρ)`, plus the β̂ at
/// the centre (warm-started forward to subsequent solves). Returns
/// `(laml_centre, grad, hess, beta_centre, inner_converged_centre)`.
#[allow(clippy::type_complexity)]
fn fd_grad_hess(
    density: &ShashDensity,
    problem: &ShashProblem,
    rho: &[f64],
    beta0: ndarray::ArrayView1<f64>,
    opts: &ShashRemlOpts,
) -> Result<(f64, Array1<f64>, Array2<f64>, Array1<f64>, bool)> {
    let d = rho.len();
    let h = opts.fd_h;

    let centre = reml_eval(density, problem, rho, beta0, opts.inner_opts)?;
    let v0 = centre.laml;
    let beta_centre = centre.beta.clone();
    // Warm-start every probe from the centre β̂ for stability/speed.
    let warm = centre.beta.view();

    // Cache the four diagonal V(ρ ± h e_i) evaluations for the Hessian diagonal.
    let mut v_plus = vec![0.0_f64; d];
    let mut v_minus = vec![0.0_f64; d];
    let mut grad = Array1::<f64>::zeros(d);
    for i in 0..d {
        let mut rp = rho.to_vec();
        let mut rm = rho.to_vec();
        rp[i] += h;
        rm[i] -= h;
        let ep = reml_eval(density, problem, &rp, warm, opts.inner_opts)?;
        let em = reml_eval(density, problem, &rm, warm, opts.inner_opts)?;
        v_plus[i] = ep.laml;
        v_minus[i] = em.laml;
        grad[i] = (ep.laml - em.laml) / (2.0 * h);
    }

    let mut hess = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        // ∂²V/∂ρ_i² ≈ (V₊ − 2V₀ + V₋)/h².
        hess[[i, i]] = (v_plus[i] - 2.0 * v0 + v_minus[i]) / (h * h);
        for j in (i + 1)..d {
            // Mixed: (V(+,+) − V(+,−) − V(−,+) + V(−,−)) / (4h²).
            let mut rpp = rho.to_vec();
            let mut rpm = rho.to_vec();
            let mut rmp = rho.to_vec();
            let mut rmm = rho.to_vec();
            rpp[i] += h;
            rpp[j] += h;
            rpm[i] += h;
            rpm[j] -= h;
            rmp[i] -= h;
            rmp[j] += h;
            rmm[i] -= h;
            rmm[j] -= h;
            let vpp = reml_eval(density, problem, &rpp, warm, opts.inner_opts)?.laml;
            let vpm = reml_eval(density, problem, &rpm, warm, opts.inner_opts)?.laml;
            let vmp = reml_eval(density, problem, &rmp, warm, opts.inner_opts)?.laml;
            let vmm = reml_eval(density, problem, &rmm, warm, opts.inner_opts)?.laml;
            let mixed = (vpp - vpm - vmp + vmm) / (4.0 * h * h);
            hess[[i, j]] = mixed;
            hess[[j, i]] = mixed;
        }
    }

    Ok((v0, grad, hess, beta_centre, centre.inner_converged))
}

#[inline]
fn inf_norm(v: ndarray::ArrayView1<f64>) -> f64 {
    v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

/// Solve `(−H_fd + τI) Δρ = g_fd` for the ascent direction, inflating the
/// diagonal by a growing `τ` (Levenberg-style) until the Cholesky succeeds —
/// mirrors [`crate::gamlss::shash_inner`]'s `solve_ascent`, here in ρ-space on
/// the small (≤4×4) FD Hessian. If `−H_fd` is already SPD this is the Newton
/// step; otherwise it blends toward gradient ascent.
fn solve_ascent_rho(hess: &Array2<f64>, grad: ndarray::ArrayView1<f64>, d: usize) -> Array1<f64> {
    let neg_h = hess.mapv(|v| -v);
    let diag_max = (0..d).map(|i| neg_h[[i, i]].abs()).fold(1.0_f64, f64::max);
    let mut tau = 0.0_f64;
    for _ in 0..32 {
        let mut a = neg_h.clone();
        if tau > 0.0 {
            for i in 0..d {
                a[[i, i]] += tau;
            }
        }
        if let Ok(fact) = CholeskySolver::factorize(a) {
            return CholeskySolver::solve(&fact, grad);
        }
        tau = if tau == 0.0 {
            1e-8 * diag_max
        } else {
            tau * 10.0
        };
    }
    // Fallback: pure gradient ascent (scaled). Should be unreachable given the
    // growing τ; we never error out of the outer loop on this.
    grad.to_owned()
}

/// Maximise the REML / LAML criterion `V(ρ)` over the log-smoothing-parameters
/// ρ with a compact damped-Newton ascent. The gradient is ANALYTIC
/// ([`reml_grad`], phase 5b); the Hessian is still the central-difference
/// `fd_grad_hess` Hessian (step `opts.fd_h`) used only to shape the Newton step.
///
/// Each outer step solves `(−H_fd)Δ = g_analytic` (with a growing diagonal bump
/// if `−H_fd` is not SPD, [`solve_ascent_rho`]) and accepts the full/halved step
/// only if `laml` strictly increases — otherwise the step is halved up to
/// `max_halvings`. Converges on `‖g_analytic‖∞ < grad_tol`, a tiny accepted
/// step, or `max_iter`. Inner solves are warm-started from the previous β̂.
pub fn fit_reml(
    density: &ShashDensity,
    problem: &ShashProblem,
    rho0: &[f64],
    beta0: ndarray::ArrayView1<f64>,
    opts: ShashRemlOpts,
) -> Result<ShashRemlFit> {
    let d = problem.n_sp();
    assert_eq!(
        rho0.len(),
        d,
        "rho0 length {} must equal the number of penalised blocks {}",
        rho0.len(),
        d
    );

    let mut rho = rho0.to_vec();
    let mut beta_warm = beta0.to_owned();
    let mut converged = false;
    let mut n_iter = 0usize;

    // Current value cache; recomputed each accepted step.
    let mut v_cur = {
        let e = reml_eval(density, problem, &rho, beta_warm.view(), opts.inner_opts)?;
        beta_warm = e.beta.clone();
        e.laml
    };

    let rho_hi = problem.rho_upper_bounds(30.0);
    while n_iter < opts.max_iter {
        n_iter += 1;
        // FD Hessian (for the Newton step) + the converged centre β̂; the FD
        // gradient it also returns is discarded — we use the ANALYTIC gradient
        // [`reml_grad`] at the centre β̂ instead (phase 5b).
        let (v0, _grad_fd, hess, beta_centre, _conv) =
            fd_grad_hess(density, problem, &rho, beta_warm.view(), &opts)?;
        beta_warm = beta_centre;
        v_cur = v0;

        // Analytic ∂V/∂ρ at the centre (FD-gated to ~1e-8 vs FD-of-laml).
        let grad = reml_grad(density, problem, &rho, beta_warm.view())?;

        let gnorm = inf_norm(grad.view());
        if gnorm < opts.grad_tol {
            converged = true;
            break;
        }

        let delta = solve_ascent_rho(&hess, grad.view(), d);

        // Step-halving line search: accept the first step that increases laml.
        // ρ is CLAMPED to a sane range: beyond `RHO_ABS_MAX` the penalty
        // `exp(ρ)·S0` overflows the penalised Hessian into finite garbage (a huge
        // bogus `laml` the search would wrongly accept — the v0.12.0 over-
        // smoothing-to-linear bug). exp(±30) already spans "no penalty" to
        // "smooth fully collapsed to its null space (EDF→Mp)", so clamping there
        // never excludes a meaningful optimum; mgcv likewise bounds sp. The upper
        // end is tighter still: `rho_hi`, where the factorisation stops being
        // computable (`ShashProblem::rho_upper_bounds`).
        const RHO_ABS_MAX: f64 = 30.0;
        let mut t = 1.0_f64;
        let mut accepted = false;
        for _ in 0..=opts.max_halvings {
            let trial: Vec<f64> = (0..d)
                .map(|i| (rho[i] + delta[i] * t).clamp(-RHO_ABS_MAX, rho_hi[i]))
                .collect();
            let e = reml_eval(density, problem, &trial, beta_warm.view(), opts.inner_opts)?;
            // Reject non-finite criteria (overflow guard, belt-and-braces).
            if e.laml.is_finite() && e.laml > v_cur {
                let step = (0..d).map(|i| (delta[i] * t).abs()).fold(0.0_f64, f64::max);
                rho = trial;
                v_cur = e.laml;
                beta_warm = e.beta.clone();
                accepted = true;
                if step < opts.step_tol {
                    converged = true;
                }
                break;
            }
            t *= 0.5;
        }
        if !accepted {
            // No increase even at the smallest step — a stationary point within
            // the FD resolution.
            converged = true;
            break;
        }
        if converged {
            break;
        }
    }

    let eval = reml_eval(density, problem, &rho, beta_warm.view(), opts.inner_opts)?;
    let _ = v_cur;
    Ok(ShashRemlFit {
        rho: Array1::from(rho),
        eval,
        n_iter,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, Array1, Array2};

    fn frac(v: f64) -> f64 {
        v - v.floor()
    }
    fn pnormal(u1: f64, u2: f64) -> f64 {
        (-2.0 * u1.max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// A synthetic shash REML problem: μ block = intercept + 2 smooth-ish
    /// covariates, τ block = intercept + 1 covariate, ε/φ intercept-only. ONE
    /// SPD penalty on the μ and τ blocks (a ridge on the non-intercept
    /// directions, with a zero null direction on the intercept ⇒ rank = p−1),
    /// none on ε/φ. Returns the problem (response embedded) and a flat β
    /// warm-start.
    fn make_problem() -> (ShashProblem, Array1<f64>) {
        let n = 80usize;
        let pw = [3usize, 2, 1, 1];
        let mut x: [Array2<f64>; 4] = [
            Array2::zeros((n, pw[0])),
            Array2::zeros((n, pw[1])),
            Array2::zeros((n, pw[2])),
            Array2::zeros((n, pw[3])),
        ];
        for i in 0..n {
            let c1 = frac((i as f64 + 0.5) * 0.6180339887498949);
            let c2 = frac((i as f64 + 0.5) * 0.3819660112501051 + 0.137);
            x[0][[i, 0]] = 1.0;
            x[0][[i, 1]] = c1;
            x[0][[i, 2]] = c2 - 0.5;
            x[1][[i, 0]] = 1.0;
            x[1][[i, 1]] = c1 - 0.5;
            x[2][[i, 0]] = 1.0;
            x[3][[i, 0]] = 1.0;
        }
        // μ penalty: rank-2 diag penalising the two non-intercept directions,
        // null on the intercept (so rank = 2 < p = 3).
        let s0_mu = arr2(&[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        // τ penalty: rank-1 penalising the single non-intercept direction.
        let s0_tau = arr2(&[[0.0, 0.0], [0.0, 1.0]]);
        let penalties = [
            Some(ShashPenalty { s0: s0_mu, rank: 2 }),
            Some(ShashPenalty {
                s0: s0_tau,
                rank: 1,
            }),
            None,
            None,
        ];

        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let c1 = x[0][[i, 1]];
            let mu = 0.4 + 1.1 * c1 + 0.6 * x[0][[i, 2]];
            let sg = (0.1 + 0.3 * x[1][[i, 1]]).exp() * 0.4;
            let u1 = frac((i as f64 + 0.5) * 0.7548776662466927 + 0.31);
            let u2 = frac((i as f64 + 0.5) * 0.5698402909980532 + 0.59);
            y[i] = mu + sg * pnormal(u1, u2);
        }

        let problem = ShashProblem {
            x,
            y,
            penalties,
            b: 1e-2,
        };
        // A reasonable flat warm-start (block widths 3,2,1,1 = 7).
        let beta0 = Array1::from(vec![0.4, 1.0, 0.5, 0.1, 0.2, 0.0, 0.0]);
        (problem, beta0)
    }

    /// FD gradient of `laml` w.r.t. each ρ at a given point (central diff).
    fn fd_laml_grad(
        density: &ShashDensity,
        problem: &ShashProblem,
        rho: &[f64],
        beta0: ndarray::ArrayView1<f64>,
        h: f64,
    ) -> Vec<f64> {
        let opts = ShashInnerOpts::default();
        let mut g = vec![0.0; rho.len()];
        for i in 0..rho.len() {
            let mut rp = rho.to_vec();
            let mut rm = rho.to_vec();
            rp[i] += h;
            rm[i] -= h;
            let vp = reml_eval(density, problem, &rp, beta0, opts).unwrap().laml;
            let vm = reml_eval(density, problem, &rm, beta0, opts).unwrap().laml;
            g[i] = (vp - vm) / (2.0 * h);
        }
        g
    }

    #[test]
    fn reml_eval_is_finite_and_edf_in_bounds() {
        let (problem, beta0) = make_problem();
        let d = ShashDensity::default();
        let tp = problem.total_p();
        // Mp = total_p − Σ rank = 7 − (2 + 1) = 4.
        let mp = tp - 3;
        for rho in [vec![-1.0, 0.5], vec![1.5, -0.5], vec![0.0, 0.0]] {
            let e = reml_eval(&d, &problem, &rho, beta0.view(), ShashInnerOpts::default())
                .expect("reml_eval");
            assert!(e.laml.is_finite(), "laml not finite at {rho:?}: {}", e.laml);
            assert!(e.edf.is_finite(), "edf not finite at {rho:?}: {}", e.edf);
            assert!(e.inner_converged, "inner did not converge at {rho:?}");
            assert!(
                e.edf > mp as f64 && e.edf < tp as f64 + 1e-6,
                "edf {} not in (Mp={}, total_p={}) at {rho:?}",
                e.edf,
                mp,
                tp
            );
        }
    }

    #[test]
    fn fd_gradient_of_laml_is_finite_and_consistent() {
        let (problem, beta0) = make_problem();
        let d = ShashDensity::default();
        // FD gradient finite at an arbitrary point.
        let g0 = fd_laml_grad(&d, &problem, &[0.3, -0.2], beta0.view(), 1e-3);
        for (i, gi) in g0.iter().enumerate() {
            assert!(gi.is_finite(), "FD grad[{i}] not finite: {gi}");
        }
        // At the fit_reml optimum the FD gradient is ~0.
        let fit = fit_reml(
            &d,
            &problem,
            &[0.0, 0.0],
            beta0.view(),
            ShashRemlOpts::default(),
        )
        .expect("fit_reml");
        let g = fd_laml_grad(
            &d,
            &problem,
            fit.rho.as_slice().unwrap(),
            fit.eval.beta.view(),
            1e-3,
        );
        let gnorm = g.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        assert!(
            gnorm < 1e-3,
            "FD gradient at fit_reml optimum not ~0: {gnorm} (rho={:?})",
            fit.rho
        );
    }

    #[test]
    fn reml_grad_matches_finite_difference() {
        // The core phase-5b correctness gate: analytic ∂V/∂ρ vs central FD of
        // reml_eval(...).laml. FD-of-laml itself carries ~1e-5 error (each laml
        // is a full inner solve), so the tolerance is ~1e-3. β̂ comes from a
        // converged reml_eval; every FD probe is warm-started from it.
        let (problem, beta0) = make_problem();
        let d = ShashDensity::default();
        let inner = ShashInnerOpts {
            grad_tol: 1e-10,
            ..Default::default()
        };
        let h = 1e-4;
        let mut max_err = 0.0_f64;
        for rho in [vec![0.3, -0.2], vec![-0.5, 0.8]] {
            // Converged β̂ at this ρ (the analytic gradient assumes stationarity).
            let centre = reml_eval(&d, &problem, &rho, beta0.view(), inner).expect("centre");
            let beta_hat = centre.beta.clone();
            let g_analytic = reml_grad(&d, &problem, &rho, beta_hat.view()).expect("reml_grad");

            for i in 0..rho.len() {
                let mut rp = rho.clone();
                let mut rm = rho.clone();
                rp[i] += h;
                rm[i] -= h;
                let vp = reml_eval(&d, &problem, &rp, beta_hat.view(), inner)
                    .expect("vp")
                    .laml;
                let vm = reml_eval(&d, &problem, &rm, beta_hat.view(), inner)
                    .expect("vm")
                    .laml;
                let fd = (vp - vm) / (2.0 * h);
                let err = (g_analytic[i] - fd).abs();
                max_err = max_err.max(err);
                assert!(
                    err < 1e-3,
                    "∂V/∂ρ[{i}] at rho={rho:?}: analytic {} vs FD {} (err {err:.2e})",
                    g_analytic[i],
                    fd
                );
            }
        }
        eprintln!("reml_grad vs FD max err {max_err:.2e}");
    }

    #[test]
    fn fit_reml_increases_laml() {
        let (problem, beta0) = make_problem();
        let d = ShashDensity::default();
        let rho0 = vec![0.0, 0.0];
        let v0 = reml_eval(&d, &problem, &rho0, beta0.view(), ShashInnerOpts::default())
            .unwrap()
            .laml;
        let fit = fit_reml(&d, &problem, &rho0, beta0.view(), ShashRemlOpts::default())
            .expect("fit_reml");
        assert!(
            fit.eval.laml >= v0 - 1e-9,
            "fit_reml did not increase laml: {} < {}",
            fit.eval.laml,
            v0
        );
        assert!(
            fit.converged,
            "fit_reml did not converge (iters={})",
            fit.n_iter
        );
    }

    // --- the mgcv confrontation (the real TDD oracle for this project) ------
    //
    // Feeds gamrs mgcv's OWN per-block design + penalty from a fitted 2-smooth
    // shash GAMLSS (scripts/r/gen_shash_reml_fixture.R) and confronts the
    // REML-selected smoothing params, total EDF and fitted linear predictors.

    #[derive(serde::Deserialize)]
    struct RemlFixture {
        n: usize,
        b: f64,
        log_sp: Vec<f64>,
        edf_total: f64,
        p: Vec<usize>,
        rank: Vec<usize>,
        #[serde(rename = "X1")]
        x1: Vec<f64>,
        #[serde(rename = "X2")]
        x2: Vec<f64>,
        #[serde(rename = "X3")]
        x3: Vec<f64>,
        #[serde(rename = "X4")]
        x4: Vec<f64>,
        #[serde(rename = "S1")]
        s1: Vec<f64>,
        #[serde(rename = "S2")]
        s2: Vec<f64>,
        eta: Vec<f64>,
        y: Vec<f64>,
    }

    fn load_reml_fixture() -> RemlFixture {
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests/fixtures/shash_reml_mgcv.json");
        let txt = std::fs::read_to_string(&p)
            .unwrap_or_else(|e| panic!("missing fixture {}: {e}", p.display()));
        serde_json::from_str(&txt).expect("malformed shash_reml fixture")
    }

    fn mat(flat: &[f64], rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_vec((rows, cols), flat.to_vec()).expect("shape")
    }

    #[test]
    fn fit_reml_matches_mgcv_two_smooth_shash() {
        let fx = load_reml_fixture();
        let n = fx.n;
        let x = [
            mat(&fx.x1, n, fx.p[0]),
            mat(&fx.x2, n, fx.p[1]),
            mat(&fx.x3, n, fx.p[2]),
            mat(&fx.x4, n, fx.p[3]),
        ];
        let penalties = [
            Some(ShashPenalty {
                s0: mat(&fx.s1, fx.p[0], fx.p[0]),
                rank: fx.rank[0],
            }),
            Some(ShashPenalty {
                s0: mat(&fx.s2, fx.p[1], fx.p[1]),
                rank: fx.rank[1],
            }),
            None,
            None,
        ];
        let y = Array1::from(fx.y.clone());
        let problem = ShashProblem {
            x: x.clone(),
            y: y.clone(),
            penalties,
            b: fx.b,
        };
        let d = ShashDensity::default();

        // Warm-start β from the phase-3 initialiser on mgcv's own designs.
        let init = crate::gamlss::shash_init::shash_init(
            x[0].view(),
            x[1].view(),
            x[2].view(),
            x[3].view(),
            y.view(),
            0.0,
        )
        .expect("init");
        let mut beta0 = Array1::<f64>::zeros(problem.total_p());
        let off = [0, fx.p[0], fx.p[0] + fx.p[1], fx.p[0] + fx.p[1] + fx.p[2]];
        beta0
            .slice_mut(ndarray::s![off[0]..off[0] + fx.p[0]])
            .assign(&init.beta_mu);
        beta0
            .slice_mut(ndarray::s![off[1]..off[1] + fx.p[1]])
            .assign(&init.beta_tau);

        let fit = fit_reml(
            &d,
            &problem,
            &[0.0, 0.0],
            beta0.view(),
            ShashRemlOpts::default(),
        )
        .expect("fit_reml");
        assert!(
            fit.converged,
            "outer REML did not converge (iters={})",
            fit.n_iter
        );
        assert!(
            fit.eval.inner_converged,
            "inner solve did not converge at ρ̂"
        );

        // 1) Selected log-smoothing-params ≈ mgcv's. gamrs recovers these to
        //    ~1e-5 here (the fixture is bs="cr", matching gamrs); 1e-3 leaves
        //    margin for FD-Newton + cross-platform FP.
        for k in 0..2 {
            assert!(
                (fit.rho[k] - fx.log_sp[k]).abs() < 1e-3,
                "log sp[{k}] = {} vs mgcv {} (sp ratio {:.4}x)",
                fit.rho[k],
                fx.log_sp[k],
                (fit.rho[k] - fx.log_sp[k]).exp()
            );
        }

        // 2) Total EDF ≈ mgcv's sum(edf) (same definition: p − tr(Hp⁻¹S));
        //    matches to ~5 dp in practice.
        assert!(
            (fit.eval.edf - fx.edf_total).abs() < 1e-2,
            "EDF = {} vs mgcv {}",
            fit.eval.edf,
            fx.edf_total
        );

        // 3) Fitted linear predictors ηᵦ = Xᵦ·β̂ᵦ ≈ mgcv's (we fit in mgcv's own
        //    extracted basis, so directly comparable; agrees to ~1e-6).
        let eta_mgcv = mat(&fx.eta, n, 4); // n×4
        let pblk = fx.p.clone();
        for b in 0..4 {
            let beta_b = fit
                .eval
                .beta
                .slice(ndarray::s![off[b]..off[b] + pblk[b]])
                .to_owned();
            let eta_b = x[b].dot(&beta_b);
            let mut max_abs = 0.0_f64;
            for i in 0..n {
                max_abs = max_abs.max((eta_b[i] - eta_mgcv[[i, b]]).abs());
            }
            assert!(
                max_abs < 1e-3,
                "fitted η block {b}: max|gamrs − mgcv| = {max_abs}",
            );
        }
    }

    #[test]
    fn edf_decreases_as_penalty_grows() {
        let (problem, beta0) = make_problem();
        let d = ShashDensity::default();
        let light = reml_eval(
            &d,
            &problem,
            &[-1.0, -1.0],
            beta0.view(),
            ShashInnerOpts::default(),
        )
        .expect("light");
        let heavy = reml_eval(
            &d,
            &problem,
            &[5.0, 5.0],
            beta0.view(),
            ShashInnerOpts::default(),
        )
        .expect("heavy");
        assert!(
            heavy.edf < light.edf,
            "EDF did not decrease as penalty grew: heavy {} vs light {}",
            heavy.edf,
            light.edf
        );
    }
}
