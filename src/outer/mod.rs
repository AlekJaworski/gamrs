//! Layer 5 — outer optimiser over θ.
//!
//! `NewtonWithHalving` implements `crate::traits::OuterSolver`. ρ-dim
//! generic via the trait surface; the concrete Phase-0 path always has
//! `dim = 1` but the same impl handles multi-θ (joint ρ + log σ²) once
//! Phase 1 wires it.
//!
//! The optimiser consumes `ScoreDerivatives::value_grad_hess` and never
//! FD-probes `value_and_grad` directly — the Hessian-FD is the score's
//! responsibility (architecture-assumptions.md §B5). This is the
//! structural defence against the closed-form-vs-FD drift bug class.

use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};

mod step;

use step::inf_norm;

use crate::error::Result;
use crate::traits::{OuterFit, OuterSolver, ScoreDerivatives};

pub struct NewtonOpts {
    pub max_iters: usize,
    /// Score-relative gradient tolerance — `|grad|_∞ < grad_tol·(|REML|+1)`.
    /// Matches mgcv's `gam.fit3.r:1644-1645` `conv.tol=1e-7` × 5 default.
    pub grad_tol: f64,
    /// Score-change tolerance — relative `|ΔREML|/|REML| < reml_tol`. Active
    /// only after iter ≥ 3, mirroring mgcv's same-line condition.
    pub reml_tol: f64,
    pub step_min: f64,
    /// Lower bound on the (positive) Hessian eigenvalues. Negative or
    /// near-zero curvature is replaced with this floor so Newton's step
    /// direction is always descent.
    pub hess_floor: f64,
    /// Cap on the L_∞ norm of a single Newton step in θ-space — keeps the
    /// outer loop from leaping into the degenerate-σ² regime.
    pub max_step: f64,
}

/// Family-facing outer-Newton tuning. The fit drivers consume this via
/// `Loss::outer_tuning()` and convert to [`NewtonOpts`]. Centralises the
/// per-family overrides mgcv ships per family (gam.fit3.r `conv.tol`,
/// `max.half`, step bounds) — defaults are mgcv-parity.
///
/// To override per family, implement `Loss::outer_tuning()` and return
/// a different struct. The base `OuterTuning::mgcv_default()` matches
/// mgcv's `conv.tol=1e-7 × 5` (gradient) and `conv.tol=1e-7` (REML
/// change) — verified against mgcv_rust smooth.rs:2545-2569.
#[derive(Debug, Clone, Copy)]
pub struct OuterTuning {
    /// Score-relative gradient tolerance: converge when
    /// `|grad|_∞ < grad_tol · (|REML| + 1)`.
    ///
    /// mgcv's extended-family outer Newton uses
    /// `conv.tol = .Machine$double.eps^.5` ≈ 1.49e-8 (`fast-REML.r:1481`),
    /// which is what `scat` goes through. The 5e-7 this used to carry was
    /// sourced from mgcv_rust (`smooth.rs:2545-2569`) — the port, not mgcv —
    /// and is 34× looser. Because the bound is scaled by the SCORE and the
    /// score is O(10²-10³) while the λ ridge can be flat to 1e-5, 5e-7 made
    /// the stopping ball wide enough to hold a real curve difference: on the
    /// real saturated-basis term the optimiser halted at |g|∞ = 3.84e-4
    /// against a bound of 4.31e-4, at a point whose score was 8.6e-6 WORSE
    /// than mgcv's. Measured against mgcv on the same (standardized) problem,
    /// tightening to `sqrt(eps)` moves the worst term from 1.09e-4 to 3.81e-5.
    ///
    /// NOTE this is DELIBERATELY ~336× tighter than mgcv's own rule and is NOT
    /// justified by parity — mgcv's `newton()` uses `conv.tol = 1e-6`
    /// (`mgcv.r:2209`) and tests `abs(grad) > score.scale*conv.tol*5`
    /// (`gam.fit3.r:1644`), i.e. `5e-6·score.scale`. The earlier citation of
    /// `fast-REML.r:1481` was wrong: that is `fast.REML.fit`, the Gaussian
    /// fREML path, not the one scat takes.
    ///
    /// It is justified EMPIRICALLY, by a sweep against mgcv 1.9.4 run directly
    /// on the ten real adjuster terms (same standardized problem, worst-term
    /// curve error):
    ///
    /// ```text
    ///   grad_tol    condition    garage_spaces    worst
    ///   5e-6 (mgcv) 1.016e-4     5.882e-4         5.882e-4
    ///   1e-6        1.016e-4     2.992e-4         2.992e-4
    ///   1e-7        1.810e-5     2.911e-5         3.809e-5   <-- the knee
    ///   1e-8        3.462e-5     3.476e-5         3.809e-5
    ///   sqrt(eps)   3.381e-5     3.476e-5         3.809e-5
    /// ```
    ///
    /// 1e-7 is the knee AND slightly better than anything tighter on both of
    /// the two worst terms, at the same iteration count — `sqrt(eps)` was
    /// 150× tighter for nothing. Below 1e-6 the worst term is pinned at
    /// 3.809e-5 by `quality`, which no tolerance moves.
    ///
    /// Costs no iterations on the scat scaling probe (identical counts at
    /// n = 500/2000/5000/10000), so this is an accuracy gain rather than a
    /// speed trade.
    pub grad_tol: f64,
    /// Score-relative REML-change tolerance: converge when
    /// `|ΔREML| / max(|REML|, 1) < reml_tol`. Active after iter ≥ 3.
    /// mgcv default 1e-7.
    pub reml_tol: f64,
    /// Cap on outer iterations before the optimiser bails out with
    /// `NotConverged`. mgcv default 200.
    pub max_iters: usize,
    /// L∞ cap on a single Newton step in θ-space. Per-axis caps from
    /// `axis_step_caps` override this when present. mgcv default 5.0.
    pub max_step: f64,
}

impl OuterTuning {
    /// mgcv R / mgcv_rust default tolerances. Match
    /// `gam.fit3.r:1644 conv.tol=1e-7` and mgcv_rust's
    /// `smooth.rs:2545-2569 reml.tol`.
    pub fn mgcv_default() -> Self {
        Self {
            grad_tol: 1.0e-7,
            reml_tol: 1.0e-7,
            max_iters: 200,
            max_step: 5.0,
        }
    }

    /// Convert to the lower-level [`NewtonOpts`] the outer Newton actually
    /// consumes. Fills in the less-commonly-tuned fields (`step_min`,
    /// `hess_floor`) with executor-side defaults.
    pub fn to_newton_opts(self) -> NewtonOpts {
        NewtonOpts {
            max_iters: self.max_iters,
            grad_tol: self.grad_tol,
            reml_tol: self.reml_tol,
            step_min: 1e-3,
            hess_floor: 1e-8,
            max_step: self.max_step,
        }
    }
}

impl Default for OuterTuning {
    fn default() -> Self {
        Self::mgcv_default()
    }
}

thread_local! {
    /// Thread-local override for the per-family [`OuterTuning`]. When
    /// `Some`, fit drivers use this in place of the family's
    /// `Loss::outer_tuning()`. Set via [`set_tuning_override`] and cleared
    /// via [`clear_tuning_override`]. Intended for the tolerance-sweep
    /// script — production callers leave it `None`.
    static OUTER_TUNING_OVERRIDE: std::cell::Cell<Option<OuterTuning>> =
        const { std::cell::Cell::new(None) };
}

/// Set the thread-local outer-tuning override. All subsequent fits on
/// this thread use the override instead of `Loss::outer_tuning()` until
/// [`clear_tuning_override`] is called.
pub fn set_tuning_override(tuning: OuterTuning) {
    OUTER_TUNING_OVERRIDE.with(|c| c.set(Some(tuning)));
}

/// Clear the thread-local outer-tuning override; subsequent fits revert
/// to `Loss::outer_tuning()`.
pub fn clear_tuning_override() {
    OUTER_TUNING_OVERRIDE.with(|c| c.set(None));
}

/// Resolve the active tuning: override if set, otherwise the family default.
/// Used by every fit driver.
pub fn resolve_tuning<L: crate::traits::Loss>(loss: &L) -> OuterTuning {
    OUTER_TUNING_OVERRIDE
        .with(|c| c.get())
        .unwrap_or_else(|| loss.outer_tuning())
}

/// Which outer optimiser to use. Maps to mgcv R's `method` argument.
///
/// - `Newton`: damped Newton on the REML score (the gamrs default — mgcv
///   `gam()` equivalent). Per-iter cost: inner PIRLS + analytic Hessian
///   assembly. Best for small-to-mid n (≤ ~10K) where the per-iter cost
///   pays off via fast convergence.
/// - `FellnerSchall`: Wood & Fasiolo (2017) multiplicative λ update
///   (mgcv `bam()`'s `fREML` equivalent). Per-iter cost: inner PIRLS + a
///   few O(p²) traces. Cheaper per-iter, slightly more iters; wins at
///   large n on GLM families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OuterAlgorithm {
    Newton,
    FellnerSchall,
}

thread_local! {
    static OUTER_ALGORITHM_OVERRIDE: std::cell::Cell<Option<OuterAlgorithm>> =
        const { std::cell::Cell::new(None) };
}

pub fn set_algorithm_override(algo: OuterAlgorithm) {
    OUTER_ALGORITHM_OVERRIDE.with(|c| c.set(Some(algo)));
}

pub fn clear_algorithm_override() {
    OUTER_ALGORITHM_OVERRIDE.with(|c| c.set(None));
}

pub fn resolved_algorithm() -> OuterAlgorithm {
    OUTER_ALGORITHM_OVERRIDE
        .with(|c| c.get())
        .unwrap_or(OuterAlgorithm::Newton)
}

/// The text every Fellner-Schall refusal starts with. The Python wrapper
/// matches on it to turn the refusal into a REML fallback — keep it in step
/// with `python/gamrs/_fitter.py::FS_UNAVAILABLE_MARKER`. The unit test at
/// the bottom of this module fails if the rendered message stops carrying it.
pub const FS_UNAVAILABLE_MARKER: &str =
    "method='fREML' (Fellner-Schall) is not implemented for the";

/// Say so when a driver that has no Fellner-Schall branch is asked for one,
/// instead of running Newton and letting the caller believe otherwise.
///
/// Fellner-Schall is a real, released solver (0.6.0) and is honoured wherever
/// it was ported: [`crate::fit::driver::fit_pirls_envelope`], the GLM envelope
/// driver, dispatches on [`resolved_algorithm`] and runs it. The port never
/// reached four other drivers — the gaussian closed-form path, the quantile
/// path, the profile-shape path (negbin) and the joint shape-parameter path
/// (scat, tweedie, ocat) — and those called `NewtonWithHalving` regardless, so
/// `method="fREML"` on them was a silent no-op. This is the loud version.
///
/// Direct Rust callers get the error. The Python wrapper catches it, warns,
/// and refits on REML, because a library that raises on a parameter it used
/// to accept breaks callers for no statistical gain: gamrs' `fREML` is the
/// Fellner-Schall optimiser (Wood & Fasiolo 2017) and mgcv's
/// `bam(method="fREML")` is the REML criterion computed the fast way — score
/// two sp vectors with `sp` pinned and bam's fREML and REML criteria return
/// identical numbers. Both are routes to the same criterion, and damped
/// Newton on the REML score is the stronger route of the two here.
pub fn reject_unsupported_algorithm(path: &str) -> crate::error::Result<()> {
    match resolved_algorithm() {
        OuterAlgorithm::Newton => Ok(()),
        OuterAlgorithm::FellnerSchall => Err(crate::error::GamrsError::InvalidParameter(format!(
            "{FS_UNAVAILABLE_MARKER} {path} fit path, which runs damped Newton on \
             the REML score instead. Pass method='REML'. Both optimise the same \
             REML criterion and Newton is generally the stronger optimiser here, \
             so this is an optimiser restriction, not a model change."
        ))),
    }
}

impl Default for NewtonOpts {
    fn default() -> Self {
        // Tolerances match mgcv's `gam.fit3.r:1644` defaults — `conv.tol=1e-7`
        // × 5 absolute on the score-relative gradient, `conv.tol=1e-7` on the
        // relative REML change. Mirrors mgcv_rust's smooth.rs:2545-2569 too.
        // Previously set to 1e-9 / 1e-10 (1000× tighter than mgcv) which
        // forced over-convergence — e.g. 10-D Gaussian took 15 outer iters
        // instead of 5-8 because the score plateaus past mgcv-relevant
        // precision. Restoring mgcv parity here keeps `rho_hat` agreement
        // to ~1e-5 (well inside the LinAlg-assembly noise floor) and
        // closes the 10-D Gaussian perf gap from 2.84× to ~1×.
        Self {
            // 200 matches v0.x's `newton_max_iter = max_outer_iter.max(200)`
            // (gam_optimized.rs:1265). Shape-aware multi-smooth fits need
            // O(80-150) iters when a λ_j saturates — at 50 we hit the
            // step-halving fallback and exit with a relaxed tolerance,
            // which produces ρ̂ differences of 1-2 units vs v0.x on the
            // saturated axis. See parity report 2026-05-27.
            max_iters: 200,
            grad_tol: 1.0e-7,
            reml_tol: 1.0e-7,
            step_min: 1e-3,
            hess_floor: 1e-8,
            max_step: 5.0,
        }
    }
}

/// Damped Newton with Wolfe-style halving. Generic over θ-dim via
/// `ScoreDerivatives` — Phase 0 always has dim=1, but the algorithm is the
/// same shape for the joint (ρ, log σ²) cases Phase 1 brings.
pub struct NewtonWithHalving {
    pub opts: NewtonOpts,
}

impl NewtonWithHalving {
    pub fn new(opts: NewtonOpts) -> Self {
        Self { opts }
    }
}

impl Default for NewtonWithHalving {
    fn default() -> Self {
        Self::new(NewtonOpts::default())
    }
}

impl OuterSolver for NewtonWithHalving {
    fn minimize<S: ScoreDerivatives>(&self, score: &S, theta0: Array1<f64>) -> Result<OuterFit> {
        let opts = &self.opts;
        let axis_caps = score.axis_step_caps();
        let axis_bounds = score.axis_bounds();
        let jump_axes = score.bound_jump_axes();
        let mut theta = theta0;
        // Clamp the initial point to any per-axis bounds — defensive
        // against `theta0` coming in from a heuristic outside the box.
        if let Some(ref bnds) = axis_bounds {
            for (i, &(lo, hi)) in bnds.iter().enumerate() {
                if i < theta.len() {
                    theta[i] = theta[i].clamp(lo, hi);
                }
            }
        }
        let (mut v, mut g, mut h) = score.value_grad_hess(&theta)?;
        let mut prev_v = f64::INFINITY;

        // Nothing to optimise. An all-parametric design has no penalties and,
        // for a fixed-dispersion family, no shape parameters either — so θ is
        // empty and there is no step to take. Return before the Newton
        // machinery, whose `active.is_empty()` safeguard would otherwise fall
        // back to `vec![0]` and index an empty gradient.
        //
        // This used to be handled incidentally: the old top-of-loop test
        // returned as soon as `grad_norm < tol`, and `inf_norm` of an empty
        // vector is 0. Adding mgcv's score-change VETO removed that accident,
        // because the veto holds convergence off for the first three iterations
        // regardless — so the empty case walked into the step builder and
        // panicked. Caught by `tests/python/test_smoke.py`'s all-parametric
        // Bernoulli case, which the Rust suite does not cover.
        if theta.is_empty() {
            return Ok(OuterFit {
                theta,
                value: v,
                grad_norm: 0.0,
                iterations: 0,
                converged: true,
            });
        }

        for iter in 0..opts.max_iters {
            if let Some(s) = score.stats() {
                s.bump_outer();
            }
            // Score-relative gradient tolerance, matching mgcv gam.fit3.r:1644.
            let score_scale = v.abs() + 1.0;
            let grad_tol_abs = opts.grad_tol * score_scale;
            let grad_norm = inf_norm(&g);
            let grad_converged = grad_norm < grad_tol_abs;
            // Score-change test — a VETO on convergence, never a trigger for
            // it. mgcv `fast-REML.r:1587-1603` sets `converged <- TRUE`, clears
            // it if any |grad| exceeds tolerance, and then clears it AGAIN if
            // the REML value is still moving (re-enabling every axis so it
            // "can't progress" otherwise). It never concludes convergence FROM
            // a small score change.
            //
            // This code had the test inverted: a small `|ΔREML|` returned
            // `converged: true`, so the loop stopped wherever the steps went
            // quiet — which on a flat λ ridge is far from the argmin. Measured
            // on the synthetic flat-ridge fixture: the criterion's argmin is at
            // ρ ≈ 21.92 and the loop was stopping at ρ ≈ 16.77, leaving 3.6e-4
            // REML units and a systematic edf overshoot on every real adjuster
            // term. The gradient test below is the one that decides.
            //
            // NB: this only helps once the criterion is right. With the
            // shipped `log|A|`, running the optimiser further makes the fit
            // WORSE (garage_spaces $577 → $807) because it converges harder
            // onto the wrong function; with the observed `log|H|` the same
            // change gives $183 → $21.
            let reml_still_moving = if iter >= 3 {
                let denom = v.abs().max(1.0);
                ((v - prev_v) / denom).abs() > opts.reml_tol
            } else {
                true
            };

            if grad_converged && !reml_still_moving {
                return Ok(OuterFit {
                    theta,
                    value: v,
                    grad_norm,
                    iterations: iter,
                    converged: true,
                });
            }

            // One iteration = build the candidate steps, probe them, accept the
            // best improvement. Everything that turns (θ, g, H, box) into a
            // candidate lives in `step.rs` so it can be tested without a score;
            // the ordering (Newton direction → project onto box → cap) is fixed
            // here and each stage documents why it sits where it does.
            let dim_tol = score_scale * step::DIM_TOL_REL;
            let active = step::active_axes(&g, &h, dim_tol);
            let raw_step = step::newton_direction(&g, &h, &active, opts.hess_floor);
            let projected = step::project_onto_box(&raw_step, &theta, axis_bounds.as_ref());
            let scaled_step = step::cap_step(&projected, axis_caps.as_ref(), opts.max_step);

            // The trial machinery is mgcv's `newton` (`gam.fit3.r:1426-1600`),
            // whose header lists four enhancements for coping with an indefinite
            // Hessian. gamrs had (i), the PSD perturbation, and (ii)/(iii). It did
            // NOT have the quadratic-model error gate that decides when the Newton
            // step is to be trusted at all, and it did not have (iv), the
            // steepest-descent trial — see the note where (iv) is declined below.
            // Both only matter when the Hessian goes indefinite, which is why 24
            // positive-definite parity fixtures never noticed; for scat / ocat /
            // tweedie an indefinite Hessian is the NORMAL state once a shape
            // parameter saturates.
            //
            // `max_half` is 30 unconditionally (`gam.fit3.r:1230`, default set at
            // `mgcv.r:2212`); the adaptive cap this used to carry — ported from
            // mgcv_rust `smooth.rs:2741-2772`, `stalled → 1 halving` — collapsed
            // the line search to a single probe exactly on the flat ridges where
            // more halving is needed, and silently made `step_min` unreachable.
            let max_half = 30;
            let (accepted, accepted_trial, accepted_v) = {
                let probe = |t: &Array1<f64>| -> Option<f64> {
                    if let Some(s) = score.stats() {
                        s.bump_line_search_trial();
                    }
                    match score.value(t) {
                        Ok(x) if x.is_finite() => Some(x),
                        _ => None,
                    }
                };
                let improves = |vt: f64| vt < v - 1e-10 * v.abs();
                let qerr_of = |s: &Array1<f64>, vt: f64| -> f64 {
                    step::quadratic_model_error(&g, &h, s, vt - v, score_scale, step::MGCV_CONV_TOL)
                };
                let mut best: Option<(Array1<f64>, f64)> = None;

                // (0) The concave-axis bound jump, probed first and exempt from
                // the quadratic-model gate — see `step::concave_bound_jump`.
                if let Some(jump) = step::concave_bound_jump(
                    &theta,
                    &g,
                    &h,
                    &scaled_step,
                    axis_bounds.as_ref(),
                    jump_axes.as_ref(),
                    dim_tol,
                ) {
                    let trial = step::clamp_to_box(&theta + &jump, axis_bounds.as_ref());
                    if let Some(vt) = probe(&trial) {
                        if improves(vt) {
                            best = Some((trial, vt));
                        }
                    }
                }

                // An accepted jump is taken as it stands rather than compared
                // against the plain Newton step: the two differ ONLY on the
                // concave axes, where the jump is at the constrained optimum and
                // Newton is a short step toward it, so the comparison would cost a
                // probe per crawling iteration to confirm what the geometry
                // already says. Measured: probing both cost ~11% wall time across
                // the sweep for the same iteration counts.
                if best.is_none() {
                    // (1) The full (modified) Newton step. mgcv tests the halving
                    // loop's condition BEFORE entering it (`gam.fit3.r:1490`), so a
                    // full step that improved the score and matched the quadratic
                    // model is never halved away.
                    let mut trial_step = scaled_step.clone();
                    let mut trial = step::clamp_to_box(&theta + &trial_step, axis_bounds.as_ref());
                    let first_ok = match probe(&trial) {
                        Some(vt)
                            if improves(vt) && qerr_of(&trial_step, vt) < step::QERROR_THRESH =>
                        {
                            best = Some((trial.clone(), vt));
                            true
                        }
                        _ => false,
                    };

                    if !first_ok {
                        // (2) mgcv's halving loop (`gam.fit3.r:1490-1552`),
                        // including its mid-loop switch to steepest descent at the
                        // fourth failed halving of an early iteration — "Newton
                        // really not working - switch to SD, but keeping step
                        // length".
                        let mut ii = 0usize;
                        let mut alpha = 1.0_f64;
                        let mut sd_used = false;
                        while ii < max_half {
                            if ii == 3 && iter < 10 && !sd_used {
                                // Built here, not per iteration: this branch is the
                                // only reader, and it needs three consecutive
                                // halvings to have already failed.
                                let s_dir = step::steepest_descent_dir(&g);
                                let len = step::l2_norm(&trial_step).min(step::MAX_S_STEP);
                                let sn = step::l2_norm(&s_dir);
                                if sn > 0.0 {
                                    trial_step = &s_dir * (len / sn);
                                    sd_used = true;
                                } else {
                                    trial_step = &trial_step * 0.5;
                                    alpha *= 0.5;
                                }
                            } else {
                                trial_step = &trial_step * 0.5;
                                alpha *= 0.5;
                            }
                            trial = step::clamp_to_box(&theta + &trial_step, axis_bounds.as_ref());
                            if let Some(vt) = probe(&trial) {
                                // mgcv `gam.fit3.r:1515` stops enforcing the gate
                                // once halving has failed more than 4 times —
                                // "don't allow step to fail altogether just
                                // because of qerror".
                                let qerr = if ii > 4 {
                                    0.0
                                } else {
                                    qerr_of(&trial_step, vt)
                                };
                                if improves(vt) && qerr < step::QERROR_THRESH {
                                    best = Some((trial, vt));
                                    break;
                                }
                            }
                            ii += 1;
                            if !sd_used && alpha < opts.step_min {
                                break;
                            }
                        }
                    }
                }

                // NOT ported: mgcv's enhancement (iv), the steepest-descent trial
                // it runs alongside Newton whenever the Hessian is indefinite
                // (`gam.fit3.r:1561-1602`). Ported, measured, removed. In gamrs's
                // setting it cost 5.2x the inner PIRLS calls — 9463 -> 49079 over
                // the 117-fit `outer_indefinite_axis` sweep — for a 2.7% reduction
                // in outer iterations and not one failure fixed, because
                // `Sstep = -grad/max|grad|` is unscaled: its largest component
                // there is the well-conditioned `log σ²` axis that is ALREADY at
                // its optimum, while the axis actually crawling (ν, `|H_ii| ~
                // 1e-3`) carries the smallest gradient. Steepest descent moves the
                // wrong axis, at up to 40 probes per iteration, on a family whose
                // Hessian is indefinite in nearly every iteration. mgcv can afford
                // it because its own gradient bar is 50x looser, so it takes far
                // fewer indefinite iterations before stopping. The cheap, targeted
                // replacement is `step::concave_bound_jump` above: one probe, and
                // it moves the axis that is actually stuck.
                //
                // The mid-loop SD switch at (2) IS kept — it costs nothing extra,
                // reusing a halving probe rather than adding a trial.
                //
                // DELIBERATE deviation, also from `gam.fit3.r:1598-1602`: mgcv
                // accepts the better of its two directions even when neither
                // improved the score, relying on `ii == maxHalf ⇒ converged` to
                // stop. gamrs keeps the score monotone and routes "nothing
                // improved" to the step-failure exit below, which classifies the
                // standing point instead of moving uphill first.
                match best {
                    Some((t, vt)) => (true, Some(t), vt),
                    None => (false, None, v),
                }
            };
            if let Some(trial) = accepted_trial {
                // One full eval at the accepted point to refresh (g, h).
                // If this fails (extremely rare — the value probe just
                // succeeded), keep the value-only success and let the next
                // iter retry with stale (g, h) — the next halving will
                // self-correct.
                if let Ok((v_full, g_full, h_full)) = score.value_grad_hess(&trial) {
                    prev_v = v;
                    theta = trial;
                    v = v_full;
                    g = g_full;
                    h = h_full;
                } else {
                    prev_v = v;
                    theta = trial;
                    v = accepted_v;
                    // g, h stale — keep last good ones.
                }
            }

            if !accepted {
                // No trial step improved the score. Whether that means "arrived"
                // or "stuck" is `step::step_failure_is_convergence`'s call — the
                // same classifier the iteration-budget exit below uses, so one
                // standing point cannot be called converged by one exit and not
                // by the other.
                let converged = step::step_failure_is_convergence(
                    &theta,
                    &g,
                    &h,
                    &raw_step,
                    axis_bounds.as_ref(),
                    v,
                    grad_norm,
                );
                return Ok(OuterFit {
                    theta,
                    value: v,
                    grad_norm,
                    iterations: iter + 1,
                    converged,
                });
            }
        }

        // Iteration budget exhausted. mgcv `gam.fit3.r:1653-1658` warns here —
        // `"Iteration limit reached without full convergence - check carefully"` —
        // and RETURNS the estimate; it does not fail. gamrs used to raise
        // `NotConverged` and discard the fit, which made this exit the only one in
        // the file that throws: the step-failure exit above returns `Ok` with a
        // classified flag, and `fellner_schall_minimize` returns
        // `Ok(converged: false)` at its own cap. Same standing point, three
        // different outcomes depending on which exit happened to fire first.
        //
        // What was being discarded, measured on `outer_indefinite_axis` seed 21:
        // at the cap the score has plateaued to 3e-8 per iteration and the fit is
        // edf 2.7660 against 2.7660 for the same fit given 20000 iterations — the
        // remaining descent is 1.3e-8 REML units and $19 on a $550k curve, spent
        // marching ν from 1369 to 21963 on data that identifies neither. Raising
        // the budget is the wrong lever; returning what mgcv would return is the
        // right one. Callers that must distinguish "converged inside the budget"
        // from "hit the cap" compare `iterations` against `max_iters`, and the
        // Python wrapper warns whenever `converged` comes back false.
        Ok(OuterFit {
            theta,
            value: v,
            grad_norm: inf_norm(&g),
            iterations: opts.max_iters,
            converged: step::relaxed_grad_converged(inf_norm(&g), v),
        })
    }
}

/// Wood & Fasiolo (2017) Fellner-Schall update.
///
/// Per smoothing parameter `λ_i`, the multiplicative update is
///
/// ```text
///   λ_new[i] = λ[i] · φ · max(rank[i]/λ[i] − tr(A⁻¹ S_i), ε) / (β' S_i β)
/// ```
///
/// where `A = X' W X + Σ_j λ_j S_j` is the inner (penalised) Hessian
/// and `φ` is the dispersion. Performed in log-space with a per-iter
/// step clamp to keep the update stable.
///
/// Port of mgcv_rust `smooth.rs:fellner_schall_step` (which is mgcv R's
/// `bam(method="fREML")` core update).
#[derive(Debug, Clone, Copy)]
pub struct FellnerSchallOpts {
    /// Max outer iterations before bailing out.
    pub max_iters: usize,
    /// Relative `log λ` change for convergence; `max_i |Δlog λ_i| < tol`.
    pub tol: f64,
    /// Per-iter cap on `|Δlog λ_i|`. mgcv default `3.0`.
    pub log_step_clamp: f64,
    /// Lower / upper bounds on λ (clamped post-update). mgcv default
    /// `(1e-9, 1e7)`.
    pub lambda_bounds: (f64, f64),
}

impl Default for FellnerSchallOpts {
    fn default() -> Self {
        Self {
            max_iters: 200,
            tol: 1e-6,
            log_step_clamp: 3.0,
            lambda_bounds: (1e-9, 1e7),
        }
    }
}

/// Run the Fellner-Schall outer loop. Returns the converged
/// `OuterFit { theta: log_lambda_hat, ... }` so the caller can plug
/// it into the same downstream pipeline as Newton's output.
///
/// `phi_fn` reads the dispersion at the current fit — for Bernoulli /
/// Poisson / NegBin it's `1.0`; for Gaussian / Gamma / IG it's the
/// Pearson φ̂ at convergence.
pub fn fellner_schall_minimize<I, S>(
    inner: &I,
    s_list: &[Array2<f64>],
    rank_s_list: &[usize],
    rho0: Array1<f64>,
    phi_fn: impl Fn(&crate::inner::GaussianInnerFit<S>) -> f64,
    opts: FellnerSchallOpts,
    stats: Option<&crate::stats::FitStats>,
) -> Result<OuterFit>
where
    I: crate::traits::InnerSolver<Fit = crate::inner::GaussianInnerFit<S>>,
    S: crate::inner::LinearSolver,
{
    let m = s_list.len();
    debug_assert_eq!(rho0.len(), m);
    debug_assert_eq!(rank_s_list.len(), m);
    let log_lo = opts.lambda_bounds.0.ln();
    let log_hi = opts.lambda_bounds.1.ln();

    let mut rho = rho0;
    for k in 0..m {
        rho[k] = rho[k].clamp(log_lo, log_hi);
    }
    let mut warm_beta: Option<Array1<f64>> = None;
    let tiny = 1e-10_f64;

    for iter in 0..opts.max_iters {
        if let Some(s) = stats {
            s.bump_outer();
        }
        // Single IRLS step at the warm-started β (mgcv R `bam(method=
        // "fREML")` port — audit `nn_exploring/src/pirls/mod.rs:4078-4097`).
        // FS λ-updates move slowly; one IRLS step from a warm β closes
        // the per-outer-iter convergence error well enough for the FS
        // update to find descent. Avoids the 2-3× redundant inner iters
        // that full-PIRLS would do.
        let fit = inner.fit_single_irls(&rho, warm_beta.as_ref())?;
        if let Some(s) = stats {
            s.record_pirls_call(fit.iterations);
        }
        let phi = phi_fn(&fit);
        let a_inv = fit.a_inv();

        // Compute the FS step per smooth and the max |log change|.
        let mut max_change = 0.0_f64;
        let mut new_rho = rho.clone();
        for i in 0..m {
            let s_i = &s_list[i];
            let lambda_i = rho[i].exp().max(1e-20);
            let rank_i = rank_s_list[i] as f64;
            // tr(A⁻¹ S_i)
            let mut tr = 0.0_f64;
            for r in 0..a_inv.nrows() {
                for c in 0..a_inv.ncols() {
                    tr += a_inv[[r, c]] * s_i[[c, r]];
                }
            }
            // β' S_i β
            let s_beta = s_i.dot(&fit.beta);
            let bsb_raw: f64 = fit.beta.iter().zip(s_beta.iter()).map(|(a, b)| a * b).sum();
            let bsb = bsb_raw.max(tiny);
            let num_raw = rank_i / lambda_i - tr;
            let num = num_raw.max(tiny);
            let log_ratio_raw = (phi * num / bsb).ln();
            let log_ratio = log_ratio_raw.clamp(-opts.log_step_clamp, opts.log_step_clamp);
            let new_log_lambda = (rho[i] + log_ratio).clamp(log_lo, log_hi);
            max_change = max_change.max((new_log_lambda - rho[i]).abs());
            new_rho[i] = new_log_lambda;
        }
        rho = new_rho;
        warm_beta = Some(fit.beta.clone());

        if max_change < opts.tol {
            return Ok(OuterFit {
                theta: rho,
                value: 0.0, // FS doesn't track the REML value directly
                grad_norm: max_change,
                iterations: iter + 1,
                converged: true,
            });
        }
    }

    // Did not converge — return last state with `converged: false`.
    let _ = warm_beta;
    Ok(OuterFit {
        theta: rho,
        value: 0.0,
        grad_norm: f64::NAN,
        iterations: opts.max_iters,
        converged: false,
    })
}

/// Project the symmetric Hessian to a positive-definite form by
/// flooring each eigenvalue at `floor`. Conservative "floor-only"
/// behaviour: negative eigenvalues get clamped to `floor` (small
/// positive), which produces tiny Newton steps in those directions
/// rather than full-magnitude flipped descent steps.
///
/// For the proper Gill-Murray-Wright recipe (ABS + relative floor),
/// see [`make_psd_gmw`]. GMW requires diagonal preconditioning to be
/// stable across families (without preconditioning, the ABS produces
/// over-large steps that destabilise scat / TDist fits).
///
/// `pub(crate)` so `fit::profile_shape` can share this canonical
/// implementation rather than maintain a duplicate.
pub(crate) fn make_psd(h: &Array2<f64>, floor: f64) -> Array2<f64> {
    let d = h.nrows();
    if d == 0 {
        return h.clone();
    }
    if d == 1 {
        let mut out = h.clone();
        if out[[0, 0]] < floor {
            out[[0, 0]] = floor;
        }
        return out;
    }
    // Symmetrise (kills FD-induced asymmetry that would otherwise make
    // eigh use only the lower triangle).
    let mut sym = h.clone();
    for i in 0..d {
        for j in i + 1..d {
            let avg = 0.5 * (sym[[i, j]] + sym[[j, i]]);
            sym[[i, j]] = avg;
            sym[[j, i]] = avg;
        }
    }
    let (eigs, vecs) = match sym.eigh(UPLO::Lower) {
        Ok(p) => p,
        Err(_) => {
            // Fallback: diagonal floor only.
            let mut out = sym;
            for i in 0..d {
                if out[[i, i]] < floor {
                    out[[i, i]] = floor;
                }
            }
            return out;
        }
    };
    let mut floored = Array2::<f64>::zeros((d, d));
    for k in 0..d {
        let lam = eigs[k].max(floor);
        let v = vecs.column(k);
        for i in 0..d {
            for j in 0..d {
                floored[[i, j]] += lam * v[i] * v[j];
            }
        }
    }
    floored
}

/// Project the symmetric Hessian onto the positive-definite cone by
/// flooring its eigenvalues at `floor`. Matches mgcv's
/// `gam.fit3.r:1397-1417` approach: if `H = Q diag(λ_i) Q'`, return
/// `Q diag(max(λ_i, floor)) Q'`. Guarantees the Newton step is a descent
/// direction even when the analytic / FD Hessian is indefinite (common
/// at the start of joint-θ scat optimisation where σ² and ν are far from
/// their optima).
///
/// 1-D case (Phase 0 / 1): degenerates to `max(h, floor)`. 2-D+ case
/// (Phase 2 scat): full eigendecomposition.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::CoordsKind;
    use ndarray::array;

    /// Test-only quadratic score: V(θ) = ½ (θ - θ*)' M (θ - θ*) with known
    /// minimum at θ*. Used to verify multi-d Newton lands at the optimum.
    struct QuadScore {
        m: Array2<f64>,
        opt: Array1<f64>,
    }
    impl ScoreDerivatives for QuadScore {
        fn dim(&self) -> usize {
            self.opt.len()
        }
        fn coords(&self) -> CoordsKind {
            CoordsKind::Identity
        }
        fn value(&self, theta: &Array1<f64>) -> Result<f64> {
            let d = theta - &self.opt;
            Ok(0.5 * d.dot(&self.m.dot(&d)))
        }
        fn value_and_grad(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>)> {
            let d = theta - &self.opt;
            let g = self.m.dot(&d);
            Ok((0.5 * d.dot(&g), g))
        }
        fn value_grad_hess(&self, theta: &Array1<f64>) -> Result<(f64, Array1<f64>, Array2<f64>)> {
            let (v, g) = self.value_and_grad(theta)?;
            Ok((v, g, self.m.clone()))
        }
    }

    #[test]
    fn newton_lands_on_2d_quadratic_optimum() {
        let m = array![[2.0, 0.5], [0.5, 1.5]];
        let opt = array![1.3, -0.7];
        let score = QuadScore {
            m: m.clone(),
            opt: opt.clone(),
        };
        let solver = NewtonWithHalving::default();
        let fit = solver.minimize(&score, array![0.0, 0.0]).unwrap();
        for i in 0..2 {
            assert!(
                (fit.theta[i] - opt[i]).abs() < 1e-8,
                "θ[{i}] = {} vs opt {}",
                fit.theta[i],
                opt[i]
            );
        }
        assert!(fit.converged);
    }

    #[test]
    fn psd_fix_lifts_negative_eigenvalues() {
        // Saddle-point Hessian — first eigenvalue negative. PSD-fix
        // should flip it to `floor`, leaving the positive one untouched.
        let h = array![[-1.0, 0.0], [0.0, 2.0]];
        let fixed = make_psd(&h, 1e-8);
        let (eigs, _) = fixed.eigh(UPLO::Lower).unwrap();
        let mut sorted = eigs.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 1e-8).abs() < 1e-12, "small eig {}", sorted[0]);
        assert!((sorted[1] - 2.0).abs() < 1e-12, "large eig {}", sorted[1]);
    }

    #[test]
    fn fellner_schall_refusal_carries_the_marker_python_matches_on() {
        // `python/gamrs/_fitter.py::FS_UNAVAILABLE_MARKER` is this same
        // string, and the wrapper turns a message carrying it into a warned
        // REML refit. Reword the message freely; drop the marker and the
        // Python fallback silently stops firing, which is why this asserts it.
        set_algorithm_override(OuterAlgorithm::FellnerSchall);
        let err = reject_unsupported_algorithm("test")
            .unwrap_err()
            .to_string();
        clear_algorithm_override();
        assert!(
            err.contains(FS_UNAVAILABLE_MARKER),
            "refusal lost the marker: {err}"
        );
        assert!(!err.contains("  "), "message has a run of spaces: {err}");
    }
}
