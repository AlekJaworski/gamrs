//! Step construction and exit classification for the outer Newton — the pure
//! half of `NewtonWithHalving`.
//!
//! Every function here is a total function of `(θ, g, H, box)`: no score
//! evaluation, no inner PIRLS, no state. That is deliberate. The driver in
//! `super` decides WHICH candidate step to probe and what to do with the
//! answer; this module decides what the candidates ARE, so each rule can be
//! tested against a hand-built Hessian instead of through a fitted model.
//!
//! Ordering matters and is fixed by the driver: build the (modified) Newton
//! direction, project it onto the feasible box, THEN cap it. Capping before
//! projecting reintroduces the defect in [`project_onto_box`]'s note.

use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, Solve, UPLO};

/// Quadratic-approximation error tolerated in a step — mgcv `gam.fit3.r:1342`
/// `qerror.thresh <- .8`.
pub(super) const QERROR_THRESH: f64 = 0.8;

/// mgcv's `conv.tol` for the path scat takes (`mgcv.r:2209`), used ONLY as the
/// noise guard in the denominator of the quadratic-model error.
///
/// Deliberately NOT `opts.grad_tol`. gamrs's `grad_tol` is its gradient test and
/// is set 50x tighter than mgcv's on purpose, but this constant answers a
/// different question — below what score change is the comparison meaningless —
/// and it must not shrink when the gradient test is tightened. Tying the two
/// together made the guard 1e-9-relative for a caller passing `grad_tol = 1e-9`
/// (`score/envelope.rs::fd_match_elf_pirls_1d` does exactly that), so near the
/// optimum, where the predicted and realised changes are both ~0 and the ratio is
/// pure noise, every remaining step was refused as a model mismatch and the
/// optimiser stopped early with a residual gradient. That test caught it: the
/// FD Hessian picked up the leftover O(g·h) and its analytic-vs-FD agreement went
/// 1.75e-4 against a 1e-4 bar.
pub(super) const MGCV_CONV_TOL: f64 = 1.0e-6;

/// Cap on the length of a steepest-descent step — mgcv's `maxSstep` argument
/// (`gam.fit3.r:1229`, default 2).
pub(super) const MAX_S_STEP: f64 = 2.0;

/// Score-relative tolerance below which an axis is treated as dead for the
/// subset filter — mgcv `gam.fit3.r:1643` `score.scale*conv.tol*.1`, which for
/// the scat path's `conv.tol = 1e-6` (`mgcv.r:2209`) is `1e-7 · score_scale`.
///
/// NB this was briefly changed to `score_scale * grad_tol * 0.1` on the belief
/// that the hardcoded 1e-7 was "67x looser than mgcv's". It was not — the 67x
/// came from comparing against `fast.REML.fit`'s `sqrt(eps)`, which drives the
/// Gaussian fREML path, not scat. Reverted, and it made no measurable difference
/// either way.
pub(super) const DIM_TOL_REL: f64 = 1.0e-7;

/// How close to a bound counts as sitting ON it, relative to the axis value.
const AT_BOUND_EPS: f64 = 1e-12;

/// Axes the Newton step is allowed to move — mgcv's `uconv.ind`
/// (`gam.fit3.r:1643`): `active_i ⇔ |g_i| > dim_tol OR |H_ii| > dim_tol`.
///
/// The `H_ii` clause keeps an axis alive when its curvature is meaningful even
/// though its gradient is small (the saddle-point case). Frozen axes get a zero
/// step, which both saves work and keeps an effectively-dead axis from polluting
/// the live axes' direction through the joint inverse. At least one axis is
/// always returned (mgcv `gam.fit3.r:1432`).
pub(super) fn active_axes(g: &Array1<f64>, h: &Array2<f64>, dim_tol: f64) -> Vec<usize> {
    let dim = g.len();
    let active: Vec<usize> = (0..dim)
        .filter(|&i| g[i].abs() > dim_tol || h[[i, i]].abs() > dim_tol)
        .collect();
    if active.is_empty() {
        let argmax = (0..dim)
            .max_by(|&a, &b| g[a].abs().total_cmp(&g[b].abs()))
            .unwrap_or(0);
        return vec![argmax];
    }
    active
}

/// The (modified) Newton direction on the active axes, padded with zeros
/// elsewhere. mgcv `gam.fit3.r:~1394-1417`, in four steps:
///
/// 1. restrict `H`, `g` to the active axes;
/// 2. diagonal preconditioning `D_ii = sqrt(|H_ii|)`, which normalises the
///    eigenvalue spectrum so step 3's relative floor is meaningful;
/// 3. Gill-Murray-Wright eigen-fix ([`make_psd_gmw`]) — absolute value of
///    negative eigenvalues plus a relative floor — so the direction is always
///    descent;
/// 4. solve in preconditioned coordinates and back-transform.
pub(super) fn newton_direction(
    g: &Array1<f64>,
    h: &Array2<f64>,
    active: &[usize],
    hess_floor: f64,
) -> Array1<f64> {
    let dim = g.len();
    let n_active = active.len();
    let mut diag_precond = Array1::<f64>::zeros(n_active);
    for (ki, &ai) in active.iter().enumerate() {
        diag_precond[ki] = h[[ai, ai]].abs().sqrt().max(hess_floor.sqrt());
    }
    let mut h_sub_pre = Array2::<f64>::zeros((n_active, n_active));
    for (ri, &ai) in active.iter().enumerate() {
        for (ci, &aj) in active.iter().enumerate() {
            h_sub_pre[[ri, ci]] = h[[ai, aj]] / (diag_precond[ri] * diag_precond[ci]);
        }
    }
    let mut g_sub_pre = Array1::<f64>::zeros(n_active);
    for (ki, &ai) in active.iter().enumerate() {
        g_sub_pre[ki] = g[ai] / diag_precond[ki];
    }
    let h_psd = make_psd_gmw(&h_sub_pre, hess_floor);
    let step_sub_pre = match h_psd.solve(&(-&g_sub_pre)) {
        Ok(s) => s,
        Err(_) => -&g_sub_pre / hess_floor.max(1.0),
    };
    let mut step = Array1::<f64>::zeros(dim);
    for (ki, &ai) in active.iter().enumerate() {
        step[ai] = step_sub_pre[ki] / diag_precond[ki];
    }
    step
}

/// Zero out the step components that push outward at an active bound.
///
/// Such a component is clamped back to the bound at every trial point anyway, so
/// it buys no movement — but the cap shrink in [`cap_step`] is a single GLOBAL
/// factor `min_i(cap_i/|s_i|)`, so leaving it in throttles every LIVE axis by its
/// magnitude. Standard projected Newton, and the reason this must run BEFORE the
/// cap.
///
/// Measured on a scat fit with ν unidentified (n=66, k=12, the
/// `scat_pinned_shape_axis_does_not_throttle_rho` fixture): the `log(ν−3)` axis
/// pins at its upper bound 10 with `|H_ii| ~ 2e-6` and asks for +19.4 every
/// iteration. Cap 1.0 ⇒ shrink 0.0515, so ρ moved 0.042 per iteration instead of
/// 0.83 and `log σ²` 5e-7 instead of 1e-5 — frozen while carrying a real,
/// FD-confirmed gradient of 5.6e-2. 200 iterations of crawl, then `NotConverged`
/// threw the fit away. With the projection: 29 iterations, converged, at a score
/// 5.5e-5 lower than where the crawl ran out of budget.
pub(super) fn project_onto_box(
    step: &Array1<f64>,
    theta: &Array1<f64>,
    bnds: Option<&Vec<(f64, f64)>>,
) -> Array1<f64> {
    let mut out = step.clone();
    if let Some(bnds) = bnds {
        for (i, &(lo, hi)) in bnds.iter().enumerate() {
            if i >= out.len() {
                continue;
            }
            if (at_upper(theta[i], hi) && out[i] > 0.0) || (at_lower(theta[i], lo) && out[i] < 0.0)
            {
                out[i] = 0.0;
            }
        }
    }
    out
}

/// Shrink a step to respect the per-axis caps, or the global L∞ cap when no
/// per-axis caps are supplied.
///
/// The per-axis form scales the WHOLE step by the tightest binding ratio rather
/// than clipping each axis, so the direction survives — mgcv's per-axis-binding
/// shrink (`smooth.r build_outer_search_vector`).
pub(super) fn cap_step(
    step: &Array1<f64>,
    axis_caps: Option<&Vec<f64>>,
    max_step: f64,
) -> Array1<f64> {
    match axis_caps {
        Some(caps) => {
            debug_assert_eq!(caps.len(), step.len(), "axis_step_caps length mismatch");
            let mut shrink = 1.0_f64;
            for (i, &si) in step.iter().enumerate() {
                if si.abs() > caps[i] && si.abs() > 0.0 {
                    shrink = shrink.min(caps[i] / si.abs());
                }
            }
            step * shrink
        }
        None => {
            let step_norm = inf_norm(step);
            if step_norm > max_step {
                step * (max_step / step_norm)
            } else {
                step.clone()
            }
        }
    }
}

/// The candidate step that takes every CONCAVE axis straight to the bound its
/// gradient points at, keeping the Newton component on all the others.
///
/// Along a saturating shape axis the curvature stays negative all the way to the
/// face, so over the box the minimum in that direction is at the face: the
/// location is known outright and does not need to be approached. That holds only
/// for the axes the score names in `jumpable` (`bound_jump_axes`); a ρ axis can be
/// concave on its way down to an interior minimum, and jumping it overshoots onto
/// a plateau the gradient cannot leave. Newton
/// instead proposes `-g_i/|H_ii|`, deliberately short for indefinite components
/// (mgcv's note at `gam.fit3.r:1403`), which on the `outer_indefinite_axis`
/// fixture is 0.02 per iteration for an axis 2.0 from its bound — ~100
/// iterations to reach a point that can be named in closed form.
///
/// `None` when there is no such axis, or when Newton is already covering the
/// distance in a few iterations. That last condition matters: the jump is worth
/// one probe when the alternative is several iterations (a probe costs ~1 inner
/// solve, an accepted iteration ~2), and is pure overhead when the axis is moving
/// freely. The `4.0` is that break-even with margin, NOT a tuned constant —
/// measured across the 117-fit sweep, the response to it is non-monotone (4 →
/// 11105 inner solves, 12 → 10880, 30 → 11316, 80 → 11793), so there is no knee
/// to find and picking it on wall time would be fitting noise.
///
/// The caller must NOT put this step through the quadratic-model gate: it is not
/// a Newton step and the model is known not to describe it. The line search still
/// has the final say — a jump that does not improve the score is discarded and
/// the ordinary Newton sequence runs unchanged.
pub(super) fn concave_bound_jump(
    theta: &Array1<f64>,
    g: &Array1<f64>,
    h: &Array2<f64>,
    capped_step: &Array1<f64>,
    bnds: Option<&Vec<(f64, f64)>>,
    jumpable: Option<&Vec<bool>>,
    dim_tol: f64,
) -> Option<Array1<f64>> {
    let (bnds, jumpable) = (bnds?, jumpable?);
    let mut jump = capped_step.clone();
    let mut jumped = false;
    for (i, &(lo, hi)) in bnds.iter().enumerate() {
        if i >= jump.len() || !jumpable.get(i).copied().unwrap_or(false) || h[[i, i]] >= -dim_tol {
            continue;
        }
        let target = if g[i] < 0.0 {
            hi
        } else if g[i] > 0.0 {
            lo
        } else {
            continue;
        };
        let dist = target - theta[i];
        if dist.abs() > AT_BOUND_EPS * theta[i].abs().max(1.0)
            && dist.abs() > 4.0 * capped_step[i].abs()
        {
            jump[i] = dist;
            jumped = true;
        }
    }
    jumped.then_some(jump)
}

/// mgcv's steepest-descent direction, `gam.fit3.r:1419`:
/// `Sstep <- -grad/max(abs(grad))`. Its L∞ norm is 1, so trial lengths along it
/// are in θ units directly.
pub(super) fn steepest_descent_dir(g: &Array1<f64>) -> Array1<f64> {
    let gmax = inf_norm(g);
    if gmax > 0.0 {
        -g / gmax
    } else {
        Array1::<f64>::zeros(g.len())
    }
}

/// How far the realised score change is from what the quadratic model predicted,
/// relative to the larger of the two — mgcv `gam.fit3.r:1461-1463`:
///
/// ```text
///   pred.change = g'·s + ½ s'·H·s
///   qerror      = |pred.change − actual| / (max(|pred.change|, |actual|) + tol)
/// ```
///
/// `H` is the RAW Hessian, not the PSD-perturbed one the step was solved from:
/// the question being asked is whether the surface actually behaves the way
/// Newton's model of it says, and a step whose answer is "no" is refused. Without
/// that refusal any strictly-decreasing step is accepted, which is how a two-cycle
/// worth ~1e-8 score units a step can hold the optimiser until it runs out of
/// iterations. The `tol` term only keeps the ratio finite when both changes are
/// ~0; mgcv uses `score.scale · conv.tol` there.
pub(super) fn quadratic_model_error(
    g: &Array1<f64>,
    h: &Array2<f64>,
    step: &Array1<f64>,
    actual_change: f64,
    score_scale: f64,
    conv_tol: f64,
) -> f64 {
    let lin: f64 = g.iter().zip(step.iter()).map(|(a, b)| a * b).sum();
    let quad: f64 = 0.5 * step.dot(&h.dot(step));
    let pred = lin + quad;
    let denom = pred.abs().max(actual_change.abs()) + score_scale * conv_tol;
    if denom <= 0.0 {
        return 0.0;
    }
    (pred - actual_change).abs() / denom
}

/// Clamp a trial point into the per-axis box, if there is one — mgcv-style
/// box-constrained Newton (`smooth.r:~1976` lo/hi clamp).
pub(super) fn clamp_to_box(mut t: Array1<f64>, bnds: Option<&Vec<(f64, f64)>>) -> Array1<f64> {
    if let Some(bnds) = bnds {
        for (i, &(lo, hi)) in bnds.iter().enumerate() {
            if i < t.len() {
                t[i] = t[i].clamp(lo, hi);
            }
        }
    }
    t
}

/// Whether a standing point at which no trial step improved the score should be
/// reported as converged. Four independent cases, any of which suffices:
///
/// - **(a) interior minimum** — `|g|_∞` small on the score scale. A
///   double-precision FD Hessian on a flat region cannot produce a
///   strictly-decreasing trial even when the gradient says the optimiser has
///   arrived. See [`relaxed_grad_converged`].
/// - **(b) strict box KKT** — the UNCONSTRAINED Newton step pushed outside the
///   box on every axis where it moved at all, so all movement was clamped away.
///   This reads `raw_step`, before [`project_onto_box`], precisely because the
///   question is what the unconstrained step wanted.
/// - **(c) projected-gradient KKT** — the general box-constrained condition:
///   axes at an active bound with their gradient pointing outward are blocked, so
///   they are dropped before the norm is measured. Covers ocat's saturating-θ
///   ridge. A tier looser than (a) because FD gradients near a box face are
///   noisier — the active-bound axis acts as a non-smooth jump in the stencil
///   that bleeds into nearby axes.
/// - **(d) rank-deficient KKT** — the analogue at outer level of mgcv `gam.fit5`
///   step 4 ("at convergence test fundamental rank on balanced version of
///   penalized Hessian"). Eigendecompose `H`, take the working subspace
///   (eigenvalues above `max|λ|·ε^0.7`), and project `g` onto it. If that is
///   small, all the gradient mass lies along a null direction — the score is flat
///   there and no step can make progress. The canonical ocat failure mode.
pub(super) fn step_failure_is_convergence(
    theta: &Array1<f64>,
    g: &Array1<f64>,
    h: &Array2<f64>,
    raw_step: &Array1<f64>,
    bnds: Option<&Vec<(f64, f64)>>,
    v: f64,
    grad_norm: f64,
) -> bool {
    relaxed_grad_converged(grad_norm, v)
        || kkt_at_boundary(theta, raw_step, bnds)
        || projected_grad_small(theta, g, bnds, v)
        || rank_deficient_grad_small(g, h, v)
}

/// Case (b) of [`step_failure_is_convergence`]. Vacuously false when the raw step
/// is ~0 everywhere — case (a) covers that.
fn kkt_at_boundary(
    theta: &Array1<f64>,
    raw_step: &Array1<f64>,
    bnds: Option<&Vec<(f64, f64)>>,
) -> bool {
    let Some(bnds) = bnds else { return false };
    let mut any_movement = false;
    let mut all_blocked = true;
    for (i, &(lo, hi)) in bnds.iter().enumerate() {
        if i >= raw_step.len() {
            continue;
        }
        let si = raw_step[i];
        // Only count "real" movement; tiny si is noise.
        if si.abs() > AT_BOUND_EPS * theta[i].abs().max(1.0) {
            any_movement = true;
            let pushes_out =
                (at_upper(theta[i], hi) && si > 0.0) || (at_lower(theta[i], lo) && si < 0.0);
            if !pushes_out {
                all_blocked = false;
            }
        }
    }
    any_movement && all_blocked
}

/// Case (c) of [`step_failure_is_convergence`]. Only fires when at least one axis
/// actually sits at a bound — otherwise case (a) is the right test.
fn projected_grad_small(
    theta: &Array1<f64>,
    g: &Array1<f64>,
    bnds: Option<&Vec<(f64, f64)>>,
    v: f64,
) -> bool {
    let Some(bnds) = bnds else { return false };
    let mut any_at_bound = false;
    let mut proj = 0.0_f64;
    for (i, &gi) in g.iter().enumerate() {
        let (lo, hi) = bnds
            .get(i)
            .copied()
            .unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
        // A looser "at bound" test than elsewhere: this asks whether the FD
        // gradient is contaminated by the face, which happens before the axis is
        // exactly on it.
        let at_lo = (theta[i] - lo).abs() <= 1e-9 * theta[i].abs().max(1.0);
        let at_hi = (hi - theta[i]).abs() <= 1e-9 * theta[i].abs().max(1.0);
        if at_lo || at_hi {
            any_at_bound = true;
        }
        let blocked = (at_hi && gi < 0.0) || (at_lo && gi > 0.0);
        if !blocked {
            proj = proj.max(gi.abs());
        }
    }
    any_at_bound && proj < 1e-1 * (v.abs() + 1.0)
}

/// Case (d) of [`step_failure_is_convergence`].
fn rank_deficient_grad_small(g: &Array1<f64>, h: &Array2<f64>, v: f64) -> bool {
    // The FD asymmetry is small enough that eigh on the lower triangle is fine.
    let Ok((eigs, vecs)) = h.eigh(UPLO::Lower) else {
        return false;
    };
    let max_abs = eigs.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let null_thresh = max_abs * f64::EPSILON.powf(0.7); // ~1.5e-11 of max
    let mut proj_max = 0.0_f64;
    for k in 0..g.len() {
        if eigs[k].abs() <= null_thresh {
            continue;
        }
        let uk = vecs.column(k);
        let utg: f64 = uk.iter().zip(g.iter()).map(|(a, b)| a * b).sum();
        proj_max = proj_max.max(utg.abs());
    }
    // Only when the Hessian is ACTUALLY rank-deficient — otherwise case (a)
    // covers it. Same tier-looser threshold as case (c).
    let has_null = eigs.iter().any(|&e| e.abs() <= null_thresh);
    has_null && proj_max < 1e-1 * (v.abs() + 1.0)
}

/// The relaxed, score-relative gradient bar the step-failure exit has always used
/// to call an interior minimum: `|g|_∞ < 1e-3·(|score| + 1)`. Looser than
/// `grad_tol` by design — a double-precision FD Hessian on a flat region cannot
/// produce a strictly-decreasing trial even when the gradient says the optimiser
/// has arrived. For scale: mgcv's own gradient test is `5·conv.tol·score.scale`
/// (`gam.fit3.r:1644`), i.e. 5e-6·score_scale, so this bar is 200x looser than
/// mgcv's and reaching it is a weak claim — which is why it decides a FLAG rather
/// than an error.
pub(super) fn relaxed_grad_converged(grad_norm: f64, v: f64) -> bool {
    grad_norm < 1e-3 * (v.abs() + 1.0)
}

fn at_upper(theta_i: f64, hi: f64) -> bool {
    (hi - theta_i).abs() <= AT_BOUND_EPS * theta_i.abs().max(1.0)
}

fn at_lower(theta_i: f64, lo: f64) -> bool {
    (theta_i - lo).abs() <= AT_BOUND_EPS * theta_i.abs().max(1.0)
}

pub(super) fn inf_norm(v: &Array1<f64>) -> f64 {
    v.iter().fold(0.0_f64, |a, &b| a.max(b.abs()))
}

pub(super) fn l2_norm(v: &Array1<f64>) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Gill-Murray-Wright modified Hessian — Practical Optimization (1981)
/// p.107-8, mgcv R `gam.fit3.r:1397-1417`, mgcv_rust `smooth.rs:2686-2712`.
///
/// ```text
///   H = U diag(λ_i) U'                       (eigendecomposition)
///   d_i ← |λ_i|                              (ABS for indefinite spectra)
///   d_i ← max(d_i, max(|λ|) · ε^0.7)         (relative floor for tiny)
///   H_psd = U diag(d_i) U'
/// ```
///
/// The ABS step is what unsticks Newton on indefinite spectra: instead
/// of flooring `−λ_i` to a tiny positive (giving a tiny step), GMW
/// preserves `|λ_i|` magnitude (giving a descent step of the right size).
///
/// **Stability caveat**: GMW takes larger steps in formerly-indefinite
/// directions, which only makes sense AFTER the Hessian has been
/// diagonally preconditioned (so the eigenvalue spectrum is uniform
/// across coordinates). Calling GMW on a raw Hessian with wildly
/// different coordinate scales (e.g. ρ and θ axes for scat) over-steps
/// in the high-curvature directions and breaks convergence. Pair with
/// the preconditioning step in the Newton driver.
///
/// Used by the joint Newton in `outer.rs` (preconditioned) and the
/// ρ-only Newton in `profile_shape.rs` (1-D, so preconditioning is
/// trivial / vacuous).
#[allow(dead_code)]
pub(super) fn make_psd_gmw(h: &Array2<f64>, floor: f64) -> Array2<f64> {
    let d = h.nrows();
    if d == 0 {
        return h.clone();
    }
    if d == 1 {
        let mut out = h.clone();
        out[[0, 0]] = out[[0, 0]].abs().max(floor);
        return out;
    }
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
            let mut out = sym;
            for i in 0..d {
                out[[i, i]] = out[[i, i]].abs().max(floor);
            }
            return out;
        }
    };
    let abs_eigs: Vec<f64> = eigs.iter().map(|e| e.abs()).collect();
    let max_abs = abs_eigs.iter().cloned().fold(0.0_f64, f64::max);
    let low_d = (max_abs * f64::EPSILON.powf(0.7)).max(floor);
    let mut floored = Array2::<f64>::zeros((d, d));
    for k in 0..d {
        let lam = abs_eigs[k].max(low_d);
        let v = vecs.column(k);
        for i in 0..d {
            for j in 0..d {
                floored[[i, j]] += lam * v[i] * v[j];
            }
        }
    }
    floored
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// The reported 0.14.x defect, at unit level: an axis pinned at its bound
    /// asking for a huge step must not shrink the axes that can still move.
    /// Pre-fix this returned `[0.0413, ...]` for the live axis instead of `0.83`.
    #[test]
    fn a_pinned_axis_does_not_throttle_the_live_ones() {
        // θ₂ sits exactly on its upper bound and Newton wants +19.4 more.
        let theta = array![8.7, -0.75, 10.0];
        let raw = array![0.83, -1e-5, 19.4];
        let bnds = vec![(-50.0, 50.0), (-50.0, 50.0), (-10.0, 10.0)];
        let caps = vec![5.0, 1.0, 1.0];

        let projected = project_onto_box(&raw, &theta, Some(&bnds));
        assert_eq!(projected[2], 0.0, "outward component at the bound must go");
        let capped = cap_step(&projected, Some(&caps), 5.0);
        assert!(
            (capped[0] - 0.83).abs() < 1e-12,
            "live axis was throttled to {}",
            capped[0]
        );

        // And the defect itself: capping the UNPROJECTED step shrinks everything
        // by 19.4x, which is what burned 200 iterations.
        let throttled = cap_step(&raw, Some(&caps), 5.0);
        assert!(
            throttled[0] < 0.05,
            "expected the old throttle to be ~0.043, got {}",
            throttled[0]
        );
    }

    #[test]
    fn projection_leaves_inward_and_interior_components_alone() {
        let theta = array![0.0, 10.0, 10.0];
        let step = array![3.0, -2.0, 2.0];
        let bnds = vec![(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)];
        let out = project_onto_box(&step, &theta, Some(&bnds));
        assert_eq!(out[0], 3.0, "interior axis untouched");
        assert_eq!(out[1], -2.0, "at the bound but pointing back inside");
        assert_eq!(out[2], 0.0, "at the bound and pointing out");
    }

    #[test]
    fn projection_is_a_no_op_without_bounds() {
        let step = array![3.0, -2.0];
        let out = project_onto_box(&step, &array![0.0, 0.0], None);
        assert_eq!(out, step);
    }

    #[test]
    fn concave_axis_jumps_to_the_bound_its_gradient_points_at() {
        let theta = array![8.7, 8.0];
        // Axis 1: negative curvature, negative gradient ⇒ score falls toward hi.
        let g = array![-1e-5, -1.9e-5];
        let h = array![[0.105, 0.0], [0.0, -9.6e-4]];
        let newton = array![0.83, 0.02];
        let bnds = vec![(-50.0, 50.0), (-10.0, 10.0)];
        let shape = vec![false, true];
        let jump = concave_bound_jump(&theta, &g, &h, &newton, Some(&bnds), Some(&shape), 7e-6)
            .expect("a concave axis 2.0 from its bound with a 0.02 step must jump");
        assert!((jump[1] - 2.0).abs() < 1e-12, "jump to hi, got {}", jump[1]);
        assert_eq!(jump[0], 0.83, "other axes keep their Newton component");

        // Sign flip on the gradient ⇒ the other face.
        let g_lo = array![-1e-5, 1.9e-5];
        let jump_lo =
            concave_bound_jump(&theta, &g_lo, &h, &newton, Some(&bnds), Some(&shape), 7e-6)
                .unwrap();
        assert!((jump_lo[1] - (-18.0)).abs() < 1e-12);
    }

    #[test]
    fn no_jump_for_convex_axes_or_when_newton_is_already_moving() {
        let theta = array![8.7, 8.0];
        let g = array![-1e-5, -1.9e-5];
        let bnds = vec![(-50.0, 50.0), (-10.0, 10.0)];
        let shape = vec![false, true];
        let convex = array![[0.105, 0.0], [0.0, 9.6e-4]];
        assert!(
            concave_bound_jump(
                &theta,
                &g,
                &convex,
                &array![0.83, 0.02],
                Some(&bnds),
                Some(&shape),
                7e-6
            )
            .is_none(),
            "a positive-curvature axis has an interior optimum to find"
        );
        let concave = array![[0.105, 0.0], [0.0, -9.6e-4]];
        assert!(
            concave_bound_jump(
                &theta,
                &g,
                &concave,
                &array![0.83, 1.5],
                Some(&bnds),
                Some(&shape),
                7e-6
            )
            .is_none(),
            "Newton covers 2.0 in ~1.3 steps here; the probe would be waste"
        );
        assert!(
            concave_bound_jump(
                &theta,
                &g,
                &concave,
                &array![0.83, 0.02],
                None,
                Some(&shape),
                7e-6
            )
            .is_none(),
            "no box, no bound to jump to"
        );
    }

    #[test]
    fn a_concave_rho_axis_is_never_jumped() {
        // Both axes concave and heading for hi, as both tensor ρ were at the start
        // of the Gaussian te(cars, feet) fit that jumped 0 -> 30 past its minimum.
        let theta = array![0.0, 0.0];
        let g = array![-2.2, -1.5];
        let h = array![[-0.3, 0.0], [0.0, -0.2]];
        let newton = array![0.4, 0.3];
        let bnds = vec![(-30.0, 30.0), (-30.0, 30.0)];
        assert!(
            concave_bound_jump(
                &theta,
                &g,
                &h,
                &newton,
                Some(&bnds),
                Some(&vec![false, false]),
                7e-6
            )
            .is_none(),
            "a ρ axis can be concave on the way down to an interior minimum"
        );
        assert!(
            concave_bound_jump(&theta, &g, &h, &newton, Some(&bnds), None, 7e-6).is_none(),
            "a score that names no jump axes gets no jump"
        );
    }

    /// On an exactly-quadratic surface the model IS the function, so the error is
    /// zero for any step — the property the gate relies on.
    #[test]
    fn quadratic_model_error_vanishes_on_a_quadratic() {
        let g = array![2.0, -1.0];
        let h = array![[4.0, 1.0], [1.0, 3.0]];
        let s = array![0.3, -0.7];
        // f(θ+s) - f(θ) for f quadratic with this g and H.
        let lin: f64 = g.iter().zip(s.iter()).map(|(a, b)| a * b).sum();
        let actual = lin + 0.5 * s.dot(&h.dot(&s));
        let qerr = quadratic_model_error(&g, &h, &s, actual, 70.0, MGCV_CONV_TOL);
        assert!(qerr < 1e-12, "expected ~0, got {qerr:.3e}");
    }

    /// A step whose realised change is nothing like the prediction is exactly what
    /// the gate must catch — a two-cycle bouncing at ~1e-8 a step.
    #[test]
    fn quadratic_model_error_flags_a_mismatched_step() {
        let g = array![2.0];
        let h = array![[4.0]];
        let s = array![0.5];
        // Predicted -0.5; realised a token -1e-8.
        let qerr = quadratic_model_error(&g, &h, &s, -1e-8, 70.0, MGCV_CONV_TOL);
        assert!(qerr > QERROR_THRESH, "expected a refusal, got {qerr:.3e}");
    }

    #[test]
    fn active_axes_keeps_a_curved_axis_whose_gradient_is_dead() {
        let g = array![1.0, 1e-12];
        let h = array![[1.0, 0.0], [0.0, 5.0]];
        assert_eq!(active_axes(&g, &h, 1e-7), vec![0, 1]);
    }

    #[test]
    fn active_axes_always_yields_the_steepest_axis() {
        let g = array![1e-12, 3e-12];
        let h = array![[1e-12, 0.0], [0.0, 1e-12]];
        assert_eq!(
            active_axes(&g, &h, 1e-7),
            vec![1],
            "everything is under tolerance, so keep the largest gradient"
        );
    }

    #[test]
    fn cap_step_preserves_direction() {
        let step = array![10.0, -5.0];
        let capped = cap_step(&step, None, 2.0);
        assert!((inf_norm(&capped) - 2.0).abs() < 1e-12);
        assert!(
            (capped[0] / capped[1] - step[0] / step[1]).abs() < 1e-12,
            "a global shrink must not rotate the step"
        );
    }

    /// The step-failure exit must call a blocked bound converged: the gradient is
    /// large but points out of the box, so no feasible step can use it.
    #[test]
    fn step_failure_at_a_blocked_bound_is_convergence() {
        let theta = array![8.7, 10.0];
        let g = array![1e-9, -0.5];
        let h = array![[0.1, 0.0], [0.0, 1e-6]];
        let raw = array![1e-9, 3.0];
        let bnds = vec![(-50.0, 50.0), (-10.0, 10.0)];
        assert!(step_failure_is_convergence(
            &theta,
            &g,
            &h,
            &raw,
            Some(&bnds),
            70.0,
            0.5
        ));
    }

    #[test]
    fn step_failure_with_a_live_interior_gradient_is_not_convergence() {
        let theta = array![0.0, 0.0];
        let g = array![5.0, 0.0];
        let h = array![[1.0, 0.0], [0.0, 1.0]];
        let raw = array![-5.0, 0.0];
        let bnds = vec![(-50.0, 50.0), (-50.0, 50.0)];
        assert!(!step_failure_is_convergence(
            &theta,
            &g,
            &h,
            &raw,
            Some(&bnds),
            70.0,
            5.0
        ));
    }

    #[test]
    fn steepest_descent_dir_is_unit_in_the_max_component() {
        let g = array![-0.5, 2.0, 1.0];
        let s = steepest_descent_dir(&g);
        assert!((inf_norm(&s) - 1.0).abs() < 1e-12);
        assert!(s[1] < 0.0, "must oppose the gradient");
        assert_eq!(steepest_descent_dir(&array![0.0, 0.0]), array![0.0, 0.0]);
    }
}
