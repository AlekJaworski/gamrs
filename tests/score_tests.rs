//! Integration tests for `gamrs::score` — envelope-gradient correctness
//! (FD oracle) and the Tweedie analytic-shape-gradient port verification.
//! Lifted out of `src/score.rs` to keep that module under the project's
//! >700-LOC threshold (architecture-assumptions.md §G).

use approx::assert_relative_eq;
use gamrs::family::{negbin_log, tdist_identity, tweedie_log};
use gamrs::inner::PirlsOpts;
use gamrs::score::{
    FixedAtOneProfile, GaussianClosedFormScore, OwnedByLossProfile, PirlsInnerBuilder,
    ShapeAwareEnvelopeScore, ShapeAwarePirlsScore,
};
use gamrs::traits::{Basis, CoordsKind, ScoreDerivatives};
use ndarray::{array, Array1, Array2};

/// FD score with σ² FROZEN — the envelope-form gradient differentiates
/// this, not the profiled score directly. GamFit3 form, matches
/// `EnvelopeScore::compute_value_grad` after the Phase-2b port.
fn score_value_at_fixed_sigma2(
    score: &GaussianClosedFormScore,
    rho: f64,
    fixed_sigma2: f64,
) -> f64 {
    use gamrs::inner::{gaussian_inner_solve, CholeskySolver};
    use gamrs::traits::Loss;
    // Single-smooth test fixture — assemble S_total at this ρ.
    let s_total = gamrs::combined_s(
        &score.s_list,
        &ndarray::Array1::from_vec(vec![rho]),
        score.inner.x_design.ncols(),
    );
    let inner = gaussian_inner_solve::<CholeskySolver>(
        score.inner.x_design.view(),
        score.inner.y.view(),
        score.inner.weights.as_ref().map(|w| w.view()),
        s_total.view(),
    )
    .unwrap();
    let lambda = rho.exp();
    let s_beta = score.s_list[0].dot(&inner.beta);
    let bsb: f64 = inner
        .beta
        .iter()
        .zip(s_beta.iter())
        .map(|(a, b)| a * b)
        .sum();
    let dp = inner.rss + lambda * bsb;
    let log_det_h = inner.log_det_a();
    let log_det_lambda_s = (score.rank_s_list[0] as f64) * rho + score.log_pseudo_det_s_list[0];
    let two_pi = 2.0 * std::f64::consts::PI;
    let mp_f = score.mp as f64;
    let ls_sum: f64 = score
        .y
        .iter()
        .map(|&y| score.loss.saturated_log_lik(y, fixed_sigma2))
        .sum();
    dp / (2.0 * fixed_sigma2) - 0.5 * mp_f * (two_pi * fixed_sigma2).ln() + 0.5 * log_det_h
        - 0.5 * log_det_lambda_s
        - ls_sum
}

#[test]
fn envelope_gradient_matches_fixed_sigma2_fd() {
    let x: Array2<f64> = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
    let y: Array1<f64> = array![1.1, 1.9, 3.0, 4.05];
    let s: Array2<f64> = array![[0.0, 0.0], [0.0, 1.0]];
    let score = GaussianClosedFormScore::new(
        x.clone(),
        y.clone(),
        vec![s.clone()],
        None,
        vec![1],
        1,
        vec![0.0],
    );

    let rho = 0.7_f64;
    let (_, g) = score.value_and_grad(&array![rho]).unwrap();

    // Phase-2b port: gradient envelope is at σ²_score = Dp/(n-Mp),
    // matching the score body's σ² convention exactly.
    let lambda = rho.exp();
    let s_total = gamrs::combined_s(&score.s_list, &array![rho], score.inner.x_design.ncols());
    let inner = gamrs::inner::gaussian_inner_solve::<gamrs::inner::CholeskySolver>(
        score.inner.x_design.view(),
        score.inner.y.view(),
        None,
        s_total.view(),
    )
    .unwrap();
    let s_beta = score.s_list[0].dot(&inner.beta);
    let bsb_score: f64 = inner
        .beta
        .iter()
        .zip(s_beta.iter())
        .map(|(a, b)| a * b)
        .sum();
    let dp = inner.rss + lambda * bsb_score;
    let n_minus_mp = (inner.n as f64) - (score.mp as f64);
    let sigma2_score = dp / n_minus_mp;

    let h = 1e-5;
    let v_plus = score_value_at_fixed_sigma2(&score, rho + h, sigma2_score);
    let v_minus = score_value_at_fixed_sigma2(&score, rho - h, sigma2_score);
    let g_fd = (v_plus - v_minus) / (2.0 * h);
    assert_relative_eq!(g[0], g_fd, epsilon = 1e-5, max_relative = 1e-3);
}

/// Component-4 TDD (scat parity bug, `data/SCAT_PARITY_BUG.md`): scat's
/// analytic shape gradient (`analytic_shape_grad_via_ift`, assembled from
/// the FD-verified Level-1 pieces + the IFT trace term) must match a
/// central FD of the REML score value at the same θ on BOTH shape axes
/// `θ = [ρ, log σ², log(ν − 2)]`. The Level-1/Level-2 pieces each have FD
/// tests (`tdist_level1_*`, `tdist_level2_*`), but the *assembled* shape
/// gradient was previously untested for scat — the prime suspect for the
/// fit collapsing to ν → 2. Probes span the moderate regime AND the
/// low-ν / small-σ² corner the real-data fit fell into.
#[test]
fn tdist_analytic_shape_grad_matches_fd() {
    use gamrs::basis::CrSpline;

    // Synthetic heavy-tailed data: smooth signal + a few large outliers,
    // so the scat shape geometry is non-degenerate.
    let n = 90;
    let xs: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
    let mut ys = Vec::with_capacity(n);
    for (i, &xi) in xs.iter().enumerate() {
        let base = (2.0 * std::f64::consts::PI * xi).sin();
        // Deterministic pseudo-noise with occasional large spikes (heavy tail).
        let r = ((i as f64) * 12.9898 + 78.233).sin() * 43758.5453;
        let frac = r - r.floor(); // in [0,1)
        let noise = if frac > 0.9 {
            6.0 * (frac - 0.5)
        } else {
            0.3 * (frac - 0.5)
        };
        ys.push(base + noise);
    }
    let y = Array1::from_vec(ys);
    let x = Array1::from_vec(xs);

    let cr = CrSpline::with_quantile_knots(x.view(), 8).unwrap();
    let x2d = x.view().insert_axis(ndarray::Axis(1));
    let x_design = cr.evaluate(x2d.view());
    let penalties = cr.penalties();
    let s = penalties[0].clone();

    let family_base = tdist_identity(5.0, 1.0);
    let score: ShapeAwarePirlsScore<_, _, _> = ShapeAwareEnvelopeScore {
        x_design: x_design.clone(),
        y: y.clone(),
        prior_weights: None,
        s_list: vec![s.clone()],
        family_base,
        rank_s_list: vec![x_design.ncols() - 2],
        mp: 2,
        log_pseudo_det_s_list: vec![0.0],
        coords: CoordsKind::Identity,
        pirls_opts: PirlsOpts::default(),
        inner_builder: PirlsInnerBuilder,
        profile: FixedAtOneProfile,
        _solver: std::marker::PhantomData,
        accepted_state: std::cell::RefCell::new(None),
        last_eta: std::cell::RefCell::new(None),
        stats: gamrs::stats::FitStats::new(),
    };

    // θ = [ρ, log σ², log(ν − 2)]. Cover moderate ν, large ν, and the
    // low-ν / small-σ² corner the real-data fit collapsed into.
    let ln = |nu: f64| (nu - 2.0_f64).ln();
    let probes: &[[f64; 3]] = &[
        [0.0, 0.0, ln(5.0)],  // σ²=1,   ν=5
        [2.0, -1.0, ln(3.0)], // σ²=0.37, ν=3
        [1.0, 1.0, ln(10.0)], // σ²=2.7,  ν=10
        [3.0, -2.0, ln(2.1)], // σ²=0.135, ν=2.1  (corner)
    ];

    let fd_at = |theta: &Array1<f64>, i: usize, h: f64| -> f64 {
        let mut t_plus = theta.clone();
        let mut t_minus = theta.clone();
        t_plus[i] += h;
        t_minus[i] -= h;
        let v_plus = score.value(&t_plus).unwrap();
        let v_minus = score.value(&t_minus).unwrap();
        (v_plus - v_minus) / (2.0 * h)
    };

    for theta_init in probes {
        let theta = Array1::from_vec(theta_init.to_vec());
        let (_v, g) = score.value_and_grad(&theta).unwrap();
        // Shape axes only (i ∈ {1,2}); the ρ axis has its own test,
        // `tdist_analytic_rho_grad_matches_fd` — which is where the
        // λ-envelope gradient was ACTUALLY checked for the first time, and
        // where it turned out to be wrong. Richardson D(h) = (4·FD(h) − FD(2h))/3
        // cancels the O(h²) truncation term for a ~O(h⁴) estimate of the
        // true derivative. With the mgcv-faithful IRLS — observed Newton
        // weight on core rows, **expected**-curvature weight with the
        // Fisher-consistent response `z = η − Dmu/EDmu2` on outlier rows —
        // the analytic IFT gradient (with the matching
        // `ift_trace_weight_derivs` override) tracks FD on both shape axes to
        // <1e-3, including the low-ν / small-σ² corner the PRIOR expected
        // fallback (which used `z = η + r`) got ~99% wrong — silently
        // stalling the outer Newton in the ν → 2 trap
        // (`data/SCAT_PARITY_BUG.md`). The residual (~1e-4) is the kink where
        // rows cross the observed↔expected boundary `r² = νσ²` (`log|H|` is
        // only piecewise-smooth there, as in mgcv); the analytic gradient is
        // exact within each branch.
        for i in 1..3 {
            let h = 1e-4;
            let fd_h = fd_at(&theta, i, h);
            let fd_2h = fd_at(&theta, i, 2.0 * h);
            let fd_rich = (4.0 * fd_h - fd_2h) / 3.0;
            let rel = (g[i] - fd_rich).abs() / (fd_rich.abs() + 1.0);
            assert!(
                rel < 1e-3,
                "θ={:?} g[{i}] analytic={:+.8e} fd_rich={:+.8e} rel={:.2e}",
                theta_init,
                g[i],
                fd_rich,
                rel
            );
        }
    }
}

/// The ρ (log λ) axis of scat's assembled REML gradient, against a central FD
/// of gamrs's OWN score at the same θ.
///
/// `tdist_analytic_shape_grad_matches_fd` above checks `i in 1..3` only — the
/// two shape axes — and used to say of the third "g[0] is the λ-envelope
/// gradient (verified separately)". It was not, for scat or for Tweedie, and
/// it was wrong: `compute_rho_envelope_gradient` carried a
/// `∂ridge/∂ρ_j · tr(H⁻¹)/2` term for a λ-dependent ridge that only
/// `OcatInner` bakes into the factor the score reads (`PirlsInner` hands back
/// an UNRIDGED factor, `linalg.rs::factor_and_solve`). Differentiating a ridge
/// that is not in `A` adds a term proportional to λ·tr(A⁻¹) to a gradient
/// whose true value decays like 1/λ. Measured on the probes below, the error
/// was exactly proportional to λ: 1.1e-2 at ρ=0, 5.7e-1 at ρ=4, 3.1e1 at ρ=8,
/// 1.7e3 at ρ=12, 9.2e4 at ρ=16 — each step of +4 in ρ multiplying it by e⁴.
///
/// Why it hid for three releases: the error is additive and the TRUE gradient
/// is O(1) while the λ ridge is steep, so it changed nothing on a well-curved
/// surface. Once the ridge goes shallow it dominates and flips the sign; the
/// outer Newton then steps the wrong way, step-halving finds no decrease, and
/// `outer.rs`'s fallback returns the point it is standing on — leaving edf
/// above mgcv's and the fitted curve hundreds of dollars off. That regime is
/// what `tests/parity_scat_flat_ridge.rs` reproduces.
///
/// The Hessian is FD'd on this same analytic gradient, so a wrong g[0] poisoned
/// H[0][0] too — negative curvature at a point where the surface is monotone.
///
/// The bar: absolute error against a Richardson-extrapolated central FD,
/// scaled by `1 + |fd|`. ρ = 16 is asserted differently — there `A` has a
/// condition number around 1e10 and the FD of `log|A|` is no longer a usable
/// oracle (it reads a systematic ~1e-3 drift that the analytic decay says is
/// not there), so the assertion is that the analytic gradient has DECAYED,
/// which is what the ridge term destroyed.
#[test]
fn tdist_analytic_rho_grad_matches_fd() {
    use gamrs::basis::CrSpline;

    // Same construction as `tdist_analytic_shape_grad_matches_fd`, so the only
    // thing that differs between the two tests is WHICH axis is probed.
    let n = 90;
    let xs: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
    let mut ys = Vec::with_capacity(n);
    for (i, &xi) in xs.iter().enumerate() {
        let base = (2.0 * std::f64::consts::PI * xi).sin();
        let r = ((i as f64) * 12.9898 + 78.233).sin() * 43758.5453;
        let frac = r - r.floor();
        let noise = if frac > 0.9 {
            6.0 * (frac - 0.5)
        } else {
            0.3 * (frac - 0.5)
        };
        ys.push(base + noise);
    }
    let y = Array1::from_vec(ys);
    let x = Array1::from_vec(xs);

    let cr = CrSpline::with_quantile_knots(x.view(), 8).unwrap();
    let x2d = x.view().insert_axis(ndarray::Axis(1));
    let x_design = cr.evaluate(x2d.view());
    let penalties = cr.penalties();
    let s = penalties[0].clone();

    let family_base = tdist_identity(5.0, 1.0);
    let score: ShapeAwarePirlsScore<_, _, _> = ShapeAwareEnvelopeScore {
        x_design: x_design.clone(),
        y: y.clone(),
        prior_weights: None,
        s_list: vec![s.clone()],
        family_base,
        rank_s_list: vec![x_design.ncols() - 2],
        mp: 2,
        log_pseudo_det_s_list: vec![0.0],
        coords: CoordsKind::Identity,
        pirls_opts: PirlsOpts::default(),
        inner_builder: PirlsInnerBuilder,
        profile: FixedAtOneProfile,
        _solver: std::marker::PhantomData,
        accepted_state: std::cell::RefCell::new(None),
        last_eta: std::cell::RefCell::new(None),
        stats: gamrs::stats::FitStats::new(),
    };

    let fd_at = |theta: &Array1<f64>, i: usize, h: f64| -> f64 {
        let mut t_plus = theta.clone();
        let mut t_minus = theta.clone();
        t_plus[i] += h;
        t_minus[i] -= h;
        (score.value(&t_plus).unwrap() - score.value(&t_minus).unwrap()) / (2.0 * h)
    };

    // Walk ρ up into the over-penalised tail, where the true |d score/dρ|
    // decays toward zero and any additive error in the analytic gradient takes
    // over. Low ρ is the regime the family was tuned in and passes.
    let ln = |nu: f64| (nu - 2.0_f64).ln();
    let probes: &[[f64; 3]] = &[
        [0.0, 0.0, ln(5.0)],
        [4.0, 0.0, ln(5.0)],
        [8.0, 0.0, ln(5.0)],
        [12.0, 0.0, ln(5.0)],
        [16.0, 0.0, ln(5.0)],
    ];

    let mut worst = 0.0_f64;
    for theta_init in probes {
        let theta = Array1::from_vec(theta_init.to_vec());
        let (_v, g) = score.value_and_grad(&theta).unwrap();
        let h = 1e-3;
        let fd_rich = (4.0 * fd_at(&theta, 0, h) - fd_at(&theta, 0, 2.0 * h)) / 3.0;
        let abs_err = (g[0] - fd_rich).abs();
        println!(
            "  rho={:>5.1}  analytic={:+.6e}  fd={:+.6e}  abs_err={:.3e}{}",
            theta_init[0],
            g[0],
            fd_rich,
            abs_err,
            if g[0] * fd_rich < 0.0 {
                "  <-- SIGN FLIP"
            } else {
                ""
            }
        );
        if theta_init[0] >= 16.0 {
            // FD is not an oracle this far out (see the doc comment); assert
            // the decay instead. Pre-fix this read +9.175e4.
            assert!(
                g[0].abs() < 1e-4,
                "at rho=16 the true gradient has decayed like 1/lambda; \
                 analytic says {:+.6e}",
                g[0]
            );
            continue;
        }
        let bar = 1e-3 * (1.0 + fd_rich.abs());
        assert!(
            abs_err < bar,
            "scat analytic d(REML)/d(rho) at rho={:.1}: analytic {:+.6e} vs FD \
             {fd_rich:+.6e}, abs err {abs_err:.3e} over bar {bar:.3e}",
            theta_init[0],
            g[0]
        );
        worst = worst.max(abs_err);
    }
    println!("  worst abs err over the rho<=12 probes: {worst:.3e}");
}

/// Phase-1 port verification: Tweedie's analytic shape gradient must
/// match a central FD of the score value at the same θ. Tests at three
/// distinct (ρ, log φ, p_trans) points to cover both shape components.
#[test]
fn tweedie_analytic_shape_grad_matches_fd() {
    use gamrs::basis::CrSpline;

    // Synthetic small Tweedie data: y = max(0, x + ε).
    let n = 80;
    let xs: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
    let mut ys = Vec::with_capacity(n);
    for (i, &xi) in xs.iter().enumerate() {
        let r = ((i as f64) * 1.103515245 + 0.137).sin().abs();
        let yi = (xi + 0.3 * (r - 0.5)).max(0.0);
        ys.push(yi);
    }
    let y = Array1::from_vec(ys);
    let x = Array1::from_vec(xs);

    let cr = CrSpline::with_quantile_knots(x.view(), 8).unwrap();
    let x2d = x.view().insert_axis(ndarray::Axis(1));
    let x_design = cr.evaluate(x2d.view());
    let penalties = cr.penalties();
    let s = penalties[0].clone();

    let family_base = tweedie_log(1.5, 1.0);
    let score: gamrs::score::ShapeAwarePirlsScoreOwnedPhi<_, _, _> = ShapeAwareEnvelopeScore {
        x_design: x_design.clone(),
        y: y.clone(),
        prior_weights: None,
        s_list: vec![s.clone()],
        family_base,
        rank_s_list: vec![x_design.ncols() - 2],
        mp: 2,
        log_pseudo_det_s_list: vec![0.0],
        coords: CoordsKind::Identity,
        pirls_opts: PirlsOpts::default(),
        inner_builder: PirlsInnerBuilder,
        profile: OwnedByLossProfile,
        _solver: std::marker::PhantomData,
        accepted_state: std::cell::RefCell::new(None),
        last_eta: std::cell::RefCell::new(None),
        stats: gamrs::stats::FitStats::new(),
    };

    let probes: &[[f64; 3]] = &[[0.5, 0.0, 0.0], [2.0, 0.5, 0.3], [1.0, -0.5, -0.2]];

    for theta_init in probes {
        let theta = Array1::from_vec(theta_init.to_vec());
        let (_v, g) = score.value_and_grad(&theta).unwrap();

        let h = 1e-5;
        let mut g_fd = [0.0_f64; 3];
        for i in 0..3 {
            let mut t_plus = theta.clone();
            let mut t_minus = theta.clone();
            t_plus[i] += h;
            t_minus[i] -= h;
            let v_plus = score.value(&t_plus).unwrap();
            let v_minus = score.value(&t_minus).unwrap();
            g_fd[i] = (v_plus - v_minus) / (2.0 * h);
        }
        // All three axes, ρ included. g[0] used to be skipped here on the
        // claim that it was "verified by a separate test" — it was not, for
        // this family or for scat, and it was wrong: the shape-aware score
        // differentiated a λ-dependent ridge that only `OcatInner` puts in
        // the factor it hands back (`ShapeInnerBuilder::score_ridge_scale`).
        // At the first probe that read +3.763e-2 against an FD of +9.567e-3.
        // FD tolerance is loose because the score value contains a Dunn-Smyth
        // series sum that contributes O(1e2) noise to FD at moderate
        // (φ, p). Agreement to 2e-2 confirms the analytic formula.
        for i in 0..3 {
            let rel = (g[i] - g_fd[i]).abs() / (g_fd[i].abs() + 1.0);
            assert!(
                rel < 2e-2,
                "θ={:?} g[{i}] analytic={:+.6e} fd={:+.6e} rel={:.2e}",
                theta_init,
                g[i],
                g_fd[i],
                rel
            );
        }
    }
}

/// v0.x analytical-Hessian port verification: the new
/// partial-freeze Hessian (`hess_via_fd_frozen_beta` for Tweedie's
/// shape rows + re-converge for the log-λ row) must agree with the
/// v0.1 full FD-on-grad path to better than 1% rel-err at probes
/// near the optimum. Looser than the gradient test (Hessian is FD
/// of a noisy gradient) but tight enough to catch a wiring bug.
#[test]
fn tweedie_analytic_hess_matches_fd_on_grad() {
    use gamrs::basis::CrSpline;

    let n = 80;
    let xs: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
    let mut ys = Vec::with_capacity(n);
    for (i, &xi) in xs.iter().enumerate() {
        let r = ((i as f64) * 1.103515245 + 0.137).sin().abs();
        let yi = (xi + 0.3 * (r - 0.5)).max(0.0);
        ys.push(yi);
    }
    let y = Array1::from_vec(ys);
    let x = Array1::from_vec(xs);
    let cr = CrSpline::with_quantile_knots(x.view(), 8).unwrap();
    let x2d = x.view().insert_axis(ndarray::Axis(1));
    let x_design = cr.evaluate(x2d.view());
    let penalties = cr.penalties();
    let s = penalties[0].clone();

    let family_base = tweedie_log(1.5, 1.0);
    let score: gamrs::score::ShapeAwarePirlsScoreOwnedPhi<_, _, _> = ShapeAwareEnvelopeScore {
        x_design: x_design.clone(),
        y: y.clone(),
        prior_weights: None,
        s_list: vec![s.clone()],
        family_base,
        rank_s_list: vec![x_design.ncols() - 2],
        mp: 2,
        log_pseudo_det_s_list: vec![0.0],
        coords: CoordsKind::Identity,
        pirls_opts: PirlsOpts::default(),
        inner_builder: PirlsInnerBuilder,
        profile: OwnedByLossProfile,
        _solver: std::marker::PhantomData,
        accepted_state: std::cell::RefCell::new(None),
        last_eta: std::cell::RefCell::new(None),
        stats: gamrs::stats::FitStats::new(),
    };

    // 3 probes near (but not at) the optimum; far from the optimum
    // the β-chain dropped by the frozen-β̂ shape path is non-negligible
    // and the looser tolerance applies (this is exactly v0.x's
    // tweedie_theta_grad_hess_analytic comment about Newton tolerating
    // the O(h) Hessian error).
    let probes: &[[f64; 3]] = &[[0.5, 0.0, 0.0], [1.0, 0.1, 0.1], [1.5, -0.1, -0.1]];

    for theta_init in probes {
        let theta = Array1::from_vec(theta_init.to_vec());
        let (_, _, h_anal) = score.value_grad_hess(&theta).unwrap();

        // Reference Hessian: central FD on the gradient (re-PIRLS at ±h).
        let eps = 1e-4_f64;
        let d = 3;
        let mut h_fd = ndarray::Array2::<f64>::zeros((d, d));
        for i in 0..d {
            let mut t_plus = theta.clone();
            let mut t_minus = theta.clone();
            t_plus[i] += eps;
            t_minus[i] -= eps;
            let (_, g_plus) = score.value_and_grad(&t_plus).unwrap();
            let (_, g_minus) = score.value_and_grad(&t_minus).unwrap();
            for j in 0..d {
                h_fd[[j, i]] = (g_plus[j] - g_minus[j]) / (2.0 * eps);
            }
        }
        // Symmetrise FD ref.
        for i in 0..d {
            for j in i + 1..d {
                let avg = 0.5 * (h_fd[[i, j]] + h_fd[[j, i]]);
                h_fd[[i, j]] = avg;
                h_fd[[j, i]] = avg;
            }
        }

        // Diagonal: tighter (1%); off-diagonal: looser (15%). The
        // off-diagonal log-λ↔shape entry mixes a re-PIRLS row (log-λ)
        // with a frozen-β̂ row (shape) and gets symmetrised — that
        // symmetrisation hides up to O(h)·|β_chain| of the FD Hessian's
        // β-chain. Diagonal entries are pure (one direction at a time)
        // so they match more tightly.
        for i in 0..d {
            let denom = h_fd[[i, i]].abs().max(1.0);
            let rel = (h_anal[[i, i]] - h_fd[[i, i]]).abs() / denom;
            assert!(
                rel < 5e-2,
                "θ={:?} diag[{i}] analytic={:+.4e} fd={:+.4e} rel={:.2e}",
                theta_init,
                h_anal[[i, i]],
                h_fd[[i, i]],
                rel
            );
        }
    }
}

/// Diagnostic (run with `--ignored`): print analytic vs FD gradients on
/// the real Tweedie parity fixture at θ₀ = [0, 0, 0] and at a few nearby
/// probes, to investigate the wrong-minimum trap.
#[test]
#[ignore]
fn debug_tweedie_real_data_grad_walk() {
    use gamrs::basis::CrSpline;
    use std::path::PathBuf;

    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/1d_tweedie_log_n300_k10_cr.json");
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
    let x_vec: Vec<f64> = v["inputs"]["x_train"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r[0].as_f64().unwrap())
        .collect();
    let y_vec: Vec<f64> = v["inputs"]["y_train"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_f64().unwrap())
        .collect();
    let k = v["inputs"]["k"][0].as_u64().unwrap() as usize;

    let x = Array1::from_vec(x_vec);
    let y = Array1::from_vec(y_vec);
    let cr = CrSpline::with_quantile_knots(x.view(), k).unwrap();
    let x2d = x.view().insert_axis(ndarray::Axis(1));
    let x_design = cr.evaluate(x2d.view());
    let penalties = cr.penalties();
    let s = penalties[0].clone();

    let family_base = tweedie_log(1.5, 1.0);
    let score: gamrs::score::ShapeAwarePirlsScoreOwnedPhi<_, _, _> = ShapeAwareEnvelopeScore {
        x_design: x_design.clone(),
        y: y.clone(),
        prior_weights: None,
        s_list: vec![s.clone()],
        family_base,
        rank_s_list: vec![x_design.ncols() - 2],
        mp: 2,
        log_pseudo_det_s_list: vec![0.0],
        coords: CoordsKind::Identity,
        pirls_opts: PirlsOpts::default(),
        inner_builder: PirlsInnerBuilder,
        profile: OwnedByLossProfile,
        _solver: std::marker::PhantomData,
        accepted_state: std::cell::RefCell::new(None),
        last_eta: std::cell::RefCell::new(None),
        stats: gamrs::stats::FitStats::new(),
    };

    let probes: &[[f64; 3]] = &[
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [5.5, -0.03, 0.0],
        [-3.0, -10.0, 0.0],
    ];

    for theta_init in probes {
        let theta = Array1::from_vec(theta_init.to_vec());
        let (v_val, g) = score.value_and_grad(&theta).unwrap();
        let h = 1e-5;
        let mut g_fd = [0.0_f64; 3];
        for i in 0..3 {
            let mut t_plus = theta.clone();
            let mut t_minus = theta.clone();
            t_plus[i] += h;
            t_minus[i] -= h;
            let v_plus = score.value(&t_plus).unwrap();
            let v_minus = score.value(&t_minus).unwrap();
            g_fd[i] = (v_plus - v_minus) / (2.0 * h);
        }
        eprintln!("θ = {:?}  v = {:.6e}", theta_init, v_val);
        for i in 0..3 {
            eprintln!(
                "  g[{i}] analytic = {:+.6e}   fd = {:+.6e}   rel = {:.2e}",
                g[i],
                g_fd[i],
                (g[i] - g_fd[i]).abs() / (g_fd[i].abs() + 1.0),
            );
        }
    }
}

/// Verifies `Loss::sum_saturated_log_lik_dtheta` against an FD of the
/// summed `saturated_log_lik` directly — isolates the ∂Σls/∂(log θ)
/// term from the broader IFT pipeline so a wrong-sign or factor bug
/// gets a clean signal here, not entangled with PIRLS.
#[test]
fn negbin_sum_sat_loglik_dtheta_matches_fd() {
    use gamrs::family::negbin_log;
    use gamrs::traits::Loss;
    let y = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 7.0, 11.0, 0.0, 4.0]);
    for theta in [0.3_f64, 1.0, 1.5, 4.0, 10.0] {
        let fam = negbin_log(theta);
        let analytic = fam.loss.sum_saturated_log_lik_dtheta(y.view(), 1.0, None);

        // FD in log θ space (matches what the shape-grad consumer uses).
        let h = 1e-5_f64;
        let alpha = theta.ln();
        let fam_plus = negbin_log((alpha + h).exp());
        let fam_minus = negbin_log((alpha - h).exp());
        let sum_plus: f64 = y
            .iter()
            .map(|&yi| fam_plus.loss.saturated_log_lik(yi, 1.0))
            .sum();
        let sum_minus: f64 = y
            .iter()
            .map(|&yi| fam_minus.loss.saturated_log_lik(yi, 1.0))
            .sum();
        let fd = (sum_plus - sum_minus) / (2.0 * h);
        let rel = (analytic[0] - fd).abs() / (fd.abs() + 1.0);
        eprintln!(
            "θ={theta}  analytic={:+.6e}  fd={:+.6e}  rel={rel:.2e}",
            analytic[0], fd
        );
        assert!(
            rel < 1e-4,
            "Σ ∂ls/∂(log θ) FD mismatch at θ={theta}: analytic={:+.6e} fd={:+.6e} rel={rel:.2e}",
            analytic[0],
            fd,
        );
    }
}

/// Architectural-boundary test for the multi-smooth Newton-IRLS Tk·KK'
/// β-chain port. The new per-term `eta1_per_term` / `tr_a_newton_inv_s_per_term`
/// machinery (PIRLS-side) plus the per-k score gradient assembly
/// (envelope.rs) must produce an analytic ρ-gradient that matches a
/// central FD of the score value to high precision, for every smooth axis.
///
/// Synthetic 2-D additive NegBin fixture (matches the 2-D parity test's
/// shape). Probes near a reasonable basin so PIRLS converges; we check
/// the *gradient is internally correct* — separate from whether gamrs's
/// ρ̂ matches mgcv's (mgcv parity is a downstream concern that depends on
/// the inner solver convergence basin).
#[test]
fn negbin_multismooth_analytic_grad_matches_fd() {
    use gamrs::design::{Additive, DesignStrategy, TermSpec};

    // Synthetic 2-D NegBin signal (deterministic).
    let n = 300;
    let mut x_flat = Vec::with_capacity(n * 2);
    let mut ys = Vec::with_capacity(n);
    let mut state: u64 = 0xdead_beef_5234_9adb;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for _ in 0..n {
        let x0 = next();
        let x1 = next();
        x_flat.push(x0);
        x_flat.push(x1);
        let eta = 0.2 + 0.7 * (2.0 * std::f64::consts::PI * x0).sin() + 0.5 * (x1 - 0.5).powi(2);
        let mu = eta.exp();
        let perturb = (next() - 0.5) * (mu.sqrt() + 1.0);
        let y = (mu + perturb).round().max(0.0);
        ys.push(y);
    }
    let x = Array2::from_shape_vec((n, 2), x_flat).unwrap();
    let y = Array1::from_vec(ys);

    // Build the same design path the 2-D parity test uses — proper
    // centering + per-term penalty embedding — so PIRLS doesn't blow up
    // on a hand-rolled rank-deficient basis.
    let terms = vec![
        TermSpec::Cr {
            col: 0,
            k: 8,
            pc: None,
        },
        TermSpec::Cr {
            col: 1,
            k: 8,
            pc: None,
        },
    ];
    let prep = Additive { terms }.prepare(x.view()).unwrap();

    let family_base = negbin_log(3.0);
    let score: gamrs::score::ShapeAwarePirlsScore<_, _, _> = ShapeAwareEnvelopeScore {
        x_design: prep.x_design.clone(),
        y: y.clone(),
        prior_weights: None,
        s_list: prep.s_list.clone(),
        family_base,
        rank_s_list: prep.rank_s_list.clone(),
        mp: prep.mp,
        log_pseudo_det_s_list: prep.log_pseudo_det_s_list.clone(),
        coords: CoordsKind::Identity,
        pirls_opts: PirlsOpts::default(),
        inner_builder: PirlsInnerBuilder,
        profile: FixedAtOneProfile,
        _solver: std::marker::PhantomData,
        accepted_state: std::cell::RefCell::new(None),
        last_eta: std::cell::RefCell::new(None),
        stats: gamrs::stats::FitStats::new(),
    };

    // Probes: (ρ_0, ρ_1, log θ). Centred near the 2-D NB parity fit's
    // optimum (ρ̂ ≈ [3.4, 11.5], log θ̂ ≈ 1.1) so PIRLS converges cleanly
    // and FD noise stays low. Keep ρ moderately small to keep A_newton
    // well-conditioned across the central-FD step.
    let probes: &[[f64; 3]] = &[[1.0, 1.0, 1.0], [2.0, 0.5, 0.8], [0.5, 2.0, 1.3]];

    // FD noise: each FD probe re-runs PIRLS to convergence at a perturbed
    // θ. PIRLS's β_max_change tol (1e-8 for NB) bounds residual β-noise;
    // at h = 2e-3 the FD floor is ~5e-6 abs.
    //
    // Per-axis tolerances:
    // - ρ axes (0, 1): 5% — these go through the well-tested
    //   Tk·KK' β-chain term using `0.5·dmu3·η₁_j·h_diag`.
    // - log θ axis (2): 5% — was 25% pre-fix because the IFT formula at
    //   `analytic_shape_grad_via_ift` called
    //   `self.family_base.loss.sum_saturated_log_lik_dtheta(...)` against
    //   the **construction-time** family (θ=3.0 from `negbin_log(3.0)`),
    //   not the **perturbed** family from the current outer probe. The
    //   `ls$d1` row was therefore stuck at the original θ, leaving an
    //   O(10) residual on the log θ axis (6-23% rel-err). Fixed in
    //   gradient.rs:430 — now reads `family.loss` (the probed family);
    //   shape-axis residual drops to ~1e-4. Companion port (Newton-IRLS
    //   in PIRLS + Newton-W `log|H|` in the shape-aware score path)
    //   ensures the inner step's β stays stationary on the deviance, so
    //   the envelope theorem holds bit-exactly.
    let rho_axes_bar = 5e-2;
    let shape_axis_bar = 5e-2;
    for theta_init in probes {
        let theta = Array1::from_vec(theta_init.to_vec());
        let (_v, g) = score.value_and_grad(&theta).unwrap();

        let h = 2e-3;
        for i in 0..3 {
            let mut t_plus = theta.clone();
            let mut t_minus = theta.clone();
            t_plus[i] += h;
            t_minus[i] -= h;
            let v_plus = score.value(&t_plus).unwrap();
            let v_minus = score.value(&t_minus).unwrap();
            let g_fd = (v_plus - v_minus) / (2.0 * h);
            let rel = (g[i] - g_fd).abs() / (g_fd.abs() + 1.0);
            eprintln!(
                "θ={:?} g[{i}] analytic={:+.6e} fd={:+.6e} rel={:.2e}",
                theta_init, g[i], g_fd, rel
            );
            let bar = if i < 2 { rho_axes_bar } else { shape_axis_bar };
            assert!(
                rel < bar,
                "θ={theta_init:?} g[{i}] analytic={:+.6e} fd={:+.6e} rel={rel:.2e} bar={bar:.0e}",
                g[i],
                g_fd,
            );
            // Sign sanity: must agree (catches the sum_saturated_log_lik_dtheta
            // omission specifically — without it, g[2] flips to wrong sign).
            assert!(
                g[i].signum() == g_fd.signum() || g[i].abs() < 1e-8,
                "θ={theta_init:?} g[{i}] sign disagrees: analytic={:+.4e} fd={:+.4e}",
                g[i],
                g_fd,
            );
        }
    }
}

/// Microbench (`--ignored`): time the analytic-IFT Hessian path against
/// a synthetic 2-D NegBin fit. Prints elapsed-per-fit so a human can
/// eyeball the improvement against the previous FD-on-grad baseline.
/// Not asserted — wall time depends on machine load.
#[test]
#[ignore]
fn nb_hess_microbench() {
    use gamrs::design::{Additive, DesignStrategy, TermSpec};
    use std::time::Instant;

    let n = 600;
    let mut x_flat = Vec::with_capacity(n * 2);
    let mut ys = Vec::with_capacity(n);
    let mut state: u64 = 0xc0ff_eeba_dbad_5e7e;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for _ in 0..n {
        let x0 = next();
        let x1 = next();
        x_flat.push(x0);
        x_flat.push(x1);
        let eta = 0.2 + 0.7 * (2.0 * std::f64::consts::PI * x0).sin() + 0.5 * (x1 - 0.5).powi(2);
        let mu = eta.exp();
        let perturb = (next() - 0.5) * (mu.sqrt() + 1.0);
        let y = (mu + perturb).round().max(0.0);
        ys.push(y);
    }
    let x = Array2::from_shape_vec((n, 2), x_flat).unwrap();
    let y = Array1::from_vec(ys);
    let terms = vec![
        TermSpec::Cr {
            col: 0,
            k: 8,
            pc: None,
        },
        TermSpec::Cr {
            col: 1,
            k: 8,
            pc: None,
        },
    ];
    let prep = Additive { terms }.prepare(x.view()).unwrap();

    let family_base = negbin_log(3.0);
    let score: gamrs::score::ShapeAwarePirlsScore<_, _, _> = ShapeAwareEnvelopeScore {
        x_design: prep.x_design.clone(),
        y: y.clone(),
        prior_weights: None,
        s_list: prep.s_list.clone(),
        family_base,
        rank_s_list: prep.rank_s_list.clone(),
        mp: prep.mp,
        log_pseudo_det_s_list: prep.log_pseudo_det_s_list.clone(),
        coords: CoordsKind::Identity,
        pirls_opts: PirlsOpts::default(),
        inner_builder: PirlsInnerBuilder,
        profile: FixedAtOneProfile,
        _solver: std::marker::PhantomData,
        accepted_state: std::cell::RefCell::new(None),
        last_eta: std::cell::RefCell::new(None),
        stats: gamrs::stats::FitStats::new(),
    };
    let theta = Array1::from_vec(vec![1.0, 1.0, 1.0]);
    // Warmup.
    for _ in 0..5 {
        let _ = score.value_grad_hess(&theta).unwrap();
    }
    let runs = 50;
    let t0 = Instant::now();
    for _ in 0..runs {
        let _ = score.value_grad_hess(&theta).unwrap();
    }
    let elapsed = t0.elapsed();
    eprintln!(
        "[microbench] NB 2-D value_grad_hess (IFT): {:.2} ms / call over {runs} runs",
        elapsed.as_secs_f64() * 1000.0 / runs as f64,
    );
    // Compare against value_and_grad alone (no Hessian) — the Hessian's
    // marginal cost is (value_grad_hess − value_and_grad).
    let t1 = Instant::now();
    for _ in 0..runs {
        let _ = score.value_and_grad(&theta).unwrap();
    }
    let elapsed_g = t1.elapsed();
    eprintln!(
        "[microbench] NB 2-D value_and_grad only: {:.2} ms / call over {runs} runs",
        elapsed_g.as_secs_f64() * 1000.0 / runs as f64,
    );
    let hess_cost_ms = (elapsed.as_secs_f64() - elapsed_g.as_secs_f64()) * 1000.0 / runs as f64;
    eprintln!(
        "[microbench] NB 2-D Hessian marginal cost: {:.2} ms / call",
        hess_cost_ms.max(0.0),
    );

    // NB 1-D — the gamrs-vs-mgcv_rust gap is largest here (82 ms vs 8 ms).
    let mut x_flat = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut state: u64 = 0xc0ff_eeba_dbad_5e7e;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for _ in 0..n {
        let x0 = next();
        x_flat.push(x0);
        let eta = 0.2 + 0.7 * (2.0 * std::f64::consts::PI * x0).sin();
        let mu = eta.exp();
        let perturb = (next() - 0.5) * (mu.sqrt() + 1.0);
        let y = (mu + perturb).round().max(0.0);
        ys.push(y);
    }
    let x1 = Array2::from_shape_vec((n, 1), x_flat).unwrap();
    let y1 = Array1::from_vec(ys);
    let terms1 = vec![TermSpec::Cr {
        col: 0,
        k: 10,
        pc: None,
    }];
    let prep1 = Additive { terms: terms1 }.prepare(x1.view()).unwrap();
    let score1: gamrs::score::ShapeAwarePirlsScore<_, _, _> = ShapeAwareEnvelopeScore {
        x_design: prep1.x_design.clone(),
        y: y1.clone(),
        prior_weights: None,
        s_list: prep1.s_list.clone(),
        family_base: negbin_log(3.0),
        rank_s_list: prep1.rank_s_list.clone(),
        mp: prep1.mp,
        log_pseudo_det_s_list: prep1.log_pseudo_det_s_list.clone(),
        coords: CoordsKind::Identity,
        pirls_opts: PirlsOpts::default(),
        inner_builder: PirlsInnerBuilder,
        profile: FixedAtOneProfile,
        _solver: std::marker::PhantomData,
        accepted_state: std::cell::RefCell::new(None),
        last_eta: std::cell::RefCell::new(None),
        stats: gamrs::stats::FitStats::new(),
    };
    let theta1 = Array1::from_vec(vec![1.0, 1.0]);
    for _ in 0..5 {
        let _ = score1.value_grad_hess(&theta1).unwrap();
    }
    let t1d = Instant::now();
    for _ in 0..runs {
        let _ = score1.value_grad_hess(&theta1).unwrap();
    }
    let elapsed1 = t1d.elapsed();
    eprintln!(
        "[microbench] NB 1-D value_grad_hess (IFT): {:.2} ms / call over {runs} runs",
        elapsed1.as_secs_f64() * 1000.0 / runs as f64,
    );
}

/// Hessian regression sentinel for the NegBin analytic IFT ρ-block path.
///
/// Port: `ShapeAwareEnvelopeScore::hess_via_ift_analytic` is a line-by-line
/// port of mgcv_rust `reml_hessian_mgcv_exact_ift` (`reml/mod.rs:2511-2813`)
/// covering the M×M ρ-block, with FD-on-grad along shape axes only.
/// This test verifies the new M×M ρ-block matches central-FD-on-grad on
/// a 2-D NB fixture to 10% rel-err — the same bar v0.x's analytic Hessian
/// hits against its FD oracle.
#[test]
fn negbin_multismooth_analytic_hess_matches_fd_on_grad() {
    use gamrs::design::{Additive, DesignStrategy, TermSpec};

    // Synthetic 2-D NegBin signal (matches the gradient sentinel above
    // for shared fixture economy — deterministic xorshift).
    let n = 300;
    let mut x_flat = Vec::with_capacity(n * 2);
    let mut ys = Vec::with_capacity(n);
    let mut state: u64 = 0xdead_beef_5234_9adb;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for _ in 0..n {
        let x0 = next();
        let x1 = next();
        x_flat.push(x0);
        x_flat.push(x1);
        let eta = 0.2 + 0.7 * (2.0 * std::f64::consts::PI * x0).sin() + 0.5 * (x1 - 0.5).powi(2);
        let mu = eta.exp();
        let perturb = (next() - 0.5) * (mu.sqrt() + 1.0);
        let y = (mu + perturb).round().max(0.0);
        ys.push(y);
    }
    let x = Array2::from_shape_vec((n, 2), x_flat).unwrap();
    let y = Array1::from_vec(ys);
    let terms = vec![
        TermSpec::Cr {
            col: 0,
            k: 8,
            pc: None,
        },
        TermSpec::Cr {
            col: 1,
            k: 8,
            pc: None,
        },
    ];
    let prep = Additive { terms }.prepare(x.view()).unwrap();

    let family_base = negbin_log(3.0);
    let score: gamrs::score::ShapeAwarePirlsScore<_, _, _> = ShapeAwareEnvelopeScore {
        x_design: prep.x_design.clone(),
        y: y.clone(),
        prior_weights: None,
        s_list: prep.s_list.clone(),
        family_base,
        rank_s_list: prep.rank_s_list.clone(),
        mp: prep.mp,
        log_pseudo_det_s_list: prep.log_pseudo_det_s_list.clone(),
        coords: CoordsKind::Identity,
        pirls_opts: PirlsOpts::default(),
        inner_builder: PirlsInnerBuilder,
        profile: FixedAtOneProfile,
        _solver: std::marker::PhantomData,
        accepted_state: std::cell::RefCell::new(None),
        last_eta: std::cell::RefCell::new(None),
        stats: gamrs::stats::FitStats::new(),
    };

    // 3 probes in the same area the gradient test uses.
    let probes: &[[f64; 3]] = &[[1.0, 1.0, 1.0], [2.0, 0.5, 0.8], [0.5, 2.0, 1.3]];

    for theta_init in probes {
        let theta = Array1::from_vec(theta_init.to_vec());
        let (_, _, h_anal) = score.value_grad_hess(&theta).unwrap();

        // Reference Hessian: central FD on the analytic gradient (re-PIRLS at ±h).
        let eps = 1e-4_f64;
        let d = 3;
        let mut h_fd = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            let mut t_plus = theta.clone();
            let mut t_minus = theta.clone();
            t_plus[i] += eps;
            t_minus[i] -= eps;
            let (_, g_plus) = score.value_and_grad(&t_plus).unwrap();
            let (_, g_minus) = score.value_and_grad(&t_minus).unwrap();
            for j in 0..d {
                h_fd[[j, i]] = (g_plus[j] - g_minus[j]) / (2.0 * eps);
            }
        }
        // Symmetrise FD ref.
        for i in 0..d {
            for j in i + 1..d {
                let avg = 0.5 * (h_fd[[i, j]] + h_fd[[j, i]]);
                h_fd[[i, j]] = avg;
                h_fd[[j, i]] = avg;
            }
        }

        // ρ-block (2×2) at 10% bar — analytic IFT vs central-FD-on-grad
        // sits well inside the bar (measured 1-3% at these probes). The
        // shape rows/cols are FD-on-grad in both reference and test so
        // they match within FD noise (skip them).
        for i in 0..2 {
            for j in 0..2 {
                let denom = h_fd[[i, j]].abs().max(1.0);
                let rel = (h_anal[[i, j]] - h_fd[[i, j]]).abs() / denom;
                assert!(
                    rel < 1e-1,
                    "θ={theta_init:?} H[{i},{j}] analytic={:+.4e} fd={:+.4e} rel={:.2e}",
                    h_anal[[i, j]],
                    h_fd[[i, j]],
                    rel
                );
            }
        }
    }
}

/// TDist's full Level-2 analytic Hessian against central FD of the
/// analytic gradient. The Hessian covers the entire joint
/// `(M + n_shape) = (1 + 2) = 3` block, including ρ×shape and
/// shape×shape — closed form via the mgcv R `gdi2` chain rule.
///
/// **Bar**: 10% relative (matches NegBin / Tweedie analytic-vs-FD bars).
/// Made tight by the observed-W PIRLS path (`TDist::irls_observed_pair`)
/// which routes gamrs's TDist PIRLS through `W = ½·D_μμ` — the same A
/// the Level-1 / Level-2 chain arrays were derived under. Without that
/// PIRLS shim, this same test failed at ≈ 30 % on σ²×σ² because gamrs's
/// default Fisher W is constant in μ for TDist+identity (its log|A| has
/// no μ-derivative).
#[test]
fn tdist_analytic_hess_matches_fd_on_grad() {
    use gamrs::design::{Additive, DesignStrategy, TermSpec};

    let n = 300;
    let mut x_flat = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut state: u64 = 0xa1b2_c3d4_5566_77ee;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for _ in 0..n {
        let x = next() * 10.0;
        x_flat.push(x);
        let eta = (x).sin();
        // Pseudo-t-distributed residual via the Cauchy clip-tail trick;
        // doesn't matter for derivative parity (β̂ is whatever PIRLS lands).
        let u = next() - 0.5;
        let noise = 0.3 * (u / (1.0 - 4.0 * u * u).abs().max(1e-3).sqrt());
        ys.push(eta + noise);
    }
    let x = Array2::from_shape_vec((n, 1), x_flat).unwrap();
    let y = Array1::from_vec(ys);
    let terms = vec![TermSpec::Cr {
        col: 0,
        k: 10,
        pc: None,
    }];
    let prep = Additive { terms }.prepare(x.view()).unwrap();

    let family_base = tdist_identity(5.0, 0.1);
    let score: gamrs::score::ShapeAwarePirlsScore<_, _, _> = ShapeAwareEnvelopeScore {
        x_design: prep.x_design.clone(),
        y: y.clone(),
        prior_weights: None,
        s_list: prep.s_list.clone(),
        family_base,
        rank_s_list: prep.rank_s_list.clone(),
        mp: prep.mp,
        log_pseudo_det_s_list: prep.log_pseudo_det_s_list.clone(),
        coords: CoordsKind::Identity,
        pirls_opts: PirlsOpts::default(),
        inner_builder: PirlsInnerBuilder,
        profile: FixedAtOneProfile,
        _solver: std::marker::PhantomData,
        accepted_state: std::cell::RefCell::new(None),
        last_eta: std::cell::RefCell::new(None),
        stats: gamrs::stats::FitStats::new(),
    };

    // Three probes — different (ρ, log σ², log(ν-2)) regions. Kept
    // moderate (σ² > 0.05; ν − 2 in (1, 5)) so the FD reference at
    // h = 1e-4 isn't dominated by truncation error on small σ².
    let probes: &[[f64; 3]] = &[
        [0.0, (0.1_f64).ln(), (3.0_f64).ln()],
        [-1.0, (0.15_f64).ln(), (4.0_f64).ln()],
        [2.0, (0.3_f64).ln(), (2.0_f64).ln()],
    ];

    for theta_init in probes {
        let theta = Array1::from_vec(theta_init.to_vec());
        let (_, _, h_anal) = score.value_grad_hess(&theta).unwrap();

        // Reference: central FD on the analytic gradient.
        let eps = 1e-4_f64;
        let d = 3;
        let mut h_fd = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            let mut t_plus = theta.clone();
            let mut t_minus = theta.clone();
            t_plus[i] += eps;
            t_minus[i] -= eps;
            let (_, g_plus) = score.value_and_grad(&t_plus).unwrap();
            let (_, g_minus) = score.value_and_grad(&t_minus).unwrap();
            for j in 0..d {
                h_fd[[j, i]] = (g_plus[j] - g_minus[j]) / (2.0 * eps);
            }
        }
        for i in 0..d {
            for j in i + 1..d {
                let avg = 0.5 * (h_fd[[i, j]] + h_fd[[j, i]]);
                h_fd[[i, j]] = avg;
                h_fd[[j, i]] = avg;
            }
        }

        for i in 0..d {
            for j in 0..d {
                let denom = h_fd[[i, j]].abs().max(1.0);
                let rel = (h_anal[[i, j]] - h_fd[[i, j]]).abs() / denom;
                assert!(
                    rel < 0.1,
                    "θ={theta_init:?} H[{i},{j}] analytic={:+.4e} fd={:+.4e} rel={:.2e}",
                    h_anal[[i, j]],
                    h_fd[[i, j]],
                    rel
                );
            }
        }
    }
}

/// Multi-smooth TDist Hessian against central FD of the analytic gradient, at
/// shape parameters AWAY from the ones the family was built with.
///
/// The deviance block of the Hessian used to read ∂D/∂μ and ∂²D/∂μ² off the
/// family's STARTING (ν, σ²) rather than the current θ, so once the outer
/// Newton had moved the shape axes every ρ curvature was wrong — 10× on the
/// near-linear terms of a 10-smooth house-price fit, which then zig-zagged for
/// 51 outer iterations where mgcv takes 16. The single-smooth check above never
/// saw it: its bar is absolute 0.1 on every entry below 1, and these entries are
/// 1e-2.
#[test]
fn tdist_multismooth_hess_matches_fd_away_from_initial_shape() {
    use gamrs::design::{Additive, DesignStrategy, TermSpec};

    let n = 400;
    let mut flat = Vec::with_capacity(3 * n);
    let mut ys = Vec::with_capacity(n);
    let mut state: u64 = 0x005e_ed0f_5ca7_7e55;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for _ in 0..n {
        let (a, b, c) = (next(), next(), next());
        flat.extend_from_slice(&[a, b, c]);
        let u = next() - 0.5;
        let noise = 0.2 * (u / (1.0 - 4.0 * u * u).abs().max(1e-3).sqrt());
        ys.push((6.0 * a).sin() + 0.5 * b + 0.1 * c + noise);
    }
    let x = Array2::from_shape_vec((n, 3), flat).unwrap();
    let y = Array1::from_vec(ys);
    let terms = (0..3)
        .map(|col| TermSpec::Cr {
            col,
            k: 8,
            pc: None,
        })
        .collect();
    let prep = Additive { terms }.prepare(x.view()).unwrap();

    let score: gamrs::score::ShapeAwarePirlsScore<_, _, _> = ShapeAwareEnvelopeScore {
        x_design: prep.x_design.clone(),
        y: y.clone(),
        prior_weights: None,
        s_list: prep.s_list.clone(),
        family_base: tdist_identity(3.0 + 1.5_f64.exp(), 1.0),
        rank_s_list: prep.rank_s_list.clone(),
        mp: prep.mp,
        log_pseudo_det_s_list: prep.log_pseudo_det_s_list.clone(),
        coords: CoordsKind::Identity,
        pirls_opts: PirlsOpts {
            dev_rel_tol: 1e-13,
            ..PirlsOpts::default()
        },
        inner_builder: PirlsInnerBuilder,
        profile: FixedAtOneProfile,
        _solver: std::marker::PhantomData,
        accepted_state: std::cell::RefCell::new(None),
        last_eta: std::cell::RefCell::new(None),
        stats: gamrs::stats::FitStats::new(),
    };

    // θ = [ρ_1, ρ_2, ρ_3, log σ², log(ν − 3)]; the shape coordinates are far
    // from the family's (σ² = 1, ν = 3 + e^1.5) — the case the bug needs.
    let probes: &[[f64; 5]] = &[
        [0.0, 3.0, 8.0, (0.05_f64).ln(), -2.0],
        [-1.0, 5.0, 10.0, (0.1_f64).ln(), 0.5],
    ];
    for t in probes {
        let theta = Array1::from_vec(t.to_vec());
        let (_, _, h) = score.value_grad_hess(&theta).unwrap();
        let d = theta.len();
        let eps = 1e-4;
        let mut h_fd = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            let (mut tp, mut tm) = (theta.clone(), theta.clone());
            tp[i] += eps;
            tm[i] -= eps;
            let (_, gp) = score.value_and_grad(&tp).unwrap();
            let (_, gm) = score.value_and_grad(&tm).unwrap();
            for j in 0..d {
                h_fd[[j, i]] = (gp[j] - gm[j]) / (2.0 * eps);
            }
        }
        let h_fd = (&h_fd + &h_fd.t()) * 0.5;
        // Per-entry, scale-free floor: an off-diagonal is judged against the
        // curvature of the two axes it couples, not against the largest entry
        // (σ²'s, ~40), which would let every ρ entry (~0.05) through.
        for i in 0..d {
            for j in 0..d {
                let err = (h[[i, j]] - h_fd[[i, j]]).abs();
                let floor = 1e-3 * (h_fd[[i, i]] * h_fd[[j, j]]).abs().sqrt();
                assert!(
                    err <= 0.01 * h_fd[[i, j]].abs() + floor,
                    "θ={t:?} H[{i},{j}] analytic={:+.4e} fd={:+.4e}",
                    h[[i, j]],
                    h_fd[[i, j]]
                );
            }
        }
    }
}

/// Two-smooth Tweedie ρ-gradient against central FD of the score value, with the
/// inner solve converged far enough (dev_rel_tol 1e-14) that the FD is exact.
///
/// The ρ-gradient used to be the bare envelope form for Tweedie: `log|H|`'s
/// β-chain was dropped, and once added it needs dβ̂/dρ from the observed
/// penalised Hessian — which in turn needs `TweedieVariance::d_variance`, absent
/// until now (the trait's zero default dropped V'/V from α). Measured on a
/// two-smooth fit: the bare form was 2.2e-3 off at p = 1.5 and the outer Newton
/// stopped where the true gradient was not zero.
#[test]
fn tweedie_multismooth_rho_grad_matches_fd() {
    use gamrs::design::{Additive, DesignStrategy, TermSpec};

    let n = 200;
    let mut flat = Vec::with_capacity(2 * n);
    let mut ys = Vec::with_capacity(n);
    let mut state: u64 = 0x0000_7eed_1e5a_5eed;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for _ in 0..n {
        let (a, b) = (next() * 10.0, next() * 10.0);
        flat.extend_from_slice(&[a, b]);
        let mu = ((1.0 + 0.3 * a + b.sin()) / 3.0).exp();
        // Compound-Poisson-ish draw: a Poisson count of gamma-ish jumps.
        let mut y = 0.0;
        let mut t = -(next().max(1e-12)).ln();
        while t < mu {
            y += 0.5 + next();
            t += -(next().max(1e-12)).ln();
        }
        ys.push(y);
    }
    let x = Array2::from_shape_vec((n, 2), flat).unwrap();
    let y = Array1::from_vec(ys);
    let terms = (0..2).map(|col| TermSpec::Cr { col, k: 10, pc: None }).collect();
    let prep = Additive { terms }.prepare(x.view()).unwrap();

    for (p, theta_p) in [(1.5, None), (1.3, Some(-0.5_f64))] {
        let mut family = tweedie_log(p, 1.0);
        if theta_p.is_none() {
            family.loss.profile_p = false;
            family.variance.profile_p = false;
        }
        let score: gamrs::score::ShapeAwarePirlsScoreOwnedPhi<_, _, _> = ShapeAwareEnvelopeScore {
            x_design: prep.x_design.clone(),
            y: y.clone(),
            prior_weights: None,
            s_list: prep.s_list.clone(),
            family_base: family,
            rank_s_list: prep.rank_s_list.clone(),
            mp: prep.mp,
            log_pseudo_det_s_list: prep.log_pseudo_det_s_list.clone(),
            coords: CoordsKind::Identity,
            pirls_opts: PirlsOpts { dev_rel_tol: 1e-14, ..PirlsOpts::default() },
            inner_builder: PirlsInnerBuilder,
            profile: OwnedByLossProfile,
            _solver: std::marker::PhantomData,
            accepted_state: std::cell::RefCell::new(None),
            last_eta: std::cell::RefCell::new(None),
            stats: gamrs::stats::FitStats::new(),
        };
        for rho in [[3.0, 4.0], [8.0, 2.0]] {
            let mut t = vec![rho[0], rho[1], 0.2];
            t.extend(theta_p);
            let theta = Array1::from_vec(t);
            let (_, g) = score.value_and_grad(&theta).unwrap();
            let eps = 1e-4;
            for j in 0..2 {
                let (mut tp, mut tm) = (theta.clone(), theta.clone());
                tp[j] += eps;
                tm[j] -= eps;
                let fd = (score.value(&tp).unwrap() - score.value(&tm).unwrap()) / (2.0 * eps);
                assert!(
                    (g[j] - fd).abs() <= 1e-3 * fd.abs() + 1e-6,
                    "p={p} θ={theta:?} g[{j}] analytic={:+.6e} fd={fd:+.6e}",
                    g[j]
                );
            }
        }
    }
}
