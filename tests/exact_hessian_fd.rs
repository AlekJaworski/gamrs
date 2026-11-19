//! The outer Newton's Hessian for inverse Gaussian (`EnvelopeScore`), Tweedie
//! and scat (shape-aware score) is assembled in closed form from the one
//! converged fit. It must be the derivative of the gradient the optimiser
//! follows: an FD of `value_and_grad` is the oracle, weighted and unweighted,
//! off the optimum.

use gamrs::design::{Additive, DesignStrategy, PreparedDesign, TermSpec};
use gamrs::family::{inverse_gaussian_log, tweedie_log, tweedie_log_fixed_p, InverseGaussian};
use gamrs::inner::{CholeskySolver, PirlsInner, PirlsOpts};
use gamrs::score::{
    EnvelopeScore, MgcvTwoSigmaProfile, OwnedByLossProfile, PirlsInnerBuilder,
    ShapeAwareEnvelopeScore,
};
use gamrs::traits::{CoordsKind, ScoreDerivatives};
use ndarray::{Array1, Array2};
use std::marker::PhantomData;

struct Data {
    prep: PreparedDesign,
    eta: Array1<f64>,
    weights: Array1<f64>,
    uniforms: Vec<f64>,
}

fn data(n: usize) -> Data {
    let mut state: u64 = 0x0000_5eed_4e55_1a2b;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut flat = Vec::with_capacity(2 * n);
    let mut eta = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    for _ in 0..n {
        let (a, b) = (next() * 10.0, next() * 10.0);
        flat.extend_from_slice(&[a, b]);
        eta.push((1.0 + 0.3 * a + b.sin()) / 3.0);
        weights.push(0.2 + 2.8 * next());
    }
    let uniforms = (0..4 * n).map(|_| next()).collect();
    let x = Array2::from_shape_vec((n, 2), flat).unwrap();
    let terms = (0..2)
        .map(|col| TermSpec::Cr {
            col,
            k: 10,
            pc: None,
            by: None,
        })
        .collect();
    Data {
        prep: Additive { terms }.prepare(x.view()).unwrap(),
        eta: Array1::from_vec(eta),
        weights: Array1::from_vec(weights),
        uniforms,
    }
}

fn tight_opts() -> PirlsOpts {
    PirlsOpts {
        dev_rel_tol: 1e-14,
        max_iters: 200,
        ..PirlsOpts::default()
    }
}

/// One inner fit per `value_grad_hess` (no FD probes), and the Hessian equal
/// to a central FD of the gradient to `tol`, relative to `max(|H_ij|, 1)`.
fn assert_hess_matches_fd<S: ScoreDerivatives>(label: &str, score: &S, probes: &[&[f64]]) {
    let mut worst = 0.0_f64;
    for probe in probes {
        let theta = Array1::from_vec(probe.to_vec());
        let stats = score.stats().unwrap();
        let before = stats.snapshot().inner_pirls_calls;
        let (_, _, h) = score.value_grad_hess(&theta).unwrap();
        let calls = stats.snapshot().inner_pirls_calls - before;
        assert_eq!(calls, 1, "{label}: {calls} inner fits for one Hessian");
        let d = theta.len();
        // Five-point stencil at a wide step: the Tweedie series carries ~1e-9
        // noise, which a narrow central difference amplifies past the bar.
        let eps = 1e-3;
        let grad_at = |i: usize, k: f64| {
            let mut t = theta.clone();
            t[i] += k * eps;
            score.value_and_grad(&t).unwrap().1
        };
        let mut h_fd = Array2::<f64>::zeros((d, d));
        for i in 0..d {
            let fd = (grad_at(i, -2.0) - grad_at(i, 2.0)
                + 8.0 * (grad_at(i, 1.0) - grad_at(i, -1.0)))
                / (12.0 * eps);
            h_fd.column_mut(i).assign(&fd);
        }
        let h_fd = (&h_fd + &h_fd.t()) * 0.5;
        for (a, b) in h.iter().zip(h_fd.iter()) {
            worst = worst.max((a - b).abs() / b.abs().max(1.0));
        }
        eprintln!("{label} θ={probe:?}\n  analytic={h:.6e}\n  fd={h_fd:.6e}");
    }
    eprintln!("{label}: worst rel {worst:.2e}");
    assert!(worst <= 1e-6, "{label}: rel {worst:.2e}");
}

fn envelope_check<L, K, V>(
    label: &str,
    d: &Data,
    y: Array1<f64>,
    weights: Option<Array1<f64>>,
    family: gamrs::family::Family<L, K, V>,
    loss: L,
) where
    L: gamrs::traits::Loss + Clone,
    K: gamrs::traits::Link + Clone,
    V: gamrs::traits::VarianceFn + Clone,
{
    let pirls = PirlsInner::<L, K, V, CholeskySolver> {
        x_design: d.prep.x_design.clone(),
        y: y.clone(),
        prior_weights: weights,
        s_list: d.prep.s_list.clone().into(),
        family,
        opts: tight_opts(),
        _solver: PhantomData,
    };
    let score = EnvelopeScore::<L, _, _, CholeskySolver>::with_inner(
        pirls,
        loss,
        MgcvTwoSigmaProfile,
        y,
        d.prep.s_list.clone(),
        d.prep.rank_s_list.clone(),
        d.prep.mp,
        d.prep.log_pseudo_det_s_list.clone(),
    );
    assert_hess_matches_fd(label, &score, &[&[0.0, 2.0], &[4.0, -1.0], &[-2.0, 6.0]]);
}

#[test]
fn inverse_gaussian_hessian_matches_fd() {
    let d = data(300);
    let y =
        Array1::from_iter((0..d.eta.len()).map(|i| d.eta[i].exp() * (0.3 + 1.4 * d.uniforms[i])));
    for (label, w) in [("ig", None), ("ig_weighted", Some(d.weights.clone()))] {
        envelope_check(
            label,
            &d,
            y.clone(),
            w,
            inverse_gaussian_log(),
            InverseGaussian,
        );
    }
}

#[test]
fn tweedie_hessian_matches_fd() {
    let d = data(300);
    // Compound Poisson–gamma draws: Poisson(μ) count of exponential(1) jumps.
    let u = &d.uniforms;
    let y = Array1::from_iter((0..d.eta.len()).map(|i| {
        let mu = d.eta[i].exp();
        let k = (mu * 2.0 * u[2 * i]).floor();
        k * (0.5 + u[2 * i + 1])
    }));
    for weights in [None, Some(d.weights.clone())] {
        for profile_p in [true, false] {
            let family = if profile_p {
                tweedie_log(1.5, 1.0)
            } else {
                tweedie_log_fixed_p(1.4, 1.0)
            };
            let label = format!(
                "tweedie profile_p={profile_p} weighted={}",
                weights.is_some()
            );
            let score: gamrs::score::ShapeAwarePirlsScoreOwnedPhi<_, _, _> =
                ShapeAwareEnvelopeScore {
                    x_design: d.prep.x_design.clone(),
                    y: y.clone(),
                    prior_weights: weights.clone(),
                    s_list: d.prep.s_list.clone().into(),
                    s_blocks: Default::default(),
                    family_base: family,
                    rank_s_list: d.prep.rank_s_list.clone(),
                    mp: d.prep.mp,
                    log_pseudo_det_s_list: d.prep.log_pseudo_det_s_list.clone(),
                    coords: CoordsKind::Identity,
                    pirls_opts: tight_opts(),
                    inner_builder: PirlsInnerBuilder,
                    profile: OwnedByLossProfile,
                    _solver: PhantomData,
                    accepted_state: std::cell::RefCell::new(None),
                    last_eta: std::cell::RefCell::new(None),
                    stats: gamrs::stats::FitStats::new(),
                };
            // θ = [ρ₀, ρ₁, log φ, (p-coordinate)].
            let probes: [&[f64]; 3] = if profile_p {
                [
                    &[0.0, 2.0, 0.0, 0.0],
                    &[4.0, -1.0, 0.3, -1.2],
                    &[1.0, 5.0, -0.4, 1.5],
                ]
            } else {
                [&[0.0, 2.0, 0.0], &[4.0, -1.0, 0.3], &[1.0, 5.0, -0.4]]
            };
            assert_hess_matches_fd(&label, &score, &probes);
        }
    }
}

#[test]
fn scat_hessian_matches_fd() {
    let d = data(300);
    let u = &d.uniforms;
    let y = Array1::from_iter((0..d.eta.len()).map(|i| {
        let spike = if u[i] > 0.9 { 4.0 } else { 0.4 };
        d.eta[i] + spike * (u[d.eta.len() + i] - 0.5)
    }));
    for weights in [None, Some(d.weights.clone())] {
        let score: gamrs::score::ShapeAwarePirlsScore<_, _, _> = ShapeAwareEnvelopeScore {
            x_design: d.prep.x_design.clone(),
            y: y.clone(),
            prior_weights: weights.clone(),
            s_list: d.prep.s_list.clone().into(),
            s_blocks: Default::default(),
            family_base: gamrs::family::tdist_identity(5.0, 0.1),
            rank_s_list: d.prep.rank_s_list.clone(),
            mp: d.prep.mp,
            log_pseudo_det_s_list: d.prep.log_pseudo_det_s_list.clone(),
            coords: CoordsKind::Identity,
            pirls_opts: tight_opts(),
            inner_builder: PirlsInnerBuilder,
            profile: gamrs::score::FixedAtOneProfile,
            _solver: PhantomData,
            accepted_state: std::cell::RefCell::new(None),
            last_eta: std::cell::RefCell::new(None),
            stats: gamrs::stats::FitStats::new(),
        };
        // θ = [ρ₀, ρ₁, log σ², log(ν − 2)].
        assert_hess_matches_fd(
            &format!("scat weighted={}", weights.is_some()),
            &score,
            &[&[0.0, 2.0, -2.0, 1.0], &[4.0, -1.0, -1.5, 0.5]],
        );
    }
}
