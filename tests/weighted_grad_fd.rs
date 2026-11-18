//! With non-unit prior weights, the REML gradient of every Newton-weight
//! family (non-canonical link: Gamma, inverse Gaussian, NegBin, Tweedie) must
//! be the derivative of the score it optimises. The score's `log|H|` reads
//! `X' diag(prior · w_newton) X + λS`; the gradient's trace and β-chain terms
//! must factor that same matrix. A central FD of `score.value` is the oracle.

use gamrs::design::{Additive, DesignStrategy, PreparedDesign, TermSpec};
use gamrs::family::{
    gamma_log, inverse_gaussian_log, negbin_log, tdist_identity, tweedie_log, Gamma,
    InverseGaussian,
};
use gamrs::inner::{CholeskySolver, PirlsInner, PirlsOpts};
use gamrs::score::{
    EnvelopeScore, FixedAtOneProfile, MgcvTwoSigmaProfile, OwnedByLossProfile, PirlsInnerBuilder,
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
    let mut state: u64 = 0x0000_91a5_3c1e_77d1;
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
    let prep = Additive { terms }.prepare(x.view()).unwrap();
    Data {
        prep,
        eta: Array1::from_vec(eta),
        weights: Array1::from_vec(weights),
        uniforms,
    }
}

fn tight_opts() -> PirlsOpts {
    PirlsOpts {
        dev_rel_tol: 1e-14,
        ..PirlsOpts::default()
    }
}

fn assert_grad_matches_fd<S: ScoreDerivatives>(
    label: &str,
    score: &S,
    probes: &[&[f64]],
    eps: f64,
    tol: f64,
) {
    let mut worst = (0.0_f64, String::new());
    for probe in probes {
        let theta = Array1::from_vec(probe.to_vec());
        let (_, g) = score.value_and_grad(&theta).unwrap();
        for j in 0..theta.len() {
            let (mut tp, mut tm) = (theta.clone(), theta.clone());
            tp[j] += eps;
            tm[j] -= eps;
            let fd = (score.value(&tp).unwrap() - score.value(&tm).unwrap()) / (2.0 * eps);
            let rel = (g[j] - fd).abs() / (fd.abs() + 1.0);
            if rel >= worst.0 {
                worst = (
                    rel,
                    format!("θ={probe:?} g[{j}] analytic={:+.6e} fd={fd:+.6e}", g[j]),
                );
            }
        }
    }
    eprintln!("{label}: worst rel {:.2e} at {}", worst.0, worst.1);
    assert!(
        worst.0 <= tol,
        "{label}: rel {:.2e} at {}",
        worst.0,
        worst.1
    );
}

fn envelope_check<L, K, V>(
    label: &str,
    d: &Data,
    y: Array1<f64>,
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
        prior_weights: Some(d.weights.clone()),
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
    // Measured before the fix: rel 1e-3 to 6e-2. After: below 1e-7.
    assert_grad_matches_fd(label, &score, &[&[0.0, 2.0], &[4.0, -1.0]], 1e-4, 1e-6);
}

#[test]
fn weighted_gamma_log_grad_matches_fd() {
    let d = data(300);
    let u = &d.uniforms;
    let y = Array1::from_iter((0..d.eta.len()).map(|i| {
        let mu = d.eta[i].exp();
        -0.5 * mu * (u[2 * i].max(1e-12).ln() + u[2 * i + 1].max(1e-12).ln())
    }));
    envelope_check("gamma_log", &d, y, gamma_log(), Gamma);
}

#[test]
fn weighted_inverse_gaussian_log_grad_matches_fd() {
    let d = data(300);
    let y =
        Array1::from_iter((0..d.eta.len()).map(|i| d.eta[i].exp() * (0.3 + 1.4 * d.uniforms[i])));
    envelope_check(
        "inverse_gaussian_log",
        &d,
        y,
        inverse_gaussian_log(),
        InverseGaussian,
    );
}

#[test]
fn weighted_negbin_grad_matches_fd() {
    let d = data(300);
    let y = Array1::from_iter((0..d.eta.len()).map(|i| {
        let mu = d.eta[i].exp();
        (mu + (d.uniforms[i] - 0.5) * (2.0 * mu.sqrt() + 1.0))
            .round()
            .max(0.0)
    }));
    let score: gamrs::score::ShapeAwarePirlsScore<_, _, _> = ShapeAwareEnvelopeScore {
        x_design: d.prep.x_design.clone(),
        y,
        prior_weights: Some(d.weights.clone()),
        s_list: d.prep.s_list.clone().into(),
        s_blocks: Default::default(),
        family_base: negbin_log(3.0),
        rank_s_list: d.prep.rank_s_list.clone(),
        mp: d.prep.mp,
        log_pseudo_det_s_list: d.prep.log_pseudo_det_s_list.clone(),
        coords: CoordsKind::Identity,
        pirls_opts: tight_opts(),
        inner_builder: PirlsInnerBuilder,
        profile: FixedAtOneProfile,
        _solver: PhantomData,
        accepted_state: std::cell::RefCell::new(None),
        last_eta: std::cell::RefCell::new(None),
        stats: gamrs::stats::FitStats::new(),
    };
    // θ = [ρ₀, ρ₁, log θ_nb].
    assert_grad_matches_fd(
        "negbin_log",
        &score,
        &[&[0.0, 2.0, 1.0], &[4.0, -1.0, 0.5]],
        1e-4,
        1e-6,
    );
}

#[test]
fn weighted_tweedie_grad_matches_fd() {
    let d = data(300);
    // Compound Poisson–gamma draws: Poisson(μ) count of exponential(1) jumps.
    let u = &d.uniforms;
    let y = Array1::from_iter((0..d.eta.len()).map(|i| {
        let mu = d.eta[i].exp();
        let k = (mu * 2.0 * u[2 * i]).floor();
        k * (0.5 + u[2 * i + 1])
    }));
    let score: gamrs::score::ShapeAwarePirlsScoreOwnedPhi<_, _, _> = ShapeAwareEnvelopeScore {
        x_design: d.prep.x_design.clone(),
        y,
        prior_weights: Some(d.weights.clone()),
        s_list: d.prep.s_list.clone().into(),
        s_blocks: Default::default(),
        family_base: tweedie_log(1.5, 1.0),
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
    // θ = [ρ₀, ρ₁, log φ, p-coordinate].
    assert_grad_matches_fd(
        "tweedie_log",
        &score,
        &[&[0.0, 2.0, 0.0, 0.0], &[4.0, -1.0, 0.3, -0.2]],
        1e-4,
        1e-6,
    );
}

#[test]
fn weighted_scat_grad_matches_fd() {
    let d = data(300);
    let u = &d.uniforms;
    let y = Array1::from_iter((0..d.eta.len()).map(|i| {
        let spike = if u[i] > 0.9 { 4.0 } else { 0.4 };
        d.eta[i] + spike * (u[d.eta.len() + i] - 0.5)
    }));
    let score: gamrs::score::ShapeAwarePirlsScore<_, _, _> = ShapeAwareEnvelopeScore {
        x_design: d.prep.x_design.clone(),
        y,
        prior_weights: Some(d.weights.clone()),
        s_list: d.prep.s_list.clone().into(),
        s_blocks: Default::default(),
        family_base: tdist_identity(5.0, 0.1),
        rank_s_list: d.prep.rank_s_list.clone(),
        mp: d.prep.mp,
        log_pseudo_det_s_list: d.prep.log_pseudo_det_s_list.clone(),
        coords: CoordsKind::Identity,
        pirls_opts: tight_opts(),
        inner_builder: PirlsInnerBuilder,
        profile: FixedAtOneProfile,
        _solver: PhantomData,
        accepted_state: std::cell::RefCell::new(None),
        last_eta: std::cell::RefCell::new(None),
        stats: gamrs::stats::FitStats::new(),
    };
    // θ = [ρ₀, ρ₁, log σ², log(ν − 2)].
    assert_grad_matches_fd(
        "scat",
        &score,
        &[&[0.0, 2.0, -2.0, 1.0], &[4.0, -1.0, -1.5, 0.5]],
        1e-4,
        1e-6,
    );
}
