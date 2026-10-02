//! The REML gradient of every GLM family must be the derivative of the score
//! it is optimising. Until 0.14.6 it was not: `log|X'WX + S|` depends on ρ
//! through β̂ as well (W is a function of μ̂), and that chain was dropped —
//! for Fisher-weight families entirely, for non-canonical links in its signed
//! α. The optimiser then stopped where the incomplete gradient vanished, off
//! mgcv's optimum by up to 1e-2 in the fitted curve (NegBin).
//!
//! A central FD of `score.value` is the oracle. Probes are away from the
//! optimum, where the gradient is large and a missing term shows.

use gamrs::design::{Additive, DesignStrategy, PreparedDesign, TermSpec};
use gamrs::family::{
    bernoulli_logit, gamma_inverse, gamma_log, inverse_gaussian_log, poisson_log, quasipoisson_log,
    Bernoulli, Gamma, InverseGaussian, Poisson, QuasiPoisson,
};
use gamrs::inner::{CholeskySolver, PirlsInner, PirlsOpts};
use gamrs::score::{EnvelopeScore, FixedAtOneProfile, MgcvTwoSigmaProfile, Profile};
use gamrs::traits::{Link, Loss, ScoreDerivatives, VarianceFn};
use ndarray::{Array1, Array2};
use std::marker::PhantomData;

/// Two covariates on [0, 10] and the linear predictor η = (1 + 0.3a + sin b)/3.
fn design(n: usize) -> (PreparedDesign, Array1<f64>, impl FnMut() -> f64) {
    let mut state: u64 = 0x0000_91a5_3c1e_77d1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut flat = Vec::with_capacity(2 * n);
    let mut eta = Vec::with_capacity(n);
    for _ in 0..n {
        let (a, b) = (next() * 10.0, next() * 10.0);
        flat.extend_from_slice(&[a, b]);
        eta.push((1.0 + 0.3 * a + b.sin()) / 3.0);
    }
    let x = Array2::from_shape_vec((n, 2), flat).unwrap();
    let terms = (0..2)
        .map(|col| TermSpec::Cr {
            col,
            k: 10,
            pc: None,
        })
        .collect();
    let prep = Additive { terms }.prepare(x.view()).unwrap();
    (prep, Array1::from_vec(eta), next)
}

fn check<L, K, V, P>(
    label: &str,
    prep: &PreparedDesign,
    y: Array1<f64>,
    family: gamrs::family::Family<L, K, V>,
    loss: L,
    profile: P,
) where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    P: Profile<L>,
{
    let pirls = PirlsInner::<L, K, V, CholeskySolver> {
        x_design: prep.x_design.clone(),
        y: y.clone(),
        prior_weights: None,
        s_list: prep.s_list.clone(),
        family,
        // The FD reference needs β̂ converged far below the FD step.
        opts: PirlsOpts {
            dev_rel_tol: 1e-14,
            ..PirlsOpts::default()
        },
        _solver: PhantomData,
    };
    let score = EnvelopeScore::<L, _, _, CholeskySolver>::with_inner(
        pirls,
        loss,
        profile,
        y,
        prep.s_list.clone(),
        prep.rank_s_list.clone(),
        prep.mp,
        prep.log_pseudo_det_s_list.clone(),
    );
    for rho in [[0.0, 2.0], [4.0, -1.0]] {
        let theta = Array1::from_vec(rho.to_vec());
        let (_, g) = score.value_and_grad(&theta).unwrap();
        let eps = 1e-4;
        for j in 0..2 {
            let (mut tp, mut tm) = (theta.clone(), theta.clone());
            tp[j] += eps;
            tm[j] -= eps;
            let fd = (score.value(&tp).unwrap() - score.value(&tm).unwrap()) / (2.0 * eps);
            // Measured worst with the chain: 4.4e-9. Without it: 2e-4 to 1.4e-3.
            assert!(
                (g[j] - fd).abs() <= 1e-6 * fd.abs() + 1e-8,
                "{label} ρ={rho:?} g[{j}] analytic={:+.6e} fd={fd:+.6e}",
                g[j]
            );
        }
    }
}

#[test]
fn poisson_log_grad_matches_fd() {
    let (prep, eta, mut next) = design(300);
    let y = eta.mapv(|e| {
        // Poisson draw by inversion.
        let (mu, u) = (e.exp(), next());
        let (mut k, mut p) = (0.0, (-mu).exp());
        let mut c = p;
        while u > c {
            k += 1.0;
            p *= mu / k;
            c += p;
        }
        k
    });
    check(
        "poisson_log",
        &prep,
        y,
        poisson_log(),
        Poisson,
        FixedAtOneProfile,
    );
}

#[test]
fn quasipoisson_log_grad_matches_fd() {
    let (prep, eta, mut next) = design(300);
    let y = eta.mapv(|e| (e.exp() * 2.0 * next()).round());
    check(
        "quasipoisson_log",
        &prep,
        y,
        quasipoisson_log(),
        QuasiPoisson,
        MgcvTwoSigmaProfile,
    );
}

#[test]
fn bernoulli_logit_grad_matches_fd() {
    let (prep, eta, mut next) = design(400);
    let y = eta.mapv(|e| {
        let p = 1.0 / (1.0 + (-(2.0 * e - 2.0)).exp());
        if next() < p {
            1.0
        } else {
            0.0
        }
    });
    check(
        "bernoulli_logit",
        &prep,
        y,
        bernoulli_logit(),
        Bernoulli,
        FixedAtOneProfile,
    );
}

#[test]
fn gamma_grad_matches_fd() {
    let (prep, eta, mut next) = design(300);
    // Shape-2 gamma: the sum of two exponentials.
    let y = eta.mapv(|e| {
        let mu = e.exp();
        -0.5 * mu * (next().max(1e-12).ln() + next().max(1e-12).ln())
    });
    check(
        "gamma_log",
        &prep,
        y.clone(),
        gamma_log(),
        Gamma,
        MgcvTwoSigmaProfile,
    );
    check(
        "gamma_inverse",
        &prep,
        y,
        gamma_inverse(),
        Gamma,
        MgcvTwoSigmaProfile,
    );
}

#[test]
fn inverse_gaussian_log_grad_matches_fd() {
    let (prep, eta, mut next) = design(300);
    let y = eta.mapv(|e| e.exp() * (0.3 + 1.4 * next()));
    check(
        "inverse_gaussian_log",
        &prep,
        y,
        inverse_gaussian_log(),
        InverseGaussian,
        MgcvTwoSigmaProfile,
    );
}
