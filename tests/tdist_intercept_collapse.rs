//! A penalty block many orders of magnitude larger than the rest must not
//! shrink the unpenalised coefficients — the β̂ solve used to ridge every
//! diagonal by `1e-12·max|A_ii|`, which on a real 10-term house-price fit put
//! 25,000 on an intercept diagonal of 2,147 and returned 0.078× the prices.

use ndarray::{Array1, Array2};

use gamrs::family::tdist_identity;
use gamrs::{fit_with_design, Additive, TermSpec};

/// Deterministic uniforms so the fixture is the same on every platform.
struct Lcg(u64);
impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn student_t5(&mut self) -> f64 {
        let chi2: f64 = (0..5).map(|_| self.normal().powi(2)).sum();
        self.normal() / (chi2 / 5.0).sqrt()
    }
}

#[test]
fn giant_penalty_block_leaves_the_intercept_alone() {
    let n = 300;
    let mut rng = Lcg(0x5eed);
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    // Column 1 is a lot-size-like covariate: integer-valued in a narrow band
    // with a few far-out values. Quantile knots then leave one interval
    // ~1e4 times longer than the others, the cardinal basis and its penalty
    // blow up on it, and `SmartInit` pins that term's λ at its 1e6 cap.
    let far_out = [1.0e6, 1.0e7, 2.5e8];
    for i in 0..n {
        let x1 = rng.uniform();
        let x2 = match far_out.get(i) {
            Some(&v) => v,
            None => 1000.0 + (rng.uniform() * 19000.0).floor(),
        };
        x[[i, 0]] = x1;
        x[[i, 1]] = x2;
        y[i] = 200_000.0
            + 50_000.0 * (2.0 * std::f64::consts::PI * x1).sin()
            + 20_000.0 * rng.student_t5();
    }
    let y_mean = y.sum() / n as f64;
    let y_var = y.iter().map(|&v| (v - y_mean).powi(2)).sum::<f64>() / n as f64;

    let terms = vec![
        TermSpec::Cr {
            col: 0,
            k: 10,
            pc: None,
        },
        TermSpec::Cr {
            col: 1,
            k: 10,
            pc: None,
        },
    ];
    let fit = fit_with_design(
        tdist_identity(5.0, y_var),
        Additive { terms },
        x.view(),
        y.view(),
        None,
    )
    .expect("scat fit failed");

    let mu = fit.predict(x.view()).expect("predict failed");
    let mu_mean = mu.sum() / n as f64;
    println!(
        "intercept = {:.0}, mean(μ)/mean(y) = {:.4}, λ = {:?}, iters = {}, converged = {}",
        fit.beta[0],
        mu_mean / y_mean,
        fit.lambda.to_vec(),
        fit.n_iters,
        fit.converged
    );
    // Sum-to-zero smooths put the response level on the intercept alone.
    assert!(
        (fit.beta[0] - 200_000.0).abs() < 10_000.0,
        "intercept {:.0} is not the response level 200,000",
        fit.beta[0]
    );
    assert!(
        (mu_mean / y_mean - 1.0).abs() < 0.02,
        "mean fitted / mean y = {:.4}",
        mu_mean / y_mean
    );
}
