//! Timing of a monotone smooth next to a large random effect, against the
//! same model with an unconstrained CR smooth in its place. Ignored by
//! default; run with
//! `cargo test --release --test sc_scaling -- --ignored --nocapture --test-threads 1`.

use std::time::Instant;

use ndarray::{Array1, Array2};

use gamrs::design::{Additive, DesignStrategy, Shape, TermSpec};
use gamrs::family::gaussian_identity;
use gamrs::fit::shape_constrained::fit_at_lambda;
use gamrs::fit_with_design;

/// Deterministic pseudo-random numbers (no rand dependency).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.next().max(1e-300), self.next());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// A flat-then-falling curve in `x` (so some monotone rows bind) plus a
/// random effect with `levels` groups.
fn data(n: usize, levels: usize) -> (Array2<f64>, Array1<f64>) {
    let mut r = Lcg(20261117 + levels as u64);
    let effects: Vec<f64> = (0..levels).map(|_| 0.08 * r.normal()).collect();
    let mut x = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    let (lo, hi) = ((300f64).ln(), (60000f64).ln());
    for i in 0..n {
        let lc = lo + r.next() * (hi - lo);
        let g = (r.next() * levels as f64).floor().min(levels as f64 - 1.0);
        x[[i, 0]] = lc;
        x[[i, 1]] = g;
        y[i] =
            12.6 - 0.06 * (lc - (1000f64).ln()).max(0.0) + effects[g as usize] + 0.1 * r.normal();
    }
    (x, y)
}

fn run(n: usize, levels: usize) {
    let (x, y) = data(n, levels);
    let re = TermSpec::Re { col: 1, by: None };
    let sc = vec![
        TermSpec::Sc {
            col: 0,
            k: 10,
            shape: Shape::Decreasing,
        },
        re.clone(),
    ];
    let cr = vec![
        TermSpec::Cr {
            col: 0,
            k: 10,
            pc: None,
            by: None,
        },
        re,
    ];

    let t = Instant::now();
    let prep = Additive { terms: sc.clone() }.prepare(x.view()).unwrap();
    let t_build = t.elapsed().as_secs_f64();
    let p = prep.x_design.ncols();
    let t = Instant::now();
    let (_, active) = fit_at_lambda(prep, y.view(), None, Array1::ones(2).view()).unwrap();
    let t_fixed = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let fit = fit_with_design(
        gaussian_identity(),
        Additive { terms: sc },
        x.view(),
        y.view(),
        None,
    )
    .unwrap();
    let t_sc = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let base = fit_with_design(
        gaussian_identity(),
        Additive { terms: cr },
        x.view(),
        y.view(),
        None,
    )
    .unwrap();
    let t_cr = t.elapsed().as_secs_f64();
    eprintln!(
        "n={n} levels={levels} p={p}: design {t_build:.2}s; one constrained solve at λ=1 \
         {t_fixed:.2}s ({} binding rows); full sc fit {t_sc:.2}s ({} Newton iterations, \
         edf {:.1}); same model with cr in place of sc {t_cr:.2}s ({} iterations, edf {:.1})",
        active.len(),
        fit.n_iters,
        fit.edf_total,
        base.n_iters,
        base.edf_total,
    );
}

#[test]
#[ignore]
fn sc_with_re_200_levels_n20000() {
    run(20_000, 200);
}

#[test]
#[ignore]
fn sc_with_re_1000_levels_n20000() {
    run(20_000, 1000);
}
