//! Robustness of the shape-constrained fit on synthetic "traffic" markets:
//! log price falls with log traffic, less so further from the road
//! (`te(log_cars, distance)` decreasing in cars, increasing in distance) plus
//! an age curve. The data are generated here.
//!
//! On markets like these mgcv's scasm criterion often has its optimum on a
//! seam where a monotonicity row starts or stops binding; a search that
//! re-optimises one binding set at a time then never settles (it failed on a
//! quarter of such markets). Flat patches of the surface also make several
//! tensor rows linearly dependent, which once let the constrained solve add
//! a dependent row and return a far-from-optimal fit. Checked here for every
//! market: the fit converges; the λ found is a local minimum of the
//! criterion; the constrained solve is optimal along a λ path; the surface is
//! monotone over the data.

use ndarray::{Array1, Array2};

use gamrs::design::{Additive, DesignStrategy, MarginKind, PreparedDesign, Shape, TermSpec};
use gamrs::family::gaussian_identity;
use gamrs::fit::shape_constrained::{fit_at_lambda, reml_at_lambda};
use gamrs::fit_with_design;

/// Deterministic pseudo-random numbers (no rand dependency).
struct Lcg(u64);
impl Lcg {
    fn unif(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.unif().max(1e-300), self.unif());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn market(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut r = Lcg(0x5eed_0000 + 1000 * n as u64 + seed);
    let mut x = Array2::<f64>::zeros((n, 3));
    let mut y = Array1::<f64>::zeros(n);
    let (lo, hi) = (300f64.ln(), 60000f64.ln());
    for i in 0..n {
        let lc = lo + r.unif() * (hi - lo);
        let d = (500.0 * r.unif()).min(450.0);
        let age = 80.0 * r.unif();
        x[[i, 0]] = lc;
        x[[i, 1]] = d;
        x[[i, 2]] = age;
        y[i] = 12.6 + 0.15 * (age / 25.0).sin()
            - 0.04 * (lc - 1000f64.ln()).max(0.0) * (1.0 - d / 450.0)
            + 0.1 * r.normal();
    }
    (x, y)
}

/// Largest wrong-way step tolerated on the grid (log-price units).
const MONO_TOL: f64 = 1e-4;

fn terms() -> Vec<TermSpec> {
    vec![
        TermSpec::Tensor {
            col_a: 0,
            col_b: 1,
            k_a: 6,
            k_b: 5,
            bs_a: MarginKind::Sc(Some(Shape::Decreasing)),
            bs_b: MarginKind::Sc(Some(Shape::Increasing)),
            by: None,
        },
        TermSpec::Cr {
            col: 2,
            k: 8,
            pc: None,
            by: None,
        },
    ]
}

fn prepare(x: &Array2<f64>) -> PreparedDesign {
    Additive { terms: terms() }.prepare(x.view()).unwrap()
}

/// Penalised least-squares objective at `rho` for coefficients `beta`.
fn objective(prep: &PreparedDesign, y: &Array1<f64>, rho: &Array1<f64>, beta: &Array1<f64>) -> f64 {
    let r = y - &prep.x_design.dot(beta);
    let mut o = r.dot(&r);
    for (j, s) in prep.s_list.iter().enumerate() {
        o += rho[j].exp() * beta.dot(&s.dot(beta));
    }
    o
}

fn check_market(n: usize, seed: u64) {
    let tag = format!("n={n} seed={seed}");
    let (x, y) = market(n, seed);
    let fit = fit_with_design(
        gaussian_identity(),
        Additive { terms: terms() },
        x.view(),
        y.view(),
        None,
    )
    .unwrap_or_else(|e| panic!("{tag}: {e}"));
    assert!(fit.converged, "{tag}: not converged");

    // A local minimum: no lower criterion a short way along any axis.
    let prep = prepare(&x);
    let v = reml_at_lambda(&prep, y.view(), None, fit.lambda.view()).unwrap();
    assert!(
        (fit.reml_value - v).abs() <= 1e-10 * v.abs(),
        "{tag}: reported REML"
    );
    for j in 0..fit.lambda.len() {
        for h in [-0.05, 0.05] {
            let mut l = fit.lambda.clone();
            l[j] *= f64::exp(h);
            let vt = reml_at_lambda(&prep, y.view(), None, l.view()).unwrap();
            assert!(
                vt >= v - 1e-8 * v.abs(),
                "{tag}: lower criterion along axis {j}"
            );
        }
    }

    // The constrained solve is optimal: along a λ path, each fit beats the
    // neighbouring fit (feasible too) on its own objective.
    let rhos: Vec<Array1<f64>> = (0..13)
        .map(|i| Array1::from_elem(3, -2.0 + 0.5 * i as f64))
        .collect();
    let betas: Vec<Array1<f64>> = rhos
        .iter()
        .map(|r| {
            fit_at_lambda(prepare(&x), y.view(), None, r.mapv(f64::exp).view())
                .unwrap()
                .0
                .beta
        })
        .collect();
    for i in 0..rhos.len() {
        let own = objective(&prep, &y, &rhos[i], &betas[i]);
        for k in [i.wrapping_sub(1), i + 1] {
            if let Some(b) = betas.get(k) {
                let other = objective(&prep, &y, &rhos[i], b);
                assert!(
                    own <= other * (1.0 + 1e-10),
                    "{tag}: constrained fit at ρ={} not optimal",
                    rhos[i][0]
                );
            }
        }
    }

    // Monotone over the data rectangle, to the tolerance scasm's tensor rows
    // allow: they make the surface monotone along each margin's coefficient
    // lines, i.e. between the reparameterised margin's nodes, and the
    // interpolating margin basis can dip slightly between nodes (scasm's own
    // surfaces on such markets go the wrong way by up to ~7e-5).
    let ap = match &fit.predictor {
        gamrs::design::Predictor::Additive(ap) => ap,
        _ => unreachable!(),
    };
    let (a, b) = ap.term_col_ranges[0];
    let (lo0, hi0) = (300f64.ln(), 60000f64.ln());
    let mut g = Array2::<f64>::zeros((30 * 20, 3));
    for i in 0..30 {
        for k in 0..20 {
            let row = i * 20 + k;
            g[[row, 0]] = lo0 + (hi0 - lo0) * i as f64 / 29.0;
            g[[row, 1]] = 450.0 * k as f64 / 19.0;
            g[[row, 2]] = 40.0;
        }
    }
    g.column_mut(0).mapv_inplace(|v| {
        v.clamp(
            x.column(0).fold(f64::INFINITY, |m, w| m.min(*w)),
            x.column(0).fold(f64::NEG_INFINITY, |m, w| m.max(*w)),
        )
    });
    g.column_mut(1).mapv_inplace(|v| {
        v.clamp(
            x.column(1).fold(f64::INFINITY, |m, w| m.min(*w)),
            x.column(1).fold(f64::NEG_INFINITY, |m, w| m.max(*w)),
        )
    });
    let f = fit
        .predictor
        .design(g.view())
        .unwrap()
        .slice(ndarray::s![.., a..b])
        .dot(&fit.beta.slice(ndarray::s![a..b]));
    for i in 0..30 {
        for k in 0..20 {
            let here = f[i * 20 + k];
            if i + 1 < 30 {
                assert!(
                    f[(i + 1) * 20 + k] - here <= MONO_TOL,
                    "{tag}: rises with traffic by {:.3e}",
                    f[(i + 1) * 20 + k] - here
                );
            }
            if k + 1 < 20 {
                assert!(
                    f[i * 20 + k + 1] - here >= -MONO_TOL,
                    "{tag}: falls with distance by {:.3e} at {i},{k}",
                    here - f[i * 20 + k + 1]
                );
            }
        }
    }
}

#[test]
fn traffic_markets_n300() {
    for seed in 1..=15 {
        check_market(300, seed);
    }
}

#[test]
fn traffic_markets_n1000() {
    for seed in 1..=15 {
        check_market(1000, seed);
    }
}
