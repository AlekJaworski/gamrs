//! `fit_bam` against mgcv `bam(method = "fREML")` on 3-D additive fixtures
//! (`scripts/r/gen_bam_freml_fixtures.R`). x2 is linear, so its λ sits on a
//! flat REML ridge where only the same Newton path lands on bam's value.
//! The 2-D tie fixtures put one observation on each of the top two knots, so
//! sweep-and-drop sees two columns of exactly equal variance.

use std::path::PathBuf;

use ndarray::{Array1, Array2};
use serde::Deserialize;

use gamrs::design::{Additive, TermSpec};
use gamrs::family::bam::BamFamily;
use gamrs::family::{gamma_log, gaussian_identity, negbin_log, tdist_identity, Family};
use gamrs::traits::{Link, Loss, VarianceFn};
use gamrs::{fit_bam, FittedGam};

#[derive(Deserialize)]
struct Fixture {
    inputs: Inputs,
    bam: Reference,
}

#[derive(Deserialize)]
struct Inputs {
    x_train: Vec<Vec<f64>>,
    y_train: Vec<f64>,
    k: Vec<usize>,
    weights: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct Reference {
    sp: Vec<f64>,
    scale: f64,
    predictions_train: Vec<f64>,
    edf_per_term: Vec<f64>,
    edf_total: f64,
    reml: f64,
    iter: usize,
    irls_iter: usize,
    theta: Option<Vec<f64>>,
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

/// mgcv's reported θ (nb θ; scat ν, σ) from gamrs's `shape_params`.
fn theta(family: &str, fit: &FittedGam) -> Vec<f64> {
    let sp = &fit.shape_params;
    match family {
        "negbin" => vec![sp[0].exp()],
        "scat" => vec![sp[1].exp() + 3.0, (sp[0] / 2.0).exp()],
        _ => vec![],
    }
}

fn check<L, K, V>(name: &str, family: Family<L, K, V>, family_name: &str, tol: Tol)
where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    Family<L, K, V>: BamFamily,
{
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(format!("tests/fixtures/{name}.json"));
    let fx: Fixture = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
    let n = fx.inputs.y_train.len();
    let d = fx.inputs.k.len();
    let x = Array2::from_shape_vec((n, d), fx.inputs.x_train.concat()).unwrap();
    let y = Array1::from_vec(fx.inputs.y_train);
    let w = fx.inputs.weights.map(Array1::from_vec);
    let terms = (0..d)
        .map(|col| TermSpec::Cr {
            col,
            k: fx.inputs.k[col],
            pc: None,
            by: None,
        })
        .collect();
    let fit = fit_bam(
        family,
        &Additive { terms },
        x.view(),
        y.view(),
        w.as_ref().map(|w| w.view()),
    )
    .unwrap();
    let r = fx.bam;

    let pred = fit
        .predict(x.view())
        .unwrap()
        .mapv(|e| fit.link_kind.inverse(e));
    let fitted = pred
        .iter()
        .zip(&r.predictions_train)
        .map(|(a, b)| rel(*a, *b))
        .fold(0.0, f64::max);
    let sp = fit
        .lambda
        .iter()
        .zip(&r.sp)
        .map(|(a, b)| rel(*a, *b))
        .fold(0.0, f64::max);
    let edf_term = fit
        .edf_per_term
        .iter()
        .zip(&r.edf_per_term)
        .map(|(a, b)| rel(*a, *b))
        .fold(0.0, f64::max);
    let report = format!(
        "{name}: fitted {fitted:.2e} sp {sp:.2e} edf/term {edf_term:.2e} edf {:.2e} \
         reml {:.2e} scale {:.2e} iter {} vs {}",
        rel(fit.edf_total, r.edf_total),
        rel(fit.reml_value, r.reml),
        rel(fit.scale, r.scale),
        fit.n_iters,
        r.iter
    );
    let th = theta(family_name, &fit)
        .iter()
        .zip(r.theta.unwrap_or_default())
        .map(|(a, b)| rel(*a, b))
        .fold(0.0, f64::max);
    eprintln!(
        "{report} theta {th:.2e} irls {}",
        fit.stats.inner_pirls_iterations_total
    );
    assert!(fitted < tol.fitted, "{report}");
    assert!(sp < tol.sp, "{report}");
    assert!(edf_term < tol.fitted, "{report}");
    assert!(rel(fit.edf_total, r.edf_total) < tol.fitted, "{report}");
    assert!(rel(fit.reml_value, r.reml) < tol.reml, "{report}");
    assert!(rel(fit.scale, r.scale) < tol.reml, "{report}");
    assert!(th < tol.fitted, "{report} theta {th:.2e}");
    assert_eq!(fit.n_iters, r.iter, "{report}");
    if family_name != "gaussian" {
        assert_eq!(
            fit.stats.inner_pirls_iterations_total, r.irls_iter,
            "{report}"
        );
    }
}

/// Bars ~10× above the measured worst (fitted/edf/θ, then λ). The nb bars
/// cover multi-threaded BLAS: its different rounding moves the flat-ridge λ
/// by 8e-7 (fitted 8e-8), against 1e-10 with one thread.
struct Tol {
    fitted: f64,
    sp: f64,
    reml: f64,
}

const GAUSSIAN: Tol = Tol {
    fitted: 1e-9,
    sp: 1e-7,
    reml: 1e-9,
};
const PIRLS: Tol = Tol {
    fitted: 1e-6,
    sp: 1e-5,
    reml: 1e-9,
};
/// A covariate spanning 1e3–2.5e8: X'WX squares the basis's ~1e10 condition
/// number, so gamrs sits 8e-6 (fitted) and 1e-8 (REML) from bam's QR.
// REML and scale move by ~4e-7 between CI CPUs: OpenBLAS picks its kernels by CPU.
const ILL_CONDITIONED: Tol = Tol {
    fitted: 1e-4,
    sp: 1e-6,
    reml: 1e-6,
};

#[test]
fn bam_gaussian_flat_ridge_term() {
    check(
        "3d_gaussian_bam_freml_n2000_k10_cr",
        gaussian_identity(),
        "gaussian",
        GAUSSIAN,
    );
}

#[test]
fn bam_gaussian_prior_weights() {
    check(
        "3d_gaussian_weighted_bam_freml_n2000_k10_cr",
        gaussian_identity(),
        "gaussian",
        GAUSSIAN,
    );
}

#[test]
fn bam_negbin_estimated_theta() {
    check(
        "3d_nb_log_bam_freml_n2000_k10_cr",
        negbin_log(1.0),
        "negbin",
        PIRLS,
    );
}

#[test]
fn bam_scat_prior_weights() {
    check(
        "3d_scat_identity_weighted_bam_freml_n1500_k10_cr",
        tdist_identity(5.0, 1.0),
        "scat",
        PIRLS,
    );
}

#[test]
fn bam_gamma_estimated_scale() {
    check(
        "3d_gamma_log_bam_freml_n2000_k10_cr",
        gamma_log(),
        "gamma",
        PIRLS,
    );
}

/// The integer lattice `gen_bam_freml_fixtures.R` fits above bam's
/// `chunk.size`: x columns, the gaussian / nb / scat responses and weights.
fn lattice(n: usize) -> (Array2<f64>, [Array1<f64>; 3], Array1<f64>) {
    let i = Array1::from_iter((1..=n).map(|v| v as f64));
    let u = |a: f64, m: f64| i.mapv(|v| (v * a).rem_euclid(m) / m);
    let (x0, x1, x2) = (u(7919.0, 10007.0), u(6037.0, 10009.0), u(4001.0, 10037.0));
    let e1 = u(7907.0, 10039.0) - 0.5;
    let e2 = u(5003.0, 10061.0) - 0.5;
    let e3 = u(3001.0, 10067.0);
    let s = 4.0 * &x0 * &(1.0 - &x0) - 2.0 * &(&x1 - 0.5) * &(&x1 - 0.5) + 0.4 * &x2;
    let yg = 2.0 + &s + 0.8 * &e1;
    let ynb = ((1.5 + 3.0 * &s) * 2.0 * &(0.5 + &e1) * &(0.5 + &e2) * 2.0).mapv(f64::floor);
    let ys = 10.0 + 3.0 * &s + 0.3 * &e1 / &(0.05 + e2.mapv(f64::abs));
    let w = 0.5 + 1.5 * &e3;
    let mut x = Array2::zeros((n, 3));
    x.column_mut(0).assign(&x0);
    x.column_mut(1).assign(&x1);
    x.column_mut(2).assign(&x2);
    (x, [yg, ynb, ys], w)
}

#[derive(Deserialize)]
struct LatticeFixture {
    inputs: LatticeInputs,
    bam: LatticeReference,
}

#[derive(Deserialize)]
struct LatticeInputs {
    n: usize,
}

#[derive(Deserialize)]
struct LatticeReference {
    sp: Vec<f64>,
    edf_total: f64,
    reml: f64,
    iter: usize,
    irls_iter: usize,
    rows: Vec<usize>,
    predictions_rows: Vec<f64>,
}

fn check_lattice<L, K, V>(
    name: &str,
    family: Family<L, K, V>,
    response: usize,
    weighted: bool,
    tol: Tol,
) where
    L: Loss + Clone,
    K: Link + Clone,
    V: VarianceFn + Clone,
    Family<L, K, V>: BamFamily,
{
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(format!("tests/fixtures/{name}.json"));
    let fx: LatticeFixture = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
    let (x, ys, w) = lattice(fx.inputs.n);
    let terms = (0..3)
        .map(|col| TermSpec::Cr {
            col,
            k: 10,
            pc: None,
            by: None,
        })
        .collect();
    let fit = fit_bam(
        family,
        &Additive { terms },
        x.view(),
        ys[response].view(),
        weighted.then(|| w.view()),
    )
    .unwrap();
    let r = fx.bam;
    let pred = fit
        .predict(x.view())
        .unwrap()
        .mapv(|e| fit.link_kind.inverse(e));
    let fitted = r
        .rows
        .iter()
        .zip(&r.predictions_rows)
        .map(|(&i, b)| rel(pred[i - 1], *b))
        .fold(0.0, f64::max);
    let sp = fit
        .lambda
        .iter()
        .zip(&r.sp)
        .map(|(a, b)| rel(*a, *b))
        .fold(0.0, f64::max);
    let report = format!(
        "{name}: fitted {fitted:.2e} sp {sp:.2e} edf {:.2e} reml {:.2e} iter {}/{} irls {}/{}",
        rel(fit.edf_total, r.edf_total),
        rel(fit.reml_value, r.reml),
        fit.n_iters,
        r.iter,
        fit.stats.inner_pirls_iterations_total,
        r.irls_iter
    );
    eprintln!("{report}");
    assert!(fitted < tol.fitted, "{report}");
    assert!(sp < tol.sp, "{report}");
    assert!(rel(fit.edf_total, r.edf_total) < tol.fitted, "{report}");
    assert!(rel(fit.reml_value, r.reml) < 1e-9, "{report}");
    assert_eq!(fit.n_iters, r.iter, "{report}");
}

#[test]
fn bam_above_chunk_size_uses_mgcvs_row_sample() {
    for n in [12000, 25000] {
        check_lattice(
            &format!("3d_gaussian_bam_freml_n{n}_lattice"),
            gaussian_identity(),
            0,
            false,
            GAUSSIAN,
        );
        check_lattice(
            &format!("3d_nb_log_bam_freml_n{n}_lattice"),
            negbin_log(1.0),
            1,
            false,
            PIRLS,
        );
        check_lattice(
            &format!("3d_scat_identity_weighted_bam_freml_n{n}_lattice"),
            tdist_identity(5.0, 1.0),
            2,
            true,
            PIRLS,
        );
    }
}

#[test]
fn bam_sweep_drop_tie_drops_rs_column() {
    check(
        "2d_gaussian_bam_freml_sweepdrop_tie_n1500",
        gaussian_identity(),
        "gaussian",
        GAUSSIAN,
    );
}

#[test]
fn bam_sweep_drop_tie_scat() {
    check(
        "2d_scat_identity_weighted_bam_freml_sweepdrop_tie_n1500",
        tdist_identity(5.0, 1.0),
        "scat",
        PIRLS,
    );
}

#[test]
fn bam_ill_conditioned_covariate_gaussian() {
    check(
        "2d_gaussian_bam_freml_illcond_n2000",
        gaussian_identity(),
        "gaussian",
        ILL_CONDITIONED,
    );
}

#[test]
fn bam_ill_conditioned_covariate_scat() {
    check(
        "2d_scat_identity_bam_freml_illcond_n2000",
        tdist_identity(5.0, 1.0),
        "scat",
        ILL_CONDITIONED,
    );
}
