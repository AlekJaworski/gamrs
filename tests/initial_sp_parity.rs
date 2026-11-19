//! `MgcvInit` is mgcv's `initial.sp`, checked on mgcv's own design and
//! penalties so the gamrs basis is not in the comparison. The fixture's third
//! term is lot-size-like; `SmartInit` started that shape past a ridge in the
//! criterion on real data.

use ndarray::{Array1, Array2};
use serde::Deserialize;

use gamrs::fit::{LambdaInit, MgcvInit};

#[derive(Deserialize)]
struct Fixture {
    n: usize,
    p: usize,
    x: Vec<Vec<f64>>,
    s_list: Vec<Vec<Vec<f64>>>,
    weights: Vec<f64>,
    log_sp_unweighted: Vec<f64>,
    log_sp_weighted: Vec<f64>,
}

fn matrix(rows: &[Vec<f64>], nrow: usize, ncol: usize) -> Array2<f64> {
    Array2::from_shape_vec((nrow, ncol), rows.iter().flatten().copied().collect()).unwrap()
}

#[test]
fn mgcv_init_matches_mgcv_initial_sp_with_and_without_weights() {
    let f: Fixture =
        serde_json::from_str(include_str!("fixtures/initial_sp_3smooth_n200.json")).unwrap();
    let x = matrix(&f.x, f.n, f.p);
    let s_list: Vec<Array2<f64>> = f.s_list.iter().map(|s| matrix(s, f.p, f.p)).collect();
    let y = Array1::<f64>::zeros(f.n);
    let cases = [
        (None, &f.log_sp_unweighted),
        (Some(Array1::from(f.weights.clone())), &f.log_sp_weighted),
    ];
    for (weights, want) in cases {
        let got = MgcvInit { weights }.init(y.view(), &x, &s_list);
        for (j, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-9, "term {j}: log sp {g} vs mgcv {w}");
        }
    }
}

#[derive(Deserialize)]
struct TermsFixture {
    x: Vec<Vec<f64>>,
    cases: Vec<TermsCase>,
}

#[derive(Deserialize)]
struct TermsCase {
    name: String,
    family: String,
    y: Vec<f64>,
    weights: Option<Vec<f64>>,
    log_sp: Vec<f64>,
    log_trace_s: Vec<f64>,
}

/// Where `gam(method = "REML")` starts, per family and term type, against
/// mgcv's `log(initial.spg(...))` on the same model. Compared as `λ_j·S_j`
/// (log λ_j + log tr S_j), since gamrs scales a te margin's penalty differently.
#[test]
fn glm_start_matches_mgcv_initial_spg_on_every_term_type() {
    use gamrs::design::{Additive, DesignStrategy, MarginKind, TermSpec};
    use gamrs::family::{
        bernoulli_logit, gamma_inverse, gamma_log, gaussian_identity, inverse_gaussian_log,
        poisson_log,
    };

    let f: TermsFixture =
        serde_json::from_str(include_str!("fixtures/initial_sp_terms_n300.json")).unwrap();
    let x = matrix(&f.x, f.x.len(), f.x[0].len());
    let cr = |col, k, by| TermSpec::Cr {
        col,
        k,
        pc: None,
        by,
    };
    let terms = vec![
        cr(0, 8, None),
        cr(1, 6, Some(4)),
        TermSpec::Re { col: 5, by: None },
        TermSpec::Tensor {
            col_a: 2,
            col_b: 3,
            k_a: 5,
            k_b: 4,
            bs_a: MarginKind::Cr,
            bs_b: MarginKind::Cr,
            by: None,
        },
    ];
    let prep = Additive { terms }.prepare(x.view()).unwrap();
    let log_trace: Vec<f64> = prep.s_list.iter().map(|s| s.diag().sum().ln()).collect();
    for case in &f.cases {
        let y = Array1::from(case.y.clone());
        let w = case.weights.clone().map(Array1::from);
        let (y, w) = (y.view(), w.as_ref().map(|w| w.view()));
        let init = match case.family.as_str() {
            "gaussian" => MgcvInit::glm(&gaussian_identity(), y, w),
            "poisson" => MgcvInit::glm(&poisson_log(), y, w),
            "binomial" => MgcvInit::glm(&bernoulli_logit(), y, w),
            "gamma_log" => MgcvInit::glm(&gamma_log(), y, w),
            "gamma_inverse" => MgcvInit::glm(&gamma_inverse(), y, w),
            "invgauss" => MgcvInit::glm(&inverse_gaussian_log(), y, w),
            other => panic!("unknown family {other}"),
        };
        let got = init.init(y, &prep.x_design, &prep.s_list);
        assert_eq!(got.len(), case.log_sp.len(), "{}", case.name);
        for j in 0..got.len() {
            let (g, want) = (got[j] + log_trace[j], case.log_sp[j] + case.log_trace_s[j]);
            assert!(
                (g - want).abs() < 1e-9,
                "{} sp {j}: log sp + log tr S {g} vs mgcv {want}",
                case.name
            );
        }
    }
}
