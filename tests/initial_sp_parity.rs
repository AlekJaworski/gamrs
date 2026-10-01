//! `MgcvInit` is mgcv's `initial.sp`, checked on mgcv's own design and
//! penalties so the gamrs basis is not in the comparison. The fixture's third
//! term is lot-size-like; `SmartInit` started that shape past a ridge in the
//! criterion on real data (docs/scat_start_basin_bug.md).

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
