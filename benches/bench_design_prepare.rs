//! Wall-time benchmark: design assembly alone (`Additive::prepare`), which
//! is the only stage the point-constraint (`pc`) change touches.
//!
//! A whole fit spends most of its time in the outer/inner solves, so a few
//! percent of design-assembly cost disappears into fit-level noise on a
//! shared box. Timing `prepare` directly magnifies whatever the constraint
//! row costs: if it is invisible here it cannot matter in a fit.
//!
//! Both arms run in ONE process, so the pc-versus-centred comparison carries
//! no cross-build variance at all. The centred arm is also the number to
//! compare against a pre-pc build of the same bench.
//!
//! Run: `cargo bench --bench bench_design_prepare` (release profile).

use std::time::Instant;

use ndarray::{Array1, Array2};

use gamrs::design::{Additive, DesignStrategy, TermSpec};

struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn synth(n: usize, seed: u64) -> Array2<f64> {
    let mut rng = Lcg(seed);
    let mut x = Array2::<f64>::zeros((n, 3));
    for i in 0..n {
        for j in 0..3 {
            x[[i, j]] = rng.next_f64();
        }
    }
    x
}

fn terms(pc: Option<f64>) -> Vec<TermSpec> {
    (0..3)
        .map(|col| TermSpec::Cr {
            col,
            k: 10,
            pc,
            by: None,
        })
        .collect()
}

/// Min and median milliseconds per `prepare`, over `batches` batches of
/// `per_batch` calls. Min is the estimator that survives interference from
/// other work on the machine (which can only ever slow a batch down).
fn time_prepare(x: &Array2<f64>, pc: Option<f64>, batches: usize, per_batch: usize) -> (f64, f64) {
    let spec = terms(pc);
    let mut ms: Vec<f64> = Vec::with_capacity(batches);
    for _ in 0..batches {
        let t0 = Instant::now();
        for _ in 0..per_batch {
            let prep = Additive {
                terms: spec.clone(),
            }
            .prepare(x.view())
            .unwrap();
            std::hint::black_box(prep.x_design.len());
        }
        ms.push(1000.0 * t0.elapsed().as_secs_f64() / (per_batch as f64));
    }
    let mut sorted = Array1::from(ms.clone()).to_vec();
    sorted.sort_by(f64::total_cmp);
    (sorted[0], sorted[sorted.len() / 2])
}

fn main() {
    println!("[bench_design_prepare] Additive::prepare, 3 Cr terms (k=10)");
    for &n in &[500usize, 2000, 5000] {
        let x = synth(n, 0x0a11_ce00_1234_5678);
        // Warm both arms before either is timed.
        let _ = time_prepare(&x, None, 1, 10);
        let _ = time_prepare(&x, Some(0.5), 1, 10);
        let (centred_min, centred_med) = time_prepare(&x, None, 7, 60);
        let (pc_min, pc_med) = time_prepare(&x, Some(0.5), 7, 60);
        println!(
            "  n={n:5}  centred min={centred_min:.4}ms med={centred_med:.4}ms  \
             pc min={pc_min:.4}ms med={pc_med:.4}ms  \
             pc/centred(min)={:.3}",
            pc_min / centred_min
        );
    }
}
