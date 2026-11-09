//! `log|Σ_j λ_j S_j|₊` and its ρ-derivatives.
//!
//! A penalty that shares no coefficient with any other contributes
//! `rank_j·ρ_j + log|S_j|₊`, linear in ρ. Penalties that overlap — the
//! margins of one tensor product — do not factor that way: their sum is
//! decomposed at each ρ, and its rank is that of the sum, not the sum of
//! the ranks.

use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::{Eigh, UPLO};

use crate::design::penalty_groups;
use crate::error::{GamrsError, Result};

enum Group {
    Single {
        j: usize,
        rank: usize,
        log_det: f64,
    },
    Overlap {
        penalties: Vec<usize>,
        /// Each penalty restricted to the group's coefficients.
        blocks: Vec<Array2<f64>>,
        rank: usize,
    },
}

pub struct PenaltyLogDet {
    groups: Vec<Group>,
    m: usize,
}

/// `log|λS|₊` at one ρ with its gradient and Hessian in ρ.
pub struct LogDetTerms {
    pub value: f64,
    pub grad: Array1<f64>,
    pub hess: Array2<f64>,
}

impl PenaltyLogDet {
    pub fn new(
        s_list: &[Array2<f64>],
        rank_s_list: &[usize],
        log_pseudo_det_s_list: &[f64],
    ) -> Self {
        let groups = penalty_groups(s_list, rank_s_list)
            .into_iter()
            .map(|g| match g.penalties.as_slice() {
                [j] => Group::Single {
                    j: *j,
                    rank: rank_s_list[*j],
                    log_det: log_pseudo_det_s_list[*j],
                },
                _ => Group::Overlap {
                    blocks: g
                        .penalties
                        .iter()
                        .map(|&j| {
                            s_list[j]
                                .select(Axis(0), &g.coefs)
                                .select(Axis(1), &g.coefs)
                        })
                        .collect(),
                    penalties: g.penalties,
                    rank: g.rank,
                },
            })
            .collect();
        Self {
            groups,
            m: s_list.len(),
        }
    }

    pub fn eval(&self, rho: &Array1<f64>) -> Result<LogDetTerms> {
        let mut value = 0.0;
        let mut grad = Array1::<f64>::zeros(self.m);
        let mut hess = Array2::<f64>::zeros((self.m, self.m));
        for g in &self.groups {
            match g {
                Group::Single { j, rank, log_det } => {
                    value += *rank as f64 * rho[*j] + log_det;
                    grad[*j] = *rank as f64;
                }
                Group::Overlap { rank: 0, .. } => {}
                Group::Overlap {
                    penalties,
                    blocks,
                    rank,
                } => {
                    let lambda: Vec<f64> = penalties.iter().map(|&j| rho[j].exp()).collect();
                    let mut total = Array2::<f64>::zeros(blocks[0].raw_dim());
                    for (b, l) in blocks.iter().zip(&lambda) {
                        total.scaled_add(*l, b);
                    }
                    let (eigs, vecs) = total.eigh(UPLO::Lower).map_err(|e| {
                        GamrsError::Linalg(format!("penalty eigendecomposition failed: {e}"))
                    })?;
                    // Ascending: the range space is the last `rank` pairs.
                    let k = eigs.len();
                    let range = vecs.slice(ndarray::s![.., k - rank..]).to_owned();
                    let kept = eigs.slice(ndarray::s![k - rank..]);
                    let floor = kept[rank - 1] * f64::EPSILON;
                    value += kept.iter().map(|e| e.max(floor).ln()).sum::<f64>();
                    let inv = kept.mapv(|e| 1.0 / e.max(floor));
                    let pinv = (&range * &inv.insert_axis(Axis(0))).dot(&range.t());
                    let pinv_s: Vec<Array2<f64>> = blocks.iter().map(|b| pinv.dot(b)).collect();
                    for (a, &i) in penalties.iter().enumerate() {
                        let gi = lambda[a] * pinv_s[a].diag().sum();
                        grad[i] = gi;
                        hess[[i, i]] += gi;
                        for (b, &j) in penalties.iter().enumerate() {
                            let tr: f64 = (&pinv_s[a] * &pinv_s[b].t()).sum();
                            hess[[i, j]] -= lambda[a] * lambda[b] * tr;
                        }
                    }
                }
            }
        }
        Ok(LogDetTerms { value, grad, hess })
    }
}
