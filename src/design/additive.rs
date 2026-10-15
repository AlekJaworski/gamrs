//! Additive multi-smooth DesignStrategy + Predictor.
//!
//! Models `y ~ s(x_{c_0}) + s(x_{c_1}) + … + te(x_{a}, x_{b}) + …` where
//! each term picks one or more columns of `x` and chooses its own basis
//! (CR spline, RE, CR-stable, tensor product). The design matrix is
//! `[1 | X_1 | X_2 | …]`, the penalty list is a flat `Vec<S_j>` (one
//! block-embedded matrix per smoothing parameter — a single-margin smooth
//! contributes ONE penalty, a tensor product term contributes TWO), and
//! the predict-time `AdditivePredictor` carries a `Vec<Predictor>` so
//! per-term reconstruction is just delegation.
//!
//! `TermSpec` is a closed-set enum (library-controlled basis set; users
//! extend `Loss/Link/VarianceFn`, not bases — see
//! `feedback_enum_vs_trait_dispatch`). New basis kinds (B-splines etc.)
//! become new variants.

use ndarray::{Array2, ArrayView2};

use crate::error::{GamrsError, Result};

use super::by::ByPredictor;
use super::cr::{CrPredictor, CrStablePredictor};
use super::re::RePredictor;
use super::tensor::TensorTermFit;
use super::{
    prepend_intercept, prepend_zero_column, rank_and_log_pseudo_det, DesignStrategy, Predictor,
    PreparedDesign,
};

/// Kind of marginal basis used inside a tensor product. Closed-set —
/// new marginal kinds extend the enum, never `Box<dyn>`. Starting with
/// `Cr` (CR + sum-to-zero centring per margin); future variants would
/// add `Bs` (B-spline) etc.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "persistence", serde(rename_all = "snake_case"))]
pub enum MarginKind {
    /// CR spline + sum-to-zero centring on this margin's column.
    Cr,
}

/// One term in an additive model. Each variant carries the column index
/// (or indices, for tensor products) into `x` and any basis-specific
/// config (e.g. `k` for CR splines).
///
/// Closed-set enum, library-controlled (per project standard — users
/// extend `Loss/Link/VarianceFn`, not bases). New basis types extend this
/// enum (a library-controlled change), not user code.
#[derive(Clone, Debug)]
pub enum TermSpec {
    /// CR spline + sum-to-zero centring on `x.column(col)` with `k` knots.
    /// `pc` is mgcv's `s(x, pc=)` point constraint — `Some(v)` replaces the
    /// centering constraint with "the smooth is zero at `x = v`".
    /// `by` is mgcv's numeric `s(x, by=w)` with `w = x.column(by)`: the term
    /// contributes `w_i · f(x_i)` and, unless `w` is constant, carries no
    /// centering constraint. Exclusive with `pc`.
    Cr {
        col: usize,
        k: usize,
        pc: Option<f64>,
        by: Option<usize>,
    },
    /// CR spline + sum-to-zero + StableReparam rotation on `x.column(col)`.
    /// `pc` and `by` as for [`Self::Cr`].
    CrStable {
        col: usize,
        k: usize,
        pc: Option<f64>,
        by: Option<usize>,
    },
    /// Random-effect (`bs="re"`) on `x.column(col)`.
    Re { col: usize },
    /// Anisotropic tensor product `te(x_{col_a}, x_{col_b})` — two
    /// smoothing parameters per term (one per margin). Marginals default
    /// to CR + sum-to-zero centring (`bs_a = bs_b = MarginKind::Cr`).
    Tensor {
        col_a: usize,
        col_b: usize,
        k_a: usize,
        k_b: usize,
        bs_a: MarginKind,
        bs_b: MarginKind,
    },
    /// N-margin anisotropic tensor product `te(x_{c_0}, ..., x_{c_{D-1}})`.
    /// `cols.len() = k.len() = bs.len() = D >= 2`. D smoothing parameters.
    TeMulti {
        cols: Vec<usize>,
        k: Vec<usize>,
        bs: Vec<MarginKind>,
    },
    /// N-margin tensor interaction `ti(x_{c_0}, ..., x_{c_{D-1}})` — pure
    /// interaction with main effects excluded (mgcv's `ti(...)`). D
    /// smoothing parameters per term.
    Ti {
        cols: Vec<usize>,
        k: Vec<usize>,
        bs: Vec<MarginKind>,
    },
    /// 2-D (or higher) isotropic thin-plate regression spline. Single
    /// smoothing parameter.
    Tps { cols: Vec<usize>, k: usize },
    /// Parametric (linear, unsmoothed) term `+ x_col` — one raw column,
    /// no spline expansion, no penalty. The coefficient retains its full
    /// degree of freedom. mgcv R's "pterms" block; equivalent to
    /// `bs="parametric"` (alias `"linear"`) in mgcv_rust 0.16+.
    Parametric { col: usize },
}

impl TermSpec {
    /// First column index this term reads from `x` (for diagnostics /
    /// duplicate-column checks). Tensor terms also read `col_b`; use
    /// [`Self::cols`] for the full list.
    pub fn col(&self) -> usize {
        match self {
            Self::Cr { col, .. } => *col,
            Self::CrStable { col, .. } => *col,
            Self::Re { col } => *col,
            Self::Parametric { col } => *col,
            Self::Tensor { col_a, .. } => *col_a,
            Self::TeMulti { cols, .. } | Self::Ti { cols, .. } | Self::Tps { cols, .. } => cols[0],
        }
    }

    /// All columns this term reads from `x`. Length 1 for univariate
    /// terms, >= 2 for tensor / interaction / TPRS terms and for a smooth
    /// with a `by` variable (`[col, by]`).
    pub fn cols(&self) -> Vec<usize> {
        match self {
            Self::Cr { col, by, .. } | Self::CrStable { col, by, .. } => {
                std::iter::once(*col).chain(*by).collect()
            }
            Self::Re { col } => vec![*col],
            Self::Parametric { col } => vec![*col],
            Self::Tensor { col_a, col_b, .. } => vec![*col_a, *col_b],
            Self::TeMulti { cols, .. } | Self::Ti { cols, .. } | Self::Tps { cols, .. } => {
                cols.clone()
            }
        }
    }
}

/// Predict-time additive design rebuilder. Holds a per-term `Predictor`,
/// the `(start, end)` column ranges in the combined design (so vcov /
/// per-term slicing can find each term), and the original column indices
/// each term reads from new x.
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct AdditivePredictor {
    /// One sub-Predictor per term. Univariate sub-predictors are invoked
    /// on a single-column slice; tensor sub-predictors are invoked on the
    /// full `x_new` (they slice their own marginal columns internally).
    pub term_predictors: Vec<Predictor>,
    /// Columns of `x_new` each term reads from. Length 1 for univariate
    /// terms, length 2 for tensor products. Length of the outer Vec equals
    /// `term_predictors.len()`.
    pub cols_used: Vec<Vec<usize>>,
    /// `(start, end)` column ranges in the combined design for each
    /// term's smooth coefficients (intercept lives at column 0, then
    /// term j occupies `term_col_ranges[j]`). Used by downstream callers
    /// that want per-term β slices or marginal predictions.
    pub term_col_ranges: Vec<(usize, usize)>,
}

impl AdditivePredictor {
    pub(crate) fn design(&self, x_new: ArrayView2<f64>) -> Result<Array2<f64>> {
        let n = x_new.nrows();
        // Each sub-predictor's design starts with an intercept column we
        // need to strip — only the leading [1 …] of the assembled
        // additive design is kept.
        let mut term_centred: Vec<Array2<f64>> = Vec::with_capacity(self.term_predictors.len());
        let mut total_smooth_cols = 0usize;
        for (j, pred) in self.term_predictors.iter().enumerate() {
            for &c in &self.cols_used[j] {
                if c >= x_new.ncols() {
                    return Err(GamrsError::InvalidParameter(format!(
                        "AdditivePredictor: term {j} expects column {c} but x has \
                         only {} columns",
                        x_new.ncols()
                    )));
                }
            }
            let sub = invoke_sub_design(pred, x_new, &self.cols_used[j])?;
            // Drop the per-term intercept column (kept at index 0 in each
            // sub-design). `sub.ncols()` is `1 + smooth_dim_j`.
            let smooth_dim = sub.ncols() - 1;
            let centred = sub.slice(ndarray::s![.., 1..]).to_owned();
            total_smooth_cols += smooth_dim;
            term_centred.push(centred);
        }
        // Assemble combined design `[1 | C_1 | C_2 | …]`.
        let p = 1 + total_smooth_cols;
        let mut design = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            design[[i, 0]] = 1.0;
        }
        let mut col_offset = 1usize;
        for centred in &term_centred {
            let k = centred.ncols();
            for i in 0..n {
                for c in 0..k {
                    design[[i, col_offset + c]] = centred[[i, c]];
                }
            }
            col_offset += k;
        }
        Ok(design)
    }

    pub(crate) fn design_deriv(&self, x_new: ArrayView2<f64>, axis: usize) -> Result<Array2<f64>> {
        // `axis` indexes into the columns of `x_new`. For each term, the
        // smooth's `∂design/∂x_axis` is nonzero only if the term reads
        // column `axis`; otherwise all entries are zero.
        let n = x_new.nrows();
        let p = 1 + self.term_col_ranges.last().map(|(_, e)| e - 1).unwrap_or(0);
        let mut d = Array2::<f64>::zeros((n, p));
        for (j, pred) in self.term_predictors.iter().enumerate() {
            let cols = &self.cols_used[j];
            if !cols.contains(&axis) {
                continue;
            }
            let sub_d1 = invoke_sub_design_deriv(pred, x_new, cols, axis)?;
            let (start, end) = self.term_col_ranges[j];
            let k = end - start;
            debug_assert_eq!(sub_d1.ncols() - 1, k);
            for i in 0..n {
                for c in 0..k {
                    d[[i, start + c]] = sub_d1[[i, 1 + c]];
                }
            }
        }
        let _ = prepend_zero_column; // referenced by other variants — keep import warm
        Ok(d)
    }
}

/// Dispatch helper: pass univariate sub-predictors a single-column slice,
/// pass tensor sub-predictors the full `x_new` (they slice marginal
/// columns themselves). Returns the sub-design with the per-term
/// intercept column still attached at index 0 — the caller strips it.
fn invoke_sub_design(
    pred: &Predictor,
    x_new: ArrayView2<f64>,
    cols: &[usize],
) -> Result<Array2<f64>> {
    match pred {
        // Multi-column sub-predictors slice their own margin columns from
        // the full design matrix using the column indices stored at
        // fit-time. `Tps::cols` may have arbitrary length (≥ 2).
        Predictor::Tensor(_) | Predictor::TensorMulti(_) | Predictor::Tps(_) | Predictor::By(_) => {
            pred.design(x_new)
        }
        _ => {
            // Univariate: pass the single configured column as a (n, 1) view.
            debug_assert_eq!(cols.len(), 1);
            let c = cols[0];
            pred.design(x_new.slice(ndarray::s![.., c..c + 1]))
        }
    }
}

fn invoke_sub_design_deriv(
    pred: &Predictor,
    x_new: ArrayView2<f64>,
    cols: &[usize],
    axis: usize,
) -> Result<Array2<f64>> {
    match pred {
        Predictor::Tensor(_) | Predictor::TensorMulti(_) | Predictor::Tps(_) | Predictor::By(_) => {
            pred.design_deriv(x_new, axis)
        }
        _ => {
            // Univariate sub-predictors take (n, 1) and an axis of 0.
            debug_assert_eq!(cols.len(), 1);
            let c = cols[0];
            pred.design_deriv(x_new.slice(ndarray::s![.., c..c + 1]), 0)
        }
    }
}

/// Additive multi-smooth `DesignStrategy`. Each [`TermSpec`] picks a
/// column of `x` and a basis; the assembled design is `[1 | X_1 | X_2 |
/// …]` with one block-embedded `(p, p)` penalty per term.
///
/// `terms.len() == 1` is allowed but redundant — prefer `Cr`, `Re`, or
/// `CrStable` directly for single-smooth fits. The inner solvers handle
/// either case uniformly via `s_list.len()`.
pub struct Additive {
    pub terms: Vec<TermSpec>,
}

impl DesignStrategy for Additive {
    fn prepare(&self, x: ArrayView2<f64>) -> Result<PreparedDesign> {
        if self.terms.is_empty() {
            return Err(GamrsError::InvalidParameter(
                "Additive: terms list must be non-empty".into(),
            ));
        }
        for (j, t) in self.terms.iter().enumerate() {
            for c in t.cols() {
                if c >= x.ncols() {
                    return Err(GamrsError::InvalidParameter(format!(
                        "Additive term {j} reads column {c} but x has only {} columns",
                        x.ncols()
                    )));
                }
            }
        }

        // First pass: build per-term centred designs + per-term penalty
        // *lists* (1 entry for univariate terms, 2 for tensor products) +
        // sub-predictors. Track per-term smooth dim so we know where each
        // block sits in the combined penalty.
        let n = x.nrows();
        let mut per_term_centred: Vec<Array2<f64>> = Vec::with_capacity(self.terms.len());
        // Per-term penalty list, in the term's own coefficient frame
        // (length = number of smoothing params for the term).
        let mut per_term_s_smooth: Vec<Vec<Array2<f64>>> = Vec::with_capacity(self.terms.len());
        let mut per_term_predictor: Vec<Predictor> = Vec::with_capacity(self.terms.len());
        let mut per_term_dim: Vec<usize> = Vec::with_capacity(self.terms.len());
        let mut cols_used: Vec<Vec<usize>> = Vec::with_capacity(self.terms.len());

        for t in &self.terms {
            match *t {
                TermSpec::Cr { col, k, pc, by } => {
                    let fit = CrPredictor::fit_centred(x, col, k, pc, by)?;
                    per_term_dim.push(fit.centred.ncols());
                    per_term_centred.push(fit.centred);
                    per_term_s_smooth.push(vec![fit.s_smooth]);
                    let pred = Predictor::Cr(CrPredictor {
                        knots: fit.knots,
                        centring: fit.centring,
                    });
                    per_term_predictor.push(ByPredictor::wrap(pred, col, by));
                    cols_used.push(t.cols());
                }
                TermSpec::CrStable { col, k, pc, by } => {
                    let fit = CrStablePredictor::fit_rotated(x, col, k, pc, by)?;
                    per_term_dim.push(fit.rotated.ncols());
                    per_term_centred.push(fit.rotated);
                    per_term_s_smooth.push(vec![fit.s_smooth]);
                    let pred = Predictor::CrStable(CrStablePredictor {
                        knots: fit.knots,
                        centring: fit.centring,
                        reparam_v: fit.reparam_v,
                    });
                    per_term_predictor.push(ByPredictor::wrap(pred, col, by));
                    cols_used.push(t.cols());
                }
                TermSpec::Re { col } => {
                    let fit = RePredictor::fit_raw(x, col)?;
                    let n_levels = fit.levels.len();
                    per_term_dim.push(n_levels);
                    per_term_centred.push(fit.raw);
                    // Identity penalty on the RE levels.
                    let mut s_re = Array2::<f64>::zeros((n_levels, n_levels));
                    for j in 0..n_levels {
                        s_re[[j, j]] = 1.0;
                    }
                    per_term_s_smooth.push(vec![s_re]);
                    per_term_predictor.push(Predictor::Re(RePredictor { levels: fit.levels }));
                    cols_used.push(vec![col]);
                }
                TermSpec::Tensor {
                    col_a,
                    col_b,
                    k_a,
                    k_b,
                    bs_a,
                    bs_b,
                } => {
                    let fit = TensorTermFit::build(x, col_a, col_b, k_a, k_b, bs_a, bs_b)?;
                    per_term_dim.push(fit.design.ncols());
                    per_term_centred.push(fit.design);
                    per_term_s_smooth.push(fit.s_list_term);
                    per_term_predictor.push(Predictor::Tensor(fit.predictor));
                    cols_used.push(vec![col_a, col_b]);
                }
                TermSpec::TeMulti { .. } | TermSpec::Ti { .. } => {
                    let (cols, k, bs, interaction) = match t {
                        TermSpec::TeMulti { cols, k, bs } => {
                            (cols.clone(), k.clone(), bs.clone(), false)
                        }
                        TermSpec::Ti { cols, k, bs } => (cols.clone(), k.clone(), bs.clone(), true),
                        _ => unreachable!(),
                    };
                    let fit = if interaction {
                        super::tensor::TensorMultiTermFit::build_ti(x, &cols, &k, &bs)?
                    } else {
                        super::tensor::TensorMultiTermFit::build_te(x, &cols, &k, &bs)?
                    };
                    per_term_dim.push(fit.design.ncols());
                    per_term_centred.push(fit.design);
                    per_term_s_smooth.push(fit.s_list_term);
                    per_term_predictor.push(Predictor::TensorMulti(fit.predictor));
                    cols_used.push(cols);
                }
                TermSpec::Tps { .. } => {
                    let (tcols, tk) = match t {
                        TermSpec::Tps { cols, k } => (cols.clone(), *k),
                        _ => unreachable!(),
                    };
                    let fit = super::tps::TpsTermFit::build(x, &tcols, tk)?;
                    per_term_dim.push(fit.design.ncols());
                    per_term_centred.push(fit.design);
                    per_term_s_smooth.push(vec![fit.s_smooth]);
                    per_term_predictor.push(Predictor::Tps(fit.predictor));
                    cols_used.push(tcols);
                }
                TermSpec::Parametric { col } => {
                    // Raw single column; no centring (would absorb effect
                    // into intercept and destroy the slope interpretation).
                    let raw = x.column(col).to_owned();
                    let mut design = Array2::<f64>::zeros((n, 1));
                    for i in 0..n {
                        design[[i, 0]] = raw[i];
                    }
                    per_term_dim.push(1);
                    per_term_centred.push(design);
                    // Empty penalty list — the coefficient is unpenalised;
                    // no smoothing parameter to optimise.
                    per_term_s_smooth.push(Vec::new());
                    per_term_predictor.push(Predictor::Parametric(
                        super::parametric::ParametricPredictor { col: 0 },
                    ));
                    cols_used.push(vec![col]);
                }
            }
        }

        // Second pass: assemble combined design `[1 | C_1 | C_2 | …]`.
        let p = 1 + per_term_dim.iter().sum::<usize>();
        let mut x_design = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            x_design[[i, 0]] = 1.0;
        }
        let mut term_col_ranges: Vec<(usize, usize)> = Vec::with_capacity(self.terms.len());
        let mut col_offset = 1usize;
        for centred in &per_term_centred {
            let k = centred.ncols();
            for i in 0..n {
                for c in 0..k {
                    x_design[[i, col_offset + c]] = centred[[i, c]];
                }
            }
            term_col_ranges.push((col_offset, col_offset + k));
            col_offset += k;
        }

        // Third pass: build the flat penalty list — for each term, embed
        // each of its per-term-frame penalties into the (p, p) combined
        // coefficient frame at the term's column range. A univariate term
        // contributes 1 entry; a tensor term contributes 2.
        let mut s_list: Vec<Array2<f64>> = Vec::new();
        let mut rank_s_list: Vec<usize> = Vec::new();
        let mut log_pseudo_det_s_list: Vec<f64> = Vec::new();
        for (j, s_list_term) in per_term_s_smooth.iter().enumerate() {
            let (start, end) = term_col_ranges[j];
            let k = end - start;
            for s_smooth in s_list_term {
                debug_assert_eq!(s_smooth.nrows(), k);
                debug_assert_eq!(s_smooth.ncols(), k);
                let mut s_block = Array2::<f64>::zeros((p, p));
                for r in 0..k {
                    for c in 0..k {
                        s_block[[start + r, start + c]] = s_smooth[[r, c]];
                    }
                }
                let (rank_j, log_det_j) = rank_and_log_pseudo_det(s_block.view())?;
                s_list.push(s_block);
                rank_s_list.push(rank_j);
                log_pseudo_det_s_list.push(log_det_j);
            }
        }

        let rank_total: usize = rank_s_list.iter().sum();
        // `mp` is the null-space dimension of the *combined* penalty
        // `Σ_j S_j`. For block-diagonal terms this equals `p − Σ rank_j`
        // only when the per-term null spaces stack to a full union (each
        // term's null space within its block contributes to the combined
        // null space). The previous single-penalty-per-term path used the
        // same calculation; keep it as a documented approximation rather
        // than a true rank-of-sum which would require an eigensolve here.
        let mp = p.saturating_sub(rank_total);

        Ok(PreparedDesign {
            x_design,
            s_list,
            rank_s_list,
            log_pseudo_det_s_list,
            mp,
            predictor: Predictor::Additive(AdditivePredictor {
                term_predictors: per_term_predictor,
                cols_used,
                term_col_ranges,
            }),
        })
    }
}

// Silence the unused-import warning on `prepend_intercept`; the
// `Additive` design assembles its own intercept column inline (the
// per-term sub-Predictors prepend one each, which we then strip).
#[allow(dead_code)]
const _PREPEND_INTERCEPT_USED: fn(ArrayView2<f64>) -> Array2<f64> = prepend_intercept;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    /// Two CR terms on a (n, 2) input: combined design width is
    /// `1 + k_smooth_0 + k_smooth_1`, and each penalty block lives in its
    /// term's column range. `s_list` length = number of terms.
    #[test]
    fn two_cr_terms_block_diagonal_penalty_assembly() {
        let n = 60;
        let mut x_vec: Vec<f64> = Vec::with_capacity(n * 2);
        for i in 0..n {
            x_vec.push((i as f64) / (n as f64));
            x_vec.push(((i as f64) * 2.7).sin());
        }
        let x = Array2::from_shape_vec((n, 2), x_vec).unwrap();
        let strategy = Additive {
            terms: vec![
                TermSpec::Cr {
                    col: 0,
                    k: 8,
                    pc: None,
                    by: None,
                },
                TermSpec::Cr {
                    col: 1,
                    k: 6,
                    pc: None,
                    by: None,
                },
            ],
        };
        let prep = strategy.prepare(x.view()).unwrap();
        assert_eq!(prep.s_list.len(), 2, "one penalty block per term");
        assert_eq!(prep.rank_s_list.len(), 2);
        assert_eq!(prep.log_pseudo_det_s_list.len(), 2);

        // Each penalty block should only touch its term's column range:
        // block 0 is nonzero in columns [1, k_smooth_0+1); block 1 in
        // [k_smooth_0+1, p). Verify by checking that S_0's last row is
        // entirely zero and S_1's first non-intercept row is zero.
        let p = prep.x_design.ncols();
        let last_row_s0: f64 = (0..p).map(|c| prep.s_list[0][[p - 1, c]].abs()).sum();
        assert!(
            last_row_s0 == 0.0,
            "S_0's last row should be zero (belongs to term 1)"
        );
        let first_row_s1: f64 = (0..p).map(|c| prep.s_list[1][[1, c]].abs()).sum();
        assert!(
            first_row_s1 == 0.0,
            "S_1's first row should be zero (belongs to term 0)"
        );

        // Mp = p - Σ_j rank_j.
        let rank_total: usize = prep.rank_s_list.iter().sum();
        assert_eq!(prep.mp, p - rank_total);
    }

    /// `Additive` with a single term should be functionally equivalent
    /// (up to FP) to the single-smooth `Cr { k }` path. Tests the
    /// rebuild/predict roundtrip.
    #[test]
    fn additive_single_term_matches_cr() {
        let n = 40;
        let mut x_vec: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            x_vec.push((i as f64) / (n as f64 - 1.0));
        }
        // Reshape to (n, 1) for the single-smooth path.
        let x1 = Array2::from_shape_vec((n, 1), x_vec.clone()).unwrap();

        let prep_cr = super::super::Cr { k: 6 }.prepare(x1.view()).unwrap();
        let prep_add = Additive {
            terms: vec![TermSpec::Cr {
                col: 0,
                k: 6,
                pc: None,
                by: None,
            }],
        }
        .prepare(x1.view())
        .unwrap();

        assert_eq!(prep_cr.x_design.shape(), prep_add.x_design.shape());
        for i in 0..n {
            for j in 0..prep_cr.x_design.ncols() {
                assert!(
                    (prep_cr.x_design[[i, j]] - prep_add.x_design[[i, j]]).abs() < 1e-12,
                    "design mismatch at [{i}, {j}]: cr={} add={}",
                    prep_cr.x_design[[i, j]],
                    prep_add.x_design[[i, j]],
                );
            }
        }
        assert_eq!(prep_cr.s_list.len(), 1);
        assert_eq!(prep_add.s_list.len(), 1);
        let p = prep_cr.x_design.ncols();
        for i in 0..p {
            for j in 0..p {
                assert!((prep_cr.s_list[0][[i, j]] - prep_add.s_list[0][[i, j]]).abs() < 1e-12,);
            }
        }
    }

    /// `AdditivePredictor::design` rebuilds the combined design on new x.
    /// Verifies the column layout `[1 | C_1 | C_2]` and that each per-
    /// term sub-predictor reads only its configured column.
    #[test]
    fn additive_predictor_rebuilds_combined_design() {
        let n = 30;
        let mut x_vec: Vec<f64> = Vec::with_capacity(n * 2);
        for i in 0..n {
            x_vec.push((i as f64) / (n as f64));
            x_vec.push(((i as f64) * 0.31).cos());
        }
        let x = Array2::from_shape_vec((n, 2), x_vec.clone()).unwrap();
        let strategy = Additive {
            terms: vec![
                TermSpec::Cr {
                    col: 0,
                    k: 5,
                    pc: None,
                    by: None,
                },
                TermSpec::Cr {
                    col: 1,
                    k: 4,
                    pc: None,
                    by: None,
                },
            ],
        };
        let prep = strategy.prepare(x.view()).unwrap();
        // Rebuild on the same x — should match the fit-time design.
        let rebuilt = prep.predictor.design(x.view()).unwrap();
        assert_eq!(rebuilt.shape(), prep.x_design.shape());
        for i in 0..n {
            for j in 0..prep.x_design.ncols() {
                assert!(
                    (rebuilt[[i, j]] - prep.x_design[[i, j]]).abs() < 1e-12,
                    "rebuilt design mismatch at [{i}, {j}]"
                );
            }
        }
    }

    /// `combined_s` is the hot-path helper inside every inner solver.
    /// Verify it produces `Σ_j exp(ρ_j) · s_list[j]` with no accidental
    /// scaling or transposition.
    #[test]
    fn combined_s_sums_per_term_lambdas() {
        let p = 4;
        let mut s0 = Array2::<f64>::zeros((p, p));
        s0[[1, 1]] = 1.0;
        let mut s1 = Array2::<f64>::zeros((p, p));
        s1[[2, 2]] = 2.0;
        let s_list = vec![s0, s1];
        let rho = Array1::from_vec(vec![0.0_f64, (3.0_f64).ln()]);
        let s_total = super::super::combined_s(&s_list, &rho, p);
        assert!((s_total[[1, 1]] - 1.0).abs() < 1e-15);
        // exp(ln 3) * 2.0 = 6.0
        assert!((s_total[[2, 2]] - 6.0).abs() < 1e-13);
        // Off-block entries stay zero.
        assert_eq!(s_total[[0, 0]], 0.0);
        assert_eq!(s_total[[3, 3]], 0.0);
    }
}
