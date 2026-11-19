# Performance

## Choosing `method`

| `method=` | Smoothing-parameter search | Use it for |
| --- | --- | --- |
| `"REML"` (default) | Damped Newton on the REML score, inner fit run to convergence | Everything; the most accurate |
| `"fREML"` / `"bam"` | mgcv's `bam(method = "fREML")`: performance iteration with Newton on the fREML criterion | Large n with gaussian, poisson, binomial, Gamma, nb or scat; matching mgcv `bam` |
| `"efs"` | Fellner–Schall updates (as in mgcv's `gam(optimizer = "efs")`), one inner step per update | Large-n binomial, Poisson, Gamma, inverse Gaussian and quasi fits |

`"REML"` and `"efs"` search the same REML criterion; `"fREML"` fits the model
`bam` fits, so its results match `bam` rather than `gam`. On a Poisson fit with
four smooths and n = 200,000, `efs` took 1.7 s against 5.0 s for `REML`, and
the linear predictors agreed to 1.3e-4, including on the two terms that are
straight lines (edf 1.00 under both).

Where bam is not available (tensors, Tweedie, inverse Gaussian, fixed θ,
quantile and location–scale fits) `method="fREML"` fits by REML and warns;
`method="bam"` raises. A covariate spanning a range of about 1e5× (lot size from 1e3 to 2.5e8,
say) can leave a bam fit about 1e-5 sd from R's: gamrs works with X'WX,
which squares the design's condition number, where bam works with a QR factor. Families without a Fellner–Schall branch (Gaussian,
`nb`, `tw`, `scat`, `ocat`, quantile and location–scale fits) fit by REML
with a warning under `method="efs"`.

## Threads

Fits release the GIL, so independent fits run in parallel on Python threads.
OpenBLAS also starts its own threads. When running several fits at once, set
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` so the fits don't compete
for cores.

## Reading `fit_stats_`

Every fitted `Gam` has a `fit_stats_` dict of solver counters:

```python
g.fit_stats_
# {'outer_iterations': 11, 'line_search_trials': 0,
#  'inner_pirls_calls': 11, 'inner_pirls_iterations_total': 11,
#  'pirls_iters_per_call': 1.0, 'no_refresh_attempts': 0,
#  'no_refresh_hits': 0, 'no_refresh_hit_rate': 0.0}
```

- **`outer_iterations` of 50 or more**: the smoothing-parameter search is
  struggling. Check `g.converged_`, the family, and whether `k` is too large
  for the data.
- **`pirls_iters_per_call` above 10**: the inner fit is ill-conditioned,
  often from strongly correlated predictors.
- **`line_search_trials` much larger than `outer_iterations`**: Newton steps
  are often rejected and halved. The fit is still correct, just slower.

## `discrete=` and `nthreads=`

Both are accepted, for code written against `mgcv_rust`, and ignored with a
warning. Use the environment variables above to control threads.
