# Performance

## Choosing `method`

| `method=` | Smoothing-parameter search | Use it for |
| --- | --- | --- |
| `"REML"` (default) | Damped Newton on the REML score, inner fit run to convergence | Everything; the most accurate |
| `"fREML"` | Fellner–Schall updates (as in `bam()`), one inner step per update | Large-n binomial, Poisson, Gamma, inverse Gaussian and quasi fits |

Both search the same REML criterion. On a Poisson fit with four smooths and
n = 200,000, `fREML` took 1.6 s against 10.6 s for `REML`. The linear
predictors agreed to 0.002, but `fREML` stopped earlier on terms that are close
to straight lines (edf 1.5 rather than 1.0).

Other families (Gaussian, `nb`, `tw`, `scat`, `ocat`, quantile and
location–scale fits) have their own solvers. Passing `method="fREML"` there
fits by REML and issues a warning.

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
