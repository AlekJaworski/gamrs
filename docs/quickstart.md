# gamrs quickstart

A walkthrough of the common fits, in order of how often you'll reach for them.
Every snippet runs against the public `gamrs` Python API.

## Install

```bash
pip install gamrs            # base wheel
pip install gamrs[quantile]  # + scipy, for SHASH-calibrated quantile fits
```

The base wheel is enough for everything in this doc except the
[quantile section](#quantile-regression-elf).

---

## 1. Single 1-D smooth

The default is a cubic regression spline (`bs="cr"`) with `k=10` knots.

```python
import numpy as np
from gamrs import Gam

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 500)
y = np.sin(x) + rng.normal(0, 0.3, 500)

g = Gam(family="gaussian").fit(x, y)

print(g.coef_.shape)        # (k,) — basis coefficients
print(g.lambda_)            # smoothing parameter(s)
print(g.edf_total_)         # effective degrees of freedom
print(g.converged_)         # outer-loop convergence flag

mu = g.predict(x)
```

`fit(X, y)` accepts a 1-D ndarray, a 2-D ndarray, or a DataFrame for `X`,
and a 1-D ndarray / Series for `y`.

## 2. Multi-smooth additive

`y ~ s(x0) + s(x1) + …`. Use the typed-term API — each `CrTerm` is one
1-D smooth with its own `k` and smoothing parameter. Column names can
be passed as strings; they're resolved against the DataFrame at fit time.

```python
import pandas as pd
from gamrs import Gam, CrTerm

df = pd.DataFrame({
    "x0": rng.uniform(0, 10, 500),
    "x1": rng.uniform(-5, 5, 500),
})
df["y"] = np.sin(df.x0) + 0.3 * df.x1**2 + rng.normal(0, 0.3, 500)

X = df[["x0", "x1"]]    # predictors only — y must NOT be a column of X
g = Gam(terms=[CrTerm("x0", k=10), CrTerm("x1", k=15)]).fit(X, df.y)

g.edf_                  # per-term effective DoF
g.lambda_               # per-term smoothing parameter
mu = g.predict(X)
```

Integer column indices (`CrTerm(0, k=10)`) are also accepted — useful
when fitting from raw `numpy` arrays.

## 3. Tensor products

For interactions across covariates with different scales. `TeTerm` is the
anisotropic 2-margin tensor product (one λ per margin); `TeMultiTerm`
generalises to N margins; `TiTerm` is the pure interaction (main effects
removed).

```python
from gamrs import TeTerm, TeMultiTerm, TiTerm

# Two-way interaction with separate smoothing per margin
g = Gam(terms=[TeTerm(cols=("x0", "x1"), k=(8, 8))]).fit(X, df.y)

# Three-way tensor product
df["x2"] = rng.uniform(0, 1, 500)
X3 = df[["x0", "x1", "x2"]]
g = Gam(terms=[TeMultiTerm(cols=("x0", "x1", "x2"), k=(5, 5, 5))]).fit(X3, df.y)

# Pure interaction — useful with explicit main effects
g = Gam(terms=[
    CrTerm("x0", k=10),
    CrTerm("x1", k=10),
    TiTerm(cols=("x0", "x1"), k=(5, 5)),
]).fit(X, df.y)
```

## 4. Thin-plate splines (2-D smooths)

`s(x0, x1, bs="tp")` — use when the two predictors share a scale (lat/lon,
xy coordinates).

```python
from gamrs import TpsTerm

g = Gam(terms=[TpsTerm(cols=("lat", "lon"), k=30)]).fit(
    df_geo[["lat", "lon"]], df_geo.elev
)
```

## 5. Random effects

`bs="re"` — drops a Gaussian random intercept per level.

```python
from gamrs import ReTerm

df["group"] = rng.integers(0, 8, len(df)).astype(float)

X_re = df[["x0", "group"]]
g = Gam(terms=[CrTerm("x0", k=10), ReTerm("group")]).fit(X_re, df.y)
```

## 5b. Parametric (linear, unsmoothed) terms

For columns you want as a raw linear effect — 0/1 indicators, integer
counts, anything you don't want smoothed. The coefficient retains its
full degree of freedom (no penalty); equivalent to mgcv R's "pterms".

```python
from gamrs import ParametricTerm

df["is_promo"] = rng.integers(0, 2, len(df)).astype(float)

# Mix smooths and parametric terms in the same fit
g = Gam(terms=[
    CrTerm("x0", k=10),
    CrTerm("x1", k=10),
    ParametricTerm("is_promo"),
]).fit(df[["x0", "x1", "is_promo"]], df.y)

# The parametric coefficient is interpretable as the slope:
print(g.coef_[-1])   # ≈ true effect of is_promo
print(len(g.lambda_))  # = number of SMOOTHS (no λ for parametric terms)
```

The `predictor_basis_map={"is_promo": "parametric"}` (alias
`"linear"`) flag spells the same thing without the typed-term API:

```python
g = Gam(
    family="gaussian",
    predictor_basis_map={"is_promo": "parametric"},
).fit(df[["x0", "is_promo"]], df.y)
```

## 6. Picking a family

| Data shape                                | Family            | Constructor                                      |
| ----------------------------------------- | ----------------- | ------------------------------------------------ |
| Continuous, ~normal residuals             | `gaussian`        | `Gam(family="gaussian")`                         |
| Binary 0/1                                | `bernoulli`       | `Gam(family="bernoulli")`                        |
| Counts, mean ≈ variance                   | `poisson`         | `Gam(family="poisson")`                          |
| Counts, mean ≪ variance (overdispersed)   | `negbin` (`nb`)   | `Gam(family="negbin")`                           |
| Strictly positive continuous              | `gamma`           | `Gam(family="gamma")`                            |
| Strictly positive, skewed                 | `inverse_gaussian`| `Gam(family="inverse_gaussian")`                 |
| Compound Poisson-Gamma (claims, rainfall) | `tweedie`         | `Gam(family="tweedie")` or `tweedie_p=1.5`       |
| Heavy-tailed continuous                   | `t-dist`          | `Gam(family="t-dist", df=4)`                     |
| Ordered categorical                       | `ocat`            | `Gam(family="ocat", r=K)`                        |
| Quantile (any τ)                          | `quantile` (ELF)  | use [`fit_quantile`](#quantile-regression-elf)   |

QuasiPoisson / QuasiBinomial are also available (`family="quasipoisson"` /
`"quasibinomial"`) for over- or under-dispersed counts/binary data when
you'd rather profile φ than commit to NegBin.

## 7. Confidence intervals & differences

`predict_ci` returns pointwise CIs from gamrs's cached `vcov` (closed-form
Wald, no sampling needed).

```python
mu, lo, hi = g.predict_ci(X, level=0.95, scale="response")
mu, lo, hi = g.predict_ci(X, level=0.95, scale="link")
```

`predict_diff` returns the η-scale difference between two design rows (or
broadcast a single baseline against many candidates), with CIs:

```python
# Δη between two rows
diff = g.predict_diff(X.iloc[[0]], X.iloc[[1]])

# With 95% CI (broadcast a single baseline against many rows)
diff, lo, hi = g.predict_diff(X.iloc[[0]], X, level=0.95, broadcast="from")
```

`predict_diff` is identity-link only — for non-identity links, sample
posteriors and difference per draw.

## 8. Subset views — isolate one smooth's contribution

`gam[["x0"]]` returns a *subset view* that masks all other terms to zero
when predicting. Use it to plot a single smooth's marginal effect or to
get a CI on just its contribution.

```python
g = Gam(terms=[CrTerm("x0", k=10), CrTerm("x1", k=15)]).fit(X, df.y)

g_x0 = g[["x0"]]                              # subset view
eta_x0 = g_x0.predict(X, scale="deviation")    # x0's η-contribution, no intercept
mu_x0, lo, hi = g_x0.predict_ci(X, level=0.95, scale="deviation")

# Include the intercept (η-scale fit with only x0's smooth and the constant)
g_x0_with_int = g[["x0", Gam.INTERCEPT]]
mu = g_x0_with_int.predict(X, scale="link")
```

A subset view supports `scale="link"` and `scale="deviation"`.
`scale="response"` only makes sense on the full model (inv-linking a
single smooth's η-component is not a meaningful prediction).

`partial_effect("x0", grid_n=100, level=0.95)` is a convenience wrapper:
it builds a grid over the training range of `x0`, holds every other
predictor at its training median, and returns a DataFrame
`{x, mean, lo, hi}` ready to plot.

```python
peff = g.partial_effect("x0", grid_n=100, level=0.95)
peff.plot(x="x", y="mean")  # or hand to matplotlib / seaborn / altair
```

## 9. Large-n fits — switch to `method="fREML"`

`gamrs`'s default is REML via damped Newton; for GLM families at large n,
`method="fREML"` switches to Fellner–Schall λ updates, as in mgcv's
`gam(optimizer = "efs")`, still on the REML criterion: results match
`gam(method = "REML")`.

```python
g = Gam(family="poisson", method="fREML").fit(X_big, y_big)
```

Pick `method="fREML"` when `n >= 50_000` and the family is binomial, Poisson,
Gamma, inverse Gaussian or quasi (see [`docs/perf.md`](perf.md)). Other
families (Gaussian, NegBin, Tweedie, scat, …) have their own optimisers; there
`method="fREML"` warns and fits by REML.

## 10. Quantile regression (ELF)

The ELF likelihood approximates an asymmetric Laplace at width σ. `gamrs`
ships qgam-style σ-calibration via K-fold pinball CV.

```python
from gamrs import fit_quantile

g_50 = fit_quantile(X, y, tau=0.5)   # median
g_95 = fit_quantile(X, y, tau=0.95)  # 95th percentile

print(g_95.sigma_)        # the σ that won CV
print(g_95.tune_info_)    # loss curve
mu_95 = g_95.predict(X)
```

For τ in the extreme tail (≳0.95), the `"fast_oos"` preset uses a SHASH
err-param heuristic (one fit instead of CV) — needs the `quantile` extra:

```python
g = fit_quantile(X, y, tau=0.99, preset="fast_oos")
```

## 11. Diagnostics

```python
g.coef_              # basis coefficients
g.vcov_              # full β covariance
g.lambda_            # smoothing parameter(s)
g.edf_               # per-term effective DoF
g.edf_total_         # total EDF
g.reml_value_        # REML score at convergence
g.converged_         # outer-loop convergence flag
g.n_iters_           # outer iterations
g.fit_stats_         # detailed counters: PIRLS calls, line-search trials, …
```

`fit_stats_` is useful when investigating slow fits — it exposes the inner
PIRLS / line-search counters that the Rust solver maintains.

## 12. Serialize & deploy

Two on-the-wire formats. Both carry the complete fitted state — β, vcov,
knots, centring, reparam — so they're round-trip equivalent for
prediction; pick by what you need at the receiving end.

```python
# Compact binary (default; production-friendly)
blob = g.serialize()                    # bytes, ~3-5× smaller than JSON
restored = Gam.deserialize(blob)        # back to a fitted Gam

# Human-debuggable JSON
text = g.to_json()                      # str, valid JSON
restored = Gam.from_json(text)

# Pickle works transparently via __reduce__
import pickle
blob = pickle.dumps(g)
restored = pickle.loads(blob)

# Inference-only deployment — strip training data
from gamrs import GamPredictor
predictor = GamPredictor.from_gam(g)
mu = predictor.predict(X_new)
```

The binary form is length-framed (`MAGIC | VERSION | LEN | bincode body`)
so a corrupt or wrong-format input fails fast with a clear error. The
JSON form is unframed — load it with `Gam.from_json(text)`, not
`Gam.deserialize(text.encode())`.

### What's actually serialized (and why it's not just the lp matrix)

A common question: "couldn't you just serialize the lp matrix and β?"
The answer is no — what we need to serialize is the *recipe* for the lp
matrix at any future X, not the lp matrix at the training X. Concretely
that's:

- the basis state (knot locations, centring constraints, reparam
  rotation matrices)
- the smoothing parameters (`λ`)
- β and vcov

`evaluate_lpmatrix(X_new)` is then deterministic at predict time. The
saved bundle is small (~kB for a typical fit) because the design matrix
isn't there — it's rebuilt on demand.

## Where next

- **[docs/perf.md](perf.md)** — REML vs fREML, large-n tips, reading `fit_stats_`.
- **tests/python/** — parity tests against `mgcv` R fixtures, also serve as
  worked examples for every family.
