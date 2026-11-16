"""Shape-constrained (monotone) terms through the `Gam` surface: `ScTerm`
and `TeScTerm` next to ordinary terms, against mgcv's `scasm()`.

The fixtures (`scripts/r/gen_scasm_internals_fixtures.R`, synthetic data)
fit `s(log_cars, bs="sc", xt="m-") + s(age, bs="cr") + s(g, bs="re")` and
`te(log_cars, distance, bs="sc", xt=list("m-", "m+")) + s(age, bs="cr")`.
gamrs picks λ by the REML of the centred model at the constrained fit,
scasm by the REML of the model reduced to the binding constraints (and stops
its iteration at a 1% tolerance), so at the optimum the fits agree to a few
thousandths in log price rather than to rounding; `tests/parity_scasm.rs`
checks everything at scasm's own λ to ~1e-12.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamrs
from conftest import load_fixture

SC_CR_RE = "scasm_internals_sc_cr_re_n3000"
TESC_CR = "scasm_internals_tesc_cr_n3000"


def data(name):
    fx = load_fixture(name)
    return (
        fx,
        fx["inputs"]["columns"],
        np.asarray(fx["inputs"]["x_train"], dtype=float),
        np.asarray(fx["inputs"]["y_train"], dtype=float),
        np.asarray(fx["inputs"]["x_new"], dtype=float),
    )


def model(name):
    if name == SC_CR_RE:
        return [
            gamrs.ScTerm("log_cars", "decreasing", k=10),
            gamrs.CrTerm("age", k=8),
            gamrs.ReTerm("g"),
        ]
    return [
        gamrs.TeScTerm(("log_cars", "distance"), ("decreasing", "increasing"), k=(6, 5)),
        gamrs.CrTerm("age", k=8),
    ]


@pytest.mark.slow
@pytest.mark.parametrize("name", [SC_CR_RE, TESC_CR])
def test_mixed_model_matches_scasm(name):
    fx, cols, X, y, X_new = data(name)
    fin = fx["final"]
    g = gamrs.Gam(predictors=cols, terms=model(name)).fit(X, y)
    # Log-price units.
    assert np.max(np.abs(g.predict(X) - fin["fitted"])) < 5e-3
    terms = g.predict(X_new, type="terms").T
    assert np.max(np.abs(terms - np.asarray(fin["term_new"]))) < 5e-3
    assert g.edf_total_ == pytest.approx(fin["edf_total"], abs=0.5)
    assert g.scale_ == pytest.approx(fin["scale"], rel=1e-2)


@pytest.mark.slow
def test_sc_term_monotone_everywhere():
    _, cols, X, y, _ = data(SC_CR_RE)
    g = gamrs.Gam(predictors=cols, terms=model(SC_CR_RE)).fit(X, y)
    lo, hi = X[:, 0].min(), X[:, 0].max()
    span = hi - lo
    # A 1-D sc smooth stays monotone past the data: it continues linearly.
    grid = np.tile(np.median(X, axis=0), (400, 1))
    grid[:, 0] = np.linspace(lo - 0.5 * span, hi + 0.5 * span, 400)
    f = g.predict(grid, type="terms")[:, 0]
    assert np.all(np.diff(f) <= 1e-10)


@pytest.mark.slow
def test_tensor_monotone_within_data_range():
    _, cols, X, y, _ = data(TESC_CR)
    g = gamrs.Gam(predictors=cols, terms=model(TESC_CR)).fit(X, y)
    lo, hi = X.min(axis=0), X.max(axis=0)
    a = np.linspace(lo[0], hi[0], 60)
    b = np.linspace(lo[1], hi[1], 60)
    A, B = np.meshgrid(a, b, indexing="ij")
    grid = np.tile(np.median(X, axis=0), (A.size, 1))
    grid[:, 0], grid[:, 1] = A.ravel(), B.ravel()
    f = g.predict(grid, type="terms")[:, 0].reshape(A.shape)
    assert np.all(np.diff(f, axis=0) <= 1e-10)  # decreasing in log_cars
    assert np.all(np.diff(f, axis=1) >= -1e-10)  # increasing in distance


@pytest.mark.smoke
def test_sc_term_needs_gaussian():
    rng = np.random.default_rng(1)
    x = rng.uniform(size=(200, 1))
    y = (rng.uniform(size=200) < x[:, 0]).astype(float)
    with pytest.raises(ValueError, match="Gaussian"):
        gamrs.Gam(terms=[gamrs.ScTerm(0, "increasing")], family="binomial").fit(x, y)


@pytest.mark.smoke
def test_sc_term_rejects_unknown_shape():
    with pytest.raises(ValueError, match="increasing"):
        gamrs.ScTerm(0, "up")
    with pytest.raises(ValueError, match="increasing"):
        gamrs.TeScTerm((0, 1), ("up", None))


@pytest.mark.smoke
def test_sc_term_small_fit_is_monotone():
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 1, size=(300, 2))
    y = np.exp(-3 * x[:, 0]) + np.sin(6 * x[:, 1]) + rng.normal(scale=0.3, size=300)
    g = gamrs.Gam(
        terms=[gamrs.ScTerm(0, "decreasing", k=8), gamrs.CrTerm(1, k=8)]
    ).fit(x, y)
    grid = np.column_stack([np.linspace(0, 1, 200), np.full(200, 0.5)])
    f = g.predict(grid, type="terms")[:, 0]
    assert np.all(np.diff(f) <= 1e-10)
