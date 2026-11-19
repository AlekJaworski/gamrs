"""mgcv's `s(x, pc=v)` through the `Gam` surface — `term_pc_mapping=` on the
predictors path and `CrTerm(pc=)` on the typed-term path.

`term_pc_mapping` was accepted and dropped on the floor until 0.14.1: two fits
differing only in it returned bit-identical predictions AND an identical
`edf_total_`. Neither is evidence of a working pc — the constraint is a
re-parameterisation of the same model space, so both are *supposed* to be
unchanged. The observable claims are that the constrained term's partial
effect is zero at `pc` and that the intercept moved to pay for it, which is
what a caller reading `gam[["__constant__", term]]` sees.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamrs

pytestmark = pytest.mark.smoke

PREDICTORS = ["gla", "concessions"]


def sample(n: int = 400, seed: int = 1):
    rng = np.random.default_rng(seed)
    conc = np.abs(rng.normal(0, 8000, n))
    gla = rng.uniform(1000, 3000, n)
    y = 120.0 * gla + 0.7 * conc + rng.normal(0, 15000, n)
    return np.column_stack([gla, conc]), y


def fit(X, y, pc, **kwargs):
    g = gamrs.Gam(
        predictors=PREDICTORS,
        family="gaussian",
        method="REML",
        k_default=6,
        min_k=2,
        term_k_mapping={"concessions": 3},
        term_pc_mapping=pc,
        **kwargs,
    )
    return g.fit(X, y)


def term_at(gam, x_gla, x_conc, term_index=1):
    """Partial effect of one term at a single covariate value."""
    row = np.array([[x_gla, x_conc]])
    return float(gam.predict(row, type="terms")[0, term_index])


def test_pc_zeroes_the_partial_effect_and_pays_for_it_from_the_intercept():
    X, y = sample()
    plain = fit(X, y, None)
    pinned = fit(X, y, {"concessions": 0.0})

    # Invariant: same model space, so the same fit. Not bit-identical —
    # the constraint row changes the inner system's conditioning, so the
    # outer Newton lands a few 1e-9 away — but far inside any parity bar
    # (mgcv's own pc-versus-plain fitted values move by ~1e-6 relative).
    mu_plain, mu_pinned = plain.predict(X), pinned.predict(X)
    assert np.max(np.abs(mu_plain - mu_pinned)) / np.max(np.abs(mu_plain)) < 1e-7
    assert plain.edf_total_ == pytest.approx(pinned.edf_total_, rel=1e-6)

    gla_mean = float(X[:, 0].mean())
    scale = abs(term_at(plain, gla_mean, float(X[:, 1].max())))
    at_pc_plain = term_at(plain, gla_mean, 0.0)
    assert abs(at_pc_plain) > 0.05 * scale, "vacuous: plain smooth already ~0 at pc"

    # Moved: the smooth is zero at pc, and the intercept took the difference.
    assert term_at(pinned, gla_mean, 0.0) == pytest.approx(0.0, abs=1e-9 * scale)
    intercept_shift = float(pinned.coef_[0]) - float(plain.coef_[0])
    assert intercept_shift == pytest.approx(at_pc_plain, rel=1e-6)

    # The shape a partial-curve consumer reads: intercept + the one term.
    view = pinned[["__constant__", "concessions"]]
    at_pc = float(view.predict(np.array([[gla_mean, 0.0]]), scale="link")[0])
    assert at_pc == pytest.approx(float(pinned.coef_[0]), rel=1e-9)


def test_pc_survives_auto_k_and_says_so_when_a_term_collapses_to_linear():
    X, y = sample()
    # auto_k rebuilds every CR term between refits; pc has to ride along.
    grown = fit(X, y, {"concessions": 0.0}, auto_k=True, max_k_auto=8)
    gla_mean = float(X[:, 0].mean())
    assert len(grown.auto_k_trace_) >= 1
    assert term_at(grown, gla_mean, 0.0) == pytest.approx(0.0, abs=1e-6)

    # A pc on a predictor with too few distinct values to carry a smooth:
    # the term becomes parametric, which has no centering constraint to
    # replace, so the pc is refused out loud rather than dropped quietly.
    X_flag = np.column_stack([X[:, 0], (X[:, 1] > 5000).astype(float)])
    with pytest.warns(UserWarning, match="is NOT applied"):
        flagged = fit(X_flag, y, {"concessions": 0.0})
    assert np.isfinite(flagged.predict(X_flag)).all()


def test_pc_on_the_typed_term_path_and_the_mapping_that_never_applies():
    X, y = sample()
    pinned = gamrs.Gam(
        predictors=PREDICTORS,
        terms=[gamrs.CrTerm("gla", k=6), gamrs.CrTerm("concessions", k=3, pc=0.0)],
    ).fit(X, y)
    gla_mean = float(X[:, 0].mean())
    assert term_at(pinned, gla_mean, 0.0) == pytest.approx(0.0, abs=1e-6)

    # term_pc_mapping is a predictors=-path knob; on the terms= path the pc
    # belongs on the term, so the mapping would silently do nothing.
    with pytest.warns(UserWarning, match="term_pc_mapping.*is ignored"):
        gamrs.Gam(
            predictors=PREDICTORS,
            terms=[gamrs.CrTerm("gla", k=6), gamrs.CrTerm("concessions", k=3)],
            term_pc_mapping={"concessions": 0.0},
        ).fit(X, y)

    # A key that names no fitted predictor is the same silent no-op.
    with pytest.warns(UserWarning, match="match no fitted predictor"):
        fit(X, y, {"concesions": 0.0})


MIXED = ["gla", "concessions", "flag"]


def mixed_sample(n: int = 400, seed: int = 2):
    """A model with genuinely both kinds of term: two CR smooths and a 0/1 indicator.

    The indicator is a real parametric term, not a smooth that collapsed into one for want of
    distinct values. That distinction is the point — a collapsed term is an accident of the data and
    exercises the refusal path, while a DECLARED parametric term sits in the fit alongside the
    constrained smooth and must come out of it untouched.
    """
    rng = np.random.default_rng(seed)
    gla = rng.uniform(1000, 3000, n)
    conc = np.abs(rng.normal(0, 8000, n))
    flag = (rng.random(n) > 0.5).astype(float)
    y = 120.0 * gla + 0.7 * conc + 25000.0 * flag + rng.normal(0, 15000, n)
    return np.column_stack([gla, conc, flag]), y


def mixed_fit(X, y, pc, **kwargs):
    return gamrs.Gam(
        predictors=MIXED,
        family="gaussian",
        method="REML",
        k_default=6,
        min_k=2,
        term_k_mapping={"concessions": 3},
        predictor_basis_map={"flag": "parametric"},
        term_pc_mapping=pc,
        **kwargs,
    ).fit(X, y)


def coef_of(gam, name):
    """One term's coefficient block, by name rather than by position."""
    first, last = next((f, l) for n, f, l in gam.get_term_indices() if n == name)
    return np.asarray(gam.coef_)[first : last + 1]


def test_pc_on_a_smooth_leaves_the_parametric_term_alone():
    """The constraint must be absorbed by the smooth's own null-space and the intercept, and reach
    nothing else. A pc that leaked into the parametric block would still zero the smooth at the
    point and still leave fitted values unchanged — so those two assertions alone cannot catch it,
    and the parametric coefficient is what has to be looked at directly.
    """
    X, y = mixed_sample()
    plain = mixed_fit(X, y, None)
    pinned = mixed_fit(X, y, {"concessions": 0.0})

    assert list(plain.bs_) == ["cr", "cr", "parametric"], "the fixture must actually be mixed"

    # Same model space: the fit does not move. (Not bit-identical — each parametrisation starts the
    # smoothing search from mgcv's initial.sp on its own columns, so they stop within the outer
    # tolerance of the shared optimum, as mgcv's own pc-versus-plain fits do.)
    mu_plain, mu_pinned = plain.predict(X), pinned.predict(X)
    assert np.max(np.abs(mu_plain - mu_pinned)) / np.max(np.abs(mu_plain)) < 1e-6
    # Both smooths are straight lines, their λ out on a flat ridge, so edf agrees to the ridge's slack.
    assert plain.edf_total_ == pytest.approx(pinned.edf_total_, abs=1e-3)

    # The parametric slope is the claim. It carries the indicator's whole 25k effect, so a leak
    # would be loud — and it must not move at all.
    flag_plain, flag_pinned = coef_of(plain, "flag"), coef_of(pinned, "flag")
    assert flag_plain.shape == (1,), "a parametric term is one coefficient"
    assert float(flag_pinned[0]) == pytest.approx(float(flag_plain[0]), rel=1e-6)
    assert abs(float(flag_plain[0])) > 1000.0, "vacuous: the indicator carries no effect to preserve"

    # The other smooth is untouched too: the constraint is one term's business.
    assert coef_of(pinned, "gla") == pytest.approx(coef_of(plain, "gla"), rel=1e-6)

    # And the constrained smooth still does what a pc means, with a parametric term in the model.
    gla_mean, conc_max = float(X[:, 0].mean()), float(X[:, 1].max())
    row = lambda c: np.array([[gla_mean, c, 1.0]])  # noqa: E731
    at = lambda g, c: float(g.predict(row(c), type="terms")[0, 1])  # noqa: E731
    scale = abs(at(plain, conc_max))
    at_pc_plain = at(plain, 0.0)
    assert abs(at_pc_plain) > 0.05 * scale, "vacuous: plain smooth already ~0 at pc"
    assert at(pinned, 0.0) == pytest.approx(0.0, abs=1e-9 * scale)
    shift = float(pinned.coef_[0]) - float(plain.coef_[0])
    assert shift == pytest.approx(at_pc_plain, abs=1e-6 * float(np.max(np.abs(mu_plain))))


def test_pc_on_a_declared_parametric_term_is_refused_on_both_paths():
    """A parametric term has no centering constraint to replace, so there is nothing for a pc to do
    to it — mgcv has no pc for parametric terms either. Refused out loud on the
    `predictor_basis_map` path and on the typed-term path, and the rest of the fit still stands.
    """
    X, y = mixed_sample()
    with pytest.warns(UserWarning, match="is NOT applied"):
        refused = mixed_fit(X, y, {"flag": 0.0})
    assert np.isfinite(refused.predict(X)).all()
    assert abs(float(coef_of(refused, "flag")[0])) > 1000.0, "the term still fits, it just has no pc"

    # ParametricTerm has no `pc` field at all, so the typed path cannot express it — the mapping is
    # the only way to ask, and on the terms= path the mapping itself is the thing that is ignored.
    assert not hasattr(gamrs.ParametricTerm("flag"), "pc")
    with pytest.warns(UserWarning, match="term_pc_mapping.*is ignored"):
        gamrs.Gam(
            predictors=MIXED,
            terms=[
                gamrs.CrTerm("gla", k=6),
                gamrs.CrTerm("concessions", k=3, pc=0.0),
                gamrs.ParametricTerm("flag"),
            ],
            term_pc_mapping={"flag": 0.0},
        ).fit(X, y)
