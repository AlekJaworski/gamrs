"""A final inner fit that runs out of iterations is redone from where the search left off.

A t fit with many tiny random-effect groups converges slowly in PIRLS: on a 54-group
golf refit the deviance change shrank only 0.911x per iteration. The outer search never
noticed, because every probe starts from the last converged eta. The final fit started
from beta = 0, needed ~110 iterations, stopped at the cap of 50 and reported the whole
fit unconverged, with fitted values $79 off mgcv's here. A final fit that converges from
beta = 0 is returned exactly as before.

The data is a synthetic stand-in of the same shape: 3,000 rows, 50 groups of 1-3 rows
under by=on, groups spread $60k, t(3.7) noise of $20k. mgcv 1.9.4,
`gam(y ~ on + s(group, by = on, bs = "re"), family = scat(), method = "REML")`:
nu 3.535077, sigma 20217.6639.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import gamrs


def golf_shaped(seed=1):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 4, size=50)
    n = 3000
    group = np.zeros(n)
    group[: sizes.sum()] = np.repeat(np.arange(1, 51), sizes)
    on = (group > 0).astype(float)
    effects = np.r_[0.0, rng.normal(0, 60_000, 50)]
    y = 400_000 + 30_000 * on + effects[group.astype(int)] + 20_000 * rng.standard_t(3.7, n)
    return np.column_stack([on, group]), y


@pytest.mark.parity
def test_a_t_fit_with_sparse_groups_reports_its_convergence():
    X, y = golf_shaped()
    gam = gamrs.Gam(
        predictors=["on", "group"],
        predictor_basis_map={"on": "parametric", "group": "re"},
        term_by_mapping={"group": "on"},
        family="t-dist",
        method="REML",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gam.fit(X, y)
    assert gam.converged_
    nu = 3 + np.exp(gam.shape_params_[1])
    sigma = np.exp(gam.shape_params_[0] / 2)
    assert nu == pytest.approx(3.535077, rel=1e-4)
    assert sigma == pytest.approx(20217.6639, rel=1e-4)
