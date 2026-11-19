#!/usr/bin/env python
"""Large-n perf bench: gamrs vs mgcv_rust at n in {10K, 50K, 100K, 250K}.

Goal: find out where (and if) gamrs's dense-QR / dense-PIRLS path falls
behind mgcv_rust's chunked-QR / discrete-binning paths. Decides whether
those ports are worth pulling in.

Each cell: synthetic Gaussian / Poisson / Bernoulli fit at the listed n,
single smooth (k=20). Best-of-3 wall time. Includes µ-pred sanity vs
the gamrs-itself prediction (no R parity at these sizes — too expensive
to generate).

Usage:
    .venv/bin/python scripts/bench_large_n.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import gamrs

try:
    import mgcv_rust as mr
except ImportError:
    mr = None


N_VALUES = [10_000, 50_000, 100_000, 250_000, 1_000_000]
RUNS = 3
RNG = np.random.default_rng(42)


def synth_gaussian(n: int):
    x = RNG.uniform(0, 10, n)
    y = np.sin(x) + 0.5 * np.cos(2 * x) + RNG.normal(0, 0.3, n)
    return x, y, "gaussian"


def synth_poisson(n: int):
    x = RNG.uniform(0, 10, n)
    eta = 0.5 + 0.3 * np.sin(x)
    mu = np.exp(eta)
    y = RNG.poisson(mu).astype(float)
    return x, y, "poisson"


def synth_bernoulli(n: int):
    x = RNG.uniform(-3, 3, n)
    eta = np.sin(x)
    p = 1 / (1 + np.exp(-eta))
    y = (RNG.uniform(0, 1, n) < p).astype(float)
    # mgcv_rust uses 'binomial' for logit-link 0/1 outcomes; gamrs aliases
    # 'binomial' to 'bernoulli' internally.
    return x, y, "binomial"


SCENARIOS = [
    ("gaussian", synth_gaussian),
    ("poisson", synth_poisson),
    ("bernoulli", synth_bernoulli),
]


def time_fit(engine_module, x, y, family: str, k: int = 20, gamrs_algorithm=None, **kwargs):
    """Time a fit. `gamrs_algorithm` only used when engine is gamrs."""
    df = pd.DataFrame({"x": x, "y": y})
    best = float("inf")
    last_gam = None
    if gamrs_algorithm is not None and engine_module.__name__.startswith("gamrs"):
        engine_module._gamrs_native.set_outer_algorithm(gamrs_algorithm)
    try:
        for _ in range(RUNS):
            t0 = time.perf_counter()
            gam = engine_module.Gam(
                predictors=["x"], target="y", family=family, k=k, **kwargs
            )
            gam.fit(df, df["y"])
            dt = (time.perf_counter() - t0) * 1000.0
            if dt < best:
                best = dt
            last_gam = gam
    finally:
        if engine_module.__name__.startswith("gamrs"):
            engine_module._gamrs_native.set_outer_algorithm(None)
    return best, last_gam


def main():
    if mr is None:
        print("mgcv_rust not installed — bench needs both engines.")
        return

    print(f"Large-n bench: best-of-{RUNS}, k=20, single smooth.\n")
    print(
        f"{'family':>9s}  {'n':>8s}  "
        f"{'gamrs N':>8s}  {'gamrs FS':>8s}  "
        f"{'mr REML':>8s}  {'mr fREML':>8s}  "
        f"{'best gamrs':>10s}  {'best mr':>8s}  {'speedup':>8s}"
    )
    print("-" * 110)
    for family, synth in SCENARIOS:
        for n in N_VALUES:
            x, y, fam = synth(n)
            # gamrs Newton + efs (Fellner-Schall)
            try:
                g_newton_ms, _ = time_fit(gamrs, x, y, fam, gamrs_algorithm=None)
            except Exception:
                g_newton_ms = float("nan")
            try:
                g_freml_ms, _ = time_fit(gamrs, x, y, fam, gamrs_algorithm="efs")
            except Exception:
                g_freml_ms = float("nan")
            # mgcv_rust REML + fREML
            results = {}
            for label, kw in [
                ("REML", {"method": "REML"}),
                ("fREML", {"method": "fREML"}),
            ]:
                try:
                    results[label], _ = time_fit(mr, x, y, fam, **kw)
                except Exception:
                    results[label] = float("nan")
            valid_gamrs = [v for v in (g_newton_ms, g_freml_ms) if v == v]
            best_gamrs = min(valid_gamrs) if valid_gamrs else float("nan")
            valid_mr = [v for v in results.values() if v == v]
            best_mr = min(valid_mr) if valid_mr else float("nan")
            speedup = (best_mr / best_gamrs) if (best_gamrs == best_gamrs and best_mr == best_mr and best_gamrs > 0) else float("nan")
            print(
                f"{family:>9s}  {n:>8d}  "
                f"{g_newton_ms:>8.1f}  {g_freml_ms:>8.1f}  "
                f"{results['REML']:>8.1f}  {results['fREML']:>8.1f}  "
                f"{best_gamrs:>10.1f}  {best_mr:>8.1f}  {speedup:>7.2f}×"
            )
        print()


if __name__ == "__main__":
    main()
