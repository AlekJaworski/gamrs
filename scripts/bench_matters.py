#!/usr/bin/env python
"""Performance bench focused on the regime where wall-time matters.

Designed around three observations:

1. **Sub-20ms fits are below the noise floor** — fractional differences
   in tiny fits don't translate into user-visible speedups. We flag
   them `[sub-noise]` and exclude them from aggregate claims.
2. **Cold caches lie** — every cell gets 3 warmup iters before the
   7 measurement iters. Wall time is `median(7)` (with `best` shown
   for context, but speedups quote medians).
3. **Coverage that maps to README claims** — three small focused
   sweeps that vary one dimension at a time, plus a shape-aware
   matrix that exercises the new convergence stack.

Sweeps:
- **n-sweep** (Gaussian / Poisson / Bernoulli, k=20, single smooth):
  n ∈ {10K, 100K, 1M}. Same structure as the README headline.
- **d-sweep** (Gaussian, k=10, n=5K): d ∈ {1, 2, 5, 10}. Multi-smooth
  additive scaling.
- **k-sweep** (Gaussian, n=2K, single smooth): k ∈ {10, 20, 50}.
- **Shape-aware** (n=2K, k=10): NegBin / Tweedie / ocat / scat.
  Tests the families that exercise the joint outer Newton stack.

Output: markdown table to stdout (paste into release notes / README)
plus a JSON dump alongside the script for downstream consumption.

Usage:
    .venv/bin/python scripts/bench_matters.py
    .venv/bin/python scripts/bench_matters.py --json bench_out.json

Target runtime: under 5 minutes on a laptop.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np

import gamrs

try:
    import mgcv_rust as mr
except ImportError:
    print("mgcv_rust not available — bench compares gamrs to itself only.")
    mr = None  # type: ignore[assignment]

RNG = np.random.default_rng(20260603)

NOISE_FLOOR_MS = 20.0  # rows below this are flagged [sub-noise]
WARMUP = 3
MEASURE = 7


# =============================================================================
# Synthetic data
# =============================================================================


def synth_gaussian(n: int, d: int = 1):
    """y = Σ_j sin(j·x_j) + N(0, 0.3)."""
    X = RNG.uniform(0, 10, (n, d))
    eta = sum(np.sin((j + 1) * X[:, j]) / (j + 1) for j in range(d))
    y = eta + RNG.normal(0, 0.3, n)
    return X, y


def synth_poisson(n: int, d: int = 1):
    X = RNG.uniform(0, 10, (n, d))
    eta = 0.5 + 0.3 * np.sin(X[:, 0])
    mu = np.exp(eta)
    y = RNG.poisson(mu).astype(float)
    return X, y


def synth_bernoulli(n: int, d: int = 1):
    X = RNG.uniform(0, 10, (n, d))
    eta = 0.5 * np.sin(X[:, 0])
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (RNG.random(n) < p).astype(float)
    return X, y


def synth_negbin(n: int, d: int = 1):
    X = RNG.uniform(0, 10, (n, d))
    mu = np.exp(0.5 * np.sin(X[:, 0]))
    y = RNG.negative_binomial(2.0, 2.0 / (2.0 + mu)).astype(float)
    return X, y


def synth_tweedie(n: int, d: int = 1):
    X = RNG.uniform(0, 10, (n, d))
    mu = np.exp(0.5 + 0.3 * np.sin(X[:, 0]))
    # Compound Poisson-Gamma at p=1.5
    n_events = RNG.poisson(mu)
    y = np.zeros(n)
    for i in range(n):
        if n_events[i] > 0:
            y[i] = RNG.gamma(n_events[i], 1.0)
    return X, y


def synth_ocat(n: int, d: int = 1, n_cats: int = 4):
    X = RNG.uniform(0, 10, (n, d))
    eta = np.sin(X[:, 0])
    if d >= 2:
        eta = eta + 0.5 * np.sin(X[:, 1] * 0.5)
    qs = np.quantile(eta, np.linspace(0, 1, n_cats + 1)[1:-1])
    y = (np.digitize(eta, qs) + 1).astype(float)
    return X, y


def synth_scat(n: int, d: int = 1, df: float = 5.0):
    X = RNG.uniform(0, 10, (n, d))
    eta = np.sin(X[:, 0])
    y = eta + RNG.standard_t(df, n) * 0.3
    return X, y


# =============================================================================
# Timing
# =============================================================================


def time_fit(builder: Callable[[], Any]) -> tuple[float, float]:
    """Return `(median_ms, best_ms)` over `MEASURE` runs after `WARMUP`."""
    for _ in range(WARMUP):
        builder()
    times = []
    for _ in range(MEASURE):
        gc.collect()
        t0 = time.perf_counter()
        builder()
        times.append((time.perf_counter() - t0) * 1000.0)
    return median(times), min(times)


# =============================================================================
# Cell runners
# =============================================================================


def cell_gamrs(family: str, X: np.ndarray, y: np.ndarray, *,
               method: str | None = None, k: int = 10, d: int = 1,
               **kw: Any) -> Callable[[], Any]:
    """Build a closure that runs one gamrs fit. d > 1 builds a
    multi-smooth additive via the typed-term API at the given k."""
    if d > 1:
        terms = [gamrs.CrTerm(i, k=k) for i in range(d)]
        kwargs = dict(family=family, terms=terms, **kw)
        if method is not None:
            kwargs["method"] = method
        return lambda: gamrs.Gam(**kwargs).fit(X, y)
    kwargs = dict(family=family, **kw)
    if method is not None:
        kwargs["method"] = method
    return lambda: gamrs.Gam(**kwargs).fit(X, y)


def cell_mr(family: str, X: np.ndarray, y: np.ndarray, *,
            method: str | None = None, k: int = 10, **kw: Any) -> Callable[[], Any]:
    """Build a closure that runs one mgcv_rust fit. mgcv_rust auto-
    detects multi-smooth from X.shape[1]. ``method`` selects REML
    (default) or fREML (mgcv R `bam()` equivalent — fast for GLMs at
    large n). For a fair speedup quote we report mgcv_rust at its
    best of REML / fREML."""
    if mr is None:
        return lambda: None
    # Map gamrs family aliases to mgcv_rust's expected names.
    mr_family = {"bernoulli": "binomial", "scat": "t-dist"}.get(family, family)
    kwargs = dict(family=mr_family, k_default=k, **kw)
    if method is not None:
        kwargs["method"] = method
    return lambda: mr.Gam(**kwargs).fit(X, y)


# =============================================================================
# Bench cells
# =============================================================================


def run_cell(label: str, family: str, X: np.ndarray, y: np.ndarray, *,
             k: int = 10, d: int = 1,
             gamrs_kw: dict[str, Any] | None = None,
             mr_kw: dict[str, Any] | None = None,
             mr_skip: bool = False) -> dict[str, Any]:
    """Run gamrs + mgcv_rust on one (n, d, k, family) cell and return
    the timing record."""
    gamrs_kw = gamrs_kw or {}
    mr_kw = mr_kw or {}
    n = X.shape[0]
    # gamrs REML
    g_med, g_best = time_fit(cell_gamrs(family, X, y, d=d, k=k, **gamrs_kw))
    # gamrs efs (Fellner-Schall; when applicable — single-smooth GLM families benefit most)
    try:
        g_fs_med, g_fs_best = time_fit(
            cell_gamrs(family, X, y, d=d, k=k, method="efs", **gamrs_kw)
        )
    except Exception:
        g_fs_med = g_fs_best = float("nan")
    # mgcv_rust REML (default).
    if mr is None or mr_skip:
        m_med = m_best = float("nan")
        m_fs_med = m_fs_best = float("nan")
    else:
        try:
            m_med, m_best = time_fit(cell_mr(family, X, y, k=k, **mr_kw))
        except Exception as e:
            print(f"  mgcv_rust REML failed on {label}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            m_med = m_best = float("nan")
        # mgcv_rust fREML is the bam() solver — only meaningful for
        # families bam() supports. For shape-aware families (negbin,
        # tweedie, scat, ocat, elf) mgcv_rust's method='fREML' accepts
        # the kwarg but returns essentially without fitting (verified
        # empirically: ocat 'fREML' returns in 8ms vs REML's 4500ms,
        # which is not fast — it's not fitting). Skip those.
        BAM_FAMILIES = {
            "gaussian", "binomial", "bernoulli", "poisson", "gamma",
            "inverse_gaussian", "inverse.gaussian", "quasipoisson",
            "quasibinomial",
        }
        if family in BAM_FAMILIES:
            try:
                m_fs_med, m_fs_best = time_fit(
                    cell_mr(family, X, y, k=k, method="fREML", **mr_kw)
                )
            except Exception:
                m_fs_med = m_fs_best = float("nan")
        else:
            m_fs_med = m_fs_best = float("nan")
    # "best gamrs" = whichever of REML / fREML is faster
    best_gamrs = g_med if not np.isfinite(g_fs_med) else min(g_med, g_fs_med)
    # "best mr" = whichever of REML / fREML is faster (apples-to-apples)
    best_mr_vals = [v for v in [m_med, m_fs_med] if np.isfinite(v)]
    best_mr = min(best_mr_vals) if best_mr_vals else float("nan")
    speedup = (best_mr / best_gamrs) if (np.isfinite(best_mr) and best_gamrs > 0) else float("nan")
    sub_noise = best_gamrs < NOISE_FLOOR_MS and (
        not np.isfinite(best_mr) or best_mr < NOISE_FLOOR_MS
    )
    return {
        "label": label,
        "family": family,
        "n": int(n),
        "d": int(d),
        "k": int(k),
        "gamrs_reml_ms": g_med,
        "gamrs_freml_ms": g_fs_med,
        "mr_reml_ms": m_med,
        "mr_freml_ms": m_fs_med,
        "best_gamrs_ms": best_gamrs,
        "best_mr_ms": best_mr,
        "speedup": speedup,
        "sub_noise": sub_noise,
    }


# =============================================================================
# Sweeps
# =============================================================================


def n_sweep() -> list[dict[str, Any]]:
    print("# n-sweep (d=1, k=20, families × n)", flush=True)
    rows: list[dict[str, Any]] = []
    for fam, synth in [("gaussian", synth_gaussian),
                       ("poisson", synth_poisson),
                       ("bernoulli", synth_bernoulli)]:
        for n in [10_000, 100_000, 1_000_000]:
            X, y = synth(n)
            label = f"{fam} n={n:,}"
            print(f"  • {label}", flush=True)
            rows.append(run_cell(label, fam, X, y, k=20, d=1))
    return rows


def d_sweep() -> list[dict[str, Any]]:
    print("# d-sweep (Gaussian, n=5000, k=10, d sweep)", flush=True)
    rows: list[dict[str, Any]] = []
    for d in [1, 2, 5, 10]:
        X, y = synth_gaussian(5000, d=d)
        label = f"gaussian d={d}"
        print(f"  • {label}", flush=True)
        rows.append(run_cell(label, "gaussian", X, y, k=10, d=d))
    return rows


def k_sweep() -> list[dict[str, Any]]:
    print("# k-sweep (Gaussian, n=2000, d=1, k sweep)", flush=True)
    rows: list[dict[str, Any]] = []
    for k in [10, 20, 50]:
        X, y = synth_gaussian(2000, d=1)
        label = f"gaussian k={k}"
        print(f"  • {label}", flush=True)
        rows.append(run_cell(label, "gaussian", X, y, k=k, d=1))
    return rows


def shape_aware_sweep() -> list[dict[str, Any]]:
    print("# shape-aware (k=10, n=2000)", flush=True)
    rows: list[dict[str, Any]] = []
    for fam_label, fam, synth, gkw, mkw, d in [
        ("negbin 1d", "negbin", synth_negbin, {"negbin_theta": 5.0},
         {"negbin_theta": 5.0}, 1),
        ("negbin 2d", "negbin", lambda n, d: synth_negbin(n, d=2),
         {"negbin_theta": 5.0}, {"negbin_theta": 5.0}, 2),
        ("tweedie 1d", "tweedie", synth_tweedie, {}, {}, 1),
        ("scat 1d", "scat", lambda n, d: synth_scat(n, d=1, df=5.0),
         {"df": 5.0}, {"df": 5.0}, 1),
        ("ocat 1d", "ocat", lambda n, d: synth_ocat(n, d=1, n_cats=4),
         {"r": 4}, {"r": 4}, 1),
        ("ocat 2d", "ocat", lambda n, d: synth_ocat(n, d=2, n_cats=4),
         {"r": 4}, {"r": 4}, 2),
    ]:
        # 2-d synth helpers ignore the `d` arg they receive (already baked in)
        X, y = synth(2000, d)
        label = fam_label
        print(f"  • {label}", flush=True)
        rows.append(
            run_cell(label, fam, X, y, k=10, d=d,
                     gamrs_kw=gkw, mr_kw=mkw)
        )
    return rows


# =============================================================================
# Output
# =============================================================================


def fmt_ms(v: float) -> str:
    if not np.isfinite(v):
        return "—"
    if v < 1.0:
        return f"{v:.2f}"
    if v < 100.0:
        return f"{v:.1f}"
    return f"{int(round(v))}"


def fmt_speedup(v: float) -> str:
    if not np.isfinite(v):
        return "—"
    return f"{v:.2f}×"


def print_section(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n## {title}\n")
    print(
        "| Fixture | n | d | k | gamrs REML | gamrs fREML "
        "| mr REML | mr fREML | best gamrs | best mr | speedup |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        sub = " [sub-noise]" if r["sub_noise"] else ""
        print(
            f"| {r['label']}{sub} | {r['n']:,} | {r['d']} | {r['k']} "
            f"| {fmt_ms(r['gamrs_reml_ms'])} "
            f"| {fmt_ms(r['gamrs_freml_ms'])} "
            f"| {fmt_ms(r['mr_reml_ms'])} "
            f"| {fmt_ms(r['mr_freml_ms'])} "
            f"| {fmt_ms(r['best_gamrs_ms'])} "
            f"| {fmt_ms(r['best_mr_ms'])} "
            f"| {fmt_speedup(r['speedup'])} |"
        )


def print_summary(all_rows: list[dict[str, Any]]) -> None:
    matters = [r for r in all_rows if not r["sub_noise"] and np.isfinite(r["speedup"])]
    if not matters:
        print("\n## Aggregate\n\nNo above-noise rows to aggregate.")
        return
    speedups = [r["speedup"] for r in matters]
    wins = sum(1 for s in speedups if s > 1.0)
    print(f"\n## Aggregate (rows above {NOISE_FLOOR_MS:.0f}ms only)\n")
    print(f"- **{wins}/{len(matters)} cells**: gamrs faster than mgcv_rust")
    print(f"- **Median speedup**: {median(speedups):.2f}×")
    print(f"- **Geometric mean**:  {np.exp(np.mean(np.log(speedups))):.2f}×")
    print(f"- **Range**: {min(speedups):.2f}× to {max(speedups):.2f}×")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None,
                        help="Write per-cell records to this JSON file")
    parser.add_argument("--skip", action="append", default=[],
                        choices=["n", "d", "k", "shape"],
                        help="Skip a sweep (one of: n, d, k, shape)")
    args = parser.parse_args()

    print(f"# gamrs perf bench (best-of-{MEASURE}, {WARMUP} warmup, "
          f"median quoted, [sub-noise] = below {NOISE_FLOOR_MS:.0f}ms)\n")
    print(f"# gamrs={gamrs.__version__ if hasattr(gamrs, '__version__') else '(unknown)'}, "
          f"mgcv_rust={mr.__version__ if (mr is not None and hasattr(mr, '__version__')) else '(unknown)'}\n")

    all_rows: list[dict[str, Any]] = []
    if "n" not in args.skip:
        rows = n_sweep()
        print_section("n-sweep — scaling vs sample size", rows)
        all_rows.extend(rows)
    if "d" not in args.skip:
        rows = d_sweep()
        print_section("d-sweep — multi-smooth scaling", rows)
        all_rows.extend(rows)
    if "k" not in args.skip:
        rows = k_sweep()
        print_section("k-sweep — basis dimension scaling", rows)
        all_rows.extend(rows)
    if "shape" not in args.skip:
        rows = shape_aware_sweep()
        print_section("Shape-aware families", rows)
        all_rows.extend(rows)

    print_summary(all_rows)

    if args.json:
        args.json.write_text(json.dumps(all_rows, indent=2))
        print(f"\n[Wrote per-cell JSON to {args.json}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
