# `scat` multi-smooth fit starts one term past a ridge and ends on the wrong optimum (OPEN)

**Status: OPEN** — found 2026-09-29, reproduced on gamrs 0.14.2 (PyPI) and on the
unreleased `fix/tdist-intercept-collapse` wheel. Not fixed. No committable
regression test yet: the reproduction uses real housing sales, kept
out of the repo under the gitignored `data/` (`data/SCAT_START_BASIN_REPRO.md`).

## The bug

On a 10-smooth `scat` (t-dist, identity link) fit of 1,000 house sales, gamrs
returns a REML score **15.3 units worse** than mgcv `gam(method="REML")` on the
same data, same bases, same `k`. Nearly all of the gap is one term, `lot_sqft`:
gamrs shrinks it to a straight line (edf 1.00, log λ ≈ 17.5) where mgcv fits a
curve (edf 4.50, log λ = −2.19). The rest of the model agrees to within
normal tolerance.

| | REML | edf total | edf `lot_sqft` | log λ `lot_sqft` | converged |
|---|---|---|---|---|---|
| mgcv `gam(REML)` | **760.848** | 38.58 | 4.498 | −2.19 | full convergence |
| gamrs 0.14.2 (PyPI) | 776.149 | 35.31 | 1.000 | 17.51 | False (22 outer its) |
| gamrs `fix/tdist-intercept-collapse` wheel | 776.151 | 35.29 | 1.000 | 19.14 | **True** (84 outer its) |

The fix-branch build is the more dangerous one: it lands in the same wrong
place and reports `converged_ = True`, so nothing downstream would notice.

This matters beyond REML units. `lot_sqft` is a priced feature, and the fitted
curve is linear instead of curved. The time term moves slightly too:
`f(0)/f(180)` on `days_ago` is 0.9074 in gamrs and 0.9090 in mgcv.

## It is not a different criterion

mgcv, with the smoothing parameters held at gamrs's landing point, scores
**776.164**. gamrs itself reports 776.149. The two libraries measure the same
function, so gamrs has simply stopped at a worse point of it.

## The criterion has two basins along `lot_sqft`

The criterion profiled along `lot_sqft`'s log λ, with every other λ held at
mgcv's optimum (mgcv evaluating):

| log λ `lot_sqft` | −4 | **−2.19** | 0 | 2 | 4 | **6** | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| REML | 762.56 | **760.85** | 765.22 | 773.34 | 777.97 | **778.47** | 777.76 | 776.87 | 776.82 | 776.82 |
| edf | 4.88 | 4.50 | 3.60 | 2.76 | 2.19 | 2.00 | 1.83 | 1.08 | 1.00 | 1.00 |

The true minimum is at −2.19. A ridge peaks near 6. Past the ridge the
criterion falls onto a flat shelf as λ → ∞, where the term is a straight line
and the gradient vanishes, so the shelf looks like an optimum. Anything that
starts past the ridge runs downhill onto the shelf.

## The cause is the starting value

gamrs's `SmartInit` (`src/fit/driver.rs`) starts `lot_sqft` at log λ =
**6.481**, on the ridge. It starts every other term between −3.1 and −0.8.
mgcv's `initial.sp` starts `lot_sqft` at **−0.113**, and the other terms
between 2.65 and 6.59.

Start mgcv from gamrs's starting point and **mgcv fails in exactly the same
way**. Swap only the `lot_sqft` start and the outcome follows it:

| mgcv run | REML | edf `lot_sqft` |
|---|---|---|
| its own start | 760.848 | 4.498 |
| gamrs's whole start vector | 776.151 | 1.000 |
| its own start, `lot_sqft` from gamrs (6.481) | 776.151 | 1.000 |
| gamrs's start, `lot_sqft` from mgcv (−0.113) | 760.848 | 4.498 |
| its own start, `lot_sqft` at 5.00 | 760.848 | 4.498 |
| its own start, `lot_sqft` at 5.25 | 776.151 | 1.000 |

The basin boundary is between 5.00 and 5.25. gamrs's 6.48 is past it; mgcv's
−0.11 is well inside the right basin. So the outer optimiser is not at fault
here. mgcv's own Newton, given gamrs's start, does the same thing.

Held at gamrs's own landing point, but with `lot_sqft` moved to mgcv's value,
the criterion scores 761.464: already 14.7 units better than where gamrs
stopped.

## Ruled out

- **The units of `lot_sqft`.** Rescaling the column by 1e-3 … 1e3 leaves gamrs's
  answer unchanged, so the start is not an artefact of square feet vs acres.
- **The ν lower bound.** gamrs ends with log(ν − 3) pinned at −10; mgcv's
  optimum has log(ν − 3) = −10.96, past that bound. But moving `lot_sqft` alone
  closes 14.7 of the 15.3 units, so the bound accounts for at most 0.6.
- **Outer tolerances.** mgcv's gradient tolerance (5e-6) gives the same landing
  point.

## What the data looks like

`lot_sqft` is heavily right-skewed. Centred: median −6,763, IQR −7,573 …
−5,472, max +387,649. There are 551 unique values in 1,000 rows. Nothing else
in the model is this skewed. Why `SmartInit`'s formula
(`λ = var(y)·‖S_j‖_F·n / ‖X‖_F²`, one global `‖X‖` for every term) puts this
term ~7 log-units above the others has **not** been established.

## Fix directions (not yet tried)

1. **Port mgcv's `initial.sp`.** It is term-local: it compares each term's own
   block of `X'X` with its `S_j`, rather than dividing every `S_j` by one global
   `‖X‖_F²`. On this data it starts every term in the right basin.
2. **Probe the shelf.** When a term ends at edf ≈ 1 with its λ at or near the
   ρ box and a vanishing gradient, re-evaluate from that term's
   `initial.sp`-style start and keep the lower score. It costs one extra
   outer run, and only on fits that land on a shelf.
3. **Stop reporting `converged` on a shelf.** At minimum, the fix branch should
   not report `converged_ = True` for a term whose gradient is zero only
   because λ ran off to infinity.

A committable restatement needs a synthetic design where one smooth's
criterion has the same two-basin shape. That is the next step once a fix is
chosen.

## Related, separate: slow `scat` fits on a small-scale response

Found in the same investigation. It affects speed, not the answer.
`scat_response_scale` (`src/fit/family_impls.rs`) floors sd(y) at 1.0. With
y = price/1e6 (sd ≈ 0.1) gamrs neither standardises y nor starts σ² at var(y).
It starts σ² about 50× too large and walks it down one capped step per outer
iteration, at 40–200 outer iterations against mgcv's 8–16. Replacing `sd > 1.0`
with `sd > 0.0` (or seeding σ² = (0.8·sd(y))² as mgcv does) brings the counts
back to raw-price levels. Raw-dollar responses are not affected.
