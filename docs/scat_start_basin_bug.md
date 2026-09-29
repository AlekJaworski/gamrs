# `scat` multi-smooth fit starts one term past a ridge and ends on the wrong optimum (FIX ON BRANCH)

**Status: FIX ON BRANCH** `fix/scat-start-basin` (on top of `fix/tdist-intercept-collapse`),
unreleased. Found 2026-09-29, reproduced on gamrs 0.14.2 (PyPI) and on the
`fix/tdist-intercept-collapse` wheel. The fix, `MgcvInit`, is mgcv's own start, pinned
by `tests/initial_sp_parity.rs` against mgcv's `initial.sp` on mgcv's own design. The
real-data reproduction stays in the gitignored `data/` (`data/SCAT_START_BASIN_REPRO.md`).

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

## The fix: start where mgcv starts

`scat` now starts from `MgcvInit` (`src/fit/driver.rs`), a port of mgcv 1.9-3's
`initial.sp`, weighted as `initial.spg` weights an extended family. Every row gets
`½·Dmu2` at `mustart = y + 0.1·[y == 0]` (`scat_initial_weights`). Each term's λ
comes from its own block, the mean of `diag(X'WX)` over the mean of `diag(S_j)`,
and then all λ are scaled together by powers of 10 until the penalised columns
are ~40% data-dominated. gamrs's init ν and σ² still differ from mgcv's
`preinitialize` (ν = 3 + e^1.5, σ = 0.8·sd(y)), which moves every starting log λ
by the same constant. That doesn't change which basin any term starts in. NegBin,
Tweedie and ocat still use `SmartInit`: nothing measured says they need to change.

On the reproduction the fit now lands exactly on mgcv's optimum: REML 760.848,
`lot_sqft` edf 4.51 at log λ −2.185.

The heatmap benchmark's 16 fits (8 markets × wide/filtered seed, y = price/1e6),
REML minus mgcv `gam(REML)`. Lower is better; negative means gamrs beat mgcv:

| fit | `fix/tdist-intercept-collapse` | + `MgcvInit` |
|---|---|---|
| m04 wide | +15.306 | **+0.001** |
| m01 wide | +14.322 | **+3.275** |
| m02 wide | **−8.856** | +1.244 |
| other 13 | within ±0.03 | within ±0.03 |

The start is mgcv's, not a better one. On m02 wide, mgcv's start leads to a
worse basin than `SmartInit` found. gamrs now matches mgcv there instead of
beating it. m01 wide is still 3.3 units behind mgcv, unconverged at 60 outer
iterations. That remainder is not a start problem, and is not investigated here.

### Considered and not done

- **Refit from a second start and keep the lower score.** It would keep m02
  wide's −8.9 as well as fixing m04. It costs a second fit on every `scat`
  call and goes beyond mgcv parity.
- **Report `converged_ = False` when a term ends on the shelf.** This can't be
  made truthful: a term whose true optimum is a straight line sits on the same
  shelf. On this very fit, `bathrooms` ends at edf 1.00, log λ 18.2, in both
  mgcv and gamrs, and that is correct. mgcv reports full convergence on the
  wrong shelf too.

A committable restatement of the two-basin shape (synthetic data) is still
missing. The parity test pins the start, not the basin.

## Related, separate: slow `scat` fits on a small-scale response

Found in the same investigation. It affects speed, not the answer.
`scat_response_scale` (`src/fit/family_impls.rs`) floors sd(y) at 1.0. With
y = price/1e6 (sd ≈ 0.1) gamrs neither standardises y nor starts σ² at var(y).
It starts σ² about 50× too large and walks it down one capped step per outer
iteration, at 40–200 outer iterations against mgcv's 8–16. Replacing `sd > 1.0`
with `sd > 0.0` (or seeding σ² = (0.8·sd(y))² as mgcv does) brings the counts
back to raw-price levels. Raw-dollar responses are not affected.
