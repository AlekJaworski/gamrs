# `Gam.predict` evaluated the linear predictor twice on the response scale

**Fixed in 0.15.1** (`1b115e2`): the unmasked response path now calls `predict_response` alone.

Found 2026-10-20 by em_segmentation (an EM mixture model fitting one gamrs `Gam` per cluster and predicting each one
on every sale, about 3,700 predicts per run).

## What happens

`Gam.predict(X)` with the default `scale="response"`, on a fit with no subset mask (`python/gamrs/_fitter.py`,
`Gam.predict`):

1. `eta = np.asarray(f.predict(x))`: builds the design at `X` and computes η natively;
2. then, because `scale == "response"` and there is no subset mask, it returns `np.asarray(f.predict_response(x))`,
   which builds the design and computes η a second time before applying the inverse link.

The first η is thrown away. Every response-scale predict costs two native evaluations.

## What it costs

Profile of one em_segmentation run (5,000 sales, gaussian identity-link fits): 3,696 predicts took 29.2 s,
14.5 s in `FittedGam.predict` and 14.2 s in `FittedGam.predict_response`. A micro-benchmark at n = 5,000 and six CR
terms gives 17.6 ms per call on `scale="response"` against 6.1 ms on `scale="link"`, with identical output for the
identity link.

## Fix

Compute η once. On the unmasked path either return `f.predict_response(x)` without the preceding `f.predict(x)`, or
compute η once and apply the inverse link in Python, as the subset path already does. The comment there asks for the
native path "to match its inverse-link conventions exactly", so keeping `predict_response` and skipping the
`f.predict` call is the smaller change.
