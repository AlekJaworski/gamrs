#!/usr/bin/env Rscript
# Multi-smooth ordered-categorical (`ocat`) parity fixture — the FIRST mgcv
# ocat reference in the repo (parity_ocat.rs was smoke-only).
#
#     Rscript scripts/r/gen_ocat_multismooth_fixtures.R
#
# CRITICAL DGP NOTE: ocat must be generated from a *noisy latent* model
#   z = η + logistic(0,1);  y = bucket(z by fixed cut points)
# NOT by cutting a noiseless η at its own quantiles. Noiseless quantile-cut
# categories are near-separable, which drives the latent scale to blow up:
# on that data mgcv itself converges to θ≈(−1, 104, 181) (degenerate) or
# crashes with "inner loop 1; can't correct step size". The proper noisy
# latent gives moderate, identifiable cut points where mgcv AND gamrs both
# converge cleanly (mgcv: 3 iters, sp≈(7, 100–1000)). This fixture pins
# that well-posed regime so multi-smooth ocat has a real parity bar.
suppressMessages({library(mgcv); library(jsonlite)})

FIX <- "tests/fixtures"

set.seed(20260609)
n <- 1500
R <- 4L
x0 <- runif(n, 0, 10); x1 <- runif(n, 0, 10)
eta <- sin(x0) + 0.5 * sin(x1 * 0.5)
z   <- eta + rlogis(n)                              # latent-variable ocat DGP
y   <- as.integer(cut(z, c(-Inf, -1.5, 0, 1.5, Inf)))  # R=4 overlapping buckets
stopifnot(sort(unique(y)) == 1:R)

fit <- gam(y ~ s(x0, k = 8, bs = "cr") + s(x1, k = 8, bs = "cr"),
           family = ocat(R = R), method = "REML")
proba <- predict(fit, type = "response")            # n x R category probs
acc   <- mean(max.col(proba) == y)
cat(sprintf("mgcv ocat: converged=%s iters=%s acc=%.3f theta=(%s) sp=(%s)\n",
            fit$converged, fit$outer.info$iter, acc,
            paste(sprintf("%.3f", as.numeric(fit$family$getTheta(TRUE))), collapse = ","),
            paste(sprintf("%.2f", fit$sp), collapse = ",")))

inputs <- list(seed = 20260609L, n = n, d = 2L, k = c(8L, 8L),
               bs = c("cr", "cr"), family = "ocat", link = "identity",
               method = "REML", r = R, weights = NULL,
               x_train = lapply(seq_len(n), function(i) c(x0[i], x1[i])),
               y_train = as.numeric(y))
mo <- list(
  # gamrs compares predict_proba to this n x R matrix (row-major list of rows).
  proba = lapply(seq_len(n), function(i) as.numeric(proba[i, ])),
  converged = fit$converged,
  accuracy = acc,
  theta = as.numeric(fit$family$getTheta(TRUE)),
  edf_total = as.numeric(sum(fit$edf)))
obj <- list(schema_version = 1L, name = "2d_ocat_r4_n1500_k8_cr",
            description = "2-D additive ocat (mgcv ocat(R=4)), noisy-latent DGP, cr k=8",
            metadata = list(mgcv_version = as.character(packageVersion("mgcv")),
                            generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")),
            inputs = inputs, mgcv_output = mo)
cat(toJSON(obj, auto_unbox = TRUE, digits = 17),
    file = file.path(FIX, "2d_ocat_r4_n1500_k8_cr.json"))
cat("  wrote 2d_ocat_r4_n1500_k8_cr.json\n")
