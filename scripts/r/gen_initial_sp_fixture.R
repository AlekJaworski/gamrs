#!/usr/bin/env Rscript
# mgcv's starting smoothing parameters on mgcv's own design, for testing the
# `MgcvInit` port (src/fit/driver.rs) without depending on gamrs's basis.
#
#     Rscript scripts/r/gen_initial_sp_fixture.R
#
# Writes tests/fixtures/initial_sp_3smooth_n200.json: the model matrix X, each
# penalty embedded in p x p at its `off`, a set of working weights, and
# log(initial.sp(X, S, off)) both unweighted and on sqrt(w)*X — which is what
# initial.spg passes for an extended family such as scat. Synthetic data; the
# third covariate is lot-size-like (a few values far out), the shape behind
# docs/scat_start_basin_bug.md.
suppressMessages({library(mgcv); library(jsonlite)})
set.seed(20260929)
n <- 200
x1 <- runif(n); x2 <- runif(n)
x3 <- c(c(4e5, 1.5e5, 9e4), 5000 + floor(runif(n - 3) * 8000))
y <- sin(2 * pi * x1) + 0.5 * x2 + 1e-5 * x3 + rt(n, 5) * 0.3
d <- data.frame(y, x1, x2, x3)
G <- gam(y ~ s(x1, bs = "cr", k = 8) + s(x2, bs = "cr", k = 6) + s(x3, bs = "cr", k = 6),
         data = d, method = "REML", fit = FALSE)
p <- ncol(G$X)
S_full <- lapply(seq_along(G$S), function(i) {
  M <- matrix(0, p, p); ix <- G$off[i]:(G$off[i] + ncol(G$S[[i]]) - 1); M[ix, ix] <- G$S[[i]]; M })
w <- 0.5 + runif(n)
out <- list(
  mgcv = as.character(packageVersion("mgcv")), n = n, p = p,
  x = unname(G$X), s_list = lapply(S_full, unname), weights = w,
  log_sp_unweighted = log(initial.sp(G$X, G$S, G$off)),
  log_sp_weighted = log(initial.sp(sqrt(w) * G$X, G$S, G$off)))
dir.create("tests/fixtures", showWarnings = FALSE)
writeLines(toJSON(out, digits = NA, matrix = "rowmajor", auto_unbox = TRUE), "tests/fixtures/initial_sp_3smooth_n200.json")
cat("log sp unweighted:", out$log_sp_unweighted, "\nlog sp weighted:  ", out$log_sp_weighted, "\n")
