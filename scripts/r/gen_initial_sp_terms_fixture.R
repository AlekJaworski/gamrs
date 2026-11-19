#!/usr/bin/env Rscript
# mgcv's starting smoothing parameters, log(initial.spg(...)) exactly as
# estimate.gam computes them for gam(method = "REML"), on one model with every
# term type whose basis gamrs builds as mgcv does: cr, cr with a numeric by, re
# and te with cr margins. One case per family, unweighted and with prior
# weights. Synthetic data, embedded so the Rust test prepares the same design.
# gamrs scales a te margin's penalty differently, which moves log sp by a
# constant and leaves sp * S where it is, so log tr(S) is written alongside.
# ti is left out: its gamrs basis is a different parametrisation of the same
# space, so its columns, and with them the start, differ from mgcv's.
#
#     Rscript scripts/r/gen_initial_sp_terms_fixture.R [out_dir]
suppressMessages({library(mgcv); library(jsonlite)})

args <- commandArgs(trailingOnly = TRUE)
FIX <- if (length(args)) args[1] else "tests/fixtures"

set.seed(20261119)
n <- 300
d <- data.frame(x1 = runif(n), x2 = runif(n), x3 = runif(n), x4 = runif(n),
                w = runif(n, 0.5, 2), g = sample(1:12, n, replace = TRUE))
eta <- sin(2 * pi * d$x1) + d$w * (d$x2 - 0.5) + 0.3 * rnorm(12)[d$g] + d$x3 * d$x4
pw <- runif(n, 0.5, 1.5)
ys <- list(
  gaussian = 3 + eta + rnorm(n) * 0.5,
  poisson = rpois(n, exp(1 + 0.5 * eta)),
  binomial = rbinom(n, 1, plogis(eta)),
  gamma = rgamma(n, shape = 4, rate = 4 / exp(1 + 0.5 * eta)),
  invgauss = exp(1 + 0.3 * eta) * rgamma(n, shape = 10, rate = 10))
fam <- list(gaussian = gaussian(), poisson = poisson(), binomial = binomial(),
            gamma_log = Gamma(link = "log"), gamma_inverse = Gamma(), invgauss = inverse.gaussian(link = "log"))
cases <- list(
  list(name = "gaussian", y = "gaussian", family = "gaussian", weighted = FALSE),
  list(name = "gaussian_weighted", y = "gaussian", family = "gaussian", weighted = TRUE),
  list(name = "poisson", y = "poisson", family = "poisson", weighted = FALSE),
  list(name = "poisson_weighted", y = "poisson", family = "poisson", weighted = TRUE),
  list(name = "binomial", y = "binomial", family = "binomial", weighted = FALSE),
  list(name = "binomial_weighted", y = "binomial", family = "binomial", weighted = TRUE),
  list(name = "gamma_log", y = "gamma", family = "gamma_log", weighted = FALSE),
  list(name = "gamma_inverse", y = "gamma", family = "gamma_inverse", weighted = FALSE),
  list(name = "invgauss_log", y = "invgauss", family = "invgauss", weighted = TRUE))

form <- y ~ s(x1, bs = "cr", k = 8) + s(x2, bs = "cr", k = 6, by = w) + s(g, bs = "re") +
  te(x3, x4, bs = "cr", k = c(5, 4))

out_cases <- lapply(cases, function(cs) {
  dd <- d; dd$y <- ys[[cs$y]]; dd$g <- factor(dd$g); dd$pw <- if (cs$weighted) pw else rep(1, n)
  G <- gam(form, family = fam[[cs$family]], data = dd, weights = pw, method = "REML", fit = FALSE)
  lsp <- log(mgcv:::initial.spg(G$X, G$y, G$w, G$family, G$S, G$rank, G$off,
                                offset = G$offset, L = G$L, lsp0 = G$lsp0, E = G$Eb))
  list(name = cs$name, family = cs$family, y = as.numeric(dd$y),
       weights = if (cs$weighted) pw else NULL, log_sp = lsp,
       log_trace_s = sapply(G$S, function(S) log(sum(diag(S)))))
})

as_rows <- function(m) lapply(seq_len(nrow(m)), function(i) as.numeric(m[i, ]))
out <- list(mgcv_version = as.character(packageVersion("mgcv")), formula = deparse(form, width.cutoff = 500L),
            columns = c("x1", "x2", "x3", "x4", "w", "g"),
            x = as_rows(as.matrix(d[, c("x1", "x2", "x3", "x4", "w", "g")])), cases = out_cases)
dir.create(FIX, showWarnings = FALSE, recursive = TRUE)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, null = "null"),
           file.path(FIX, "initial_sp_terms_n300.json"))
for (cs in out_cases) cat(sprintf("%-18s %s\n", cs$name, paste(sprintf("%8.4f", cs$log_sp), collapse = " ")))
