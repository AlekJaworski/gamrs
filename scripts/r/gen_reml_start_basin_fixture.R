#!/usr/bin/env Rscript
# A gaussian REML fit with two local optima, where the smoothing-parameter
# search started at lambda = 1 ends in the worse one and mgcv's start
# (initial.sp) in the better one. Nine cr smooths on synthetic house-sale-like
# data; the lot-size covariate is heavy-tailed and its true effect is linear.
# From lambda = 1 the lot smooth keeps edf 3.47 and REML is 2.4 units worse
# than mgcv's fit, where it is a straight line (edf 1.00).
#
#     Rscript scripts/r/gen_reml_start_basin_fixture.R [out_dir]
suppressMessages({library(mgcv); library(jsonlite)})

args <- commandArgs(trailingOnly = TRUE)
FIX <- if (length(args)) args[1] else "tests/fixtures"

n <- 1500
set.seed(92)
gla <- round(exp(rnorm(n, 7.45, 0.38)))
baths <- pmin(8, pmax(1, round(2 * (gla / 950 + rnorm(n, 0, 0.35))) / 2))
beds <- pmin(8, pmax(1, round(gla / 600 + rnorm(n, 0, 0.7))))
garage <- sample(0:4, n, TRUE, prob = c(.15, .2, .5, .1, .05))
lot <- round(exp(rnorm(n, 8.9, 0.55)) + ifelse(runif(n) < 0.01, exp(rnorm(n, 11, 0.6)), 0))
stories <- sample(1:3, n, TRUE, prob = c(.75, .23, .02))
yb <- pmin(2024, round(1990 + rnorm(n, 0, 22)))
days <- runif(n, 0, 1825)
conc <- ifelse(runif(n) < 0.7, 0, round(runif(n, 0, 20000)))
y <- 1e5 + 160 * gla + 2.5e4 * baths - 6e3 * beds + 1.2e4 * garage + 1.5 * lot - 1.5e4 * stories +
  900 * (yb - 1970) - 60 * days + 3e4 * sin(days / 250) - 0.5 * conc + 7e4 * rt(n, 3) * (gla / 1700)
d <- data.frame(baths, beds, days, garage, gla, lot, stories, yb, conc)
k <- c(baths = 6, beds = 6, days = 25, garage = 5, gla = 6, lot = 6, stories = 3, yb = 6, conc = 6)
form <- as.formula(paste("y ~", paste(sprintf("s(%s, bs = \"cr\", k = %d)", names(k), k), collapse = " + ")))
fit <- gam(form, data = cbind(y = y, d), method = "REML")

blocks <- lapply(fit$smooth, function(s) s$first.para:s$last.para)
out <- list(
  schema_version = 1L, name = "reml_start_basin_gaussian_n1500",
  metadata = list(mgcv_version = as.character(packageVersion("mgcv")), formula = deparse(form, width.cutoff = 500L)),
  inputs = list(columns = names(d), k = unname(k),
                x_train = lapply(seq_len(n), function(i) as.numeric(d[i, ])), y_train = y),
  mgcv_output = list(fitted = as.numeric(fitted(fit)), reml = as.numeric(fit$gcv.ubre),
                     edf_per_term = sapply(blocks, function(b) sum(fit$edf[b])), sp = as.numeric(fit$sp)))
dir.create(FIX, showWarnings = FALSE, recursive = TRUE)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE), file.path(FIX, "reml_start_basin_gaussian_n1500.json"))
cat("REML", out$mgcv_output$reml, "\nedf", round(out$mgcv_output$edf_per_term, 3), "\n")
