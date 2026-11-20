#!/usr/bin/env Rscript
# REML fits for the extended families (nb, tw) whose criterion has two local
# optima, so where the smoothing-parameter search starts decides which one a fit
# ends in. mgcv starts at initial.spg: initial.sp on sqrt(w)*X with the working
# weights prior*mu'(eta)^2/V(mu) at the family's mustart, V at the starting theta
# (nb: theta = 1; tw: p = 1.5).
#
#   reml_start_nb_lot_n2000.json  nb, one covariate spanning 1e3 to 2.5e8 (lot-like)
#   reml_start_nb_n1200.json      nb, five house-sale-like smooths; the other
#                                 optimum is 0.10 worse
#   reml_start_tw_n1200.json      tw, the same covariates; the other optimum is
#                                 0.30 better, but mgcv lands on its own
#
#     Rscript scripts/r/gen_reml_start_extended_fixtures.R [out_dir]
suppressMessages({library(mgcv); library(jsonlite)})

args <- commandArgs(trailingOnly = TRUE)
FIX <- if (length(args)) args[1] else "tests/fixtures"
dir.create(FIX, showWarnings = FALSE, recursive = TRUE)

emit <- function(name, d, k, fit) {
  blocks <- lapply(fit$smooth, function(s) s$first.para:s$last.para)
  out <- list(
    schema_version = 1L, name = name,
    metadata = list(mgcv_version = as.character(packageVersion("mgcv")), family = fit$family$family),
    inputs = list(columns = names(d), k = unname(k),
                  x_train = lapply(seq_len(nrow(d)), function(i) as.numeric(d[i, ])), y_train = as.numeric(fit$y)),
    mgcv_output = list(fitted = as.numeric(fitted(fit)), reml = as.numeric(fit$gcv.ubre),
                       edf_per_term = sapply(blocks, function(b) sum(fit$edf[b])), sp = as.numeric(fit$sp),
                       theta = as.numeric(fit$family$getTheta()), scale = as.numeric(fit$sig2)))
  writeLines(toJSON(out, digits = NA, auto_unbox = TRUE), file.path(FIX, paste0(name, ".json")))
  cat(sprintf("%-26s REML %.4f edf %s\n", name, fit$gcv.ubre, paste(round(out$mgcv_output$edf_per_term, 3), collapse = " ")))
}

# Lot-like: from mgcv's start the lot smooth has edf 5.6. Started with lambda
# near the top of its range on the lot smooth, the search stopped after one step
# at edf 2.0, REML 228 worse.
set.seed(1)
n <- 2000
xl <- pmax(exp(rnorm(n, log(8000), 0.6)), 1e3); xl[sample(n, 3)] <- c(3e6, 4e7, 2.5e8)
x2 <- runif(n)
s <- log(xl); s <- (s - mean(s)) / sd(s)
d <- data.frame(xl = xl, x2 = x2)
y <- rnbinom(n, size = 3, mu = exp(1 + 0.5 * s))
emit("reml_start_nb_lot_n2000", d, c(xl = 10, x2 = 10),
     gam(y ~ s(xl, k = 10, bs = "cr") + s(x2, k = 10, bs = "cr"), data = cbind(y = y, d), family = nb(), method = "REML"))

house <- function(seed, n) {
  set.seed(seed)
  gla <- round(exp(rnorm(n, 7.45, 0.38)))
  lot <- round(exp(rnorm(n, 8.9, 0.55)) + ifelse(runif(n) < 0.01, exp(rnorm(n, 11, 0.6)), 0))
  yb <- pmin(2024, round(1990 + rnorm(n, 0, 22)))
  days <- runif(n, 0, 1825)
  baths <- pmin(6, pmax(1, round(2 * (gla / 950 + rnorm(n, 0, 0.35))) / 2))
  eta <- 0.6 * (log(gla) - 7.45) / 0.38 + 0.1 * (log(lot) - 8.9) + 0.3 * sin(days / 250) +
    0.004 * (yb - 1990) + 0.1 * baths
  list(d = data.frame(baths, days, gla, lot, yb), mu = exp(1 + eta))
}
k <- c(baths = 6, days = 12, gla = 8, lot = 8, yb = 8)
form <- as.formula(paste("y ~", paste(sprintf("s(%s, bs = \"cr\", k = %d)", names(k), k), collapse = " + ")))

h <- house(16, 1200)
y <- rnbinom(1200, mu = h$mu, size = 2)
emit("reml_start_nb_n1200", h$d, k, gam(form, data = cbind(y = y, h$d), family = nb(), method = "REML"))

h <- house(17, 1200)
y <- rTweedie(h$mu, p = 1.4, phi = 1.5)
emit("reml_start_tw_n1200", h$d, k, gam(form, data = cbind(y = y, h$d), family = tw(), method = "REML"))
