#!/usr/bin/env Rscript
# Parity fixtures for numeric `by=` random effects, Gaussian REML:
#   y ~ golf + s(course, by = golf, bs = "re") + s(x, k = 8, bs = "cr")
# with golf a 0/1 indicator, eight courses of unequal size (down to one row),
# and every non-golf row carrying a placeholder course 0, and
#   y ~ w + s(g, by = w, bs = "re") + s(x, k = 8, bs = "cr")
# a random slope in a covariate w that is 0 on a fifth of the rows, with
# one group seen only where w = 0. Data is embedded so the Rust and Python
# tests fit exactly what mgcv saw. Run from the repo root.
suppressMessages({library(mgcv); library(jsonlite)})

args <- commandArgs(trailingOnly = TRUE)
FIX <- if (length(args)) args[1] else "tests/fixtures"

emit <- function(path, name, desc, formula, train, new, group) {
  as_factor <- function(d) {
    d[[group]] <- factor(d[[group]], levels = sort(unique(train[[group]])))
    d
  }
  fit <- gam(formula, method = "REML", data = as_factor(train),
             control = gam.control(epsilon = 1e-12, newton = list(conv.tol = 1e-10)))
  cols <- setdiff(names(train), "y")
  terms_train <- predict(fit, type = "terms")
  terms_new <- predict(fit, newdata = as_factor(new), type = "terms")
  smooths <- vapply(fit$smooth, function(s) s$label, "")
  blocks <- lapply(fit$smooth, function(s) s$first.para:s$last.para)
  as_rows <- function(d) lapply(seq_len(nrow(d)), function(i) as.numeric(d[i, cols]))
  inputs <- list(n = nrow(train), columns = cols, method = "REML",
                 x_train = as_rows(train), y_train = as.numeric(train$y),
                 x_new = as_rows(new))
  mo <- list(fitted = as.numeric(fitted(fit)),
             smooth_labels = I(smooths),
             term_train = lapply(smooths, function(l) as.numeric(terms_train[, l])),
             term_new = lapply(smooths, function(l) as.numeric(terms_new[, l])),
             predictions_new = as.numeric(predict(fit, newdata = as_factor(new))),
             coef_parametric = as.numeric(coef(fit)[1:2]),
             coef_re = as.numeric(coef(fit)[blocks[[1]]]),
             edf_per_term = I(sapply(blocks, function(b) sum(fit$edf[b]))),
             n_coef_per_term = I(sapply(blocks, length)),
             reml = as.numeric(fit$gcv.ubre),
             scale = as.numeric(fit$sig2),
             sp = I(as.numeric(fit$sp)))
  obj <- list(schema_version = 1L, name = name, description = desc,
              metadata = list(mgcv_version = as.character(packageVersion("mgcv")),
                              formula = deparse(formula)),
              inputs = inputs, mgcv_output = mo)
  cat(toJSON(obj, auto_unbox = TRUE, digits = 17), file = file.path(FIX, path))
  cat(sprintf("  %-34s re coefs=%d edf=(%s) reml=%.6f sp=(%s)\n", name,
              length(blocks[[1]]),
              paste(sprintf("%.3f", mo$edf_per_term), collapse = ", "), mo$reml,
              paste(sprintf("%.4g", mo$sp), collapse = ", ")))
}

set.seed(20261110)
n <- 400

# Golf rows first: courses 1..8 with 60, 40, 25, 15, 10, 5, 2 and 1 rows.
sizes <- c(60, 40, 25, 15, 10, 5, 2, 1)
course <- c(rep(seq_along(sizes), sizes), rep(0, n - sum(sizes)))
golf <- as.numeric(course > 0)
x <- runif(n)
b <- c(0.6, -0.4, 0.3, -0.7, 0.5, -0.2, 0.9, -0.5)
y <- 1 + 0.8 * golf + ifelse(golf == 1, b[pmax(course, 1)], 0) + sin(2 * pi * x) +
  rnorm(n, sd = 0.3)
new <- expand.grid(golf = c(1, 0), course = 0:8, x = c(0.1, 0.55, 0.9))
new <- new[new$golf == 0 & new$course == 0 | new$golf == 1, ]
emit("re_by_golf_gaussian_n400_cr8.json", "re_by_golf_gaussian_n400_cr8",
     "golf + s(course, by = golf, bs = 're') + s(x), golf 0/1, non-golf rows on placeholder course 0, Gaussian REML",
     y ~ golf + s(course, by = golf, bs = "re") + s(x, k = 8, bs = "cr"),
     data.frame(golf = golf, course = course, x = x, y = y), new, "course")

# Six groups of unequal size; group 6 only ever has w = 0.
g <- sample(1:6, n, replace = TRUE, prob = c(0.3, 0.25, 0.2, 0.15, 0.06, 0.04))
w <- ifelse(runif(n) < 0.2 | g == 6, 0, rnorm(n, 1, 0.6))
x <- runif(n)
slope <- c(0.5, -0.3, 0.8, -0.6, 0.2, 1.0)
y <- 1 + 0.4 * w + slope[g] * w + sin(2 * pi * x) + rnorm(n, sd = 0.3)
new <- expand.grid(w = c(1.3, 0, -0.4), g = 1:6, x = c(0.2, 0.7))
emit("re_by_slope_gaussian_n400_cr8.json", "re_by_slope_gaussian_n400_cr8",
     "w + s(g, by = w, bs = 're') + s(x), w a covariate that is 0 on a fifth of the rows, group 6 only at w = 0, Gaussian REML",
     y ~ w + s(g, by = w, bs = "re") + s(x, k = 8, bs = "cr"),
     data.frame(w = w, g = g, x = x, y = y), new, "g")
