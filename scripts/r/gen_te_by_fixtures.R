#!/usr/bin/env Rscript
# Parity fixtures for numeric `by=` tensor products, Gaussian REML:
#   y ~ te(x, z, k = c(4, 3), bs = "cr", by = x)
# with x >= 0 and a block of rows at x = 0, so the surface is pinned to 0
# along x = 0, and
#   y ~ te(x, z, k = c(4, 3), bs = "cr", by = w) + s(v, k = 8, bs = "cr")
# with w a covariate of its own that is 0 on a fifth of the rows. Data is
# embedded so the Rust and Python tests fit exactly what mgcv saw. Run from
# the repo root.
suppressMessages({library(mgcv); library(jsonlite)})

args <- commandArgs(trailingOnly = TRUE)
FIX <- if (length(args)) args[1] else "tests/fixtures"

emit <- function(path, name, desc, formula, train, new) {
  # Both fits put one margin's sp on a ridge towards infinity, where mgcv's
  # default conv.tol stops ~2e-5 above the REML minimum; converge it.
  fit <- gam(formula, method = "REML", data = train,
             control = gam.control(epsilon = 1e-12, newton = list(conv.tol = 1e-10)))
  cols <- setdiff(names(train), "y")
  terms_train <- predict(fit, type = "terms")
  terms_new <- predict(fit, newdata = new, type = "terms")
  blocks <- lapply(fit$smooth, function(s) s$first.para:s$last.para)
  as_rows <- function(d) lapply(seq_len(nrow(d)), function(i) as.numeric(d[i, cols]))
  inputs <- list(n = nrow(train), columns = cols, method = "REML",
                 x_train = as_rows(train), y_train = as.numeric(train$y),
                 x_new = as_rows(new))
  mo <- list(fitted = as.numeric(fitted(fit)),
             term_train = lapply(seq_len(ncol(terms_train)), function(j) as.numeric(terms_train[, j])),
             term_new = lapply(seq_len(ncol(terms_new)), function(j) as.numeric(terms_new[, j])),
             predictions_new = as.numeric(predict(fit, newdata = new)),
             intercept = as.numeric(coef(fit)[1]),
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
  cat(sprintf("  %-34s edf=(%s) reml=%.6f sp=(%s)\n", name,
              paste(sprintf("%.3f", mo$edf_per_term), collapse = ", "), mo$reml,
              paste(sprintf("%.4g", mo$sp), collapse = ", ")))
}

set.seed(20261109)
n <- 400

# x is 0 on a quarter of the rows and positive otherwise; z is capped, so a
# block of rows sits on the cap.
x <- ifelse(runif(n) < 0.25, 0, 2.5 * runif(n))
z <- pmin(500 * runif(n), 450)
y <- 1 + x * 1.2 * exp(-z / 150) + rnorm(n, sd = 0.3)
grid <- expand.grid(x = c(0, 0.3, 1, 1.8, 2.5), z = c(0, 100, 250, 450))
emit("te_by_self_gaussian_n400_k4x3_cr.json", "te_by_self_gaussian_n400_k4x3_cr",
     "te(x, z, by = x), x >= 0 with x = 0 on a quarter of the rows, Gaussian REML, cr k = (4, 3)",
     y ~ te(x, z, k = c(4, 3), bs = "cr", by = x),
     data.frame(x = x, z = z, y = y), grid)

x <- runif(n); z <- runif(n); v <- runif(n)
w <- ifelse(runif(n) < 0.2, 0, rnorm(n, 1, 0.6))
y <- 1 + w * sin(2 * pi * x) * cos(pi * z) + 0.5 * cos(pi * v) + rnorm(n, sd = 0.3)
x_new <- seq(0.02, 0.98, length.out = 24)
new <- data.frame(x = x_new, z = rev(x_new), w = rep(c(1.3, 0, -0.4, 0.8), length.out = 24),
                  v = (x_new * 7) %% 1)
emit("te_by_covariate_gaussian_n400_k4x3_cr.json", "te_by_covariate_gaussian_n400_k4x3_cr",
     "te(x, z, by = w) + s(v), w a covariate that is 0 on a fifth of the rows, Gaussian REML, cr k = (4, 3) and 8",
     y ~ te(x, z, k = c(4, 3), bs = "cr", by = w) + s(v, k = 8, bs = "cr"),
     data.frame(x = x, z = z, w = w, v = v, y = y), new)
