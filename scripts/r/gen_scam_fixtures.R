#!/usr/bin/env Rscript
# Parity fixtures for shape-constrained smooths (R package scam), Gaussian.
#
# Synthetic data only, shaped like a busy-road effect on log sale price:
# log_cars (log daily traffic), distance to the road capped at 450, an
# ordinary covariate age, and a discount that is 0 below 1,000 cars, grows
# with log cars and fades linearly to 0 at the 450 cap.
#
#   scam_mpd_*    y1 ~ s(log_cars, bs = "mpd", k = 10)                [+ s(age, cr)]
#   scam_mpi_*    the same fit on 25.2 - y1 with bs = "mpi" (sign handling)
#   scam_tedmd_*  y  ~ s(log_cars, neg_distance, bs = "tedmd", k = c(6, 5))
#                 decreasing in cars and in -distance = increasing in distance
#   scam_tedi_*   y  ~ s(log_cars, distance, bs = "tedi", k = c(6, 5))
#                 the "direct" mixed variant; kept as a contrast fixture: its
#                 reparameterisation forces d2f/dcars ddistance <= 0, the
#                 opposite sign to this surface, and it collapses to additive
#   scasm_*       mgcv::scasm references for the same 1-D and tensor fits
#                 (linear inequality constraints, EFS / REML), the only
#                 REML-type shape-constrained fitter there is to compare with
#
# scam selects smoothing parameters by GCV (scale unknown) with BFGS. Its
# inner Newton tolerance is tightened below; its BFGS is left at the default,
# because on the tensor fits it does not reach a GCV minimum anyway: at n = 500
# it stalls (200 iterations, gradient ~1e-5, analytic gradient checked
# against finite differences) at a point where fresh fits along the descent
# direction score 0.4% lower GCV. So the endpoint is recorded as "what scam
# returns", and every fixture except the mpi and tedi ones also records fresh
# fixed-sp fits on a grid of sp around it (log10 factors -2..2 per sp). The
# grid is the parity target for the criterion and the inner fit; its best
# point is the reference for the optimiser.
# The slope (0.06 per log car) is steeper than real busy-road discounts so the
# n = 500 fits have an interior GCV optimum instead of sitting on the
# straight-line limit.
# Data is embedded. Run from the repo root.
suppressMessages({library(scam); library(mgcv); library(jsonlite)})

args <- commandArgs(trailingOnly = TRUE)
FIX <- if (length(args)) args[1] else "tests/fixtures"

CTL <- list(devtol.fit = 1e-10, steptol.fit = 1e-10)

gen <- function(n, seed) {
  set.seed(seed)
  log_cars <- runif(n, log(300), log(60000))
  distance <- pmin(500 * runif(n), 450)
  age <- runif(n, 0, 80)
  f_age <- 0.15 * sin(age / 25)
  d1 <- 0.06 * pmax(0, log_cars - log(1000))
  e <- rnorm(n, sd = 0.10)
  data.frame(log_cars = log_cars, distance = distance, neg_distance = -distance,
             age = age, y1 = 12.6 + f_age - d1 + e,
             y = 12.6 + f_age - d1 * (1 - distance / 450) + e)
}

grid_1d <- function() data.frame(log_cars = seq(log(300), log(60000), length.out = 25), age = 40)
extrap_1d <- function() data.frame(log_cars = c(log(200), log(90000)), age = 40)
grid_2d <- function() {
  g <- expand.grid(log_cars = seq(log(300), log(60000), length.out = 8),
                   distance = seq(0, 450, length.out = 7))
  g$neg_distance <- -g$distance
  g$age <- 40
  g
}

cols_of <- function(m) lapply(seq_len(ncol(m)), function(j) as.numeric(m[, j]))
rows_of <- function(m) lapply(seq_len(nrow(m)), function(i) as.numeric(m[i, ]))

# Largest monotonicity violation of the fitted surface on a dense grid, in the
# direction each covariate is constrained (positive = violated).
mono_check <- function(fit, kind) {
  if (kind == "1d") {
    nd <- data.frame(log_cars = seq(log(300), log(60000), length.out = 2001), age = 40)
    p <- predict(fit, nd)
    return(list(grid = "log_cars 2001 points", max_increase_in_cars = max(diff(p))))
  }
  nd <- expand.grid(log_cars = seq(log(300), log(60000), length.out = 101),
                    distance = seq(0, 450, length.out = 101))
  nd$neg_distance <- -nd$distance
  nd$age <- 40
  M <- matrix(predict(fit, nd), 101, 101)  # rows: log_cars, columns: distance
  cross <- diff(t(diff(M)))
  list(grid = "101 x 101 log_cars x distance",
       max_increase_in_cars = max(diff(M)),
       max_decrease_in_distance = max(-t(diff(t(M)))),
       cross_difference_range = c(min(cross), max(cross)))
}

smooth_meta <- function(fit) lapply(fit$smooth, function(s) {
  kn <- if (is.list(s$knots)) lapply(s$knots, as.numeric) else as.numeric(s$knots)
  list(label = s$label, class = class(s)[1], first_para = s$first.para,
       last_para = s$last.para, bs_dim = I(s$bs.dim), m = if (is.null(s$m)) NULL else I(s$m),
       knots = kn, cmX = if (is.null(s$cmX)) NULL else as.numeric(s$cmX))
})

base_inputs <- function(train, cols, response, new, extrap) {
  list(n = nrow(train), family = "gaussian", link = "identity", columns = cols,
       response = response,
       x_train = rows_of(as.matrix(train[, cols, drop = FALSE])),
       y_train = as.numeric(train[[response]]),
       x_new = rows_of(as.matrix(new[, cols, drop = FALSE])),
       x_extrap = if (is.null(extrap)) NULL else rows_of(as.matrix(extrap[, cols, drop = FALSE])))
}

write_fixture <- function(path, obj, summary) {
  cat(toJSON(obj, auto_unbox = TRUE, digits = 17, null = "null"), file = file.path(FIX, path))
  cat(sprintf("  %-40s %s\n", sub("\\.json$", "", path), summary))
}

meta <- function(formula, fitter) list(
  scam_version = as.character(packageVersion("scam")),
  mgcv_version = as.character(packageVersion("mgcv")),
  r_version = R.version.string, generated_at = format(Sys.Date()),
  fitter = fitter, formula = paste(deparse(formula), collapse = " "),
  data = "synthetic, generated by scripts/r/gen_scam_fixtures.R")

emit_scam <- function(path, desc, formula, train, cols, response, kind, ladder = TRUE) {
  t0 <- proc.time()[3]
  fit <- scam(formula, data = train, control = CTL)
  secs <- proc.time()[3] - t0
  new <- if (kind == "1d") grid_1d() else grid_2d()
  extrap <- if (kind == "1d") extrap_1d() else NULL
  blocks <- lapply(fit$smooth, function(s) s$first.para:s$last.para)
  n <- nrow(train)
  out <- list(
    fitted = as.numeric(fitted(fit)),
    term_train = cols_of(predict(fit, type = "terms")),
    term_new = cols_of(predict(fit, newdata = new, type = "terms")),
    predictions_new = as.numeric(predict(fit, newdata = new)),
    predictions_new_se = as.numeric(predict(fit, newdata = new, se.fit = TRUE)$se.fit),
    predictions_extrap = if (is.null(extrap)) NULL else as.numeric(predict(fit, newdata = extrap)),
    lpmatrix_new = rows_of(predict(fit, newdata = new, type = "lpmatrix")),
    coefficients = as.numeric(fit$coefficients),      # beta tilde: the fitted, unconstrained scale
    coefficients_t = as.numeric(fit$coefficients.t),  # exp() applied where p_ident
    p_ident = as.logical(fit$p.ident),
    intercept = as.numeric(fit$coefficients.t[1]),
    edf = as.numeric(fit$edf), edf1 = as.numeric(fit$edf1),
    edf_per_term = I(sapply(blocks, function(b) sum(fit$edf[b]))),
    edf_total = sum(fit$edf), trA = as.numeric(fit$trA),
    n_coef_per_term = I(sapply(blocks, length)),
    sp = I(as.numeric(fit$sp)),
    criterion = "GCV", gcv = as.numeric(fit$gcv.ubre),
    deviance = as.numeric(fit$deviance), sig2 = as.numeric(fit$sig2),
    Vp = rows_of(fit$Vp), Vp_t = rows_of(fit$Vp.t),
    optimizer = list(conv = fit$bfgs.info$conv, iter = fit$bfgs.info$iter,
                     grad = I(as.numeric(fit$bfgs.info$grad)),
                     score_history = I(as.numeric(fit$bfgs.info$score.hist)),
                     inner_converged = isTRUE(fit$conv), seconds = as.numeric(secs)),
    monotone_check = mono_check(fit, kind),
    smooths = smooth_meta(fit))
  if (ladder) {
    # Fresh fits at fixed sp: every combination of log10 factors -2..2 on
    # each smoothing parameter, centred on scam's endpoint.
    factors <- as.matrix(expand.grid(rep(list(-2:2), length(fit$sp))))
    out$sp_grid <- lapply(seq_len(nrow(factors)), function(i) {
      sp <- fit$sp * 10^factors[i, ]
      f <- scam(formula, data = train, sp = sp, control = CTL)
      list(log10_factor = I(as.numeric(factors[i, ])), sp = I(as.numeric(sp)),
           gcv = n * f$deviance / (n - f$trA)^2, edf_total = sum(f$edf),
           trA = as.numeric(f$trA), deviance = as.numeric(f$deviance),
           inner_converged = isTRUE(f$conv), inner_iter = f$iter,
           coefficients = as.numeric(f$coefficients))
    })
    gcvs <- sapply(out$sp_grid, `[[`, "gcv")
    out$sp_grid_best <- list(index = which.min(gcvs), gcv = min(gcvs),
                             sp = out$sp_grid[[which.min(gcvs)]]$sp,
                             endpoint_minus_best = out$gcv - min(gcvs))
  }
  obj <- list(schema_version = 1L, name = sub("\\.json$", "", path), description = desc,
              metadata = c(meta(formula, "scam"), list(control = CTL)),
              inputs = base_inputs(train, cols, response, new, extrap),
              scam_output = out)
  write_fixture(path, obj, sprintf("edf=%.3f gcv=%.8g sp=(%s) %s, %d iters, %.1fs%s",
                                   out$edf_total, out$gcv,
                                   paste(sprintf("%.4g", out$sp), collapse = ", "),
                                   out$optimizer$conv, out$optimizer$iter, secs,
                                   if (ladder) sprintf(" | grid best %.8g (endpoint - best %.2e)",
                                                       out$sp_grid_best$gcv, out$sp_grid_best$endpoint_minus_best) else ""))
}

emit_scasm <- function(path, desc, formula, train, cols, response, kind) {
  fit <- scasm(formula, data = train)
  new <- if (kind == "1d") grid_1d() else grid_2d()
  blocks <- lapply(fit$smooth, function(s) s$first.para:s$last.para)
  out <- list(
    fitted = as.numeric(fitted(fit)),
    term_train = cols_of(predict(fit, type = "terms")),
    term_new = cols_of(predict(fit, newdata = new, type = "terms")),
    predictions_new = as.numeric(predict(fit, newdata = new)),
    coefficients = as.numeric(coef(fit)),
    edf = as.numeric(fit$edf), edf_per_term = I(sapply(blocks, function(b) sum(fit$edf[b]))),
    edf_total = sum(fit$edf), sp = I(as.numeric(fit$sp)),
    criterion = "REML (EFS; scasm's laml)", reml = as.numeric(fit$gcv.ubre),
    scale = as.numeric(fit$sig2), Vp = rows_of(fit$Vp),
    monotone_check = mono_check(fit, kind))
  obj <- list(schema_version = 1L, name = sub("\\.json$", "", path), description = desc,
              metadata = meta(formula, "mgcv::scasm"),
              inputs = base_inputs(train, cols, response, new, NULL),
              mgcv_scasm_output = out)
  write_fixture(path, obj, sprintf("edf=%.3f reml=%.8g sp=(%s)", out$edf_total, out$reml,
                                   paste(sprintf("%.4g", out$sp), collapse = ", ")))
}

for (n in c(500, 3000)) {
  d <- gen(n, 20261116 + n)
  d$y1_up <- 25.2 - d$y1
  cat(sprintf("n = %d\n", n))
  emit_scam(sprintf("scam_mpd_gaussian_n%d_k10.json", n),
            "1-D monotone decreasing scam smooth, Gaussian GCV, mpd k = 10; synthetic traffic-like data",
            y1 ~ s(log_cars, bs = "mpd", k = 10), d, c("log_cars"), "y1", "1d")
  emit_scam(sprintf("scam_mpd_cr_gaussian_n%d_k10.json", n),
            "monotone decreasing mpd k = 10 plus an unconstrained s(age, cr, k = 8), Gaussian GCV; synthetic",
            y1 ~ s(log_cars, bs = "mpd", k = 10) + s(age, bs = "cr", k = 8), d,
            c("log_cars", "age"), "y1", "1d")
  emit_scam(sprintf("scam_tedmd_gaussian_n%d_k6x5.json", n),
            "doubly decreasing scam tensor on (log_cars, -distance), i.e. decreasing in cars and increasing in distance, Gaussian GCV, k = (6, 5); synthetic",
            y ~ s(log_cars, neg_distance, bs = "tedmd", k = c(6, 5)), d,
            c("log_cars", "neg_distance"), "y", "2d")
  emit_scasm(sprintf("scasm_mono_gaussian_n%d_k10.json", n),
             "mgcv::scasm reference: s(log_cars, bs = 'sc', xt = 'm-', k = 10), REML by EFS; synthetic",
             y1 ~ s(log_cars, bs = "sc", xt = "m-", k = 10), d, c("log_cars"), "y1", "1d")
  emit_scasm(sprintf("scasm_te_gaussian_n%d_k6x5.json", n),
             "mgcv::scasm reference: te(log_cars, distance, bs = 'sc', xt = list('m-', 'm+'), k = (6, 5)), REML by EFS; synthetic",
             y ~ te(log_cars, distance, bs = "sc", xt = list("m-", "m+"), k = c(6, 5)), d,
             c("log_cars", "distance"), "y", "2d")
}

d <- gen(500, 20261116 + 500)
d$y1_up <- 25.2 - d$y1
cat("n = 500, sign and mixed-direction fixtures\n")
emit_scam("scam_mpi_gaussian_n500_k10.json",
          "1-D monotone increasing scam smooth on the mirrored response 25.2 - y1, Gaussian GCV, mpi k = 10; synthetic",
          y1_up ~ s(log_cars, bs = "mpi", k = 10), d, c("log_cars"), "y1_up", "1d", ladder = FALSE)
emit_scam("scam_tedi_gaussian_n500_k6x5.json",
          "CONTRAST: scam tedi on (log_cars, distance). Its reparameterisation forces a non-positive cross difference, the wrong sign for this surface; the fit collapses to additive. Gaussian GCV, k = (6, 5); synthetic",
          y ~ s(log_cars, distance, bs = "tedi", k = c(6, 5)), d,
          c("log_cars", "distance"), "y", "2d", ladder = FALSE)
emit_scam("scam_tedmd_cr_gaussian_n500_k6x5.json",
          "doubly decreasing tensor on (log_cars, -distance) plus s(age, cr, k = 8), Gaussian GCV; synthetic",
          y ~ s(log_cars, neg_distance, bs = "tedmd", k = c(6, 5)) + s(age, bs = "cr", k = 8), d,
          c("log_cars", "neg_distance", "age"), "y", "2d")
