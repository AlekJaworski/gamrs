#!/usr/bin/env Rscript
# Stage-by-stage fixtures for porting mgcv::scasm (Gaussian), on the same
# synthetic data as the scasm_*.json fixtures (read from them, so the data is
# identical). Each file records, for one model:
#   setup   the prefit object from scasm(fit = FALSE): model matrix (n = 500
#           only; colMeans and the first rows otherwise), penalties after
#           mgcv's penalty rescaling, sum-to-zero rows C, inequality rows
#           Ain >= bin, the LP-feasible start beta0, knots, and the
#           prediction matrix on new points including extrapolation
#   fixed   scasm.pirls at fixed sp (log10 factors -2..2 of the fitted sp):
#           coefficients and the active inequality set
#   efs     every iteration of scasm.fit's extended Fellner-Schall loop
#           (a copy of that loop, Gaussian branch, recording its state)
#   final   coefficients, sp, edf, Vp, scale, REML (-laml), fitted values,
#           terms on new data
# Run from the repo root after gen_scam_fixtures.R.
suppressMessages({library(mgcv); library(jsonlite)})

args <- commandArgs(trailingOnly = TRUE)
FIX <- if (length(args)) args[1] else "tests/fixtures"
ns <- asNamespace("mgcv")
for (f in c("scasm.pirls", "getZ", "getVb", "trSiSZ", "Stot", "lpdgDet", "initial.spg"))
  assign(f, get(f, ns))

rows_of <- function(m) lapply(seq_len(nrow(m)), function(i) as.numeric(m[i, ]))

trace_efs <- function(G, control = gam.control()) {
  G$family <- fix.family.ls(fix.family.link(G$family))
  G$Eb <- Stot(G, TRUE)$R
  lsp <- log(initial.spg(G$X, G$y, G$w, G$family, G$S, G$rank, G$off,
                         offset = G$offset, L = G$L, lsp0 = G$lsp0, E = G$Eb))
  lsp_initial <- lsp
  L <- if (is.null(G$L)) diag(1, length(lsp)) else G$L
  start <- NULL; scale <- 1; G$sp <- exp(L %*% lsp + G$lsp0)
  dlsp <- lsp * 0; alpha <- dlsp + 1; eps <- .01
  iters <- list()
  for (i in 1:200) {
    fit <- scasm.pirls(G, log(G$sp), start = start, control = control, eps = eps)
    active <- attr(fit$coef, "active")
    zob <- getZ(G, active)
    v <- getVb(G, H = fit$H, p = fit$coef, zob = zob, project = TRUE)
    scale <- fit$pearson / (nrow(G$X) - sum(v$edf))
    trS <- trSiSZ(G, zob)
    redf <- drop(t(L) %*% ((trS - v$sedf) * G$sp))
    if (any(redf <= 0)) redf[redf < 1e-7] <- 1e-7
    sedf <- drop(t(L) %*% (v$sedf * G$sp))
    dlsp0 <- dlsp
    dlsp <- log(scale) + log(redf) - log(drop(t(L) %*% (v$pSp * G$sp)))
    maxstep <- max(abs(dlsp))
    if (i == 1 && maxstep > 8) dlsp <- 8 * dlsp / maxstep
    if (i > 2 && maxstep > 4) dlsp <- 4 * dlsp / maxstep
    dlsp[(redf < .05 & dlsp > 0) | (sedf < .05 & dlsp < 0)] <- 0
    if (i > 1) {
      jj <- dlsp * dlsp0 > 0
      alpha[!jj] <- alpha[!jj] / 2
      if (i > 2) { ii <- jj & jj.old; alpha[ii] <- pmin(2, alpha[ii] * 1.2) }
      jj.old <- jj
    }
    lsp1 <- dlsp * alpha + lsp
    if (i == 7) eps <- 1e-3
    conv <- (i > 1 && max(abs(pSpold - v$pSp) / pmax(1e-4, v$pSp)) < .01) || (max(abs(lsp1 - lsp)) < .01)
    iters[[i]] <- list(lsp = I(as.numeric(lsp)), active = I(as.integer(active)),
                       coefficients = as.numeric(fit$coef), edf_total = sum(v$edf),
                       scale = scale, trS = I(as.numeric(trS)), sedf = I(as.numeric(v$sedf)),
                       pSp = I(as.numeric(v$pSp)), redf = I(as.numeric(redf)),
                       dlsp = I(as.numeric(dlsp)), alpha = I(as.numeric(alpha)),
                       lsp_next = I(as.numeric(lsp1)), conv = conv)
    if (conv) { if (eps < .005) break else eps <- 1e-3 }
    lsp <- lsp1
    G$sp <- exp(drop(L %*% lsp + G$lsp0))
    start <- fit$coef
    pSpold <- v$pSp
  }
  list(lsp_initial = I(as.numeric(lsp_initial)), iterations = iters)
}

emit <- function(src, path, formula, cols, new) {
  fx <- fromJSON(file.path(FIX, src), simplifyVector = FALSE)
  xt <- do.call(rbind, lapply(fx$inputs$x_train, unlist))
  d <- as.data.frame(xt); names(d) <- cols
  d[[fx$inputs$response]] <- unlist(fx$inputs$y_train)
  n <- nrow(d)
  G <- scasm(formula, data = d, fit = FALSE)
  fit <- scasm(formula, data = d)
  sm <- lapply(G$smooth, function(s) list(
    label = s$label, first_para = s$first.para, last_para = s$last.para,
    bs_dim = I(s$bs.dim), m = I(s$m),
    knots = if (is.list(s$knots)) lapply(s$knots, as.numeric) else as.numeric(s$knots),
    S_scale = I(as.numeric(s$S.scale))))
  setup <- list(
    p = ncol(G$X), off = I(as.integer(G$off)),
    X = if (n <= 500) rows_of(G$X) else NULL,
    X_head = rows_of(G$X[1:20, , drop = FALSE]), X_colmeans = as.numeric(colMeans(G$X)),
    S = lapply(G$S, rows_of), C = rows_of(G$C), Ain = rows_of(G$Ain), bin = as.numeric(G$bin),
    beta0 = as.numeric(G$beta0), smooths = sm,
    lpmatrix_new = rows_of(predict(fit, newdata = new, type = "lpmatrix")))
  fixed <- lapply(-2:2, function(k) {
    Gf <- G; Gf$family <- fix.family.ls(fix.family.link(Gf$family))
    sp <- fit$sp * 10^k
    Gf$sp <- sp
    pf <- scasm.pirls(Gf, log(sp), control = gam.control())
    list(log10_factor = k, sp = I(as.numeric(sp)), coefficients = as.numeric(pf$coef),
         active = I(as.integer(attr(pf$coef, "active"))), pdev = pf$pdev,
         pearson = pf$pearson)
  })
  efs <- trace_efs(G)
  last <- efs$iterations[[length(efs$iterations)]]
  final <- list(coefficients = as.numeric(coef(fit)), sp = I(as.numeric(fit$sp)),
                edf = as.numeric(fit$edf), edf_total = sum(fit$edf), scale = fit$scale,
                reml = as.numeric(fit$gcv.ubre), laml = fit$laml, Vp = rows_of(fit$Vp),
                fitted = as.numeric(fitted(fit)),
                term_new = lapply(seq_len(ncol(tn <- predict(fit, newdata = new, type = "terms"))),
                                  function(j) as.numeric(tn[, j])),
                term_new_se = lapply(seq_len(ncol(ts <- predict(fit, newdata = new, type = "terms", se.fit = TRUE)$se.fit)),
                                     function(j) as.numeric(ts[, j])),
                predictions_new = as.numeric(predict(fit, newdata = new)),
                efs_iterations = length(efs$iterations),
                trace_matches_fit = isTRUE(all.equal(last$coefficients, as.numeric(coef(fit)), tolerance = 0)))
  obj <- list(schema_version = 1L, name = sub("\\.json$", "", path),
              description = paste("mgcv::scasm stage-by-stage internals for", deparse(formula), "on the data of", src),
              metadata = list(mgcv_version = as.character(packageVersion("mgcv")),
                              r_version = R.version.string, generated_at = format(Sys.Date()),
                              formula = paste(deparse(formula), collapse = " "),
                              data = "synthetic, read from the source fixture"),
              inputs = list(n = n, columns = cols, response = fx$inputs$response,
                            source_fixture = src,
                            x_new = rows_of(as.matrix(new[, cols, drop = FALSE]))),
              setup = setup, fixed_sp = fixed, efs = efs, final = final)
  cat(toJSON(obj, auto_unbox = TRUE, digits = 17, null = "null"), file = file.path(FIX, path))
  cat(sprintf("  %-36s efs iters %d (trace == fit: %s) sp=(%s) edf=%.4f reml=%.10g\n", path,
              length(efs$iterations), final$trace_matches_fit,
              paste(sprintf("%.6g", fit$sp), collapse = ", "), final$edf_total, final$reml))
}

new_1d <- data.frame(log_cars = c(log(200), seq(log(300), log(60000), length.out = 23), log(90000)))
for (n in c(500, 3000))
  emit(sprintf("scasm_mono_gaussian_n%d_k10.json", n), sprintf("scasm_internals_mono_n%d_k10.json", n),
       y1 ~ s(log_cars, bs = "sc", xt = "m-", k = 10), "log_cars", new_1d)
