"""Tweedie parity diagnostic between v0.x mgcv_rust and gamrs.

Layer-by-layer cross-evaluation, following the ocat parity methodology.

Needs both `gamrs` and `mgcv_rust` importable:
    python scripts/diagnostics/tweedie_parity_layered.py
"""

import numpy as np
import pandas as pd

import mgcv_rust as mr
import gamrs


def main():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"x0": rng.uniform(0, 1, n), "x1": rng.uniform(0, 1, n)})
    mu = np.exp(np.sin(2 * np.pi * df["x0"]) + 0.3 * df["x1"])
    df["y"] = rng.gamma(2.0, mu / 2.0)

    # Use tweedie_p=None to ENABLE v0.x's profile-p (so it optimises p
    # like gamrs does) — apples-to-apples for the layered analysis.
    # Note: with tweedie_p=1.5 v0.x FIXES p at 1.5 (no profiling).
    print("==== Fit v0.x (profile-p) ====")
    g_v0 = mr.Gam(
        predictors=["x0", "x1"],
        target="y",
        family="tweedie",
        tweedie_p=None,
        k_default=6,
    ).fit(df[["x0", "x1"]], df["y"])

    v0_native_gam = g_v0._native
    v0_lambdas = np.asarray(v0_native_gam.get_all_lambdas(), dtype=np.float64)
    # v0.x exposes phi via get_scale (if avail) or omits it; family params
    # carry the current Tweedie p.
    v0_phi = None
    for name in ("get_scale", "scale"):
        if hasattr(v0_native_gam, name):
            try:
                v0_phi = float(getattr(v0_native_gam, name)() if callable(getattr(v0_native_gam, name)) else getattr(v0_native_gam, name))
            except Exception:
                pass
            if v0_phi is not None:
                break
    fp = v0_native_gam.get_family_params()
    v0_p = float(fp.get("p", 1.5))
    print(f"v0.x lambdas: {v0_lambdas}")
    print(f"v0.x phi:     {v0_phi}")
    print(f"v0.x p:       {v0_p}")
    y_pred_v0 = g_v0.predict(df[["x0", "x1"]])
    print(f"v0.x μ-RMSE vs μ_true: {np.sqrt(np.mean((y_pred_v0 - mu)**2)):.6f}")

    print("\n==== Fit gamrs ====")
    g_rs = gamrs.Gam(
        predictors=["x0", "x1"],
        target="y",
        family="tweedie",
        tweedie_p=1.5,
        k_default=6,
    ).fit(df[["x0", "x1"]], df["y"])
    fg = g_rs._fitted
    rs_lambdas = np.asarray(getattr(fg, "lambda"), dtype=np.float64)
    rs_rho = np.asarray(fg.rho, dtype=np.float64)
    rs_shape = np.asarray(fg.shape_params, dtype=np.float64)
    rs_log_phi, rs_p_trans = float(rs_shape[0]), float(rs_shape[1])
    rs_phi = float(np.exp(rs_log_phi))
    rs_s = 1.0 / (1.0 + np.exp(-rs_p_trans))
    rs_p = float(np.clip(1.0 + rs_s, 1.05, 1.95))
    print(f"gamrs lambdas: {rs_lambdas}")
    print(f"gamrs phi:     {rs_phi}")
    print(f"gamrs p:       {rs_p}  (p_transform={rs_p_trans})")
    y_pred_rs = g_rs.predict(df[["x0", "x1"]])
    print(f"gamrs μ-RMSE vs μ_true: {np.sqrt(np.mean((y_pred_rs - mu)**2)):.6f}")
    print(
        f"\nLayer 6 (predictions): μ-RMSE(gamrs vs v0.x) = "
        f"{np.sqrt(np.mean((y_pred_rs - y_pred_v0)**2)):.6f}  "
        f"({100 * np.sqrt(np.mean((y_pred_rs - y_pred_v0)**2)) / max(1e-12, np.sqrt(np.mean(y_pred_v0**2))):.4f}%)"
    )

    # Both engines work on raw (n × 2) feature matrix.
    x = df[["x0", "x1"]].values.astype(np.float64)
    y = df["y"].values.astype(np.float64)

    # =================================================================
    # LAYER 4A: each engine's components at its OWN converged θ
    # =================================================================
    theta_rs = np.concatenate([rs_rho, rs_shape])
    rs_self = fg.evaluate_reml_at_tweedie(y, x, theta_rs, [6, 6])
    print("\n---- Layer 4A: gamrs components @ gamrs converged θ ----")
    for k in ("score", "deviance", "bsb_total", "dp", "log_det_h",
              "log_det_lambda_s", "ls", "phi", "p", "mp", "iters", "converged"):
        v = rs_self[k]
        if isinstance(v, (list, np.ndarray)):
            v = list(v)
        print(f"  {k:20s} = {v}")

    p_for_v0 = v0_p if v0_p is not None else 1.5
    v0_self = v0_native_gam.evaluate_reml_tweedie_components_at(
        y, list(v0_lambdas), [p_for_v0]
    )
    print(f"\n---- Layer 4A: v0.x components @ v0.x converged θ "
          f"(λ={v0_lambdas}, p={p_for_v0}) ----")
    for k in ("score", "d_total", "p_total", "dp", "log_det_h",
              "log_det_s_plus", "log_pseudo_det_sum", "log_lambda_sum",
              "ls", "phi", "mp", "ridge", "max_diag_post_pen", "iters",
              "converged"):
        v = v0_self[k]
        print(f"  {k:20s} = {v}")

    # =================================================================
    # LAYER 4B: cross-evaluate each engine at the OTHER engine's θ
    # =================================================================
    # Convert v0.x (λ, p) into gamrs θ = [ρ, log_φ, p_transform].
    p_cross = float(np.clip(p_for_v0, 1.05, 1.95))
    s_cross = p_cross - 1.0
    p_trans_cross = float(np.log(s_cross / max(1.0 - s_cross, 1e-15)))
    log_phi_cross = float(np.log(v0_phi)) if v0_phi else rs_log_phi
    theta_cross_rs = np.concatenate(
        [np.log(v0_lambdas), [log_phi_cross, p_trans_cross]]
    )
    rs_at_v0 = fg.evaluate_reml_at_tweedie(y, x, theta_cross_rs, [6, 6])
    print(f"\n---- Layer 4B: gamrs components @ v0.x θ "
          f"(λ={v0_lambdas}, p={p_cross}, log_φ={log_phi_cross:.4f}) ----")
    for k in ("score", "deviance", "dp", "log_det_h", "log_det_lambda_s",
              "ls", "phi", "p"):
        v = rs_at_v0[k]
        print(f"  {k:20s} = {v}")

    # v0.x at gamrs's (λ, p)
    v0_at_rs = v0_native_gam.evaluate_reml_tweedie_components_at(
        y, list(rs_lambdas), [rs_p]
    )
    print(f"\n---- Layer 4B: v0.x components @ gamrs θ (λ={rs_lambdas}, p={rs_p}) ----")
    for k in ("score", "d_total", "p_total", "dp", "log_det_h",
              "log_det_s_plus", "ls", "phi", "mp"):
        v = v0_at_rs[k]
        print(f"  {k:20s} = {v}")

    # =================================================================
    # Component diff table at v0.x converged θ.
    # =================================================================
    print("\n=== Layer 4 component diffs (v0.x vs gamrs @ v0.x converged θ) ===")
    pairs = [
        ("d_total", "deviance"),
        ("dp", "dp"),
        ("log_det_h", "log_det_h"),
        ("log_det_s_plus", "log_det_lambda_s"),
        ("ls", "ls"),
        ("phi", "phi"),
        ("score", "score"),
    ]
    for v0_key, rs_key in pairs:
        v0_v = float(v0_self[v0_key])
        rs_v = float(rs_at_v0[rs_key])
        diff = rs_v - v0_v
        rel_d = diff / max(1e-12, abs(v0_v))
        print(
            f"  {v0_key:18s}: v0={v0_v:14.6f}  rs={rs_v:14.6f}  "
            f"diff={diff:+.4e}  rel={rel_d:+.3e}"
        )

    # =================================================================
    # LAYER 3: β residual at fixed θ. Compare beta vectors via L2.
    # =================================================================
    beta_rs = np.asarray(rs_at_v0["beta"], dtype=np.float64)
    beta_v0 = np.asarray(v0_self["beta"], dtype=np.float64)
    if beta_rs.shape == beta_v0.shape:
        diff_beta = np.linalg.norm(beta_rs - beta_v0)
        denom = np.linalg.norm(beta_v0) + 1e-12
        print(f"\n=== Layer 3: β L2 at v0.x converged θ ===")
        print(f"  ||β_rs - β_v0||₂ = {diff_beta:.4e}, "
              f"||β_v0||₂ = {np.linalg.norm(beta_v0):.4e}, "
              f"rel = {diff_beta / denom:.3e}")
    else:
        print(f"\n=== Layer 3: β shapes mismatch ({beta_rs.shape} vs {beta_v0.shape}) ===")

    # =================================================================
    # LAYER 5: converged θ comparison summary
    # =================================================================
    print("\n=== Layer 5: converged (λ, p, φ) ===")
    print(f"  λ:    v0={v0_lambdas}   rs={rs_lambdas}")
    print(f"  p:    v0={v0_p}                rs={rs_p}")
    print(f"  φ:    v0={v0_phi}        rs={rs_phi}")


if __name__ == "__main__":
    main()
