"""Tweedie parity diagnostic — fixed-p mode (tweedie_p=1.5).

Documents the FIXED-p mode comparison: this is where the prior
"1.14% parity floor" was observed. With v0.x's tweedie_p=1.5 v0.x fixes
p at 1.5; gamrs always profiles p, so they end up at different (p, λ)
and the prediction gap reflects that mismatch — NOT a structural Rust
bug.
"""
import numpy as np
import pandas as pd

import mgcv_rust as mr
import gamrs


def fit_compare(tweedie_p, label):
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"x0": rng.uniform(0, 1, n), "x1": rng.uniform(0, 1, n)})
    mu = np.exp(np.sin(2 * np.pi * df["x0"]) + 0.3 * df["x1"])
    df["y"] = rng.gamma(2.0, mu / 2.0)
    print(f"\n==== {label} (v0.x tweedie_p={tweedie_p}) ====")
    g_v0 = mr.Gam(
        predictors=["x0", "x1"],
        target="y",
        family="tweedie",
        tweedie_p=tweedie_p,
        k_default=6,
    ).fit(df[["x0", "x1"]], df["y"])
    g_rs = gamrs.Gam(
        predictors=["x0", "x1"],
        target="y",
        family="tweedie",
        tweedie_p=1.5 if tweedie_p is not None else None,
        k_default=6,
    ).fit(df[["x0", "x1"]], df["y"])
    y_pred_v0 = g_v0.predict(df[["x0", "x1"]])
    y_pred_rs = g_rs.predict(df[["x0", "x1"]])
    v0_p = float(g_v0._native.get_family_params().get("p", 1.5))
    fg = g_rs._fitted
    rs_shape = np.asarray(fg.shape_params, dtype=np.float64)
    rs_log_phi, rs_p_trans = float(rs_shape[0]), float(rs_shape[1])
    s_sig = 1.0 / (1.0 + np.exp(-rs_p_trans))
    rs_p = float(np.clip(1.0 + s_sig, 1.05, 1.95))
    print(f"  v0.x p:    {v0_p:.4f}    λ: {np.asarray(g_v0._native.get_all_lambdas())}")
    print(f"  gamrs p:   {rs_p:.4f}    λ: {np.asarray(getattr(fg, 'lambda'))}")
    print(f"  μ-RMSE vs truth — v0.x: {np.sqrt(np.mean((y_pred_v0-mu)**2)):.5f}, "
          f"gamrs: {np.sqrt(np.mean((y_pred_rs-mu)**2)):.5f}")
    diff = np.sqrt(np.mean((y_pred_rs - y_pred_v0)**2))
    rel = 100 * diff / max(1e-12, np.sqrt(np.mean(y_pred_v0**2)))
    print(f"  Layer 6: μ-RMSE(gamrs vs v0.x) = {diff:.6f}  ({rel:.4f}%)")


if __name__ == "__main__":
    fit_compare(1.5, "FIXED p=1.5 (v0.x) vs profile-p (gamrs)")
    fit_compare(None, "Profile-p (both)")
