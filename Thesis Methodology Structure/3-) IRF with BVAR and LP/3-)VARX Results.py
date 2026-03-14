import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# VARX IRFs (ROBUSTNESS, noEBP) — FULL SCRIPT
#
# You said: BVAR is main; VARX/LP are robustness.
# This script implements VARX as robustness with the SAME spirit/specs:
#   - EXCLUDES EBP completely
#   - PRE vs POST (split at 2008-11-01) for WX and KR
#   - PRE-only: AD and Hybrid (auto-detected)
#   - PRE 4-shock comparison (WX, KR, AD, Hybrid) — median lines
#   - POST-only overlay: WX vs KR
#
# VARX model:
#   y_t = c + A1 y_{t-1} + ... + Ap y_{t-p} + B0 x_t + B1 x_{t-1} + ... + Bq x_{t-q} + u_t
#
# Identification/IRF (dynamic multiplier):
#   shock path: x_t = 1, x_{t+k}=0 for k>0
#   response recursion:
#     IRF(h) = sum_{i=1..p} A_i IRF(h-i) + (B_h if h<=q else 0)
#   with IRF(h<0)=0.
#
# Uncertainty:
#   - (Optional) pairs bootstrap over regression rows
#   - Percentile bands (16–84) like your BVAR script
#
# ============================================================

# -------------------------
# PATHS
# -------------------------
BASE_DIR = os.getcwd()
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FULL_DATA_FILENAME = "bvar_monthly_dataset_FULL_with_EBP_with_ALL_meeting_shockSUM.csv"
DATA_PATH = os.path.join(OUTPUTS_DIR, FULL_DATA_FILENAME)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find source data file:\n{DATA_PATH}\n"
        f"Check FULL_DATA_FILENAME or that the file is in outputs/."
    )

OUTDIR = os.path.join(BASE_DIR, "varx_outputs_noEBP")
os.makedirs(OUTDIR, exist_ok=True)

print(f"✅ Using source data: {DATA_PATH}")
print(f"✅ Outputs folder   : {OUTDIR}")

# -------------------------
# CONFIG (match your BVAR/LP spirit)
# -------------------------
DATE_COL_CANDIDATES = ["MS", "DATE", "Date", "date"]

# Shocks (confirmed)
SHOCK_WX_COL = "shock_Wu_meeting_sum"
SHOCK_KR_COL = "shock_Kr_meeting_sum"

# PRE-only shock detection candidates
AD_CANDIDATES = ["AD", "shock_AD_meeting_sum", "ad_meeting_sum", "aruoba", "aruoba_drechsel"]
HYBRID_CANDIDATES = ["hybrid", "shock_hybrid_meeting_sum", "hybrid_meeting_sum", "policy_hybrid"]

# Split date
SPLIT_MS = "2008-11-01"

# Variables (no EBP)
MACRO_VARS = ["y1", "log_sp500", "log_rgdp", "log_pgdp", "unemp"]

# VARX lags (keep same as your BVAR baseline; if POST is short, reduce P_POST)
P_LAGS_PRE = 12
P_LAGS_POST = 12

# Lags on x in VARX; simplest: q = p (recommended for comparability)
Q_XLAGS_PRE = 12
Q_XLAGS_POST = 12

# IRF horizon
HORIZON = 48

# Winsorization
CLIP_Q_LO, CLIP_Q_HI = 0.001, 0.999

# Shock normalization: 1-std within subsample
NORMALIZE_SHOCK_TO_1STD = True

# Sign convention: tightening => y1 impact at h=0 positive
SIGN_ALIGN_TO_Y1_POSITIVE = True

# Bootstrap (optional; set 0 to skip and just plot point IRFs)
BOOTSTRAP_DRAWS = 400   # 0 disables CIs; 400–1000 is typical for robustness
CI_LO, CI_HI = 16, 84

# RNG
SEED = 123
rng = np.random.default_rng(SEED)

# ============================================================
# Helpers
# ============================================================
def read_csv_auto(path: str) -> pd.DataFrame:
    """
    Robust CSV reader for the 'one big column' issue (semicolon-separated).
    Detect delimiter by inspecting header line.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(path, sep=sep)

def detect_date_col(df: pd.DataFrame):
    for c in DATE_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    dc = detect_date_col(df)
    if dc is not None:
        df[dc] = pd.to_datetime(df[dc], errors="coerce")
        df = df.dropna(subset=[dc]).sort_values(dc).set_index(dc)
        return df
    df.index = pd.to_datetime(df.index, errors="coerce")
    if df.index.isna().all():
        raise ValueError("No usable date column and index not datetime.")
    return df.sort_index()

def clip_series(s: pd.Series, qlo=CLIP_Q_LO, qhi=CLIP_Q_HI) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.quantile([qlo, qhi])
    return s.clip(lo, hi)

def detect_col_fuzzy(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    lower_cols = [c.lower() for c in cols]
    lower_map = {c.lower(): c for c in cols}

    # exact
    for cand in candidates:
        if cand in cols:
            return cand
    # case-insensitive exact
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    # contains match
    for cand in candidates:
        key = cand.lower()
        for i, lc in enumerate(lower_cols):
            if key in lc:
                return cols[i]
    return None

def build_lags(df: pd.DataFrame, col: str, max_lag: int) -> pd.DataFrame:
    out = {}
    for L in range(1, max_lag + 1):
        out[f"{col}_L{L}"] = df[col].shift(L)
    return pd.DataFrame(out, index=df.index)

def prepare_varx_design(df_sub: pd.DataFrame, y_cols, x_col, p_lags: int, q_lags: int):
    """
    Returns:
      Y_t (T x n)
      X_t (T x k)  design with constant + lags(y) + x_t + lags(x)
    aligned so each row corresponds to time t (current y_t depends on lagged stuff & current x)
    """
    d = df_sub.copy()

    need = y_cols + [x_col]
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Available: {list(d.columns)}")

    # numeric, winsorize
    for c in need:
        d[c] = clip_series(d[c])

    # shocks: missing months => 0
    d[x_col] = d[x_col].fillna(0.0)

    # normalize shock to 1 std within sample
    x_std = float(d[x_col].std(ddof=0))
    if NORMALIZE_SHOCK_TO_1STD and x_std > 0:
        d["_x"] = d[x_col] / x_std
        x_scale_note = "1-std normalized"
    else:
        d["_x"] = d[x_col]
        x_scale_note = "raw units"

    # Build lagged Y blocks
    lag_blocks = []
    for y in y_cols:
        lag_blocks.append(build_lags(d, y, p_lags))
    Ylags = pd.concat(lag_blocks, axis=1)

    # Build X lags
    Xlags = build_lags(d, "_x", q_lags)

    # contemporaneous x
    X0 = d[["_x"]].rename(columns={"_x": "x0"})

    # Design
    X = pd.concat([pd.Series(1.0, index=d.index, name="const"), Ylags, X0, Xlags], axis=1)
    Y = d[y_cols]

    # drop NA from lags
    reg = pd.concat([Y, X], axis=1).dropna()
    Yv = reg[y_cols].values
    Xv = reg[X.columns].values

    return Yv, Xv, X.columns.tolist(), x_scale_note

def ols_multivariate(Yv: np.ndarray, Xv: np.ndarray):
    """
    OLS for Y = X B + U
    Returns:
      Bhat: (k x n)
      Uhat: (T x n)
    """
    # B = (X'X)^{-1} X'Y
    XtX = Xv.T @ Xv
    XtY = Xv.T @ Yv
    Bhat = np.linalg.solve(XtX, XtY)
    Uhat = Yv - Xv @ Bhat
    return Bhat, Uhat

def unpack_varx_coeffs(Bhat: np.ndarray, xcols: list, n_y: int, p_lags: int, q_lags: int, y_cols: list):
    """
    Parse Bhat into:
      c  (n,)
      A_list: list of p matrices (n x n) for y-lags
      B_list: list of (q+1) vectors/matrices for x terms:
              B_list[0] is coeff on x_t (n x 1)
              B_list[h] is coeff on x_{t-h} (n x 1) for h=1..q
    """
    k = len(xcols)
    B = Bhat  # k x n

    # locate blocks by column names
    col_to_idx = {name: i for i, name in enumerate(xcols)}

    # constant
    c = B[col_to_idx["const"], :].copy()

    # A_lags
    A_list = []
    for L in range(1, p_lags + 1):
        # order in design: for each y in y_cols we created y_L1..y_Lp
        # but we concatenated build_lags per y, so columns are:
        # y1_L1..y1_Lp, log_sp500_L1..log_sp500_Lp, ...
        A_L = np.zeros((n_y, n_y))
        for j, yj in enumerate(y_cols):  # column variable j
            for i, yi in enumerate(y_cols):  # equation i
                cname = f"{yj}_L{L}"
                A_L[i, j] = B[col_to_idx[cname], i]
        A_list.append(A_L)

    # B_x lags
    B_list = []
    # x0 column name in design is "x0"
    B0 = B[col_to_idx["x0"], :].reshape(n_y, 1)
    B_list.append(B0)
    for L in range(1, q_lags + 1):
        cname = f"_x_L{L}"
        if cname in col_to_idx:
            BL = B[col_to_idx[cname], :].reshape(n_y, 1)
        else:
            BL = np.zeros((n_y, 1))
        B_list.append(BL)

    return c, A_list, B_list

def varx_irf(A_list, B_list, H: int):
    """
    Dynamic multiplier IRF for y to an exogenous x shock at time 0:
      x_0 = 1, x_{k>0}=0.
    With x-lags, there is a direct term B_h at horizon h (if h<=q).
    Recursion:
      irf[h] = sum_{i=1..p} A_i irf[h-i] + B_h (if h<=q else 0)
    """
    p = len(A_list)
    q = len(B_list) - 1
    n = B_list[0].shape[0]

    irf = np.zeros((H + 1, n))
    for h in range(0, H + 1):
        direct = B_list[h] if h <= q else np.zeros((n, 1))
        accum = direct.copy()
        for i in range(1, p + 1):
            if h - i >= 0:
                accum += A_list[i - 1] @ irf[h - i].reshape(n, 1)
        irf[h, :] = accum.ravel()
    return irf

def sign_align_irf(irf: np.ndarray, y_cols: list):
    if not SIGN_ALIGN_TO_Y1_POSITIVE:
        return irf, False
    if "y1" not in y_cols:
        return irf, False
    y1_idx = y_cols.index("y1")
    if np.isfinite(irf[0, y1_idx]) and irf[0, y1_idx] < 0:
        return -irf, True
    return irf, False

def bootstrap_varx_irf(Yv, Xv, xcols, y_cols, p_lags, q_lags, H, nboot: int):
    """
    Pairs bootstrap (resample rows with replacement), re-estimate, recompute IRF.
    Returns: irfs_boot (nboot x (H+1) x n_y)
    """
    T = Yv.shape[0]
    n_y = Yv.shape[1]
    irfs = np.full((nboot, H + 1, n_y), np.nan)

    for b in range(nboot):
        idx = rng.integers(0, T, size=T)
        Yb = Yv[idx, :]
        Xb = Xv[idx, :]

        try:
            Bhat, _ = ols_multivariate(Yb, Xb)
            _, A_list, B_list = unpack_varx_coeffs(Bhat, xcols, n_y, p_lags, q_lags, y_cols)
            irf = varx_irf(A_list, B_list, H)
            irf, _ = sign_align_irf(irf, y_cols)
            irfs[b, :, :] = irf
        except np.linalg.LinAlgError:
            continue

    return irfs

# ============================================================
# Plotting
# ============================================================
def plot_irf_panels(h, med, lo, hi, y_cols, tag, out_fig):
    n = len(y_cols)
    nrows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for j, y in enumerate(y_cols):
        ax = axes[j]
        ax.plot(h, med[:, j], linewidth=2)
        if lo is not None and hi is not None:
            ax.fill_between(h, lo[:, j], hi[:, j], alpha=0.20)
        ax.axhline(0, linewidth=1)
        ax.set_title(f"{y} | {tag}")
        ax.grid(True)

    for k in range(n, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.show()
    print(f"✅ Saved figure → {out_fig}")

def plot_two_overlay(h, A, B, y_cols, labelA, labelB, title_prefix, out_fig):
    medA, loA, hiA = A
    medB, loB, hiB = B

    n = len(y_cols)
    nrows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for j, y in enumerate(y_cols):
        ax = axes[j]
        ax.plot(h, medA[:, j], linewidth=2, label=labelA)
        if loA is not None and hiA is not None:
            ax.fill_between(h, loA[:, j], hiA[:, j], alpha=0.12)
        ax.plot(h, medB[:, j], linewidth=2, linestyle="--", label=labelB)
        if loB is not None and hiB is not None:
            ax.fill_between(h, loB[:, j], hiB[:, j], alpha=0.12)

        ax.axhline(0, linewidth=1)
        ax.set_title(f"{y} | {title_prefix}")
        ax.grid(True)
        ax.legend()

    for k in range(n, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.show()
    print(f"✅ Saved overlay → {out_fig}")

def plot_four_pre(h, y_cols, irf_dict, title_prefix, out_fig):
    """
    Median-only plot with 4 shocks.
    irf_dict: label -> (med, lo, hi)
    """
    n = len(y_cols)
    nrows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    linestyles = ["-", "--", ":", "-."]

    labels = list(irf_dict.keys())
    for j, y in enumerate(y_cols):
        ax = axes[j]
        for k, lab in enumerate(labels):
            med, _, _ = irf_dict[lab]
            ax.plot(h, med[:, j], linewidth=2, linestyle=linestyles[k % len(linestyles)], label=lab)
        ax.axhline(0, linewidth=1)
        ax.set_title(f"{y} | {title_prefix}")
        ax.grid(True)
        ax.legend()

    for k in range(n, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.show()
    print(f"✅ Saved 4-shock PRE figure → {out_fig}")

# ============================================================
# VARX runner (one sample, one shock)
# ============================================================
def run_varx_one(df_sub, shock_col, p_lags, q_lags, tag, out_csv, out_fig):
    t0 = time.time()

    Yv, Xv, xcols, x_scale_note = prepare_varx_design(
        df_sub=df_sub,
        y_cols=MACRO_VARS,
        x_col=shock_col,
        p_lags=p_lags,
        q_lags=q_lags
    )

    T_eff = Yv.shape[0]
    print(f"\n[{tag}] T_eff={T_eff} | p={p_lags} q={q_lags} | shock={shock_col} ({x_scale_note})")

    # estimate
    Bhat, _ = ols_multivariate(Yv, Xv)
    _, A_list, B_list = unpack_varx_coeffs(Bhat, xcols, len(MACRO_VARS), p_lags, q_lags, MACRO_VARS)

    # point IRF
    h = np.arange(HORIZON + 1)
    irf = varx_irf(A_list, B_list, HORIZON)
    irf, flipped = sign_align_irf(irf, MACRO_VARS)
    if flipped:
        print(f"   [{tag}] Sign-aligned (flipped) so y1 impact at h=0 is positive.")

    # bootstrap bands
    lo = hi = None
    if BOOTSTRAP_DRAWS and BOOTSTRAP_DRAWS > 0:
        irfs_boot = bootstrap_varx_irf(Yv, Xv, xcols, MACRO_VARS, p_lags, q_lags, HORIZON, BOOTSTRAP_DRAWS)
        # drop failed draws (all-nan)
        ok = np.isfinite(irfs_boot).all(axis=(1, 2))
        if ok.sum() < max(30, int(0.25 * BOOTSTRAP_DRAWS)):
            print(f"   [{tag}] ⚠️ Many bootstrap failures ({ok.sum()}/{BOOTSTRAP_DRAWS}). Bands may be noisy.")
        irfs_ok = irfs_boot[ok, :, :] if ok.any() else irfs_boot

        lo = np.nanpercentile(irfs_ok, CI_LO, axis=0)
        hi = np.nanpercentile(irfs_ok, CI_HI, axis=0)

    # save table
    out = pd.DataFrame({"h": h})
    for j, y in enumerate(MACRO_VARS):
        out[f"{y}_med"] = irf[:, j]
        if lo is not None and hi is not None:
            out[f"{y}_lo"] = lo[:, j]
            out[f"{y}_hi"] = hi[:, j]
    out.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"✅ Saved VARX table → {out_csv}")

    # plot
    plot_irf_panels(h, irf, lo, hi, MACRO_VARS, tag, out_fig)

    print(f"   [{tag}] elapsed {time.time() - t0:.1f}s")
    return h, (irf, lo, hi), MACRO_VARS

# ============================================================
# MAIN
# ============================================================
def main():
    df = read_csv_auto(DATA_PATH)
    df = to_datetime_index(df)

    # Drop EBP entirely
    if "EBP" in df.columns:
        df = df.drop(columns=["EBP"])
        print("✅ Dropped EBP column (VARX spec).")

    # Validate core cols
    required = [SHOCK_WX_COL, SHOCK_KR_COL] + MACRO_VARS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")

    # Detect AD / Hybrid
    ad_col = detect_col_fuzzy(df, AD_CANDIDATES)
    hy_col = detect_col_fuzzy(df, HYBRID_CANDIDATES)
    print(f"AD detected: {ad_col} | Hybrid detected: {hy_col}")

    # Split
    split = pd.to_datetime(SPLIT_MS)
    df_pre = df.loc[df.index < split].copy()
    df_post = df.loc[df.index >= split].copy()

    # -------------------------
    # WX + KR: PRE vs POST
    # -------------------------
    pre_irfs = {}
    post_irfs = {}

    # WX
    h_pre, wx_pre, _ = run_varx_one(
        df_pre, SHOCK_WX_COL,
        p_lags=P_LAGS_PRE, q_lags=Q_XLAGS_PRE,
        tag="VARX_WX_PRE_noEBP",
        out_csv=os.path.join(OUTDIR, "varx_WX_PRE.csv"),
        out_fig=os.path.join(OUTDIR, "varx_WX_PRE.png"),
    )
    pre_irfs["WX"] = wx_pre

    h_post, wx_post, _ = run_varx_one(
        df_post, SHOCK_WX_COL,
        p_lags=P_LAGS_POST, q_lags=Q_XLAGS_POST,
        tag="VARX_WX_POST_noEBP",
        out_csv=os.path.join(OUTDIR, "varx_WX_POST.csv"),
        out_fig=os.path.join(OUTDIR, "varx_WX_POST.png"),
    )
    post_irfs["WX"] = (h_post, wx_post)

    plot_two_overlay(
        h_pre, wx_pre, wx_post, MACRO_VARS,
        labelA="Pre-2008", labelB="Post-2008",
        title_prefix="WX | VARX | noEBP",
        out_fig=os.path.join(OUTDIR, "overlay_WX_PRE_vs_POST.png")
    )

    # KR
    h_pre, kr_pre, _ = run_varx_one(
        df_pre, SHOCK_KR_COL,
        p_lags=P_LAGS_PRE, q_lags=Q_XLAGS_PRE,
        tag="VARX_KR_PRE_noEBP",
        out_csv=os.path.join(OUTDIR, "varx_KR_PRE.csv"),
        out_fig=os.path.join(OUTDIR, "varx_KR_PRE.png"),
    )
    pre_irfs["KR"] = kr_pre

    h_post, kr_post, _ = run_varx_one(
        df_post, SHOCK_KR_COL,
        p_lags=P_LAGS_POST, q_lags=Q_XLAGS_POST,
        tag="VARX_KR_POST_noEBP",
        out_csv=os.path.join(OUTDIR, "varx_KR_POST.csv"),
        out_fig=os.path.join(OUTDIR, "varx_KR_POST.png"),
    )
    post_irfs["KR"] = (h_post, kr_post)

    plot_two_overlay(
        h_pre, kr_pre, kr_post, MACRO_VARS,
        labelA="Pre-2008", labelB="Post-2008",
        title_prefix="KR | VARX | noEBP",
        out_fig=os.path.join(OUTDIR, "overlay_KR_PRE_vs_POST.png")
    )

    # POST-only overlay: WX vs KR
    if ("WX" in post_irfs) and ("KR" in post_irfs):
        h_wx, wxp = post_irfs["WX"]
        h_kr, krp = post_irfs["KR"]
        h_use = h_wx if (len(h_wx) == len(h_kr) and np.allclose(h_wx, h_kr)) else h_wx
        plot_two_overlay(
            h_use, wxp, krp, MACRO_VARS,
            labelA="Wu–Xia (POST)", labelB="Krippner (POST)",
            title_prefix="POST: WX vs KR | VARX | noEBP",
            out_fig=os.path.join(OUTDIR, "overlay_POST_WX_vs_KR.png")
        )

    # -------------------------
    # PRE-only AD + Hybrid + 4-shock comparison
    # -------------------------
    if ad_col is not None:
        _, ad_pre, _ = run_varx_one(
            df_pre, ad_col,
            p_lags=P_LAGS_PRE, q_lags=Q_XLAGS_PRE,
            tag="VARX_AD_PRE_noEBP",
            out_csv=os.path.join(OUTDIR, "varx_AD_PRE.csv"),
            out_fig=os.path.join(OUTDIR, "varx_AD_PRE.png"),
        )
        pre_irfs["AD"] = ad_pre

    if hy_col is not None:
        _, hy_pre, _ = run_varx_one(
            df_pre, hy_col,
            p_lags=P_LAGS_PRE, q_lags=Q_XLAGS_PRE,
            tag="VARX_HYBRID_PRE_noEBP",
            out_csv=os.path.join(OUTDIR, "varx_HYBRID_PRE.csv"),
            out_fig=os.path.join(OUTDIR, "varx_HYBRID_PRE.png"),
        )
        pre_irfs["Hybrid"] = hy_pre

    if set(pre_irfs.keys()) >= {"WX", "KR", "AD", "Hybrid"}:
        plot_four_pre(
            h_pre,
            MACRO_VARS,
            {
                "WX": pre_irfs["WX"],
                "KR": pre_irfs["KR"],
                "AD": pre_irfs["AD"],
                "Hybrid": pre_irfs["Hybrid"],
            },
            title_prefix="PRE: 4 shocks | VARX | noEBP (median lines)",
            out_fig=os.path.join(OUTDIR, "PRE_4shock_comparison_VARX.png"),
        )
    else:
        print("⚠️ Skipping PRE 4-shock figure (need WX, KR, AD, Hybrid). Have:", list(pre_irfs.keys()))

    print("\n✅ DONE.")
    print(f"Outputs saved in: {OUTDIR}")
    print("\nNOTE:")
    print("- If POST looks unstable or too noisy, reduce P_LAGS_POST/Q_XLAGS_POST (e.g., 6) and rerun.")
    print("- BOOTSTRAP_DRAWS=0 will speed up runs (point IRFs only).")

if __name__ == "__main__":
    main()
