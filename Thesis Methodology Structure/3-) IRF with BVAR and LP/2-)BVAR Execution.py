import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# ============================================================
# BVAR IRFs (PRODUCTION):
#   - INCLUDES EBP (EBP is part of the VAR)
#   - PRE vs POST 2008 for WX + KR
#   - PRE-only: AD vs Hybrid + 4-shock comparison figure (WX, KR, AD, HYBRID)
#
# NEW (only addition requested):
#   - One extra POST-only overlay: WX vs KR in the same panels
#
# Uses dataset:
#   outputs/bvar_monthly_dataset_FULL_with_EBP_with_ALL_meeting_shockSUM.csv
# ============================================================

# ================== PATHS ==================
BASE_DIR = os.getcwd()
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

FULL_DATA_FILENAME = "bvar_monthly_dataset_FULL_with_EBP_with_ALL_meeting_shockSUM.csv"
DATA_PATH = os.path.join(OUTPUTS_DIR, FULL_DATA_FILENAME)

OUTDIR = os.path.join(BASE_DIR, "irf_outputs_EBP_final")
os.makedirs(OUTDIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find source data file:\n{DATA_PATH}\n"
        f"Check FULL_DATA_FILENAME or that the file is in outputs/."
    )

print(f"✅ Using source data: {DATA_PATH}")
print(f"✅ Outputs folder    : {OUTDIR}")

# ================== CONFIG ==================
DATE_COL_CANDIDATES = ["MS", "DATE", "Date", "date"]

# Explicit meeting-shock columns (your confirmed names)
SHOCK_WX_COL = "shock_Wu_meeting_sum"
SHOCK_KR_COL = "shock_Kr_meeting_sum"

# PRE-only: detect these robustly (case-insensitive + contains-match)
AD_CANDIDATES = ["AD", "shock_AD_meeting_sum", "ad_meeting_sum", "aruoba", "aruoba_drechsel"]
HYBRID_CANDIDATES = ["hybrid", "shock_hybrid_meeting_sum", "hybrid_meeting_sum", "policy_hybrid"]

# Split
SPLIT_MS = "2008-11-01"

# BVAR settings
HORIZON = 48
DRAWS = 4000

P_LAGS_PRE = 12
P_LAGS_POST = 12

# Priors (simple shrinkage, stable)
DF0 = 10
S0_SCALE = 0.2
LAMBDA_B = 0.2

# Credible intervals
CI_LO, CI_HI = 16, 84

# Winsorization
CLIP_Q_LO, CLIP_Q_HI = 0.001, 0.999

# Estimation options
STANDARDIZE = True
NORMALIZE_SHOCK_IMPACT_TO_ONE = True
BACKTRANSFORM_TO_ORIGINAL_UNITS = True

# Sign convention: tightening => y1 impact positive at h=0
SIGN_ALIGN_TO_Y1_POSITIVE = True

# ================== DISPLAY (THESIS UNITS) ==================
# Match y-axis labels to thesis figure conventions.
DISPLAY_LABELS = {
    "y1": "1-year bond yield (%)",
    "log_sp500": "S&P500 (100 × log)",
    "log_rgdp": "Real GDP (100 × log)",
    "unemp": "Unemployment (%)",
    "log_pgdp": "GDP deflator (100 × log)",
    "EBP": "EBP (%)",
}

def pretty_ylabel(varname: str) -> str:
    return DISPLAY_LABELS.get(varname, varname)

# Optional: if your log_* columns are in plain log units (NOT 100×log),
# set this True so plots match your thesis figure units.
SCALE_LOGS_BY_100 = False
LOG_VARS_TO_SCALE = ["log_sp500", "log_rgdp", "log_pgdp"]

# RNG seed
SEED = 123
rng = np.random.default_rng(SEED)

# ================== UTILITIES ==================
def detect_date_col(df):
    for c in DATE_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def to_datetime_index(df):
    dc = detect_date_col(df)
    if dc is not None:
        df[dc] = pd.to_datetime(df[dc], errors="coerce")
        df = df.dropna(subset=[dc]).sort_values(dc).set_index(dc)
        return df
    df.index = pd.to_datetime(df.index, errors="coerce")
    if df.index.isna().all():
        raise ValueError("No usable date column and index not datetime.")
    return df.sort_index()

def clip_series(s, qlo=CLIP_Q_LO, qhi=CLIP_Q_HI):
    lo, hi = s.quantile([qlo, qhi])
    return s.clip(lo, hi)

def zscore_with_stats(X):
    X = np.asarray(X, float)
    m = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=0)
    sd[sd == 0] = 1.0
    Z = (X - m) / sd
    return Z, m, sd

def make_lag_matrix(Y, p):
    T, k = Y.shape
    Xlags = []
    for L in range(1, p + 1):
        Xlags.append(Y[p - L:T - L, :])
    return np.hstack(Xlags)

def companion_matrix(A_list):
    k = A_list[0].shape[0]
    p = len(A_list)
    top = np.hstack(A_list)
    if p == 1:
        return top
    I = np.eye(k * (p - 1))
    zeros = np.zeros((k * (p - 1), k))
    bottom = np.hstack([I, zeros])
    return np.vstack([top, bottom])

def is_stable(A_list):
    F = companion_matrix(A_list)
    ev = np.linalg.eigvals(F)
    return np.max(np.abs(ev)) < 0.9999

def wishart_rnd(df, V):
    V = np.asarray(V, float)
    L = np.linalg.cholesky(V)
    p = V.shape[0]
    A = np.zeros((p, p))
    for i in range(p):
        A[i, i] = np.sqrt(rng.chisquare(df - i))
        for j in range(i):
            A[i, j] = rng.normal()
    LA = L @ A
    return LA @ LA.T

def invwishart_rnd(df, S):
    S = np.asarray(S, float)
    Sinv = np.linalg.inv(S)
    W = wishart_rnd(df, Sinv)
    return np.linalg.inv(W)

def sample_matrix_normal(Bbar, V, Sigma):
    Bbar = np.asarray(Bbar, float)
    V = np.asarray(V, float)
    Sigma = np.asarray(Sigma, float)
    Lv = np.linalg.cholesky(V)
    Ls = np.linalg.cholesky(Sigma)
    m, k = Bbar.shape
    Z = rng.normal(size=(m, k))
    return Bbar + (Lv @ Z @ Ls.T)

def bvar_draws_irf(Y, p, H, draws, lambda_b=LAMBDA_B, df0=DF0, s0_scale=S0_SCALE, tag=""):
    """
    Returns irfs: (n_kept_draws, H+1, k)
    Identification: Cholesky, shock is first variable.
    """
    t0 = time.time()

    T, k = Y.shape
    Xlags = make_lag_matrix(Y, p)
    Ydep = Y[p:, :]
    X = np.hstack([np.ones((T - p, 1)), Xlags])

    ncoef = X.shape[1]
    B0 = np.zeros((ncoef, k))
    V0_inv = np.eye(ncoef) / (lambda_b ** 2)
    S0 = (s0_scale ** 2) * np.eye(k)

    XtX = X.T @ X
    XtY = X.T @ Ydep

    V_post = np.linalg.inv(V0_inv + XtX)
    B_post = V_post @ (V0_inv @ B0 + XtY)

    U = Ydep - X @ B_post
    S_post = S0 + (U.T @ U) + (B_post - B0).T @ V0_inv @ (B_post - B0)
    df_post = df0 + (T - p)

    def parse_A_list(Bmat):
        A_list = []
        start = 1
        for L in range(p):
            block = Bmat[start + L * k : start + (L + 1) * k, :].T
            A_list.append(block)
        return A_list

    keep_irfs = []
    kept = 0
    milestones = {250, 500, 1000, 2000, 3000, 4000}

    for d in range(draws):
        Sigma = invwishart_rnd(df_post, S_post)
        Bdraw = sample_matrix_normal(B_post, V_post, Sigma)

        A_list = parse_A_list(Bdraw)
        if not is_stable(A_list):
            continue

        P = np.linalg.cholesky(Sigma)
        impact = P[:, 0].copy()

        F = companion_matrix(A_list)
        J = np.hstack([np.eye(k), np.zeros((k, k*(p-1)))])

        irf = np.zeros((H + 1, k))
        irf[0, :] = impact

        Fh = np.eye(k * p)
        for hh in range(1, H + 1):
            Fh = Fh @ F
            Phi_h = J @ Fh @ J.T
            irf[hh, :] = Phi_h @ impact

        keep_irfs.append(irf)
        kept += 1

        if kept in milestones:
            elapsed = time.time() - t0
            print(f"   [{tag}] kept {kept} stable draws (attempt {d+1}/{draws}) | elapsed {elapsed:.1f}s")

    if len(keep_irfs) == 0:
        raise RuntimeError(
            f"[{tag}] No stable posterior draws kept.\n"
            "Try smaller p, or stronger shrinkage (lower LAMBDA_B)."
        )

    elapsed = time.time() - t0
    print(f"   [{tag}] kept {len(keep_irfs)} stable draws out of {draws} attempts | elapsed {elapsed:.1f}s")
    return np.array(keep_irfs)

def normalize_and_backtransform(irfs, sd_vec, normalize_shock=True, backtransform=True):
    irfs2 = irfs.copy()

    if normalize_shock:
        impacts = irfs2[:, 0, 0].copy()
        impacts[np.abs(impacts) < 1e-12] = np.nan
        scale = 1.0 / impacts
        irfs2 = irfs2 * scale[:, None, None]

    if backtransform:
        for j in range(1, irfs2.shape[2]):
            irfs2[:, :, j] = irfs2[:, :, j] * sd_vec[j]

    return irfs2

def sign_align_to_y1(irfs, y1_index=1):
    med_y1_impact = np.median(irfs[:, 0, y1_index])
    if med_y1_impact < 0:
        return -irfs, True
    return irfs, False

def detect_col_fuzzy(df, candidates):
    """
    Robust detection:
    - exact match first
    - case-insensitive exact
    - contains-match (case-insensitive) as fallback
    """
    cols = list(df.columns)
    lower_cols = [c.lower() for c in cols]
    lower_map = {c.lower(): c for c in cols}

    for cand in candidates:
        if cand in cols:
            return cand

    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]

    for cand in candidates:
        key = cand.lower()
        for i, lc in enumerate(lower_cols):
            if key in lc:
                return cols[i]

    return None

# ================== IRF PIPELINE ==================
def run_bvar_irf(df_sub, shock_col, var_order, tag, p_lags, out_csv, out_fig):
    need = var_order
    miss = [c for c in need if c not in df_sub.columns]
    if miss:
        raise ValueError(f"[{tag}] Missing columns: {miss}\nAvailable: {list(df_sub.columns)}")

    d = df_sub.copy()

    for c in need:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan)

    d[shock_col] = d[shock_col].fillna(0.0)

    macro_cols = [c for c in var_order if c != shock_col]
    d = d.dropna(subset=macro_cols).copy()

    nonzero = int((d[shock_col] != 0).sum())
    print(f"\n[{tag}] p={p_lags} | T={len(d)} {d.index.min().date()} → {d.index.max().date()} | nonzero shock months={nonzero}")

    for c in need:
        d[c] = clip_series(d[c])

    Y_raw = d[var_order].values.astype(float)

    if STANDARDIZE:
        Y, _, sd_vec = zscore_with_stats(Y_raw)
    else:
        Y = Y_raw
        sd_vec = np.ones(Y.shape[1])

    irfs = bvar_draws_irf(Y, p=p_lags, H=HORIZON, draws=DRAWS, tag=tag)

    irfs = normalize_and_backtransform(
        irfs,
        sd_vec=sd_vec,
        normalize_shock=NORMALIZE_SHOCK_IMPACT_TO_ONE,
        backtransform=BACKTRANSFORM_TO_ORIGINAL_UNITS
    )

    if SIGN_ALIGN_TO_Y1_POSITIVE:
        irfs, flipped = sign_align_to_y1(irfs, y1_index=1)
        if flipped:
            print(f"   [{tag}] Sign-aligned: flipped IRFs so y1 impact (h=0) is positive.")

    h = np.arange(HORIZON + 1)
    med = np.median(irfs, axis=0)
    lo = np.percentile(irfs, CI_LO, axis=0)
    hi = np.percentile(irfs, CI_HI, axis=0)

    out = pd.DataFrame({"h": h})
    for j in range(1, len(var_order)):
        v = var_order[j]
        out[f"{v}_med"] = med[:, j]
        out[f"{v}_lo"] = lo[:, j]
        out[f"{v}_hi"] = hi[:, j]
    out.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"✅ Saved IRF table → {out_csv}")

    macro_vars = var_order[1:]
    n = len(macro_vars)
    nrows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for i, v in enumerate(macro_vars):
        ax = axes[i]
        j = i + 1
        ax.plot(h, med[:, j], linewidth=2)
        ax.fill_between(h, lo[:, j], hi[:, j], alpha=0.20)
        ax.axhline(0, linewidth=1)
        ax.set_title(f"{v} — {tag}")
        ax.set_ylabel(pretty_ylabel(v))
        ax.grid(True)

    for k in range(n, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.show()
    print(f"✅ Saved figure → {out_fig}")

    return h, (med, lo, hi), macro_vars

# ================== OVERLAYS ==================
def plot_two_irf_overlay(h, A, B, macro_vars, labelA, labelB, title_prefix, out_fig):
    med_A, lo_A, hi_A = A
    med_B, lo_B, hi_B = B

    n = len(macro_vars)
    nrows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for i, v in enumerate(macro_vars):
        ax = axes[i]
        j = i + 1
        ax.plot(h, med_A[:, j], linewidth=2, label=labelA)
        ax.fill_between(h, lo_A[:, j], hi_A[:, j], alpha=0.15)
        ax.plot(h, med_B[:, j], linewidth=2, linestyle="--", label=labelB)
        ax.fill_between(h, lo_B[:, j], hi_B[:, j], alpha=0.15)
        ax.axhline(0, linewidth=1)
        ax.set_title(f"{v} — {title_prefix}")
        ax.set_ylabel(pretty_ylabel(v))
        ax.grid(True)
        ax.legend()

    for k in range(n, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.show()
    print(f"✅ Saved overlay → {out_fig}")

def plot_four_shocks_pre(h, macro_vars, irf_dict, title_prefix, out_fig):
    """
    irf_dict: {label: (med, lo, hi)}; plots median lines only.
    """
    n = len(macro_vars)
    nrows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    linestyles = ["-", "--", ":", "-."]

    labels = list(irf_dict.keys())
    for i, v in enumerate(macro_vars):
        ax = axes[i]
        j = i + 1
        for k, lab in enumerate(labels):
            med, _, _ = irf_dict[lab]
            ax.plot(h, med[:, j], linewidth=2, linestyle=linestyles[k % len(linestyles)], label=lab)

        ax.axhline(0, linewidth=1)
        ax.set_title(f"{v} — {title_prefix}")
        ax.set_ylabel(pretty_ylabel(v))
        ax.grid(True)
        ax.legend()

    for k in range(n, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.show()
    print(f"✅ Saved 4-shock PRE figure → {out_fig}")

# ================== DIAGNOSTICS ==================
def shock_diagnostics(df_pre, df_post, shock_col, label, pre_only=False):
    pre = pd.to_numeric(df_pre[shock_col], errors="coerce").fillna(0.0)
    print(f"\n--- Shock diagnostics: {label} ({shock_col}) ---")
    print(f"PRE : nonzero={int((pre!=0).sum())} | mean={pre.mean():.6f} | std={pre.std(ddof=0):.6f} | min={pre.min():.6f} | max={pre.max():.6f}")
    if not pre_only:
        post = pd.to_numeric(df_post[shock_col], errors="coerce").fillna(0.0)
        print(f"POST: nonzero={int((post!=0).sum())} | mean={post.mean():.6f} | std={post.std(ddof=0):.6f} | min={post.min():.6f} | max={post.max():.6f}")

# ================== MAIN ==================
def main():
    df = pd.read_csv(DATA_PATH)
    df = to_datetime_index(df)

    # ---- INCLUDE EBP ----
    if "EBP" not in df.columns:
        raise ValueError(f"EBP column not found in dataset.\nAvailable columns: {list(df.columns)}")
    print("✅ Keeping EBP column for estimation (EBP spec).")

    # Macro vars (WITH EBP)
    macro_vars = ["y1", "log_sp500", "log_rgdp", "log_pgdp", "unemp", "EBP"]

    required_base = [SHOCK_WX_COL, SHOCK_KR_COL] + macro_vars
    missing = [c for c in required_base if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable columns: {list(df.columns)}")

    # Optional: enforce thesis units for log variables (100 × log)
    if SCALE_LOGS_BY_100:
        for c in LOG_VARS_TO_SCALE:
            if c in df.columns:
                df[c] = 100.0 * pd.to_numeric(df[c], errors="coerce")
        print("✅ Scaled log_* variables to 100×log for thesis-unit consistency.")

    split = pd.to_datetime(SPLIT_MS)
    df_pre = df.loc[df.index < split].copy()
    df_post = df.loc[df.index >= split].copy()

    # ---------- Thesis-ready naming helpers ----------
    def sample_label(is_pre):
        return "Pre2008" if is_pre else "Post2008"

    def shock_full_label(sh):
        return {"WX": "WuXia", "KR": "Krippner", "AD": "AruobaDrechsel", "HYB": "Hybrid"}.get(sh, sh)

    def fig_irf(shock_short, is_pre, p):
        return os.path.join(
            OUTDIR,
            f"Figure_IRF_{shock_full_label(shock_short)}_{sample_label(is_pre)}_p{p}_H{HORIZON}_EBP.png"
        )

    def tab_irf(shock_short, is_pre, p):
        return os.path.join(
            OUTDIR,
            f"Table_IRF_{shock_full_label(shock_short)}_{sample_label(is_pre)}_p{p}_H{HORIZON}_EBP.csv"
        )

    def fig_overlay_prepost(shock_short):
        return os.path.join(
            OUTDIR,
            f"Figure_Overlay_PreVsPost_{shock_full_label(shock_short)}_pPre{P_LAGS_PRE}_pPost{P_LAGS_POST}_H{HORIZON}_EBP.png"
        )

    def fig_overlay_post_wxkr():
        return os.path.join(
            OUTDIR,
            f"Figure_Overlay_Post_WXvsKR_p{P_LAGS_POST}_H{HORIZON}_EBP.png"
        )

    def fig_compare_pre_four():
        return os.path.join(
            OUTDIR,
            f"Figure_Compare_Pre_FourShocks_p{P_LAGS_PRE}_H{HORIZON}_EBP.png"
        )

    # Detect AD/Hybrid robustly
    ad_col = detect_col_fuzzy(df, AD_CANDIDATES)
    hy_col = detect_col_fuzzy(df, HYBRID_CANDIDATES)

    if ad_col is None or hy_col is None:
        print("\n⚠️ PRE-only shocks not found (will skip AD/hybrid if missing).")
        print("   Detected AD  :", ad_col)
        print("   Detected HYB :", hy_col)
    else:
        print(f"\n✅ Detected PRE-only shocks: AD='{ad_col}', Hybrid='{hy_col}'")

    # Diagnostics
    shock_diagnostics(df_pre, df_post, SHOCK_WX_COL, "WX")
    shock_diagnostics(df_pre, df_post, SHOCK_KR_COL, "KR")
    if ad_col is not None:
        shock_diagnostics(df_pre, df_post, ad_col, "AD (pre-only)", pre_only=True)
    if hy_col is not None:
        shock_diagnostics(df_pre, df_post, hy_col, "HYBRID (pre-only)", pre_only=True)

    baseline_order = ["SHOCK"] + macro_vars

    def build_order(shock_col, template):
        return [shock_col if x == "SHOCK" else x for x in template]

    pre_irfs = {}

    # =========================================================
    # WX + KR: PRE vs POST
    # =========================================================
    # store POST IRFs too (for NEW post-only overlay)
    post_irfs = {}

    for shock_col, shock_label in [(SHOCK_WX_COL, "WX"), (SHOCK_KR_COL, "KR")]:
        order = build_order(shock_col, baseline_order)

        h_pre, irf_pre, macro_vars_out = run_bvar_irf(
            df_pre, shock_col, order,
            tag=f"{shock_full_label(shock_label)} (Pre-2008, EBP)",
            p_lags=P_LAGS_PRE,
            out_csv=tab_irf(shock_label, True, P_LAGS_PRE),
            out_fig=fig_irf(shock_label, True, P_LAGS_PRE),
        )
        pre_irfs[shock_label] = irf_pre

        h_post, irf_post, _ = run_bvar_irf(
            df_post, shock_col, order,
            tag=f"{shock_full_label(shock_label)} (Post-2008, EBP)",
            p_lags=P_LAGS_POST,
            out_csv=tab_irf(shock_label, False, P_LAGS_POST),
            out_fig=fig_irf(shock_label, False, P_LAGS_POST),
        )
        post_irfs[shock_label] = (h_post, irf_post)

        # existing PRE vs POST overlays (same structure; thesis naming)
        plot_two_irf_overlay(
            h_pre, irf_pre, irf_post, macro_vars_out,
            labelA="Pre-2008", labelB="Post-2008",
            title_prefix=f"{shock_full_label(shock_label)}: Pre vs Post (EBP)",
            out_fig=fig_overlay_prepost(shock_label),
        )

    # =========================================================
    # PRE-only: AD + Hybrid + 4-shock PRE figure
    # =========================================================
    if ad_col is not None:
        order_ad = build_order(ad_col, baseline_order)
        _, irf_ad, _ = run_bvar_irf(
            df_pre, ad_col, order_ad,
            tag="AruobaDrechsel (Pre-2008, EBP)",
            p_lags=P_LAGS_PRE,
            out_csv=tab_irf("AD", True, P_LAGS_PRE),
            out_fig=fig_irf("AD", True, P_LAGS_PRE),
        )
        pre_irfs["AD"] = irf_ad

    if hy_col is not None:
        order_hy = build_order(hy_col, baseline_order)
        _, irf_hy, _ = run_bvar_irf(
            df_pre, hy_col, order_hy,
            tag="Hybrid (Pre-2008, EBP)",
            p_lags=P_LAGS_PRE,
            out_csv=tab_irf("HYB", True, P_LAGS_PRE),
            out_fig=fig_irf("HYB", True, P_LAGS_PRE),
        )
        pre_irfs["Hybrid"] = irf_hy

    needed = {"WX", "KR", "AD", "Hybrid"}
    if set(pre_irfs.keys()) >= needed:
        plot_four_shocks_pre(
            h_pre,
            macro_vars_out,
            irf_dict={
                "WX": pre_irfs["WX"],
                "KR": pre_irfs["KR"],
                "AD": pre_irfs["AD"],
                "Hybrid": pre_irfs["Hybrid"],
            },
            title_prefix="Pre-2008: Four shocks comparison (median IRFs, EBP)",
            out_fig=fig_compare_pre_four(),
        )
    else:
        print("\n⚠️ Skipping 4-line PRE figure because not all shocks were detected.")
        print("   Have:", list(pre_irfs.keys()), "Need:", list(needed))

    # =========================================================
    # NEW: POST-only overlay WX vs KR (requested addition)
    # =========================================================
    if ("WX" in post_irfs) and ("KR" in post_irfs):
        h_wx, irf_wx_post = post_irfs["WX"]
        h_kr, irf_kr_post = post_irfs["KR"]

        if len(h_wx) != len(h_kr) or not np.allclose(h_wx, h_kr):
            print("\n⚠️ POST horizons differ between WX and KR; using WX horizon for overlay.")
            h_overlay = h_wx
        else:
            h_overlay = h_wx

        plot_two_irf_overlay(
            h_overlay,
            irf_wx_post,
            irf_kr_post,
            macro_vars_out,
            labelA="Wu–Xia (POST)",
            labelB="Krippner (POST)",
            title_prefix="Post-2008: Wu–Xia vs Krippner (EBP)",
            out_fig=fig_overlay_post_wxkr(),
        )
    else:
        print("\n⚠️ Skipping POST WX vs KR overlay (missing WX or KR POST IRFs).")

    print("\n✅ All done.")
    print(f"Source data: {DATA_PATH}")
    print(f"Outputs saved in: {OUTDIR}")

if __name__ == "__main__":
    main()