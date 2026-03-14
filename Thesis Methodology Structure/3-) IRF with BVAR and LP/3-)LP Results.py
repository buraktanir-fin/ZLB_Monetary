import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac

# ============================================================
# LOCAL PROJECTIONS (PRODUCTION, WITH EBP) — FULL SCRIPT
#
# ONLY CHANGE FROM noEBP VERSION:
#   - EBP is retained in MACRO_VARS (not dropped)
#   - Output directories renamed to reflect EBP spec
#   - Everything else (linear LP, split NL LP, overlays,
#     sign alignment, HAC, normalization) is IDENTICAL
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

OUTDIR_LINEAR = os.path.join(BASE_DIR, "lp_outputs_EBP_linear")
OUTDIR_NL     = os.path.join(BASE_DIR, "lp_outputs_EBP_nonlinear_split")
os.makedirs(OUTDIR_LINEAR, exist_ok=True)
os.makedirs(OUTDIR_NL, exist_ok=True)

print(f"✅ Using source data: {DATA_PATH}")
print(f"✅ Linear outputs   : {OUTDIR_LINEAR}")
print(f"✅ Nonlinear outputs: {OUTDIR_NL}")

# -------------------------
# CONFIG
# -------------------------
DATE_COL_CANDIDATES = ["MS", "DATE", "Date", "date"]

SHOCK_WX_COL = "shock_Wu_meeting_sum"
SHOCK_KR_COL = "shock_Kr_meeting_sum"

AD_CANDIDATES     = ["AD", "shock_AD_meeting_sum", "ad_meeting_sum", "aruoba", "aruoba_drechsel"]
HYBRID_CANDIDATES = ["hybrid", "shock_hybrid_meeting_sum", "hybrid_meeting_sum", "policy_hybrid"]

SPLIT_MS    = "2008-11-01"
SAMPLE_END  = "2019-12-01"   # post-2008 sample end: Nov 2008 – Dec 2019 = 133 months

HORIZON      = 48
P_LAGS_PRE   = 12
P_LAGS_POST  = 6    # 133 post months − 6 lags − 48 horizon = 79 obs vs 44 regressors ✅

NW_LAGS_RULE  = "h+1"
NW_LAGS_FIXED = 12

CLIP_Q_LO, CLIP_Q_HI = 0.001, 0.999

NORMALIZE_SHOCK_TO_1STD  = True
SIGN_ALIGN_TO_Y1_POSITIVE = True

# ── Only change: EBP included ────────────────────────────────
MACRO_VARS = ["y1", "log_sp500", "log_rgdp", "log_pgdp", "unemp", "EBP"]

# -------------------------
# HELPERS
# -------------------------
def read_csv_auto(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(path, sep=sep)

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
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.quantile([qlo, qhi])
    return s.clip(lo, hi)

def detect_col_fuzzy(df, candidates):
    cols      = list(df.columns)
    lower_cols = [c.lower() for c in cols]
    lower_map  = {c.lower(): c for c in cols}
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

def nw_lags_for_h(h: int) -> int:
    if NW_LAGS_RULE == "h":
        return max(0, h)
    if NW_LAGS_RULE == "h+1":
        return max(0, h + 1)
    if NW_LAGS_RULE == "fixed":
        return int(max(0, NW_LAGS_FIXED))
    raise ValueError(f"Unknown NW_LAGS_RULE: {NW_LAGS_RULE}")

def hac_se_safe(results, maxlags):
    nobs     = int(results.nobs)
    k_params = int(results.df_model) + 1
    if nobs <= k_params + 1:
        return np.full(len(results.params), np.nan)
    nlags = int(max(0, min(maxlags, nobs - 1)))
    V  = cov_hac(results, nlags=nlags)
    se = np.sqrt(np.diag(V))
    return se

# -------------------------
# CORE LP ROUTINES
# -------------------------
def lp_linear(df_sub: pd.DataFrame, shock_col: str, p_lags: int, tag: str,
              out_csv: str, out_fig: str):
    t0 = time.time()
    d  = df_sub.copy()

    needed  = [shock_col] + MACRO_VARS
    missing = [c for c in needed if c not in d.columns]
    if missing:
        raise ValueError(f"[{tag}] Missing columns: {missing}\nAvailable: {list(d.columns)}")

    for c in needed:
        d[c] = clip_series(d[c])
    d[shock_col] = d[shock_col].fillna(0.0)

    shock_std = float(d[shock_col].std(ddof=0))
    if NORMALIZE_SHOCK_TO_1STD and shock_std > 0:
        d["_shock_used"] = d[shock_col] / shock_std
        shock_label = f"{shock_col} (1-std normalized)"
    else:
        d["_shock_used"] = d[shock_col]
        shock_label = f"{shock_col} (raw units)"

    controls_base = ["_shock_used"] + MACRO_VARS
    for L in range(1, p_lags + 1):
        for c in controls_base:
            d[f"{c}_L{L}"] = d[c].shift(L)

    H     = HORIZON
    hgrid = np.arange(H + 1)

    b  = np.full((H + 1, len(MACRO_VARS)), np.nan)
    se = np.full((H + 1, len(MACRO_VARS)), np.nan)

    for hi in range(H + 1):
        nw    = nw_lags_for_h(hi)
        Xcols = ["_shock_used"] + [f"{c}_L{L}" for L in range(1, p_lags + 1) for c in controls_base]
        X     = d[Xcols].copy()

        for j, y in enumerate(MACRO_VARS):
            ydep = d[y].shift(-hi)
            reg  = pd.concat([ydep.rename("ydep"), X], axis=1).dropna()
            if reg.empty:
                continue

            yv = reg["ydep"].values
            Xv = reg[Xcols].values
            Xv = sm.add_constant(Xv, has_constant="add")

            res      = sm.OLS(yv, Xv).fit()
            nobs     = int(res.nobs)
            k_params = Xv.shape[1]
            if nobs <= k_params + 1:
                continue

            se_vec   = hac_se_safe(res, maxlags=nw)
            b[hi, j] = res.params[1]
            se[hi, j] = se_vec[1] if np.isfinite(se_vec[1]) else np.nan

    if SIGN_ALIGN_TO_Y1_POSITIVE:
        y1_idx = MACRO_VARS.index("y1")
        if np.isfinite(b[0, y1_idx]) and b[0, y1_idx] < 0:
            b = -b

    lo = b - se
    hi = b + se

    out = pd.DataFrame({"h": hgrid})
    for j, y in enumerate(MACRO_VARS):
        out[f"{y}_b"]  = b[:, j]
        out[f"{y}_lo"] = lo[:, j]
        out[f"{y}_hi"] = hi[:, j]
    out.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"✅ Saved LP table → {out_csv}")

    n     = len(MACRO_VARS)
    nrows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for j, y in enumerate(MACRO_VARS):
        ax = axes[j]
        ax.plot(hgrid, b[:, j], linewidth=2)
        ax.fill_between(hgrid, lo[:, j], hi[:, j], alpha=0.20)
        ax.axhline(0, linewidth=1)
        ax.set_title(f"{y} | {tag}")
        ax.grid(True)

    for k in range(n, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.show()
    print(f"✅ Saved LP figure → {out_fig}")
    print(f"   [{tag}] shock scaling: {shock_label} | p={p_lags} | elapsed {time.time()-t0:.1f}s")

    return hgrid, (b, lo, hi), MACRO_VARS


def lp_nonlinear_split(df_full: pd.DataFrame, shock_col: str, p_lags: int, tag: str,
                       out_csv: str, out_fig: str, split_date: str):
    t0 = time.time()
    d  = df_full.copy()

    needed  = [shock_col] + MACRO_VARS
    missing = [c for c in needed if c not in d.columns]
    if missing:
        raise ValueError(f"[{tag}] Missing columns: {missing}\nAvailable: {list(d.columns)}")

    for c in needed:
        d[c] = clip_series(d[c])
    d[shock_col] = d[shock_col].fillna(0.0)

    split  = pd.to_datetime(split_date)
    d["_D"] = (d.index >= split).astype(int)

    shock_std = float(d[shock_col].std(ddof=0))
    if NORMALIZE_SHOCK_TO_1STD and shock_std > 0:
        d["_shock_used"] = d[shock_col] / shock_std
        shock_label = f"{shock_col} (1-std normalized, FULL-sample)"
    else:
        d["_shock_used"] = d[shock_col]
        shock_label = f"{shock_col} (raw units)"

    d["_shock_post"] = d["_shock_used"] * d["_D"]
    d["_shock_pre"]  = d["_shock_used"] * (1 - d["_D"])

    controls_base = ["_shock_used"] + MACRO_VARS
    for L in range(1, p_lags + 1):
        for c in controls_base:
            d[f"{c}_L{L}"] = d[c].shift(L)

    H     = HORIZON
    hgrid = np.arange(H + 1)

    b_pre   = np.full((H + 1, len(MACRO_VARS)), np.nan)
    b_post  = np.full((H + 1, len(MACRO_VARS)), np.nan)
    se_pre  = np.full((H + 1, len(MACRO_VARS)), np.nan)
    se_post = np.full((H + 1, len(MACRO_VARS)), np.nan)
    se_diff = np.full((H + 1, len(MACRO_VARS)), np.nan)
    diff    = np.full((H + 1, len(MACRO_VARS)), np.nan)

    for hi in range(H + 1):
        nw    = nw_lags_for_h(hi)
        Xcols = ["_shock_pre", "_shock_post"] + [
            f"{c}_L{L}" for L in range(1, p_lags + 1) for c in controls_base
        ]
        X = d[Xcols].copy()

        for j, y in enumerate(MACRO_VARS):
            ydep = d[y].shift(-hi)
            reg  = pd.concat([ydep.rename("ydep"), X], axis=1).dropna()
            if reg.empty:
                continue

            yv = reg["ydep"].values
            Xv = reg[Xcols].values
            Xv = sm.add_constant(Xv, has_constant="add")

            res      = sm.OLS(yv, Xv).fit()
            nobs     = int(res.nobs)
            k_params = Xv.shape[1]
            if nobs <= k_params + 1:
                continue

            nlags  = int(max(0, min(nw, nobs - 1)))
            V      = cov_hac(res, nlags=nlags)
            params = res.params

            b_pre[hi, j]  = params[1]
            b_post[hi, j] = params[2]

            se_pre[hi, j]  = np.sqrt(V[1, 1]) if np.isfinite(V[1, 1]) else np.nan
            se_post[hi, j] = np.sqrt(V[2, 2]) if np.isfinite(V[2, 2]) else np.nan

            diff[hi, j]   = params[2] - params[1]
            var_diff       = V[2, 2] + V[1, 1] - 2.0 * V[1, 2]
            se_diff[hi, j] = np.sqrt(var_diff) if (np.isfinite(var_diff) and var_diff >= 0) else np.nan

    if SIGN_ALIGN_TO_Y1_POSITIVE:
        y1_idx = MACRO_VARS.index("y1")
        anchor = b_pre[0, y1_idx]
        if np.isfinite(anchor) and anchor < 0:
            b_pre, b_post, diff = -b_pre, -b_post, -diff

    lo_pre,  hi_pre  = b_pre  - se_pre,  b_pre  + se_pre
    lo_post, hi_post = b_post - se_post, b_post + se_post
    lo_diff, hi_diff = diff   - se_diff, diff   + se_diff

    out = pd.DataFrame({"h": hgrid})
    for j, y in enumerate(MACRO_VARS):
        out[f"{y}_b_pre"]             = b_pre[:, j]
        out[f"{y}_lo_pre"]            = lo_pre[:, j]
        out[f"{y}_hi_pre"]            = hi_pre[:, j]
        out[f"{y}_b_post"]            = b_post[:, j]
        out[f"{y}_lo_post"]           = lo_post[:, j]
        out[f"{y}_hi_post"]           = hi_post[:, j]
        out[f"{y}_diff_post_minus_pre"] = diff[:, j]
        out[f"{y}_lo_diff"]           = lo_diff[:, j]
        out[f"{y}_hi_diff"]           = hi_diff[:, j]
    out.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"✅ Saved Split-NL LP table → {out_csv}")

    n     = len(MACRO_VARS)
    nrows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for j, y in enumerate(MACRO_VARS):
        ax = axes[j]
        ax.plot(hgrid, b_pre[:, j],  linewidth=2, label="Pre-2008 (b_pre)")
        ax.fill_between(hgrid, lo_pre[:, j], hi_pre[:, j], alpha=0.12)
        ax.plot(hgrid, b_post[:, j], linewidth=2, linestyle="--", label="Post-2008 (b_post)")
        ax.fill_between(hgrid, lo_post[:, j], hi_post[:, j], alpha=0.12)
        ax.axhline(0, linewidth=1)
        ax.set_title(f"{y} | {tag}")
        ax.grid(True)
        ax.legend()

    for k in range(n, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    plt.show()
    print(f"✅ Saved Split-NL LP figure → {out_fig}")
    print(f"   [{tag}] state=pre/post split @ {split_date} | shock scaling: {shock_label} | p={p_lags} | elapsed {time.time()-t0:.1f}s")

    return hgrid, ((b_pre, lo_pre, hi_pre), (b_post, lo_post, hi_post), (diff, lo_diff, hi_diff)), MACRO_VARS


# -------------------------
# PLOTTING HELPERS
# -------------------------
def plot_two_overlay(h, A, B, macro_vars, labelA, labelB, title_prefix, out_fig):
    bA, loA, hiA = A
    bB, loB, hiB = B

    n     = len(macro_vars)
    nrows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for j, y in enumerate(macro_vars):
        ax = axes[j]
        ax.plot(h, bA[:, j], linewidth=2, label=labelA)
        ax.fill_between(h, loA[:, j], hiA[:, j], alpha=0.12)
        ax.plot(h, bB[:, j], linewidth=2, linestyle="--", label=labelB)
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

def plot_four_pre(h, macro_vars, irf_dict, title_prefix, out_fig):
    n     = len(macro_vars)
    nrows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    linestyles = ["-", "--", ":", "-."]
    labels = list(irf_dict.keys())

    for j, y in enumerate(macro_vars):
        ax = axes[j]
        for k, lab in enumerate(labels):
            b, _, _ = irf_dict[lab]
            ax.plot(h, b[:, j], linewidth=2,
                    linestyle=linestyles[k % len(linestyles)], label=lab)
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

# -------------------------
# MAIN
# -------------------------
def main():
    df = read_csv_auto(DATA_PATH)
    df = to_datetime_index(df)

    # ── EBP retained ─────────────────────────────────────────
    if "EBP" not in df.columns:
        raise ValueError(
            f"EBP column not found in dataset.\nAvailable: {list(df.columns)}"
        )
    print("✅ EBP column retained for LP estimation.")

    required = [SHOCK_WX_COL, SHOCK_KR_COL] + MACRO_VARS
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")

    ad_col = detect_col_fuzzy(df, AD_CANDIDATES)
    hy_col = detect_col_fuzzy(df, HYBRID_CANDIDATES)
    print(f"AD detected: {ad_col} | Hybrid detected: {hy_col}")

    split      = pd.to_datetime(SPLIT_MS)
    sample_end = pd.to_datetime(SAMPLE_END)
    df_pre  = df.loc[df.index < split].copy()
    df_post = df.loc[(df.index >= split) & (df.index <= sample_end)].copy()
    print(f"✅ Post-2008 sample: {df_post.index.min().date()} → "
          f"{df_post.index.max().date()} ({len(df_post)} months)")

    # -------------------------
    # LINEAR LP
    # -------------------------
    pre_irfs  = {}
    post_irfs = {}

    # WX
    h, wx_pre, _ = lp_linear(
        df_pre, SHOCK_WX_COL, p_lags=P_LAGS_PRE, tag="LP_LINEAR_WX_PRE_EBP",
        out_csv=os.path.join(OUTDIR_LINEAR, "lp_WX_PRE.csv"),
        out_fig=os.path.join(OUTDIR_LINEAR, "lp_WX_PRE.png"),
    )
    pre_irfs["WX"] = wx_pre

    h2, wx_post, _ = lp_linear(
        df_post, SHOCK_WX_COL, p_lags=P_LAGS_POST, tag="LP_LINEAR_WX_POST_EBP",
        out_csv=os.path.join(OUTDIR_LINEAR, "lp_WX_POST.csv"),
        out_fig=os.path.join(OUTDIR_LINEAR, "lp_WX_POST.png"),
    )
    post_irfs["WX"] = (h2, wx_post)

    plot_two_overlay(
        h, wx_pre, wx_post, MACRO_VARS,
        labelA="Pre-2008", labelB="Post-2008",
        title_prefix="WX | Linear LP | EBP",
        out_fig=os.path.join(OUTDIR_LINEAR, "overlay_WX_PRE_vs_POST.png"),
    )

    # KR
    h, kr_pre, _ = lp_linear(
        df_pre, SHOCK_KR_COL, p_lags=P_LAGS_PRE, tag="LP_LINEAR_KR_PRE_EBP",
        out_csv=os.path.join(OUTDIR_LINEAR, "lp_KR_PRE.csv"),
        out_fig=os.path.join(OUTDIR_LINEAR, "lp_KR_PRE.png"),
    )
    pre_irfs["KR"] = kr_pre

    h2, kr_post, _ = lp_linear(
        df_post, SHOCK_KR_COL, p_lags=P_LAGS_POST, tag="LP_LINEAR_KR_POST_EBP",
        out_csv=os.path.join(OUTDIR_LINEAR, "lp_KR_POST.csv"),
        out_fig=os.path.join(OUTDIR_LINEAR, "lp_KR_POST.png"),
    )
    post_irfs["KR"] = (h2, kr_post)

    plot_two_overlay(
        h, kr_pre, kr_post, MACRO_VARS,
        labelA="Pre-2008", labelB="Post-2008",
        title_prefix="KR | Linear LP | EBP",
        out_fig=os.path.join(OUTDIR_LINEAR, "overlay_KR_PRE_vs_POST.png"),
    )

    # POST-only overlay WX vs KR
    if ("WX" in post_irfs) and ("KR" in post_irfs):
        h_wx, wxp = post_irfs["WX"]
        h_kr, krp = post_irfs["KR"]
        h_use = h_wx if (len(h_wx) == len(h_kr) and np.allclose(h_wx, h_kr)) else h_wx
        plot_two_overlay(
            h_use, wxp, krp, MACRO_VARS,
            labelA="Wu–Xia (POST)", labelB="Krippner (POST)",
            title_prefix="POST: WX vs KR | Linear LP | EBP",
            out_fig=os.path.join(OUTDIR_LINEAR, "overlay_POST_WX_vs_KR.png"),
        )

    # PRE-only AD + Hybrid + 4-shock figure
    if ad_col is not None:
        _, ad_pre, _ = lp_linear(
            df_pre, ad_col, p_lags=P_LAGS_PRE, tag="LP_LINEAR_AD_PRE_EBP",
            out_csv=os.path.join(OUTDIR_LINEAR, "lp_AD_PRE.csv"),
            out_fig=os.path.join(OUTDIR_LINEAR, "lp_AD_PRE.png"),
        )
        pre_irfs["AD"] = ad_pre

    if hy_col is not None:
        _, hy_pre, _ = lp_linear(
            df_pre, hy_col, p_lags=P_LAGS_PRE, tag="LP_LINEAR_HYBRID_PRE_EBP",
            out_csv=os.path.join(OUTDIR_LINEAR, "lp_HYBRID_PRE.csv"),
            out_fig=os.path.join(OUTDIR_LINEAR, "lp_HYBRID_PRE.png"),
        )
        pre_irfs["Hybrid"] = hy_pre

    if set(pre_irfs.keys()) >= {"WX", "KR", "AD", "Hybrid"}:
        plot_four_pre(
            h, MACRO_VARS,
            {
                "WX":     pre_irfs["WX"],
                "KR":     pre_irfs["KR"],
                "AD":     pre_irfs["AD"],
                "Hybrid": pre_irfs["Hybrid"],
            },
            title_prefix="PRE: 4 shocks | Linear LP | EBP (median lines)",
            out_fig=os.path.join(OUTDIR_LINEAR, "PRE_4shock_comparison.png"),
        )
    else:
        print("⚠️ Skipping PRE 4-shock figure. Have:", list(pre_irfs.keys()))

    # -------------------------
    # NON-LINEAR / STATE-DEPENDENT LP
    # -------------------------
    shocks_for_split_nl = [(SHOCK_WX_COL, "WX"), (SHOCK_KR_COL, "KR")]
    if ad_col is not None:
        shocks_for_split_nl.append((ad_col, "AD"))
    if hy_col is not None:
        shocks_for_split_nl.append((hy_col, "Hybrid"))

    for shock_col, lab in shocks_for_split_nl:
        lp_nonlinear_split(
            df_full    = df.loc[df.index <= sample_end].copy(),
            shock_col  = shock_col,
            p_lags     = P_LAGS_PRE,
            tag        = f"LP_SPLIT_NL_{lab}_FULL_EBP",
            out_csv    = os.path.join(OUTDIR_NL, f"split_nl_lp_{lab}_FULL.csv"),
            out_fig    = os.path.join(OUTDIR_NL, f"split_nl_lp_{lab}_FULL.png"),
            split_date = SPLIT_MS,
        )

    print("\n✅ DONE.")
    print("Split-NL LP runs on FULL sample; produces PRE and POST IRFs from one regression per horizon.")

if __name__ == "__main__":
    main()