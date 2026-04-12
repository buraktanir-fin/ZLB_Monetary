"""
================================================================================
lp_results.py
================================================================================
Thesis:  "When Rates Hit Zero: Identifying Monetary Shocks with Shadow Rates"
Author:  Burak Tanir
Date:    April 2026

Purpose
-------
Estimates Local Projection (LP) impulse response functions for all four
monetary policy shock series and produces the robustness figures reported
in Section 7 of the thesis (Alternative Model: Local Projection Robustness).
The LP results are compared against the BVAR baseline to confirm that the
main findings are robust to the choice of estimation method.

The script implements two LP specifications:

    Linear LP (Section 7.1 and 7.2)
        For each shock series and each period (PRE / POST), a separate
        LP regression is estimated:
            y_{t+h} = alpha + beta_h * shock_t + controls_t + e_{t+h}
        where controls include p lags of all variables. beta_h traces the
        impulse response at horizon h. HAC (Newey-West) standard errors
        are computed with a horizon-dependent lag truncation rule.

    Non-linear State-Dependent LP (split specification)
        A single LP regression is estimated over the full sample with
        the shock interacted with a post-2008 dummy:
            y_{t+h} = alpha + b_pre * (shock * (1-D)) + b_post * (shock * D)
                     + controls + e_{t+h}
        where D = 1 after November 2008. This yields pre- and post-2008
        impulse responses from a single regression, enabling a formal
        test of regime differences via the coefficient on (b_post - b_pre).

Methodological details
-----------------------
    Estimator        : OLS with HAC (Newey-West) standard errors
    Lag truncation   : h+1 (horizon-dependent, following Ramey 2016)
    Shock normalisation: divided by full-sample standard deviation before
                         regression (NORMALIZE_SHOCK_TO_1STD = True)
    Sign convention  : y1 coefficient at h=0 forced positive (contractionary)
    Confidence bands : ±1 HAC standard error (approximately 68% band)
    Winsorisation    : 0.1th–99.9th percentile before estimation

Sample periods
--------------
    Pre-2008  : January 1982 – October 2008 (p = 12 lags)
    Post-2008 : November 2008 – December 2019 (133 months, p = 6 lags)
                p = 6 chosen so that T - p - H = 133 - 6 - 48 = 79 > k

Shock series estimated
-----------------------
    WX      — Wu-Xia shadow rate shock (main post-2008 instrument)
    KR      — Krippner shadow rate shock (post-2008 robustness)
    AD      — Aruoba-Drechsel (2024) shock (pre-2008 benchmark)
    Hybrid  — Replicated FFR-based shock (pre-2008 benchmark)

Output files
------------
    lp_outputs_EBP_linear/
        lp_{shock}_{period}.csv     — LP coefficient table (h, b, lo, hi)
        lp_{shock}_{period}.png     — LP figure per shock and period
        overlay_{shock}_PRE_vs_POST.png   — pre vs post overlay
        overlay_POST_WX_vs_KR.png         — post-2008 WX vs KR overlay
        PRE_4shock_comparison.png         — four-shock pre-2008 figure

    lp_outputs_EBP_nonlinear_split/
        split_nl_lp_{shock}_FULL.csv  — split-NL LP table (pre, post, diff)
        split_nl_lp_{shock}_FULL.png  — split-NL LP figure

Input
-----
    outputs/bvar_monthly_dataset_FULL_with_EBP_with_ALL_meeting_shockSUM.csv
        Produced by bvar_dataset_builder.py. Must contain DATE index,
        all macro variable columns, and all shock columns.

Dependencies
------------
    os, time  (standard library)
    numpy, pandas, matplotlib  — pip install numpy pandas matplotlib
    statsmodels                — pip install statsmodels

Usage
-----
    python lp_results.py

    Run from the same working directory as bvar_dataset_builder.py.
    Both output folders are created automatically.
================================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac


# ============================================================
# SECTION 1 — PATH CONFIGURATION
# ============================================================
# Input dataset is read from the outputs/ folder produced by
# bvar_dataset_builder.py. LP outputs are written to two separate
# folders: one for the linear LP and one for the non-linear split LP.

BASE_DIR    = os.getcwd()
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FULL_DATA_FILENAME = "bvar_monthly_dataset_FULL_with_EBP_with_ALL_meeting_shockSUM.csv"
DATA_PATH = os.path.join(OUTPUTS_DIR, FULL_DATA_FILENAME)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find source data file:\n{DATA_PATH}\n"
        f"Check FULL_DATA_FILENAME or that the file is in outputs/."
    )

# Output folders — one per LP specification
OUTDIR_LINEAR = os.path.join(BASE_DIR, "lp_outputs_EBP_linear")
OUTDIR_NL     = os.path.join(BASE_DIR, "lp_outputs_EBP_nonlinear_split")
os.makedirs(OUTDIR_LINEAR, exist_ok=True)
os.makedirs(OUTDIR_NL, exist_ok=True)

print(f"Using source data: {DATA_PATH}")
print(f"Linear outputs   : {OUTDIR_LINEAR}")
print(f"Nonlinear outputs: {OUTDIR_NL}")


# ============================================================
# SECTION 2 — COLUMN IDENTIFICATION
# ============================================================
# Wu-Xia and Krippner shock column names are hardcoded (fixed by
# bvar_dataset_builder.py). AD and Hybrid columns are detected
# robustly via fuzzy matching to handle naming variants.

DATE_COL_CANDIDATES = ["MS", "DATE", "Date", "date"]

# Shadow rate shock columns — hardcoded names
SHOCK_WX_COL = "shock_Wu_meeting_sum"
SHOCK_KR_COL = "shock_Kr_meeting_sum"

# AD and Hybrid — detected case-insensitively
AD_CANDIDATES     = ["AD", "shock_AD_meeting_sum", "ad_meeting_sum", "aruoba", "aruoba_drechsel"]
HYBRID_CANDIDATES = ["hybrid", "shock_hybrid_meeting_sum", "hybrid_meeting_sum", "policy_hybrid"]


# ============================================================
# SECTION 3 — SAMPLE CONFIGURATION
# ============================================================
# PRE sample  : full history up to October 2008
# POST sample : November 2008 – December 2019 (133 months)
#
# P_LAGS_POST = 6 is chosen so that the effective regression sample
# is large enough to estimate 44 regressors:
#   T - p - H = 133 - 6 - 48 = 79 observations > 44 regressors ✅

SPLIT_MS    = "2008-11-01"   # first POST month
SAMPLE_END  = "2019-12-01"   # last month of the thesis sample

HORIZON     = 48   # maximum IRF horizon in months
P_LAGS_PRE  = 12   # VAR lag order for the pre-2008 sample
P_LAGS_POST = 6    # VAR lag order for the post-2008 sample (see note above)


# ============================================================
# SECTION 4 — ESTIMATION SETTINGS
# ============================================================

# Newey-West HAC lag truncation rule.
# "h+1" : lags = h + 1 at horizon h (Ramey 2016 recommended approach)
# "h"   : lags = h
# "fixed": fixed at NW_LAGS_FIXED regardless of horizon
NW_LAGS_RULE  = "h+1"
NW_LAGS_FIXED = 12

# Winsorisation quantiles applied before estimation
CLIP_Q_LO, CLIP_Q_HI = 0.001, 0.999

# Divide shock by its full-sample standard deviation before regression
# so that beta_h is interpretable as a one-standard-deviation impulse
NORMALIZE_SHOCK_TO_1STD  = True

# Flip all IRFs if y1 coefficient at h=0 is negative (contractionary convention)
SIGN_ALIGN_TO_Y1_POSITIVE = True

# BVAR variable set — EBP included, matching bvar_irf_estimation.py
MACRO_VARS = ["y1", "log_sp500", "log_rgdp", "log_pgdp", "unemp", "EBP"]


# ============================================================
# SECTION 5 — UTILITY FUNCTIONS
# ============================================================

def read_csv_auto(path: str) -> pd.DataFrame:
    """
    Read a CSV file with automatic delimiter detection.

    Detects whether the file uses comma or semicolon as delimiter by
    comparing counts in the header row. Handles both formats without
    requiring the user to specify the separator.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(path, sep=sep)


def detect_date_col(df):
    """
    Find the date column by searching DATE_COL_CANDIDATES in order.

    Returns the first matching column name, or None if not found.
    """
    for c in DATE_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def to_datetime_index(df):
    """
    Convert a date column (or the existing index) to a DatetimeIndex.

    Parses, sorts, and sets the date column as the DataFrame index.
    Falls back to parsing the existing index if no date column is found.
    """
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
    """
    Winsorise a numeric Series at the specified quantiles.

    Limits the influence of outliers on LP estimates without removing
    observations. Applied to all variables before estimation.
    """
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.quantile([qlo, qhi])
    return s.clip(lo, hi)


def detect_col_fuzzy(df, candidates):
    """
    Robustly find a column in df matching any candidate name.

    Detection priority:
        1. Exact match (case-sensitive)
        2. Exact match (case-insensitive)
        3. Contains-match (case-insensitive substring)
    """
    cols       = list(df.columns)
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
    """
    Compute the Newey-West HAC lag truncation for horizon h.

    Applies the rule specified in NW_LAGS_RULE:
        "h+1"  : lags = h + 1  (recommended by Ramey 2016)
        "h"    : lags = h
        "fixed": lags = NW_LAGS_FIXED

    Parameters
    ----------
    h : int  Current LP horizon.

    Returns
    -------
    int  Number of HAC lags to use.
    """
    if NW_LAGS_RULE == "h":
        return max(0, h)
    if NW_LAGS_RULE == "h+1":
        return max(0, h + 1)
    if NW_LAGS_RULE == "fixed":
        return int(max(0, NW_LAGS_FIXED))
    raise ValueError(f"Unknown NW_LAGS_RULE: {NW_LAGS_RULE}")


def hac_se_safe(results, maxlags):
    """
    Compute HAC standard errors safely, returning NaN if infeasible.

    Checks that the sample has enough observations to estimate the
    HAC covariance matrix (nobs > k_params + 1). Caps the number of
    HAC lags at nobs - 1 to avoid rank deficiency.

    Parameters
    ----------
    results : statsmodels OLS results object.
    maxlags : int  Maximum number of Newey-West lags.

    Returns
    -------
    np.ndarray  HAC standard errors; NaN where computation is infeasible.
    """
    nobs     = int(results.nobs)
    k_params = int(results.df_model) + 1
    if nobs <= k_params + 1:
        return np.full(len(results.params), np.nan)
    nlags = int(max(0, min(maxlags, nobs - 1)))
    V  = cov_hac(results, nlags=nlags)
    se = np.sqrt(np.diag(V))
    return se


# ============================================================
# SECTION 6 — LINEAR LP ESTIMATION
# ============================================================

def lp_linear(df_sub: pd.DataFrame, shock_col: str, p_lags: int, tag: str,
              out_csv: str, out_fig: str):
    """
    Estimate linear Local Projection impulse responses.

    For each horizon h in [0, HORIZON], estimates:
        y_{t+h} = alpha + beta_h * shock_t
                  + sum_{l=1}^{p} (gamma_l * [shock, macro]_{t-l})
                  + e_{t+h}

    where the shock is optionally normalised to 1 standard deviation.
    HAC standard errors use the h+1 lag truncation rule by default.

    The impulse response at horizon h is the OLS coefficient beta_h.
    Confidence bands are ±1 HAC standard error (approximately 68%).

    Parameters
    ----------
    df_sub    : pd.DataFrame  Subsample (PRE or POST).
    shock_col : str           Name of the shock column.
    p_lags    : int           Number of control lags.
    tag       : str           Label for console output and figure titles.
    out_csv   : str           Output path for the LP coefficient CSV.
    out_fig   : str           Output path for the LP figure PNG.

    Returns
    -------
    hgrid     : np.ndarray    Horizon array [0, 1, ..., HORIZON].
    irf_stats : tuple         (b, lo, hi) coefficient arrays, each (H+1 × k).
    MACRO_VARS: list of str   Variable names.
    """
    t0 = time.time()
    d  = df_sub.copy()

    # Validate required columns
    needed  = [shock_col] + MACRO_VARS
    missing = [c for c in needed if c not in d.columns]
    if missing:
        raise ValueError(f"[{tag}] Missing columns: {missing}\nAvailable: {list(d.columns)}")

    # Winsorise all variables and fill shock NaNs with zero
    for c in needed:
        d[c] = clip_series(d[c])
    d[shock_col] = d[shock_col].fillna(0.0)

    # Normalise shock to 1 standard deviation for interpretable beta_h
    shock_std = float(d[shock_col].std(ddof=0))
    if NORMALIZE_SHOCK_TO_1STD and shock_std > 0:
        d["_shock_used"] = d[shock_col] / shock_std
        shock_label = f"{shock_col} (1-std normalized)"
    else:
        d["_shock_used"] = d[shock_col]
        shock_label = f"{shock_col} (raw units)"

    # Build lag matrix: p lags of shock and all macro variables
    controls_base = ["_shock_used"] + MACRO_VARS
    for L in range(1, p_lags + 1):
        for c in controls_base:
            d[f"{c}_L{L}"] = d[c].shift(L)

    H     = HORIZON
    hgrid = np.arange(H + 1)

    # Pre-allocate coefficient and standard error arrays
    b  = np.full((H + 1, len(MACRO_VARS)), np.nan)
    se = np.full((H + 1, len(MACRO_VARS)), np.nan)

    # ── Horizon-by-horizon LP regression ─────────────────────────────────────
    for hi in range(H + 1):
        nw    = nw_lags_for_h(hi)
        Xcols = ["_shock_used"] + [f"{c}_L{L}" for L in range(1, p_lags + 1) for c in controls_base]
        X     = d[Xcols].copy()

        for j, y in enumerate(MACRO_VARS):
            ydep = d[y].shift(-hi)      # h-step ahead dependent variable
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

            # HAC standard error for the shock coefficient (position 1)
            se_vec   = hac_se_safe(res, maxlags=nw)
            b[hi, j] = res.params[1]
            se[hi, j] = se_vec[1] if np.isfinite(se_vec[1]) else np.nan

    # Sign alignment: flip all IRFs if y1 impact at h=0 is negative
    if SIGN_ALIGN_TO_Y1_POSITIVE:
        y1_idx = MACRO_VARS.index("y1")
        if np.isfinite(b[0, y1_idx]) and b[0, y1_idx] < 0:
            b = -b

    # Confidence bands: ±1 HAC standard error
    lo = b - se
    hi = b + se

    # Save LP coefficient table to CSV
    out = pd.DataFrame({"h": hgrid})
    for j, y in enumerate(MACRO_VARS):
        out[f"{y}_b"]  = b[:, j]
        out[f"{y}_lo"] = lo[:, j]
        out[f"{y}_hi"] = hi[:, j]
    out.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"Saved LP table → {out_csv}")

    # Save LP figure
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
    print(f"Saved LP figure → {out_fig}")
    print(f"   [{tag}] shock scaling: {shock_label} | p={p_lags} | elapsed {time.time()-t0:.1f}s")

    return hgrid, (b, lo, hi), MACRO_VARS


# ============================================================
# SECTION 7 — NON-LINEAR STATE-DEPENDENT LP (SPLIT SPECIFICATION)
# ============================================================

def lp_nonlinear_split(df_full: pd.DataFrame, shock_col: str, p_lags: int, tag: str,
                       out_csv: str, out_fig: str, split_date: str):
    """
    Estimate a state-dependent Local Projection over the full sample.

    Interacts the shock with a post-2008 regime dummy D (D=1 after
    split_date) so that a single regression per horizon yields separate
    pre- and post-2008 impulse responses:

        y_{t+h} = alpha
                  + b_pre  * shock_t * (1 - D_t)
                  + b_post * shock_t * D_t
                  + controls + e_{t+h}

    The regime difference (b_post - b_pre) and its HAC standard error
    are also computed and saved, enabling a formal test of whether the
    two regimes produce statistically different impulse responses.

    Advantages over running two separate LPs:
        - Uses a common coefficient on the lagged controls
        - Enables a direct test of H0: b_pre = b_post
        - Avoids sample-selection bias in standard error estimation

    Sign alignment is based on the pre-2008 y1 coefficient at h=0
    (the conventional policy era anchor).

    Parameters
    ----------
    df_full    : pd.DataFrame  Full sample dataset.
    shock_col  : str           Name of the shock column.
    p_lags     : int           Number of control lags.
    tag        : str           Label for console output and figure titles.
    out_csv    : str           Output path for the coefficient CSV.
    out_fig    : str           Output path for the figure PNG.
    split_date : str           Date string defining the regime boundary
                               (e.g. '2008-11-01').

    Returns
    -------
    hgrid     : np.ndarray    Horizon array [0, 1, ..., HORIZON].
    irf_stats : tuple         ((b_pre, lo_pre, hi_pre),
                               (b_post, lo_post, hi_post),
                               (diff, lo_diff, hi_diff))
    MACRO_VARS: list of str   Variable names.
    """
    t0 = time.time()
    d  = df_full.copy()

    # Validate required columns
    needed  = [shock_col] + MACRO_VARS
    missing = [c for c in needed if c not in d.columns]
    if missing:
        raise ValueError(f"[{tag}] Missing columns: {missing}\nAvailable: {list(d.columns)}")

    # Winsorise and fill shock NaNs
    for c in needed:
        d[c] = clip_series(d[c])
    d[shock_col] = d[shock_col].fillna(0.0)

    # Post-2008 regime dummy: D=1 from split_date onwards
    split  = pd.to_datetime(split_date)
    d["_D"] = (d.index >= split).astype(int)

    # Normalise shock on full-sample standard deviation
    shock_std = float(d[shock_col].std(ddof=0))
    if NORMALIZE_SHOCK_TO_1STD and shock_std > 0:
        d["_shock_used"] = d[shock_col] / shock_std
        shock_label = f"{shock_col} (1-std normalized, FULL-sample)"
    else:
        d["_shock_used"] = d[shock_col]
        shock_label = f"{shock_col} (raw units)"

    # Regime-interacted shock terms
    d["_shock_post"] = d["_shock_used"] * d["_D"]         # post-2008 component
    d["_shock_pre"]  = d["_shock_used"] * (1 - d["_D"])   # pre-2008 component

    # Build lag matrix
    controls_base = ["_shock_used"] + MACRO_VARS
    for L in range(1, p_lags + 1):
        for c in controls_base:
            d[f"{c}_L{L}"] = d[c].shift(L)

    H     = HORIZON
    hgrid = np.arange(H + 1)

    # Pre-allocate arrays for pre, post, and difference coefficients
    b_pre   = np.full((H + 1, len(MACRO_VARS)), np.nan)
    b_post  = np.full((H + 1, len(MACRO_VARS)), np.nan)
    se_pre  = np.full((H + 1, len(MACRO_VARS)), np.nan)
    se_post = np.full((H + 1, len(MACRO_VARS)), np.nan)
    se_diff = np.full((H + 1, len(MACRO_VARS)), np.nan)
    diff    = np.full((H + 1, len(MACRO_VARS)), np.nan)

    # ── Horizon-by-horizon split-NL LP regression ─────────────────────────────
    for hi in range(H + 1):
        nw    = nw_lags_for_h(hi)
        # _shock_pre (position 1) and _shock_post (position 2) in regressor matrix
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

            # b_pre = params[1], b_post = params[2]
            b_pre[hi, j]  = params[1]
            b_post[hi, j] = params[2]

            se_pre[hi, j]  = np.sqrt(V[1, 1]) if np.isfinite(V[1, 1]) else np.nan
            se_post[hi, j] = np.sqrt(V[2, 2]) if np.isfinite(V[2, 2]) else np.nan

            # Regime difference: b_post - b_pre with joint HAC variance
            diff[hi, j]   = params[2] - params[1]
            var_diff       = V[2, 2] + V[1, 1] - 2.0 * V[1, 2]
            se_diff[hi, j] = np.sqrt(var_diff) if (np.isfinite(var_diff) and var_diff >= 0) else np.nan

    # Sign alignment anchored on pre-2008 y1 impact at h=0
    if SIGN_ALIGN_TO_Y1_POSITIVE:
        y1_idx = MACRO_VARS.index("y1")
        anchor = b_pre[0, y1_idx]
        if np.isfinite(anchor) and anchor < 0:
            b_pre, b_post, diff = -b_pre, -b_post, -diff

    # Confidence bands: ±1 HAC standard error
    lo_pre,  hi_pre  = b_pre  - se_pre,  b_pre  + se_pre
    lo_post, hi_post = b_post - se_post, b_post + se_post
    lo_diff, hi_diff = diff   - se_diff, diff   + se_diff

    # Save split-NL LP coefficient table to CSV
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
    print(f"Saved Split-NL LP table → {out_csv}")

    # Save split-NL LP figure
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
    print(f"Saved Split-NL LP figure → {out_fig}")
    print(f"   [{tag}] state=pre/post split @ {split_date} | shock scaling: {shock_label} | p={p_lags} | elapsed {time.time()-t0:.1f}s")

    return hgrid, ((b_pre, lo_pre, hi_pre), (b_post, lo_post, hi_post), (diff, lo_diff, hi_diff)), MACRO_VARS


# ============================================================
# SECTION 8 — OVERLAY PLOTTING FUNCTIONS
# ============================================================

def plot_two_overlay(h, A, B, macro_vars, labelA, labelB, title_prefix, out_fig):
    """
    Plot two LP IRF sets in the same panels for direct comparison.

    Series A is a solid line; series B is dashed. Both ±1 HAC standard
    error bands are shown with 12% opacity. Used for:
        - Pre vs Post overlays for each shadow rate shock
        - Post-2008 Wu-Xia vs Krippner overlay

    Parameters
    ----------
    h           : np.ndarray   Horizon array.
    A, B        : tuple        (b, lo, hi) each (H+1 × k).
    macro_vars  : list of str  Variable names.
    labelA, labelB : str       Legend labels.
    title_prefix : str         Subplot title prefix.
    out_fig      : str         Output PNG path.
    """
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
    print(f"Saved overlay → {out_fig}")


def plot_four_pre(h, macro_vars, irf_dict, title_prefix, out_fig):
    """
    Plot median LP IRFs from four pre-2008 shock series in the same panels.

    Displays coefficient lines only (no bands) with distinct line styles
    for unambiguous identification. Used for the four-shock pre-2008
    comparison in Section 7.

    Parameters
    ----------
    h           : np.ndarray  Horizon array.
    macro_vars  : list of str Variable names.
    irf_dict    : dict        {label: (b, lo, hi)} for each shock.
    title_prefix : str        Subplot title prefix.
    out_fig      : str        Output PNG path.
    """
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
    print(f"Saved 4-shock PRE figure → {out_fig}")


# ============================================================
# SECTION 9 — MAIN EXECUTION
# ============================================================

def main():
    """
    Main execution function orchestrating all LP estimations.

    Execution order:
        1.  Load and index the monthly dataset
        2.  Validate EBP and required shock columns are present
        3.  Detect AD and Hybrid columns robustly
        4.  Split dataset into PRE and POST subsamples
        5.  Linear LP — Wu-Xia: PRE, POST, pre vs post overlay
        6.  Linear LP — Krippner: PRE, POST, pre vs post overlay
        7.  Linear LP — POST-only Wu-Xia vs Krippner overlay
        8.  Linear LP — AD and Hybrid (PRE only); four-shock comparison
        9.  Non-linear split LP — all four shocks over full sample
    """
    df = read_csv_auto(DATA_PATH)
    df = to_datetime_index(df)

    # Confirm EBP is present — required for this specification
    if "EBP" not in df.columns:
        raise ValueError(
            f"EBP column not found in dataset.\nAvailable: {list(df.columns)}"
        )
    print("EBP column retained for LP estimation.")

    # Validate all required shock and macro columns are present
    required = [SHOCK_WX_COL, SHOCK_KR_COL] + MACRO_VARS
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable: {list(df.columns)}")

    # Detect AD and Hybrid columns
    ad_col = detect_col_fuzzy(df, AD_CANDIDATES)
    hy_col = detect_col_fuzzy(df, HYBRID_CANDIDATES)
    print(f"AD detected: {ad_col} | Hybrid detected: {hy_col}")

    # Split into PRE and POST subsamples
    split      = pd.to_datetime(SPLIT_MS)
    sample_end = pd.to_datetime(SAMPLE_END)
    df_pre  = df.loc[df.index < split].copy()
    df_post = df.loc[(df.index >= split) & (df.index <= sample_end)].copy()
    print(f"Post-2008 sample: {df_post.index.min().date()} → "
          f"{df_post.index.max().date()} ({len(df_post)} months)")

    # ── Linear LP ────────────────────────────────────────────────────────────
    pre_irfs  = {}
    post_irfs = {}

    # ── Wu-Xia: PRE and POST ─────────────────────────────────────────────────
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

    # ── Krippner: PRE and POST ────────────────────────────────────────────────
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

    # ── POST-only overlay: Wu-Xia vs Krippner ────────────────────────────────
    if ("WX" in post_irfs) and ("KR" in post_irfs):
        h_wx, wxp = post_irfs["WX"]
        h_kr, krp = post_irfs["KR"]
        h_use = h_wx if (len(h_wx) == len(h_kr) and np.allclose(h_wx, h_kr)) else h_wx
        plot_two_overlay(
            h_use, wxp, krp, MACRO_VARS,
            labelA="Wu-Xia (POST)", labelB="Krippner (POST)",
            title_prefix="POST: WX vs KR | Linear LP | EBP",
            out_fig=os.path.join(OUTDIR_LINEAR, "overlay_POST_WX_vs_KR.png"),
        )

    # ── PRE-only: AD and Hybrid; four-shock comparison ────────────────────────
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
        print("Skipping PRE 4-shock figure. Have:", list(pre_irfs.keys()))

    # ── Non-linear state-dependent LP — all four shocks over full sample ──────
    # Runs on the full sample (up to SAMPLE_END) with pre/post interaction terms
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

    print("\nDone.")
    print("Split-NL LP runs on FULL sample; produces PRE and POST IRFs from one regression per horizon.")


if __name__ == "__main__":
    main()