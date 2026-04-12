"""
================================================================================
================================================================================
Thesis:  "When Rates Hit Zero: Identifying Monetary Shocks with Shadow Rates"
Author:  Burak Tanir
Date:    April 2026

Purpose
-------
Estimates Bayesian VAR impulse response functions for three monetary policy
shock series — AD, FFR, and Wu-Xia — over an identical sample window of
March 1990 to October 2008, and produces a single comparison figure
displaying all three sets of IRFs in the same panels.

This is the sample-controlled robustness exercise reported in Section 6.1
of the thesis (Alternative Shock Series, Pre-2008 Period). By holding the
estimation sample constant across all three instruments, any differences
between the Wu-Xia responses and the FFR-based responses (AD, FFR) reflect
the properties of the shock series themselves rather than differences in
sample length or coverage.

Economic rationale
------------------
The full pre-2008 sample runs from January 1982 to October 2008, but the
Wu-Xia shadow rate is not available before March 1990. Running the full-sample
BVAR with Wu-Xia would therefore compare Wu-Xia over 1990-2008 against AD
and FFR over 1982-2008 — confounding instrument differences with sample
differences. This script eliminates that confound by restricting all three
series to the 1990-2008 window.

Three shock series estimated
-----------------------------
    AD      — original Aruoba-Drechsel (2024) shock; detected by fuzzy
              column name matching (AD_CANDIDATES)
    FFR     — replicated FFR-based (Hybrid) shock; detected by fuzzy
              column name matching (HYBRID_CANDIDATES); labeled as "FFR"
    Wu-Xia  — shadow rate shock; column name hardcoded as SHOCK_WX_COL

BVAR specification
------------------
    Variable order : shock, y1, log_sp500, log_rgdp, log_pgdp, unemp, EBP
    Lag order      : 12 months (P_LAGS_PRE)
    Horizon        : 48 months (HORIZON)
    Posterior draws: 4,000 (DRAWS); only stable draws retained
    Prior          : Minnesota-style (Normal-Inverse-Wishart)
    Credible bands : 16th–84th percentile (68% posterior credible interval)
    Sign convention: median y1 impact at h=0 normalised to positive

Sample window
-------------
    Start : March 1990  (SAMPLE_START — first date Wu-Xia is available)
    End   : November 2008 (SAMPLE_END — consistent with pre/post split)

Input
-----
    outputs/bvar_monthly_dataset_FULL_with_EBP_with_ALL_meeting_shockSUM.csv
        Produced by bvar_dataset_builder.py. Must contain DATE index,
        all macro variable columns, and shock columns for AD, Hybrid,
        and Wu-Xia.

Output
------
    irf_outputs_EBP_final/
        Analysis1_1990_AD_FFR_WuXia_p{p}_H{H}_EBP.png
            One figure with six panels (one per macro variable), each
            showing three IRF series (AD, FFR, Wu-Xia) with 68% credible
            bands over the 1990-2008 sample.

Dependencies
------------
    numpy, pandas, matplotlib  — pip install numpy pandas matplotlib
    Python 3.7+ compatible

Usage
-----
    python bvar_1990_2008_comparison.py

    Must be run after bvar_dataset_builder.py has produced the input
    dataset. Run from the same working directory.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time


# ============================================================
# SECTION 1 — PATH CONFIGURATION
# ============================================================
# Input dataset is read from the outputs/ folder produced by
# bvar_dataset_builder.py. The IRF figure is saved to the same
# irf_outputs_EBP_final/ folder used by bvar_irf_estimation.py.

BASE_DIR    = os.getcwd()
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

FULL_DATA_FILENAME = "bvar_monthly_dataset_FULL_with_EBP_with_ALL_meeting_shockSUM.csv"
DATA_PATH = os.path.join(OUTPUTS_DIR, FULL_DATA_FILENAME)

# Output folder shared with bvar_irf_estimation.py
OUTDIR = os.path.join(BASE_DIR, "irf_outputs_EBP_final")
os.makedirs(OUTDIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find source data file:\n{DATA_PATH}\n"
        f"Check FULL_DATA_FILENAME or that the file is in outputs/."
    )

print(f"Using source data: {DATA_PATH}")
print(f"Outputs folder   : {OUTDIR}")


# ============================================================
# SECTION 2 — COLUMN IDENTIFICATION
# ============================================================
# Date column candidates searched in order of priority.
# Wu-Xia shock column name is hardcoded (fixed by shock_comparison.py).
# AD and Hybrid column names are detected robustly to handle variants.

DATE_COL_CANDIDATES = ["MS", "DATE", "Date", "date"]

# Wu-Xia shock column — fixed name from bvar_dataset_builder.py output
SHOCK_WX_COL  = "shock_Wu_meeting_sum"
SHOCK_KR_COL  = "shock_Kr_meeting_sum"

# AD and Hybrid detected case-insensitively (labeled as "FFR" in plots)
AD_CANDIDATES     = ["AD", "shock_AD_meeting_sum", "ad_meeting_sum", "aruoba", "aruoba_drechsel"]
HYBRID_CANDIDATES = ["hybrid", "shock_hybrid_meeting_sum", "hybrid_meeting_sum", "policy_hybrid"]


# ============================================================
# SECTION 3 — SAMPLE WINDOW
# ============================================================
# The window is chosen to align all three shock series on identical dates.
# SAMPLE_START: March 1990 — first date Wu-Xia shadow rate is available
# SAMPLE_END  : November 2008 — consistent with the pre/post ZLB split

SAMPLE_START = "1990-03-01"   # start: Wu-Xia availability
SAMPLE_END   = "2008-11-01"   # end:   pre-2008 split (exclusive)


# ============================================================
# SECTION 4 — BVAR ESTIMATION SETTINGS
# ============================================================
# All settings are identical to bvar_irf_estimation.py to ensure
# that differences in IRFs across the three shock series reflect
# instrument characteristics and not specification differences.

HORIZON    = 48    # IRF horizon in months
DRAWS      = 4000  # posterior draws attempted; only stable draws kept
P_LAGS_PRE = 12    # VAR lag order for the 1990-2008 sample

# Minnesota prior hyperparameters
DF0      = 10   # Inverse-Wishart prior degrees of freedom
S0_SCALE = 0.2  # Inverse-Wishart prior scale (S0 = scale^2 * I)
LAMBDA_B = 0.2  # Normal prior tightness on VAR coefficients

# Credible interval percentiles (68% posterior credible interval)
CI_LO, CI_HI = 16, 84

# Winsorisation quantiles applied before estimation
CLIP_Q_LO, CLIP_Q_HI = 0.001, 0.999

# Estimation options
STANDARDIZE                     = True   # z-score standardise before estimation
NORMALIZE_SHOCK_IMPACT_TO_ONE   = True   # normalise h=0 shock impact to 1 std
BACKTRANSFORM_TO_ORIGINAL_UNITS = True   # convert IRFs back to original units
SIGN_ALIGN_TO_Y1_POSITIVE       = True   # flip if median y1 impact at h=0 < 0

# Set True if log variables need to be multiplied by 100 for thesis units
SCALE_LOGS_BY_100 = False
LOG_VARS_TO_SCALE = ["log_sp500", "log_rgdp", "log_pgdp"]


# ============================================================
# SECTION 5 — DISPLAY SETTINGS
# ============================================================

DISPLAY_LABELS = {
    "y1":        "1-year bond yield (%)",
    "log_sp500": "S&P500 (100 × log)",
    "log_rgdp":  "Real GDP (100 × log)",
    "unemp":     "Unemployment (%)",
    "log_pgdp":  "GDP deflator (100 × log)",
    "EBP":       "EBP (%)",
}

def pretty_ylabel(v):
    """Return the thesis-standard y-axis label for a BVAR variable."""
    return DISPLAY_LABELS.get(v, v)

# Random number generator seed for reproducibility
SEED = 123
rng  = np.random.default_rng(SEED)


# ============================================================
# SECTION 6 — UTILITY FUNCTIONS: DATA HANDLING
# ============================================================

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

    Limits the influence of outliers on posterior estimates without
    removing observations from the sample.
    """
    lo, hi = s.quantile([qlo, qhi])
    return s.clip(lo, hi)


def zscore_with_stats(X):
    """
    Z-score standardise a matrix and return standardisation statistics.

    Returns the standardised matrix, column means, and population
    standard deviations for use in back-transforming IRFs.
    """
    X  = np.asarray(X, float)
    m  = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return (X - m) / sd, m, sd


def detect_col_fuzzy(df, candidates):
    """
    Robustly find a column in df matching any candidate name.

    Detection priority:
        1. Exact match (case-sensitive)
        2. Exact match (case-insensitive)
        3. Contains-match (case-insensitive substring)

    Used to detect AD and Hybrid columns across naming variants.
    """
    cols      = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for cand in candidates:
        for lc, orig in lower_map.items():
            if cand.lower() in lc:
                return orig
    return None


# ============================================================
# SECTION 7 — BVAR ALGEBRA UTILITIES
# ============================================================

def make_lag_matrix(Y, p):
    """
    Construct the regressor matrix of lagged endogenous variables.

    Stacks lags 1 through p horizontally into a ((T-p) × k*p) matrix.
    """
    T, k = Y.shape
    return np.hstack([Y[p - L : T - L, :] for L in range(1, p + 1)])


def companion_matrix(A_list):
    """
    Build the companion form matrix from VAR coefficient matrices.

    Used for stability checking and multi-step IRF computation.
    """
    k   = A_list[0].shape[0]
    p   = len(A_list)
    top = np.hstack(A_list)
    if p == 1:
        return top
    I     = np.eye(k * (p - 1))
    zeros = np.zeros((k * (p - 1), k))
    return np.vstack([top, np.hstack([I, zeros])])


def is_stable(A_list):
    """
    Return True if the spectral radius of the companion matrix < 0.9999.

    Unstable draws are discarded to ensure IRFs do not explode.
    """
    return np.max(np.abs(np.linalg.eigvals(companion_matrix(A_list)))) < 0.9999


def wishart_rnd(df, V):
    """
    Sample from a Wishart distribution W(df, V) via Bartlett decomposition.
    """
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
    """
    Sample from an Inverse-Wishart distribution IW(df, S).

    Used to draw the posterior error covariance matrix Sigma in the
    Normal-Inverse-Wishart Gibbs sampler.
    """
    return np.linalg.inv(
        wishart_rnd(df, np.linalg.inv(np.asarray(S, float)))
    )


def sample_matrix_normal(Bbar, V, Sigma):
    """
    Sample from a matrix Normal distribution MN(Bbar, V, Sigma).

    Used to draw the posterior VAR coefficient matrix B.
    """
    Bbar, V, Sigma = [np.asarray(x, float) for x in (Bbar, V, Sigma)]
    Lv = np.linalg.cholesky(V)
    Ls = np.linalg.cholesky(Sigma)
    m, k = Bbar.shape
    Z = rng.normal(size=(m, k))
    return Bbar + (Lv @ Z @ Ls.T)


# ============================================================
# SECTION 8 — BVAR POSTERIOR SAMPLING AND IRF EXTRACTION
# ============================================================

def bvar_draws_irf(Y, p, H, draws, lambda_b=LAMBDA_B,
                   df0=DF0, s0_scale=S0_SCALE, tag=""):
    """
    Draw posterior BVAR IRFs via the Normal-Inverse-Wishart closed-form posterior.

    Identification: Cholesky decomposition of Sigma; the monetary policy
    shock is the first variable in var_order. Only stable posterior draws
    (spectral radius < 0.9999) are retained.

    Parameters
    ----------
    Y        : np.ndarray  (T × k) standardised BVAR data matrix.
    p        : int         VAR lag order.
    H        : int         IRF horizon in months.
    draws    : int         Number of posterior draws to attempt.
    lambda_b : float       Minnesota prior tightness on coefficients.
    df0      : int         Prior degrees of freedom for Inverse-Wishart.
    s0_scale : float       Prior scale for Inverse-Wishart.
    tag      : str         Label for console progress output.

    Returns
    -------
    np.ndarray  (n_stable_draws × H+1 × k) array of IRFs.
    """
    t0   = time.time()
    T, k = Y.shape

    # Construct OLS-style regressor matrix with intercept and p lags
    Xlags   = make_lag_matrix(Y, p)
    Ydep    = Y[p:, :]
    X       = np.hstack([np.ones((T - p, 1)), Xlags])
    ncoef   = X.shape[1]

    # Minnesota prior: B0 = 0 (shrink toward zero), V0 = lambda^2 * I
    B0     = np.zeros((ncoef, k))
    V0_inv = np.eye(ncoef) / (lambda_b ** 2)
    S0     = (s0_scale ** 2) * np.eye(k)

    # Closed-form Normal-Inverse-Wishart posterior
    XtX    = X.T @ X
    XtY    = X.T @ Ydep
    V_post = np.linalg.inv(V0_inv + XtX)
    B_post = V_post @ (V0_inv @ B0 + XtY)

    U       = Ydep - X @ B_post
    S_post  = (S0 + U.T @ U
               + (B_post - B0).T @ V0_inv @ (B_post - B0))
    df_post = df0 + (T - p)

    def parse_A_list(Bmat):
        """Extract list of (k × k) VAR coefficient matrices from a posterior draw."""
        A_list, start = [], 1
        for L in range(p):
            A_list.append(Bmat[start + L*k : start + (L+1)*k, :].T)
        return A_list

    keep_irfs  = []
    milestones = {250, 500, 1000, 2000, 3000, 4000}

    for d in range(draws):
        Sigma  = invwishart_rnd(df_post, S_post)
        Bdraw  = sample_matrix_normal(B_post, V_post, Sigma)
        A_list = parse_A_list(Bdraw)
        if not is_stable(A_list):
            continue

        # Cholesky identification: shock = first column of Cholesky factor
        P      = np.linalg.cholesky(Sigma)
        impact = P[:, 0].copy()
        F      = companion_matrix(A_list)
        J      = np.hstack([np.eye(k), np.zeros((k, k*(p-1)))])

        # Propagate impulse via companion matrix
        irf    = np.zeros((H + 1, k))
        irf[0] = impact
        Fh     = np.eye(k * p)
        for hh in range(1, H + 1):
            Fh      = Fh @ F
            irf[hh] = (J @ Fh @ J.T) @ impact

        keep_irfs.append(irf)
        if len(keep_irfs) in milestones:
            print(f"   [{tag}] kept {len(keep_irfs)} draws | "
                  f"attempt {d+1}/{draws} | "
                  f"elapsed {time.time()-t0:.1f}s")

    if not keep_irfs:
        raise RuntimeError(f"[{tag}] No stable posterior draws kept.")

    print(f"   [{tag}] DONE — kept {len(keep_irfs)} / {draws} | "
          f"elapsed {time.time()-t0:.1f}s")
    return np.array(keep_irfs)


# ============================================================
# SECTION 9 — IRF POST-PROCESSING
# ============================================================

def normalize_and_backtransform(irfs, sd_vec,
                                 normalize_shock=True,
                                 backtransform=True):
    """
    Normalise the shock impact to 1 std and back-transform to original units.

    normalize_shock : scales each draw so that the h=0 shock impact = 1.
    backtransform   : multiplies columns 1:k by training standard deviations,
                      converting IRFs from standardised to original units.
    """
    irfs2 = irfs.copy()
    if normalize_shock:
        impacts = irfs2[:, 0, 0].copy()
        impacts[np.abs(impacts) < 1e-12] = np.nan
        irfs2 = irfs2 * (1.0 / impacts)[:, None, None]
    if backtransform:
        for j in range(1, irfs2.shape[2]):
            irfs2[:, :, j] *= sd_vec[j]
    return irfs2


def sign_align_to_y1(irfs, y1_index=1):
    """
    Flip all IRFs if the median y1 impact at h=0 is negative.

    Ensures a contractionary shock raises the short-term yield on impact,
    consistent with the sign convention used throughout the thesis.
    """
    if np.median(irfs[:, 0, y1_index]) < 0:
        return -irfs, True
    return irfs, False


# ============================================================
# SECTION 10 — CORE BVAR PIPELINE
# ============================================================

def run_bvar_irf(df_sub, shock_col, var_order, tag, p_lags):
    """
    Full IRF estimation pipeline for one shock series over the 1990-2008 window.

    Steps:
        1. Validate required columns
        2. Fill shock NaNs with zero (no-meeting months)
        3. Drop rows with NaN in any macro variable
        4. Winsorise and z-score standardise
        5. Draw posterior IRFs via bvar_draws_irf()
        6. Normalise and back-transform
        7. Sign-align to y1
        8. Compute median and credible bands

    Parameters
    ----------
    df_sub    : pd.DataFrame  Subsample restricted to 1990-2008.
    shock_col : str           Name of the shock column.
    var_order : list of str   Ordered variable list with shock first.
    tag       : str           Label for console progress output.
    p_lags    : int           VAR lag order.

    Returns
    -------
    h          : np.ndarray   Horizon array [0, 1, ..., HORIZON].
    irf_stats  : tuple        (median, lo, hi) IRF arrays, each (H+1 × k).
    macro_vars : list of str  Variable names excluding the shock.
    """
    miss = [c for c in var_order if c not in df_sub.columns]
    if miss:
        raise ValueError(f"[{tag}] Missing columns: {miss}")

    d = df_sub[var_order].copy()
    for c in var_order:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan)

    # Months with no FOMC meeting receive shock = 0
    d[shock_col] = d[shock_col].fillna(0.0)

    # Drop rows where any macro variable is missing
    macro_cols = [c for c in var_order if c != shock_col]
    d = d.dropna(subset=macro_cols).copy()

    print(f"\n[{tag}] p={p_lags} | T={len(d)} "
          f"{d.index.min().date()} → {d.index.max().date()} | "
          f"nonzero shock months={(d[shock_col] != 0).sum()}")

    for c in var_order:
        d[c] = clip_series(d[c])

    Y_raw = d[var_order].values.astype(float)
    if STANDARDIZE:
        Y, _, sd_vec = zscore_with_stats(Y_raw)
    else:
        Y, sd_vec = Y_raw, np.ones(Y_raw.shape[1])

    irfs = bvar_draws_irf(Y, p=p_lags, H=HORIZON, draws=DRAWS, tag=tag)
    irfs = normalize_and_backtransform(
        irfs, sd_vec,
        normalize_shock=NORMALIZE_SHOCK_IMPACT_TO_ONE,
        backtransform=BACKTRANSFORM_TO_ORIGINAL_UNITS,
    )
    if SIGN_ALIGN_TO_Y1_POSITIVE:
        irfs, flipped = sign_align_to_y1(irfs)
        if flipped:
            print(f"   [{tag}] Sign-aligned (flipped).")

    h   = np.arange(HORIZON + 1)
    med = np.median(irfs, axis=0)
    lo  = np.percentile(irfs, CI_LO, axis=0)
    hi  = np.percentile(irfs, CI_HI, axis=0)

    return h, (med, lo, hi), var_order[1:]


# ============================================================
# SECTION 11 — PLOTTING
# ============================================================

def plot_three_lines_1990(h, irf_dict, macro_vars, out_fig):
    """
    Plot three IRF series estimated over the same 1990-2008 window.

    Displays posterior median lines with 68% credible bands (12% opacity)
    for all three shock series in a two-column panel grid. Each series
    has a distinct colour and line style for unambiguous identification
    in black-and-white reproduction.

    Parameters
    ----------
    h          : np.ndarray  Horizon array.
    irf_dict   : dict        {label: (med, lo, hi)} for each shock series.
    macro_vars : list of str Variable names (excluding shock, in order).
    out_fig    : str         Output PNG file path.
    """
    # Colour and line style scheme — distinguishable in greyscale
    styles = {
        "AD (1990–2008)":     dict(color="#1f77b4", ls="-",  lw=2.2),
        "FFR (1990–2008)":    dict(color="#ff7f0e", ls="--", lw=2.2),
        "Wu-Xia (1990–2008)": dict(color="#2ca02c", ls=":",  lw=2.2),
    }

    n     = len(macro_vars)
    nrows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 4.5 * nrows), sharex=True)
    axes  = np.array(axes).reshape(-1)

    fig.suptitle(
        "Sample-Controlled Comparison: AD vs. FFR vs. Wu-Xia (1990–2008)\n"
        "Pre-2008 | BVAR with EBP | Median IRFs with 68% credible intervals",
        fontsize=11, y=1.01
    )

    for i, v in enumerate(macro_vars):
        ax = axes[i]
        j  = i + 1
        for lab, (med, lo, hi) in irf_dict.items():
            st = styles[lab]
            ax.plot(h, med[:, j], label=lab,
                    color=st["color"], ls=st["ls"], lw=st["lw"])
            ax.fill_between(h, lo[:, j], hi[:, j],
                            color=st["color"], alpha=0.12)
        ax.axhline(0, color="steelblue", lw=0.9)
        ax.set_ylabel(pretty_ylabel(v), fontsize=9)
        ax.set_title(v, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, loc="best")

    for k in range(n, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved figure → {out_fig}")


# ============================================================
# SECTION 12 — MAIN EXECUTION
# ============================================================

def main():
    """
    Main execution function for the 1990-2008 sample-controlled comparison.

    Execution order:
        1.  Load and index the monthly BVAR dataset
        2.  Validate EBP column is present
        3.  Optionally scale log variables to 100 × log
        4.  Detect AD and Hybrid (FFR) shock columns robustly
        5.  Restrict dataset to the 1990-2008 window
        6.  Run 1: AD shocks (1990-2008)
        7.  Run 2: FFR (Hybrid) shocks (1990-2008)
        8.  Run 3: Wu-Xia shocks (1990-2008)
        9.  Produce and save the three-series comparison figure
    """
    df = pd.read_csv(DATA_PATH)
    df = to_datetime_index(df)

    # Confirm EBP is present — required for this BVAR specification
    if "EBP" not in df.columns:
        raise ValueError(f"EBP column not found. Available: {list(df.columns)}")

    # Optional: scale log variables from plain log to 100 × log
    if SCALE_LOGS_BY_100:
        for c in LOG_VARS_TO_SCALE:
            if c in df.columns:
                df[c] = 100.0 * pd.to_numeric(df[c], errors="coerce")

    # BVAR variable order — shock placeholder replaced per run
    macro_vars     = ["y1", "log_sp500", "log_rgdp", "log_pgdp", "unemp", "EBP"]
    baseline_order = ["SHOCK"] + macro_vars

    def build_order(sc):
        """Replace 'SHOCK' placeholder with the actual shock column name."""
        return [sc if x == "SHOCK" else x for x in baseline_order]

    # ── Detect AD and FFR (Hybrid) shock columns ──────────────────────────────
    ad_col = detect_col_fuzzy(df, AD_CANDIDATES)
    hy_col = detect_col_fuzzy(df, HYBRID_CANDIDATES)

    if ad_col is None:
        raise ValueError(
            f"Could not detect AD shock column.\n"
            f"Tried: {AD_CANDIDATES}\nAvailable: {list(df.columns)}"
        )
    if hy_col is None:
        raise ValueError(
            f"Could not detect FFR/Hybrid shock column.\n"
            f"Tried: {HYBRID_CANDIDATES}\nAvailable: {list(df.columns)}"
        )

    print(f"AD column detected  : '{ad_col}'")
    print(f"FFR column detected : '{hy_col}'")

    # ── Restrict dataset to the 1990-2008 window ──────────────────────────────
    start  = pd.to_datetime(SAMPLE_START)
    end    = pd.to_datetime(SAMPLE_END)
    df_sub = df.loc[(df.index >= start) & (df.index < end)].copy()

    print(f"\nSubsample: {df_sub.index.min().date()} → "
          f"{df_sub.index.max().date()} "
          f"({len(df_sub)} months)")

    # ── Run 1: AD shocks (1990-2008) ──────────────────────────────────────────
    print("\n" + "="*60)
    print("RUN 1 of 3 — AD shocks | 1990–2008")
    print("="*60)
    h, irf_ad, mv = run_bvar_irf(
        df_sub, ad_col, build_order(ad_col),
        tag="AD 1990-2008", p_lags=P_LAGS_PRE,
    )

    # ── Run 2: FFR (Hybrid) shocks (1990-2008) ────────────────────────────────
    print("\n" + "="*60)
    print("RUN 2 of 3 — FFR (Hybrid) shocks | 1990–2008")
    print("="*60)
    _, irf_ffr, _ = run_bvar_irf(
        df_sub, hy_col, build_order(hy_col),
        tag="FFR 1990-2008", p_lags=P_LAGS_PRE,
    )

    # ── Run 3: Wu-Xia shocks (1990-2008) ─────────────────────────────────────
    print("\n" + "="*60)
    print("RUN 3 of 3 — Wu-Xia shocks | 1990–2008")
    print("="*60)
    _, irf_wx, _ = run_bvar_irf(
        df_sub, SHOCK_WX_COL, build_order(SHOCK_WX_COL),
        tag="WuXia 1990-2008", p_lags=P_LAGS_PRE,
    )

    # ── Produce three-series comparison figure ────────────────────────────────
    out_fig = os.path.join(
        OUTDIR,
        f"Analysis1_1990_AD_FFR_WuXia_"
        f"p{P_LAGS_PRE}_H{HORIZON}_EBP.png"
    )

    plot_three_lines_1990(
        h,
        irf_dict={
            "AD (1990–2008)":     irf_ad,
            "FFR (1990–2008)":    irf_ffr,
            "Wu-Xia (1990–2008)": irf_wx,
        },
        macro_vars=mv,
        out_fig=out_fig,
    )

    print("\nAnalysis 1 complete.")
    print(f"   Figure saved: {out_fig}")


if __name__ == "__main__":
    main()