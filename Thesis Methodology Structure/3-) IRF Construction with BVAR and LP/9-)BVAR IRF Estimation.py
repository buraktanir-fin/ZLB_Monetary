"""
================================================================================
bvar_irf_estimation.py
================================================================================
Thesis:  "When Rates Hit Zero: Identifying Monetary Shocks with Shadow Rates"
Author:  Burak Tanir
Date:    April 2026

Purpose
-------
Estimates Bayesian Vector Autoregressive (BVAR) Impulse Response Functions
(IRFs) for the six macroeconomic variables described in Section 4.4 of the
thesis, using four monetary policy shock series as external instruments.
This script produces all IRF figures and tables reported in Sections 5, 6,
and 7 of the thesis.

The BVAR is estimated separately for the pre-2008 (conventional policy) and
post-2008 (zero lower bound) subsamples. Identification relies on a Cholesky
decomposition in which the monetary policy shock is ordered first.

Shock series estimated
----------------------
    Wu-Xia (WX)   — shadow rate shock; main post-2008 instrument
    Krippner (KR) — shadow rate shock; post-2008 robustness alternative
    AD             — original Aruoba-Drechsel (2024) shock; pre-2008 benchmark
    Hybrid         — FFR-based replicated shock; pre-2008 benchmark

BVAR variable set (ordering in the VAR)
----------------------------------------
    Position 0  : Monetary policy shock (external instrument, ordered first)
    Position 1  : y1         — 1-year Treasury yield (%)
    Position 2  : log_sp500  — log S&P 500 index (100 × log)
    Position 3  : log_rgdp   — log real GDP (100 × log)
    Position 4  : log_pgdp   — log GDP deflator (100 × log)
    Position 5  : unemp      — unemployment rate (%)
    Position 6  : EBP        — excess bond premium (%)

Figures and tables produced
-----------------------------
    For each shock series × each period (PRE / POST):
        Figure_IRF_{shock}_{period}_p{lags}_H{horizon}_EBP.png
        Table_IRF_{shock}_{period}_p{lags}_H{horizon}_EBP.csv

    Overlay figures comparing pre vs post for each shadow rate shock:
        Figure_Overlay_PreVsPost_{shock}_pPre{p}_pPost{p}_H{horizon}_EBP.png

    Four-shock comparison figure (pre-2008, median IRFs only):
        Figure_Compare_Pre_FourShocks_p{lags}_H{horizon}_EBP.png

    Post-2008 overlay comparing Wu-Xia vs Krippner:
        Figure_Overlay_Post_WXvsKR_p{lags}_H{horizon}_EBP.png

    All outputs saved to: irf_outputs_EBP_final/

BVAR estimation details
------------------------
    Prior           : Minnesota-style independent Normal-Inverse-Wishart
                      (B0 = 0, V0 = lambda^2 * I, S0 = sigma^2 * I)
    Posterior       : Closed-form Normal-Inverse-Wishart
    Draws           : 4,000 attempted; only stable draws retained
    Stability check : Spectral radius of companion matrix < 0.9999
    Identification  : Cholesky decomposition; shock ordered first
    Horizon         : 48 months
    Lags            : 12 (both PRE and POST)
    Credible bands  : 16th–84th percentile (68% posterior credible interval)
    Sign convention : Median y1 impact at h=0 normalised to positive
                      (contractionary shock raises short-term yields)

Pre-processing pipeline
------------------------
    1. Winsorise each variable at the 0.1th and 99.9th percentile
    2. Z-score standardise all variables (fit on training sample)
    3. Estimate BVAR and extract Cholesky IRFs
    4. Normalise shock impact at h=0 to 1 standard deviation
    5. Back-transform IRFs to original units (multiply by training std)
    6. Sign-align: flip if median y1 impact at h=0 is negative

Input
-----
    outputs/bvar_monthly_dataset_FULL_with_EBP_with_ALL_meeting_shockSUM.csv
        Produced by bvar_dataset_builder.py. Must contain DATE index,
        all macro variable columns, and all shock columns.

Output
------
    irf_outputs_EBP_final/
        All IRF CSV tables, PNG figures, and overlay comparison figures.

Key configuration parameters
------------------------------
    HORIZON   : IRF horizon in months (default: 48)
    DRAWS     : Number of posterior draws attempted (default: 4,000)
    P_LAGS_PRE / P_LAGS_POST : VAR lag order for each period (default: 12)
    LAMBDA_B  : Minnesota prior tightness (default: 0.2)
    CI_LO / CI_HI : Credible interval percentiles (default: 16, 84)
    SEED      : Random number generator seed for reproducibility (default: 123)
    SCALE_LOGS_BY_100 : Set True if log variables are in plain log units
                        and need to be multiplied by 100 for thesis units

Dependencies
------------
    numpy, pandas, matplotlib  — pip install numpy pandas matplotlib
    Python 3.7+ compatible (no f-string walrus operators)

Usage
-----
    python bvar_irf_estimation.py

    Ensure bvar_dataset_builder.py has been run first so that the input
    dataset exists in the outputs/ folder. The irf_outputs_EBP_final/
    folder is created automatically.
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
# bvar_dataset_builder.py. All IRF outputs are written to a
# dedicated subfolder to keep them separate from the input datasets.

BASE_DIR    = os.getcwd()
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

FULL_DATA_FILENAME = "bvar_monthly_dataset_FULL_with_EBP_with_ALL_meeting_shockSUM.csv"
DATA_PATH = os.path.join(OUTPUTS_DIR, FULL_DATA_FILENAME)

# All IRF figures and tables are saved here
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
# The date column is searched under several common name variants.
# Wu-Xia and Krippner shock column names are specified explicitly
# since they have fixed names set by shock_comparison.py.
# AD and Hybrid column names are detected robustly (case-insensitive)
# to handle minor naming variations across dataset versions.

# Common date column name variants to search for
DATE_COL_CANDIDATES = ["MS", "DATE", "Date", "date"]

# Explicit shock column names (from shock_comparison.py output)
SHOCK_WX_COL = "shock_Wu_meeting_sum"    # Wu-Xia shadow rate shock
SHOCK_KR_COL = "shock_Kr_meeting_sum"    # Krippner shadow rate shock

# AD and Hybrid: detected case-insensitively to handle naming variants
AD_CANDIDATES     = ["AD", "shock_AD_meeting_sum", "ad_meeting_sum", "aruoba", "aruoba_drechsel"]
HYBRID_CANDIDATES = ["hybrid", "shock_hybrid_meeting_sum", "hybrid_meeting_sum", "policy_hybrid"]


# ============================================================
# SECTION 3 — SAMPLE SPLIT
# ============================================================
# The pre/post split is set at November 2008, consistent with the
# CUTOFF_DATE used in shock_comparison.py. Months before this date
# form the PRE (conventional policy) sample; months from this date
# onwards form the POST (ZLB) sample.

SPLIT_MS = "2008-11-01"   # first POST month (November 2008)


# ============================================================
# SECTION 4 — BVAR ESTIMATION SETTINGS
# ============================================================

HORIZON = 48      # IRF horizon in months
DRAWS   = 4000    # number of posterior draws attempted (stable draws are kept)

P_LAGS_PRE  = 12  # VAR lag order for the pre-2008 subsample
P_LAGS_POST = 12  # VAR lag order for the post-2008 subsample

# Minnesota-style prior hyperparameters
# DF0      : degrees of freedom for the Inverse-Wishart prior on Sigma
# S0_SCALE : scale of the Inverse-Wishart prior (S0 = S0_SCALE^2 * I)
# LAMBDA_B : tightness of the Normal prior on VAR coefficients (smaller = tighter)
DF0      = 10
S0_SCALE = 0.2
LAMBDA_B = 0.2

# Credible interval percentiles (68% band: 16th–84th)
CI_LO, CI_HI = 16, 84

# Winsorisation quantiles applied before estimation to limit outlier influence
CLIP_Q_LO, CLIP_Q_HI = 0.001, 0.999

# Estimation options
STANDARDIZE                  = True   # z-score standardise before estimation
NORMALIZE_SHOCK_IMPACT_TO_ONE = True  # normalise h=0 shock impact to 1 std
BACKTRANSFORM_TO_ORIGINAL_UNITS = True # multiply IRFs by training std after normalisation

# Sign convention: positive shock = contractionary = y1 rises at h=0
# If the median y1 impact is negative, all IRFs are flipped
SIGN_ALIGN_TO_Y1_POSITIVE = True

# Random number generator seed for reproducibility across runs
SEED = 123
rng  = np.random.default_rng(SEED)


# ============================================================
# SECTION 5 — DISPLAY SETTINGS (THESIS FIGURE UNITS)
# ============================================================
# Y-axis labels match the units reported in the thesis figures.
# If log_* variables are stored in plain log units (not 100 × log),
# set SCALE_LOGS_BY_100 = True to convert before estimation.

DISPLAY_LABELS = {
    "y1":         "1-year bond yield (%)",
    "log_sp500":  "S&P500 (100 × log)",
    "log_rgdp":   "Real GDP (100 × log)",
    "unemp":      "Unemployment (%)",
    "log_pgdp":   "GDP deflator (100 × log)",
    "EBP":        "EBP (%)",
}

def pretty_ylabel(varname: str) -> str:
    """Return the thesis-standard y-axis label for a BVAR variable."""
    return DISPLAY_LABELS.get(varname, varname)

# Set True if log variables are in plain log units and need × 100 for thesis plots
SCALE_LOGS_BY_100   = False
LOG_VARS_TO_SCALE   = ["log_sp500", "log_rgdp", "log_pgdp"]


# ============================================================
# SECTION 6 — UTILITY FUNCTIONS: DATA HANDLING
# ============================================================

def detect_date_col(df):
    """
    Find the date column in df by searching DATE_COL_CANDIDATES.

    Returns the first matching column name, or None if not found.
    """
    for c in DATE_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def to_datetime_index(df):
    """
    Convert a date column (or the existing index) to a DatetimeIndex.

    Searches for the date column using detect_date_col(), parses it
    to datetime, drops unparseable rows, and sets it as the index.
    Falls back to parsing the existing index if no date column is found.

    Parameters
    ----------
    df : pd.DataFrame  Raw DataFrame from CSV read.

    Returns
    -------
    pd.DataFrame  DataFrame with a sorted DatetimeIndex.
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

    Clips extreme values before BVAR estimation to limit the influence
    of outliers on the posterior draws without removing observations.

    Parameters
    ----------
    s   : pd.Series  Numeric series to winsorise.
    qlo : float      Lower quantile (default: 0.001).
    qhi : float      Upper quantile (default: 0.999).

    Returns
    -------
    pd.Series  Clipped series.
    """
    lo, hi = s.quantile([qlo, qhi])
    return s.clip(lo, hi)


def zscore_with_stats(X):
    """
    Z-score standardise a matrix and return the standardisation statistics.

    Standardisation is applied column-wise using the population mean and
    standard deviation (ddof=0). Columns with zero variance receive
    sd = 1 to avoid division by zero. The returned mean and std vectors
    are used later to back-transform IRFs to original units.

    Parameters
    ----------
    X : array-like  (T × k) matrix of BVAR variables.

    Returns
    -------
    Z    : np.ndarray  Standardised matrix (T × k).
    m    : np.ndarray  Column means (k,).
    sd   : np.ndarray  Column standard deviations (k,).
    """
    X  = np.asarray(X, float)
    m  = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=0)
    sd[sd == 0] = 1.0
    Z = (X - m) / sd
    return Z, m, sd


def detect_col_fuzzy(df, candidates):
    """
    Robustly find a column in df matching any of the candidate names.

    Detection priority:
        1. Exact match (case-sensitive)
        2. Exact match (case-insensitive)
        3. Contains-match (case-insensitive substring)

    Used to detect AD and Hybrid columns which may appear under
    slightly different names across dataset versions.

    Parameters
    ----------
    df         : pd.DataFrame  Dataset to search.
    candidates : list of str   Candidate column names to match.

    Returns
    -------
    str or None  Matched column name (original capitalisation), or None.
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


# ============================================================
# SECTION 7 — UTILITY FUNCTIONS: BVAR ALGEBRA
# ============================================================

def make_lag_matrix(Y, p):
    """
    Construct the regressor matrix of lagged endogenous variables.

    Stacks lags 1 through p horizontally. The resulting matrix has
    (T - p) rows, aligned to the dependent variable block Y[p:, :].

    Parameters
    ----------
    Y : np.ndarray  (T × k) matrix of BVAR variables.
    p : int         Number of lags.

    Returns
    -------
    np.ndarray  ((T-p) × k*p) lag matrix.
    """
    T, k   = Y.shape
    Xlags  = []
    for L in range(1, p + 1):
        Xlags.append(Y[p - L:T - L, :])
    return np.hstack(Xlags)


def companion_matrix(A_list):
    """
    Build the companion form matrix from a list of VAR coefficient matrices.

    The companion matrix is used for stability checking and for computing
    multi-step IRFs via repeated matrix multiplication.

    Parameters
    ----------
    A_list : list of np.ndarray  VAR coefficient matrices [A1, A2, ..., Ap],
                                  each of shape (k × k).

    Returns
    -------
    np.ndarray  (k*p × k*p) companion matrix.
    """
    k   = A_list[0].shape[0]
    p   = len(A_list)
    top = np.hstack(A_list)
    if p == 1:
        return top
    I      = np.eye(k * (p - 1))
    zeros  = np.zeros((k * (p - 1), k))
    bottom = np.hstack([I, zeros])
    return np.vstack([top, bottom])


def is_stable(A_list):
    """
    Check whether a VAR is covariance-stationary.

    Computes the spectral radius of the companion matrix. A VAR is stable
    (covariance-stationary) if and only if all eigenvalues of the companion
    matrix lie strictly inside the unit circle. A strict threshold of 0.9999
    is used to avoid near-unit-root instability in the IRFs.

    Parameters
    ----------
    A_list : list of np.ndarray  VAR coefficient matrices.

    Returns
    -------
    bool  True if the spectral radius is strictly less than 0.9999.
    """
    F  = companion_matrix(A_list)
    ev = np.linalg.eigvals(F)
    return np.max(np.abs(ev)) < 0.9999


def wishart_rnd(df, V):
    """
    Sample from a Wishart distribution W(df, V) using the Bartlett decomposition.

    Parameters
    ----------
    df : int         Degrees of freedom.
    V  : np.ndarray  Scale matrix (p × p), positive definite.

    Returns
    -------
    np.ndarray  (p × p) Wishart draw.
    """
    V  = np.asarray(V, float)
    L  = np.linalg.cholesky(V)
    p  = V.shape[0]
    A  = np.zeros((p, p))
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

    Parameters
    ----------
    df : int         Degrees of freedom.
    S  : np.ndarray  Scale matrix (p × p), positive definite.

    Returns
    -------
    np.ndarray  (p × p) Inverse-Wishart draw.
    """
    S    = np.asarray(S, float)
    Sinv = np.linalg.inv(S)
    W    = wishart_rnd(df, Sinv)
    return np.linalg.inv(W)


def sample_matrix_normal(Bbar, V, Sigma):
    """
    Sample from a matrix Normal distribution MN(Bbar, V, Sigma).

    Used to draw the posterior VAR coefficient matrix B in the
    Normal-Inverse-Wishart Gibbs sampler. Samples are generated via
    Cholesky factorisation for numerical stability.

    Parameters
    ----------
    Bbar  : np.ndarray  Posterior mean of B (m × k).
    V     : np.ndarray  Left covariance factor (m × m).
    Sigma : np.ndarray  Right covariance factor (k × k).

    Returns
    -------
    np.ndarray  (m × k) draw from the matrix Normal distribution.
    """
    Bbar  = np.asarray(Bbar,  float)
    V     = np.asarray(V,     float)
    Sigma = np.asarray(Sigma, float)
    Lv    = np.linalg.cholesky(V)
    Ls    = np.linalg.cholesky(Sigma)
    m, k  = Bbar.shape
    Z     = rng.normal(size=(m, k))
    return Bbar + (Lv @ Z @ Ls.T)


# ============================================================
# SECTION 8 — BVAR POSTERIOR SAMPLING AND IRF EXTRACTION
# ============================================================

def bvar_draws_irf(Y, p, H, draws, lambda_b=LAMBDA_B, df0=DF0, s0_scale=S0_SCALE, tag=""):
    """
    Draw posterior BVAR IRFs via the Normal-Inverse-Wishart closed-form posterior.

    Identification: Cholesky decomposition of Sigma; the monetary policy
    shock is the innovation to the first variable (position 0 in var_order).
    Only stable posterior draws (spectral radius < 0.9999) are retained.

    Algorithm:
        For each draw:
            1. Draw Sigma ~ IW(df_post, S_post)
            2. Draw B     ~ MN(B_post, V_post, Sigma)
            3. Check stability of companion matrix; discard if unstable
            4. Compute Cholesky impact vector P[:, 0]
            5. Propagate via companion matrix: IRF(h) = J F^h J' impact

    Parameters
    ----------
    Y        : np.ndarray  (T × k) standardised BVAR data matrix.
    p        : int         VAR lag order.
    H        : int         IRF horizon in months.
    draws    : int         Number of posterior draws to attempt.
    lambda_b : float       Minnesota prior tightness on coefficients.
    df0      : int         Prior degrees of freedom for Inverse-Wishart.
    s0_scale : float       Prior scale for Inverse-Wishart (S0 = scale^2 * I).
    tag      : str         Label for progress print statements.

    Returns
    -------
    np.ndarray  (n_stable_draws × H+1 × k) array of IRFs.

    Raises
    ------
    RuntimeError  If no stable posterior draws are found.
    """
    t0 = time.time()

    T, k    = Y.shape
    Xlags   = make_lag_matrix(Y, p)
    Ydep    = Y[p:, :]
    X       = np.hstack([np.ones((T - p, 1)), Xlags])

    ncoef   = X.shape[1]
    B0      = np.zeros((ncoef, k))
    V0_inv  = np.eye(ncoef) / (lambda_b ** 2)
    S0      = (s0_scale ** 2) * np.eye(k)

    XtX     = X.T @ X
    XtY     = X.T @ Ydep

    # Closed-form Normal-Inverse-Wishart posterior
    V_post  = np.linalg.inv(V0_inv + XtX)
    B_post  = V_post @ (V0_inv @ B0 + XtY)

    U       = Ydep - X @ B_post
    S_post  = S0 + (U.T @ U) + (B_post - B0).T @ V0_inv @ (B_post - B0)
    df_post = df0 + (T - p)

    def parse_A_list(Bmat):
        """Extract list of (k × k) VAR coefficient matrices from posterior draw."""
        A_list = []
        start  = 1
        for L in range(p):
            block = Bmat[start + L * k : start + (L + 1) * k, :].T
            A_list.append(block)
        return A_list

    keep_irfs  = []
    kept       = 0
    milestones = {250, 500, 1000, 2000, 3000, 4000}

    for d in range(draws):
        Sigma  = invwishart_rnd(df_post, S_post)
        Bdraw  = sample_matrix_normal(B_post, V_post, Sigma)

        A_list = parse_A_list(Bdraw)
        if not is_stable(A_list):
            continue

        # External instrument identification: unit impulse to the pre-constructed shock
        impact = np.zeros(k)
        impact[0] = 1.0

        # Propagate via companion matrix
        F  = companion_matrix(A_list)
        J  = np.hstack([np.eye(k), np.zeros((k, k*(p-1)))])

        irf    = np.zeros((H + 1, k))
        irf[0, :] = impact

        Fh = np.eye(k * p)
        for hh in range(1, H + 1):
            Fh        = Fh @ F
            Phi_h     = J @ Fh @ J.T
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


# ============================================================
# SECTION 9 — IRF POST-PROCESSING
# ============================================================

def normalize_and_backtransform(irfs, sd_vec, normalize_shock=True, backtransform=True):
    """
    Normalise the shock impact and back-transform IRFs to original units.

    Step 1 (normalise_shock=True): scales each draw so that the shock's
    own impact at h=0 equals 1. This makes IRFs interpretable as responses
    to a one-standard-deviation shock.

    Step 2 (backtransform=True): multiplies columns 1 through k by their
    respective training standard deviations, converting IRFs from
    standardised units back to the original economic units of each variable.
    Column 0 (the shock itself) is not back-transformed.

    Parameters
    ----------
    irfs            : np.ndarray  (draws × H+1 × k) raw IRF array.
    sd_vec          : np.ndarray  (k,) training standard deviations.
    normalize_shock : bool        Whether to normalise h=0 shock impact to 1.
    backtransform   : bool        Whether to convert to original units.

    Returns
    -------
    np.ndarray  (draws × H+1 × k) processed IRF array.
    """
    irfs2 = irfs.copy()

    if normalize_shock:
        impacts = irfs2[:, 0, 0].copy()
        impacts[np.abs(impacts) < 1e-12] = np.nan
        scale   = 1.0 / impacts
        irfs2   = irfs2 * scale[:, None, None]

    if backtransform:
        for j in range(1, irfs2.shape[2]):
            irfs2[:, :, j] = irfs2[:, :, j] * sd_vec[j]

    return irfs2


def sign_align_to_y1(irfs, y1_index=1):
    """
    Align the sign of all IRFs so that the median y1 impact at h=0 is positive.

    A contractionary monetary shock should raise the short-term interest rate
    (y1) on impact. If the median posterior impact is negative, all IRFs
    are flipped (multiplied by -1), which is equivalent to relabelling the
    shock as expansionary.

    Parameters
    ----------
    irfs     : np.ndarray  (draws × H+1 × k) IRF array.
    y1_index : int         Column index of the y1 variable (default: 1).

    Returns
    -------
    irfs    : np.ndarray  Sign-corrected IRF array.
    flipped : bool        True if the sign was flipped.
    """
    med_y1_impact = np.median(irfs[:, 0, y1_index])
    if med_y1_impact < 0:
        return -irfs, True
    return irfs, False


# ============================================================
# SECTION 10 — IRF PIPELINE: ESTIMATION, SAVING, AND PLOTTING
# ============================================================

def run_bvar_irf(df_sub, shock_col, var_order, tag, p_lags, out_csv, out_fig):
    """
    Full IRF estimation pipeline for one shock series and one sample period.

    Steps:
        1. Validate required columns are present
        2. Fill shock NaNs with zero (no-meeting months)
        3. Drop rows with NaN in any macro variable
        4. Winsorise and z-score standardise
        5. Draw posterior IRFs via bvar_draws_irf()
        6. Normalise and back-transform
        7. Sign-align to y1
        8. Compute median and credible bands
        9. Save IRF table to CSV
        10. Save IRF figure to PNG

    Parameters
    ----------
    df_sub    : pd.DataFrame  Subsample DataFrame (PRE or POST).
    shock_col : str           Name of the shock column in df_sub.
    var_order : list of str   Ordered variable list with shock first.
    tag       : str           Label for console progress output.
    p_lags    : int           VAR lag order.
    out_csv   : str           Output path for the IRF table CSV.
    out_fig   : str           Output path for the IRF figure PNG.

    Returns
    -------
    h          : np.ndarray   Horizon array [0, 1, ..., HORIZON].
    irf_stats  : tuple        (median, lo, hi) IRF arrays, each (H+1 × k).
    macro_vars : list of str  Variable names excluding the shock (positions 1:k).
    """
    need = var_order
    miss = [c for c in need if c not in df_sub.columns]
    if miss:
        raise ValueError(f"[{tag}] Missing columns: {miss}\nAvailable: {list(df_sub.columns)}")

    d = df_sub.copy()

    for c in need:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan)

    # Months with no FOMC meeting receive shock = 0
    d[shock_col] = d[shock_col].fillna(0.0)

    # Drop rows where any macro variable is missing
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
        Y      = Y_raw
        sd_vec = np.ones(Y.shape[1])

    irfs = bvar_draws_irf(Y, p=p_lags, H=HORIZON, draws=DRAWS, tag=tag)

    irfs = normalize_and_backtransform(
        irfs,
        sd_vec              = sd_vec,
        normalize_shock     = NORMALIZE_SHOCK_IMPACT_TO_ONE,
        backtransform       = BACKTRANSFORM_TO_ORIGINAL_UNITS
    )

    if SIGN_ALIGN_TO_Y1_POSITIVE:
        irfs, flipped = sign_align_to_y1(irfs, y1_index=1)
        if flipped:
            print(f"   [{tag}] Sign-aligned: flipped IRFs so y1 impact (h=0) is positive.")

    h   = np.arange(HORIZON + 1)
    med = np.median(irfs, axis=0)
    lo  = np.percentile(irfs, CI_LO, axis=0)
    hi  = np.percentile(irfs, CI_HI, axis=0)

    # Save IRF table
    out = pd.DataFrame({"h": h})
    for j in range(1, len(var_order)):
        v             = var_order[j]
        out[f"{v}_med"] = med[:, j]
        out[f"{v}_lo"]  = lo[:, j]
        out[f"{v}_hi"]  = hi[:, j]
    out.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"Saved IRF table → {out_csv}")

    # Save IRF figure
    macro_vars = var_order[1:]
    n          = len(macro_vars)
    nrows      = int(np.ceil(n / 2))
    fig, axes  = plt.subplots(nrows, 2, figsize=(12, 4 * nrows), sharex=True)
    axes       = np.array(axes).reshape(-1)

    for i, v in enumerate(macro_vars):
        ax = axes[i]
        j  = i + 1
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
    print(f"Saved figure → {out_fig}")

    return h, (med, lo, hi), macro_vars


# ============================================================
# SECTION 11 — OVERLAY PLOTTING FUNCTIONS
# ============================================================

def plot_two_irf_overlay(h, A, B, macro_vars, labelA, labelB, title_prefix, out_fig):
    """
    Plot two IRF sets in the same panels for direct comparison.

    Series A is shown as a solid line; series B as a dashed line.
    Both 68% credible bands are shown with 15% opacity. Used for:
        - Pre vs Post comparison for the same shock series
        - Wu-Xia vs Krippner comparison in the post-2008 period

    Parameters
    ----------
    h           : np.ndarray    Horizon array.
    A, B        : tuple         (median, lo, hi) each (H+1 × k).
    macro_vars  : list of str   Variable names (excluding shock).
    labelA, labelB : str        Legend labels for the two sets.
    title_prefix : str          Prefix for each subplot title.
    out_fig      : str          Output PNG path.
    """
    med_A, lo_A, hi_A = A
    med_B, lo_B, hi_B = B

    n          = len(macro_vars)
    nrows      = int(np.ceil(n / 2))
    fig, axes  = plt.subplots(nrows, 2, figsize=(12, 4 * nrows), sharex=True)
    axes       = np.array(axes).reshape(-1)

    for i, v in enumerate(macro_vars):
        ax = axes[i]
        j  = i + 1
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
    print(f"Saved overlay → {out_fig}")


def plot_four_shocks_pre(h, macro_vars, irf_dict, title_prefix, out_fig):
    """
    Plot median IRFs from four shock series in the same panels.

    Displays median lines only (no credible bands) to avoid visual clutter
    when comparing four series simultaneously. Each series receives a
    distinct line style. Used for the pre-2008 four-shock comparison figure
    reported in Section 6.1 of the thesis.

    Parameters
    ----------
    h           : np.ndarray      Horizon array.
    macro_vars  : list of str     Variable names (excluding shock).
    irf_dict    : dict            {label: (median, lo, hi)} for each shock.
    title_prefix : str            Prefix for each subplot title.
    out_fig      : str            Output PNG path.
    """
    n          = len(macro_vars)
    nrows      = int(np.ceil(n / 2))
    fig, axes  = plt.subplots(nrows, 2, figsize=(12, 4 * nrows), sharex=True)
    axes       = np.array(axes).reshape(-1)

    linestyles = ["-", "--", ":", "-."]

    labels = list(irf_dict.keys())
    for i, v in enumerate(macro_vars):
        ax = axes[i]
        j  = i + 1
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
    print(f"Saved 4-shock PRE figure → {out_fig}")


# ============================================================
# SECTION 12 — DIAGNOSTIC UTILITY
# ============================================================

def shock_diagnostics(df_pre, df_post, shock_col, label, pre_only=False):
    """
    Print summary statistics for a shock series in the PRE and POST samples.

    Reports the number of non-zero months, mean, standard deviation, and
    range. Non-zero month counts confirm that the monthly aggregation from
    meeting-level shocks worked correctly (expected: ~8-12 non-zero months
    per year).

    Parameters
    ----------
    df_pre    : pd.DataFrame  Pre-2008 subsample.
    df_post   : pd.DataFrame  Post-2008 subsample.
    shock_col : str           Shock column name.
    label     : str           Display label for the print output.
    pre_only  : bool          If True, skip POST diagnostics (AD/Hybrid).
    """
    pre = pd.to_numeric(df_pre[shock_col], errors="coerce").fillna(0.0)
    print(f"\n--- Shock diagnostics: {label} ({shock_col}) ---")
    print(f"PRE : nonzero={int((pre!=0).sum())} | mean={pre.mean():.6f} | std={pre.std(ddof=0):.6f} | min={pre.min():.6f} | max={pre.max():.6f}")
    if not pre_only:
        post = pd.to_numeric(df_post[shock_col], errors="coerce").fillna(0.0)
        print(f"POST: nonzero={int((post!=0).sum())} | mean={post.mean():.6f} | std={post.std(ddof=0):.6f} | min={post.min():.6f} | max={post.max():.6f}")


# ============================================================
# SECTION 13 — MAIN EXECUTION
# ============================================================

def main():
    """
    Main execution function orchestrating the full IRF estimation workflow.

    Execution order:
        1.  Load and index the monthly BVAR dataset
        2.  Validate required columns; optionally scale log variables
        3.  Split into PRE and POST subsamples
        4.  Detect AD and Hybrid columns robustly
        5.  Print shock diagnostics for all four series
        6.  Estimate PRE and POST IRFs for Wu-Xia and Krippner
        7.  Produce pre vs post overlay figures for each shadow rate
        8.  Estimate PRE IRFs for AD and Hybrid
        9.  Produce four-shock comparison figure (PRE, median only)
        10. Produce POST overlay comparing Wu-Xia vs Krippner
    """
    df = pd.read_csv(DATA_PATH)
    df = to_datetime_index(df)

    # Confirm EBP is present — required for this BVAR specification
    if "EBP" not in df.columns:
        raise ValueError(f"EBP column not found in dataset.\nAvailable columns: {list(df.columns)}")
    print("Keeping EBP column for estimation (EBP spec).")

    # BVAR variable order: shock first, then macro variables
    macro_vars = ["y1", "log_sp500", "log_rgdp", "log_pgdp", "unemp", "EBP"]

    required_base = [SHOCK_WX_COL, SHOCK_KR_COL] + macro_vars
    missing = [c for c in required_base if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable columns: {list(df.columns)}")

    # Optional: scale log variables from plain log units to 100 × log
    if SCALE_LOGS_BY_100:
        for c in LOG_VARS_TO_SCALE:
            if c in df.columns:
                df[c] = 100.0 * pd.to_numeric(df[c], errors="coerce")
        print("Scaled log_* variables to 100×log for thesis-unit consistency.")

    # Split into PRE and POST subsamples
    split    = pd.to_datetime(SPLIT_MS)
    df_pre   = df.loc[df.index < split].copy()
    df_post  = df.loc[df.index >= split].copy()

    # ── Thesis-ready output file naming helpers ────────────────────────────────
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

    # Robustly detect AD and Hybrid columns
    ad_col = detect_col_fuzzy(df, AD_CANDIDATES)
    hy_col = detect_col_fuzzy(df, HYBRID_CANDIDATES)

    if ad_col is None or hy_col is None:
        print("\nPRE-only shocks not found (will skip AD/hybrid if missing).")
        print("   Detected AD  :", ad_col)
        print("   Detected HYB :", hy_col)
    else:
        print(f"\nDetected PRE-only shocks: AD='{ad_col}', Hybrid='{hy_col}'")

    # Print shock diagnostics before estimation
    shock_diagnostics(df_pre, df_post, SHOCK_WX_COL, "WX")
    shock_diagnostics(df_pre, df_post, SHOCK_KR_COL, "KR")
    if ad_col is not None:
        shock_diagnostics(df_pre, df_post, ad_col, "AD (pre-only)", pre_only=True)
    if hy_col is not None:
        shock_diagnostics(df_pre, df_post, hy_col, "HYBRID (pre-only)", pre_only=True)

    baseline_order = ["SHOCK"] + macro_vars

    def build_order(shock_col, template):
        """Replace the 'SHOCK' placeholder with the actual shock column name."""
        return [shock_col if x == "SHOCK" else x for x in template]

    pre_irfs  = {}   # stores PRE IRF tuples keyed by shock label
    post_irfs = {}   # stores POST IRF tuples keyed by shock label

    # ── Wu-Xia and Krippner: estimate PRE and POST IRFs ───────────────────────
    for shock_col, shock_label in [(SHOCK_WX_COL, "WX"), (SHOCK_KR_COL, "KR")]:
        order = build_order(shock_col, baseline_order)

        h_pre, irf_pre, macro_vars_out = run_bvar_irf(
            df_pre, shock_col, order,
            tag    = f"{shock_full_label(shock_label)} (Pre-2008, EBP)",
            p_lags = P_LAGS_PRE,
            out_csv = tab_irf(shock_label, True,  P_LAGS_PRE),
            out_fig = fig_irf(shock_label, True,  P_LAGS_PRE),
        )
        pre_irfs[shock_label] = irf_pre

        h_post, irf_post, _ = run_bvar_irf(
            df_post, shock_col, order,
            tag    = f"{shock_full_label(shock_label)} (Post-2008, EBP)",
            p_lags = P_LAGS_POST,
            out_csv = tab_irf(shock_label, False, P_LAGS_POST),
            out_fig = fig_irf(shock_label, False, P_LAGS_POST),
        )
        post_irfs[shock_label] = (h_post, irf_post)

        # Pre vs post overlay for each shadow rate shock
        plot_two_irf_overlay(
            h_pre, irf_pre, irf_post, macro_vars_out,
            labelA       = "Pre-2008",
            labelB       = "Post-2008",
            title_prefix = f"{shock_full_label(shock_label)}: Pre vs Post (EBP)",
            out_fig      = fig_overlay_prepost(shock_label),
        )

    # ── AD and Hybrid: estimate PRE IRFs only ─────────────────────────────────
    if ad_col is not None:
        order_ad = build_order(ad_col, baseline_order)
        _, irf_ad, _ = run_bvar_irf(
            df_pre, ad_col, order_ad,
            tag     = "AruobaDrechsel (Pre-2008, EBP)",
            p_lags  = P_LAGS_PRE,
            out_csv = tab_irf("AD",  True, P_LAGS_PRE),
            out_fig = fig_irf("AD",  True, P_LAGS_PRE),
        )
        pre_irfs["AD"] = irf_ad

    if hy_col is not None:
        order_hy = build_order(hy_col, baseline_order)
        _, irf_hy, _ = run_bvar_irf(
            df_pre, hy_col, order_hy,
            tag     = "Hybrid (Pre-2008, EBP)",
            p_lags  = P_LAGS_PRE,
            out_csv = tab_irf("HYB", True, P_LAGS_PRE),
            out_fig = fig_irf("HYB", True, P_LAGS_PRE),
        )
        pre_irfs["Hybrid"] = irf_hy

    # ── Four-shock PRE comparison figure ─────────────────────────────────────
    needed = {"WX", "KR", "AD", "Hybrid"}
    if set(pre_irfs.keys()) >= needed:
        plot_four_shocks_pre(
            h_pre,
            macro_vars_out,
            irf_dict = {
                "WX":     pre_irfs["WX"],
                "KR":     pre_irfs["KR"],
                "AD":     pre_irfs["AD"],
                "Hybrid": pre_irfs["Hybrid"],
            },
            title_prefix = "Pre-2008: Four shocks comparison (median IRFs, EBP)",
            out_fig      = fig_compare_pre_four(),
        )
    else:
        print("\nSkipping 4-line PRE figure because not all shocks were detected.")
        print("   Have:", list(pre_irfs.keys()), "Need:", list(needed))

    # ── POST-only overlay: Wu-Xia vs Krippner ────────────────────────────────
    if ("WX" in post_irfs) and ("KR" in post_irfs):
        h_wx, irf_wx_post = post_irfs["WX"]
        h_kr, irf_kr_post = post_irfs["KR"]

        if len(h_wx) != len(h_kr) or not np.allclose(h_wx, h_kr):
            print("\nPOST horizons differ between WX and KR; using WX horizon for overlay.")
            h_overlay = h_wx
        else:
            h_overlay = h_wx

        plot_two_irf_overlay(
            h_overlay,
            irf_wx_post,
            irf_kr_post,
            macro_vars_out,
            labelA       = "Wu-Xia (POST)",
            labelB       = "Krippner (POST)",
            title_prefix = "Post-2008: Wu-Xia vs Krippner (EBP)",
            out_fig      = fig_overlay_post_wxkr(),
        )
    else:
        print("\nSkipping POST WX vs KR overlay (missing WX or KR POST IRFs).")

    print("\nAll done.")
    print(f"Source data: {DATA_PATH}")
    print(f"Outputs saved in: {OUTDIR}")


if __name__ == "__main__":
    main()