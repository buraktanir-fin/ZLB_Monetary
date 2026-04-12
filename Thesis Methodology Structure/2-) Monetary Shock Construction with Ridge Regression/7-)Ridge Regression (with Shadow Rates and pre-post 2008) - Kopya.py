"""
================================================================================
================================================================================
Thesis:  "When Rates Hit Zero: Identifying Monetary Shocks with Shadow Rates"
Author:  Burak Tanir
Date:    April 2026

Purpose
-------
Constructs and compares four monetary policy shock series across the pre-2008
conventional policy period and the post-2008 zero lower bound period. The
shock series produced by this script are the primary empirical outputs of the
thesis and serve as direct inputs to the Bayesian VAR impulse response
analysis reported in Sections 5 and 6.

The four shock series constructed are:
    Hybrid  — Ridge regression residual using FFR changes as the dependent
              variable. Pre-2008 benchmark; replicates Aruoba and Drechsel
              (2024) identification approach.
    AD      — Original Aruoba and Drechsel (2024) shock series loaded
              directly from their data file. Not re-estimated; used as a
              validation reference only.
    Wu      — Ridge regression residual using Wu-Xia shadow rate changes as
              the dependent variable. Primary post-2008 policy instrument.
    Kr      — Ridge regression residual using Krippner shadow rate changes as
              the dependent variable. Post-2008 robustness specification.

Pipeline
--------
    Step 1  — Load the full regressor dataset from dataset_builder.py
    Step 2  — Identify regressor column groups (forecasts, sentiments, FFR lags)
    Step 3  — Load the original Aruoba-Drechsel shock series for benchmarking
    Step 4  — Load and align Wu-Xia and Krippner shadow rate levels
    Step 5  — Construct four shock series via ridge regression
    Step 6  — Split into PRE and POST panels; compute availability statistics
    Step 6+ — PRE-2008 multi-window correlation table (1982, 1990, 1995 starts)
    Step 7  — Compute correlations and error metrics within and across panels
    Step 8  — Generate overlay time series plots and correlation heatmaps
    Step 9  — Generate additional PRE-window graphs

Input
-----
    Forecast and Sentiment Results/AD_full_dataset_ready.csv
    Aruoba Drechsel Data/Aruoba_Drechsel_Data.xlsx  (sheet: Shocks and FFR by meeting)
    Shadow Rates/WuXiaShadowRate.xlsx               (sheet: Data)
    Shadow Rates/krippner_shadow_rate.xlsx           (sheet: D. Month-end SSR series)

Output — all saved to Shock Results/Shadow Rate Results/Pre-Post 2008/
--------------------------------------------------------------------------
    shock_series_pre_post_2008.csv       model_details_pre_post_2008.csv
    corr_PRE.csv    corr_POST.csv        availability_by_period.csv
    metrics_PRE_vs_AD.csv               metrics_PRE_vs_Hybrid.csv
    metrics_POST_Wu_vs_Kr.csv           PRE_window_corr_table.csv
    PRE_overlay.png  POST_overlay.png   PRE_corr.png  POST_corr.png
    PRE_scatter_{series}_AD.png         POST_scatter_Wu_Kr.png
    PRE_{window}_overlay/corr/scatter PNGs

Key settings
------------
    CUTOFF_DATE   : regime split date (2008-10-29)
    ALPHAS        : 30-point log-spaced ridge penalty grid [10^-3, 10^3]
    CV_FOLDS      : cross-validation folds for RidgeCV (default: 10)
    MIN_TRAIN_OBS : minimum non-NaN y observations to fit the ridge

Dependencies
------------
    os, numpy, pandas, matplotlib  (standard / pip install)
    scikit-learn — pip install scikit-learn
    openpyxl     — pip install openpyxl

Usage
-----
    python shock_comparison.py

    Run from the directory containing Forecast and Sentiment Results/,
    Aruoba Drechsel Data/, and Shadow Rates/ folders.
    All output folders are created automatically.
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

# ============================================================
# PRE / POST OCT-2008 COMPARISON FOR SHOCKS
# - Builds: Hybrid (FFR-based), AD original, Wu-Xia shadow, Krippner shadow
# - Handles different start dates by:
#     * aligning each comparison on the intersection of available dates
#     * producing separate PRE and POST panels
# - POST period: compares Wu-Xia vs Krippner (since no FFR after Oct-2008)
# - Saves all outputs into: Shadow Rate Results/Pre-Post 2008/
#
# CHANGES FROM ORIGINAL (two fixes only):
#   FIX 1 — align_monthly_levels_to_meetings() replaced with
#            align_wx_to_meetings_interpolated() which linearly interpolates
#            Wu-Xia to the exact FOMC meeting date instead of using the
#            stale last-month-end value. Krippner uses the original backward
#            merge as before (no change there).
#   FIX 2 — X_wu now uses ffr_lags instead of wu_lags so that Wu-Xia shock
#            ridge removes the same predictable component as the Hybrid shock,
#            making the two residuals directly comparable.
# ============================================================

# ============================================================
# SECTION 1 — DIRECTORIES
# ============================================================
# All folders are resolved relative to the current working directory.
# The output folder hierarchy is created automatically if absent.

BASE_DIR = os.getcwd()

CONTROL_RESULTS_DIR = os.path.join(BASE_DIR, "Forecast and Sentiment Results")  # full regressor dataset
ARUOBA_DIR = os.path.join(BASE_DIR, "Aruoba Drechsel Data")                     # original AD data
SHADOW_DIR = os.path.join(BASE_DIR, "Shadow Rates")                             # Wu-Xia and Krippner

SHOCK_RESULTS_DIR = os.path.join(BASE_DIR, "Shock Results")
SHADOW_OUT_DIR = os.path.join(SHOCK_RESULTS_DIR, "Shadow Rate Results")
OUT_DIR = os.path.join(SHADOW_OUT_DIR, "Pre-Post 2008")                         # all outputs

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# SECTION 2 — INPUT FILES
# ============================================================

FULL_DATASET_PATH = os.path.join(CONTROL_RESULTS_DIR, "AD_full_dataset_ready.csv")

ARUOBA_XLSX = os.path.join(ARUOBA_DIR, "Aruoba_Drechsel_Data.xlsx")
ARUOBA_SHEET = "Shocks and FFR by meeting"
ARUOBA_DATE_COL = "FOMC meeting (scheduled)"
ARUOBA_SHOCK_COL = "Shock"

WU_XIA_PATH = os.path.join(SHADOW_DIR, "WuXiaShadowRate.xlsx")         # sheet: Data
KRIPPNER_PATH = os.path.join(SHADOW_DIR, "krippner_shadow_rate.xlsx")  # sheet: B. Month end SSR series

# ============================================================
# SECTION 3 — TIME SPLIT
# ============================================================
# CUTOFF_DATE separates the conventional (PRE) and ZLB (POST) regimes.
# October 29, 2008 is the last FOMC meeting before the Federal Funds Rate
# reached the effective zero lower bound.

CUTOFF_DATE = pd.to_datetime("2008-10-29")

# ============================================================
# SECTION 4 — RIDGE SETTINGS
# ============================================================
# ALPHAS    : 30 penalty values log-spaced between 10^-3 and 10^3.
#             The optimal value is selected by CV_FOLDS-fold cross-validation.
# CV_FOLDS  : number of folds for RidgeCV (default: 10).
# MIN_TRAIN_OBS : minimum non-NaN y observations needed to fit the ridge.
#                 Must exceed CV_FOLDS to allow valid cross-validation.

ALPHAS = np.logspace(-3, 3, 30)
CV_FOLDS = 10
MIN_TRAIN_OBS = max(30, CV_FOLDS + 5)

# ============================================================
# SECTION 5 — OUTPUT FILE PATHS
# ============================================================

OUT_SHOCKS_CSV = os.path.join(OUT_DIR, "shock_series_pre_post_2008.csv")
OUT_MODEL_DETAILS_CSV = os.path.join(OUT_DIR, "model_details_pre_post_2008.csv")

OUT_PRE_CORR_CSV = os.path.join(OUT_DIR, "corr_PRE.csv")
OUT_POST_CORR_CSV = os.path.join(OUT_DIR, "corr_POST.csv")
OUT_AVAIL_CSV = os.path.join(OUT_DIR, "availability_by_period.csv")

OUT_PRE_METRICS_VS_AD = os.path.join(OUT_DIR, "metrics_PRE_vs_AD.csv")
OUT_PRE_METRICS_VS_HYB = os.path.join(OUT_DIR, "metrics_PRE_vs_Hybrid.csv")
OUT_POST_METRICS_WU_VS_KR = os.path.join(OUT_DIR, "metrics_POST_Wu_vs_Kr.csv")


# ============================================================
# SECTION 6 — HELPER FUNCTIONS
# ============================================================

def ensure_datetime(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    """
    Parse a column to datetime, drop unparseable rows, sort chronologically,
    and deduplicate on that column. Applied consistently to every input
    dataset to guarantee a clean, ordered date index before merging.
    """
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=[col]).sort_values(col)
    df = df.drop_duplicates(subset=[col]).reset_index(drop=True)
    return df


def standardize_series(s: pd.Series) -> pd.Series:
    """
    Standardise a numeric series to zero mean and unit standard deviation
    (population std, ddof=0). Returns the original series unchanged if the
    standard deviation is zero. Used to place shock series on a common scale
    for overlay plots.
    """
    x = pd.to_numeric(s, errors="coerce")
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=0)
    if sd and sd > 0:
        return (x - mu) / sd
    return x


def corr_metrics(a: pd.Series, b: pd.Series):
    """
    Compute Pearson and Spearman correlations and error metrics between
    two shock series on their overlapping non-NaN observations.

    Returns a dict with: n, pearson, spearman, mae, rmse.
    All metrics are NaN if fewer than 10 overlapping observations exist.
    """
    valid = a.notna() & b.notna()
    n = int(valid.sum())
    if n < 10:
        return {"n": n, "pearson": np.nan, "spearman": np.nan, "mae": np.nan, "rmse": np.nan}
    av = a[valid].astype(float)
    bv = b[valid].astype(float)
    diff = av - bv
    return {
        "n": n,
        "pearson": float(av.corr(bv, method="pearson")),
        "spearman": float(av.corr(bv, method="spearman")),
        "mae": float(diff.abs().mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
    }


def ridge_shock(df: pd.DataFrame, y_col: str, x_cols: list, model_name: str):
    """
    AD-style ridge residual:
      - drop all-NaN X columns
      - fill remaining NaN with column mean
      - standardize (fit on training subset)
      - RidgeCV
      - shock = y - y_hat

    Parameters
    ----------
    df         : DataFrame containing the dependent variable and regressors
    y_col      : name of the dependent variable column (e.g. 'FFR_change')
    x_cols     : list of regressor column names
    model_name : label used in error messages for diagnostics

    Returns
    -------
    shock    : np.ndarray of residuals (NaN where y is NaN)
    y_hat    : np.ndarray of fitted values
    alpha    : selected ridge penalty value
    x_cols   : regressor columns actually used after dropping all-NaN cols
    """
    work = df[["Date", y_col] + x_cols].copy()
    y = pd.to_numeric(work[y_col], errors="coerce").astype(float).to_numpy()
    X = work[x_cols].apply(pd.to_numeric, errors="coerce").to_numpy()

    if X.size == 0:
        return np.full_like(y, np.nan, float), np.full_like(y, np.nan, float), np.nan, []

    col_nan_all = np.isnan(X).all(axis=0)
    if col_nan_all.any():
        X = X[:, ~col_nan_all]
        x_cols = [c for i, c in enumerate(x_cols) if not col_nan_all[i]]

    for j in range(X.shape[1]):
        col = X[:, j]
        m = np.isnan(col)
        if m.any():
            mean_val = np.nanmean(col)
            if np.isnan(mean_val):
                mean_val = 0.0
            col[m] = mean_val
            X[:, j] = col

    if np.isnan(X).any():
        raise ValueError(f"[{model_name}] Still NaN in X after cleaning.")

    valid_y = ~np.isnan(y)
    if valid_y.sum() < MIN_TRAIN_OBS:
        return np.full_like(y, np.nan, float), np.full_like(y, np.nan, float), np.nan, x_cols

    X_train = X[valid_y]
    y_train = y[valid_y]

    # Fit scaler on training rows only; transform all rows for prediction
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_all_std = scaler.transform(X)

    ridge = RidgeCV(alphas=ALPHAS, cv=CV_FOLDS)
    ridge.fit(X_train_std, y_train)

    y_hat = ridge.predict(X_all_std)
    shock = y - y_hat
    return shock, y_hat, float(ridge.alpha_), x_cols


def add_lags(df: pd.DataFrame, base_col: str, prefix: str):
    """
    Add a one-meeting lag and its square of `base_col` to `df`.
    Returns a list of the two new column names created.
    Used to add instrument-specific autoregressive controls for the
    Krippner ridge regression (Krippner lags remain instrument-specific
    and are unchanged from the original version).
    """
    x = pd.to_numeric(df[base_col], errors="coerce")
    df[f"{prefix}_lag1"] = x.shift(1)
    df[f"{prefix}_lag1_sq"] = df[f"{prefix}_lag1"] ** 2
    return [f"{prefix}_lag1", f"{prefix}_lag1_sq"]


def plot_overlay(df_plot: pd.DataFrame, cols: list, title: str, outpath: str):
    """
    Plot multiple standardised shock series on a shared time axis and save
    as a 200 dpi PNG. Each series appears as a separate labelled line.
    """
    plt.figure(figsize=(12, 5))
    for c in cols:
        plt.plot(df_plot["Date"], df_plot[c], label=c, linewidth=1)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Std")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_heatmap(corr_df: pd.DataFrame, title: str, outpath: str):
    """
    Plot a Pearson correlation matrix as a colour-mapped heatmap and save
    as a 200 dpi PNG. Used to visualise cross-shock correlations within
    each policy regime.
    """
    plt.figure(figsize=(9, 7))
    plt.imshow(corr_df.values, aspect="auto")
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr_df.index)), corr_df.index, fontsize=7)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ============================================================
# SECTION 7 — SHADOW RATE LOADERS
# ============================================================

def load_wu_xia_levels(path: str) -> pd.DataFrame:
    """
    Load monthly Wu-Xia shadow rate levels from the 'Data' sheet.

    Auto-detects the date column (first datetime-typed column, or first
    column as fallback) and the level column (first column whose name
    contains both 'wu' and 'shadow'; fallback: most-populated numeric
    column). Returns a DataFrame with columns ['Date', 'level'].
    """
    # MUST use sheet: Data
    raw = pd.read_excel(path, sheet_name="Data")
    raw.columns = [str(c).strip() for c in raw.columns]

    date_col = None
    for c in raw.columns:
        if pd.api.types.is_datetime64_any_dtype(raw[c]):
            date_col = c
            break
    if date_col is None:
        date_col = raw.columns[0]

    level_col = None
    for c in raw.columns:
        if c == date_col:
            continue
        cl = str(c).lower()
        if ("wu" in cl) and ("shadow" in cl):
            if pd.to_numeric(raw[c], errors="coerce").notna().sum() > 0:
                level_col = c
                break
    if level_col is None:
        best, best_n = None, -1
        for c in raw.columns:
            if c == date_col:
                continue
            n = pd.to_numeric(raw[c], errors="coerce").notna().sum()
            if n > best_n:
                best, best_n = c, n
        level_col = best

    out = raw[[date_col, level_col]].copy()
    out.columns = ["Date", "level"]
    out = ensure_datetime(out, "Date")
    out["level"] = pd.to_numeric(out["level"], errors="coerce")
    out = out.dropna(subset=["level"])[["Date", "level"]].copy()
    return out


def load_krippner_levels_point(path: str) -> pd.DataFrame:
    """
    Load monthly Krippner shadow rate levels from the
    'D. Month-end SSR series' sheet.

    Locates the 'US SSR' header cell to find the data start row, then
    extracts the date and level columns immediately beneath it.
    Returns a DataFrame with columns ['Date', 'level'].
    """
    raw = pd.read_excel(path, sheet_name="D. Month-end SSR series", header=None)

    hit = np.where(raw.astype(str).values == "US SSR")
    if len(hit[0]) == 0:
        raise ValueError("Could not locate 'US SSR' header in Krippner month-end sheet.")
    header_row = int(hit[0][0])

    data = raw.iloc[header_row + 1:, [0, 1]].copy()
    data.columns = ["Date", "level"]

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["level"] = pd.to_numeric(data["level"], errors="coerce")
    data = data.dropna(subset=["Date", "level"]).sort_values("Date")
    data = data.drop_duplicates(subset=["Date"]).reset_index(drop=True)
    return data[["Date", "level"]].copy()


# ============================================================
# SECTION 8 — SHADOW RATE ALIGNMENT FUNCTIONS
# ============================================================
# FIX 1 applied here: Wu-Xia now uses linear interpolation to exact meeting
# dates. Krippner continues to use the original backward merge (unchanged).

def align_wx_to_meetings_interpolated(
    meeting_dates: pd.Series,
    monthly_levels: pd.DataFrame,
    out_level_col: str,
) -> pd.DataFrame:
    """
    Interpolate monthly Wu-Xia levels to exact FOMC meeting dates using
    linear interpolation on a daily grid (FIX 1 — replaces backward merge).

    OLD behaviour: merge_asof(backward) — stale last-month-end value.
        e.g. meeting on Nov 3  gets Oct 31 WX level (3 days stale)
             meeting on Nov 28 gets Oct 31 WX level (28 days stale)

    NEW behaviour: linearly interpolate WX to a daily series, then snap
        to the exact meeting date — eliminates staleness entirely.
        Krippner still uses the original backward merge (unchanged).
    """
    # Build a daily series by resampling and linearly interpolating
    wx_daily = (
        monthly_levels
        .set_index("Date")["level"]
        .resample("D")
        .interpolate(method="linear")
    )

    # Snap to each exact meeting date (nearest available day)
    mtg = pd.to_datetime(meeting_dates)
    aligned_levels = wx_daily.reindex(mtg, method="nearest")

    result = pd.DataFrame({
        "Date": mtg.values,
        out_level_col: aligned_levels.values,
    })
    return result


# Keep original function for Krippner — no change
def align_monthly_levels_to_meetings(
    meeting_dates: pd.Series,
    monthly_levels: pd.DataFrame,
    out_level_col: str,
) -> pd.DataFrame:
    """
    Align by last available monthly observation on or before each meeting
    date using a backward merge. Used for Krippner — unchanged from the
    original version.
    """
    m = pd.DataFrame({"Date": pd.to_datetime(meeting_dates)}).sort_values("Date").reset_index(drop=True)
    x = monthly_levels.sort_values("Date").reset_index(drop=True)

    aligned = pd.merge_asof(m, x, on="Date", direction="backward")
    aligned = aligned.rename(columns={"level": out_level_col})
    return aligned[["Date", out_level_col]].copy()


def meeting_to_meeting_change(df_dates: pd.Series, level_df: pd.DataFrame, level_col: str, out_change_col: str):
    """
    Compute the meeting-to-meeting first difference of a shadow rate level
    series aligned to the FOMC meeting calendar. Change at meeting t equals
    level(t) minus level(t-1).
    """
    tmp = pd.DataFrame({"Date": pd.to_datetime(df_dates)}).merge(level_df, on="Date", how="left")
    tmp[out_change_col] = tmp[level_col] - tmp[level_col].shift(1)
    return tmp[["Date", out_change_col]].copy()


# ============================================================
# SECTION 9 — STEP 1: LOAD FULL DATASET
# ============================================================
# Loads the wide-format regressor dataset produced by dataset_builder.py.
# Contains Date, FFR_change, Shock, forecast regressors, sentiment
# regressors, lagged variables, and quadratic terms.

if not os.path.exists(FULL_DATASET_PATH):
    raise FileNotFoundError(f"Full dataset not found: {FULL_DATASET_PATH}")

df = pd.read_csv(FULL_DATASET_PATH)
if "Date" not in df.columns:
    raise ValueError("Dataset must have a Date column.")
df = ensure_datetime(df, "Date")

# ============================================================
# SECTION 10 — STEP 2: IDENTIFY X GROUPS
# ============================================================
# Three regressor groups are identified by column naming conventions:
#   forecast_cols  — staff forecast levels, first differences, and squares
#   sentiment_cols — z-scored concept scores, lags, and squares
#   ffr_lags       — one-meeting lag of FFR_change and its square
#
# FIX 2: X_wu uses ffr_lags (not wu_lags) so the Wu and Hybrid residuals
# remove the same predictable autoregressive component, making them directly
# comparable across regimes.

forecast_cols = [
    c for c in df.columns
    if (("B" in c or "F" in c) and ("_diff" in c or "_level" in c or c.endswith("_sq")))
]

sentiment_cols = [
    c for c in df.columns
    if ("_score_per_10k_words_z" in c) or
       ("_lag" in c and "_score" in c) or
       (c.endswith("_sq") and "_score" in c)
]

ffr_lags = [c for c in ["FFR_lag1", "FFR_lag1_sq"] if c in df.columns]

print("Forecast cols:", len(forecast_cols))
print("Sentiment cols:", len(sentiment_cols))
print("FFR lags:", ffr_lags)

# ============================================================
# SECTION 11 — STEP 3: LOAD ARUOBA ORIGINAL SHOCKS
# ============================================================
# The AD shock series is loaded for benchmarking only — it is not
# re-estimated. Missing meetings (dates not in the AD dataset) receive NaN.

if not os.path.exists(ARUOBA_XLSX):
    raise FileNotFoundError(f"Aruoba file not found: {ARUOBA_XLSX}")

aru = pd.read_excel(ARUOBA_XLSX, sheet_name=ARUOBA_SHEET)
if ARUOBA_DATE_COL not in aru.columns or ARUOBA_SHOCK_COL not in aru.columns:
    raise ValueError(f"Aruoba sheet must contain '{ARUOBA_DATE_COL}' and '{ARUOBA_SHOCK_COL}'.")

aru["Date"] = pd.to_datetime(aru[ARUOBA_DATE_COL].astype(str).str.replace("_", "-"), errors="coerce")
aru = ensure_datetime(aru, "Date")
aru["AD"] = pd.to_numeric(aru[ARUOBA_SHOCK_COL], errors="coerce")
aru = aru.dropna(subset=["AD"])[["Date", "AD"]].copy()

df = df.merge(aru, on="Date", how="left")

# ============================================================
# SECTION 12 — STEP 4: BUILD SHADOW RATE CHANGES ON MEETING GRID
# ============================================================
# Wu-Xia uses FIX 1 (interpolated alignment to exact meeting dates).
# Krippner uses the original backward merge — unchanged.

meeting_dates = df["Date"].copy()

if not os.path.exists(WU_XIA_PATH):
    raise FileNotFoundError(f"Wu-Xia file not found: {WU_XIA_PATH}")
wu_levels_m = load_wu_xia_levels(WU_XIA_PATH)
wu_aligned  = align_wx_to_meetings_interpolated(meeting_dates, wu_levels_m, "Wu_level")
wu_change = meeting_to_meeting_change(meeting_dates, wu_aligned, "Wu_level", "Wu_change")
df = df.merge(wu_change, on="Date", how="left")

if not os.path.exists(KRIPPNER_PATH):
    raise FileNotFoundError(f"Krippner file not found: {KRIPPNER_PATH}")

kr_levels_m = load_krippner_levels_point(KRIPPNER_PATH)
kr_aligned = align_monthly_levels_to_meetings(meeting_dates, kr_levels_m, "Kr_level")
kr_change = meeting_to_meeting_change(meeting_dates, kr_aligned, "Kr_level", "Kr_change")
df = df.merge(kr_change, on="Date", how="left")

# ============================================================
# SECTION 13 — STEP 5: BUILD SHOCK SERIES
# ============================================================
# Each shock = policy instrument change minus ridge-predicted systematic
# component. Model details (selected alpha, regressor count) are recorded.

details = []

shock = pd.DataFrame({"Date": df["Date"]})

# 5.1 Hybrid (FFR-based, pre-2008 benchmark)
if "FFR_change" in df.columns:
    X_hyb = forecast_cols + sentiment_cols + ffr_lags
    hyb, hyb_hat, alpha, used = ridge_shock(df, "FFR_change", X_hyb, "Hybrid")
    shock["Hybrid"] = hyb
    details.append({"model": "Hybrid", "y": "FFR_change", "alpha": alpha, "k": len(used)})
else:
    shock["Hybrid"] = np.nan
    details.append({"model": "Hybrid", "y": "FFR_change", "alpha": np.nan, "k": np.nan})

# 5.2 Aruoba original shock (benchmark, not re-estimated)
shock["AD"] = df["AD"]
details.append({"model": "AD", "y": "Aruoba", "alpha": np.nan, "k": np.nan})

# 5.3 Wu-Xia — FIX 2: use ffr_lags instead of wu_lags in X
# Rationale: using the same lag controls as the Hybrid shock means the ridge
# removes the same predictable component from Wu-Xia changes, making the
# two residuals directly comparable rather than residuals from different
# autoregressive structures.
# wu_lags = add_lags(df, "Wu_change", "Wu")   <- REMOVED
X_wu = forecast_cols + sentiment_cols + ffr_lags  # CHANGED: ffr_lags not wu_lags
wu_sh, wu_hat, wu_alpha, used = ridge_shock(df, "Wu_change", X_wu, "Wu")
shock["Wu"] = wu_sh
details.append({"model": "Wu", "y": "Wu_change", "alpha": wu_alpha, "k": len(used)})

# 5.4 Krippner — unchanged
kr_lags = add_lags(df, "Kr_change", "Kr")
X_kr = forecast_cols + sentiment_cols + kr_lags
kr_sh, kr_hat, kr_alpha, used = ridge_shock(df, "Kr_change", X_kr, "Kr")
shock["Kr"] = kr_sh
details.append({"model": "Kr", "y": "Kr_change", "alpha": kr_alpha, "k": len(used)})

# Save series + details
shock.to_csv(OUT_SHOCKS_CSV, index=False)
pd.DataFrame(details).to_csv(OUT_MODEL_DETAILS_CSV, index=False)
print("Saved:", OUT_SHOCKS_CSV)
print("Saved:", OUT_MODEL_DETAILS_CSV)

# ============================================================
# SECTION 14 — STEP 6: PRE / POST SPLIT AND AVAILABILITY
# ============================================================
# Observations on or before CUTOFF_DATE are labelled PRE; later dates POST.
# Availability statistics (non-missing count, start, end) are saved for
# each series and period to support sample description in the thesis.

shock["period"] = np.where(shock["Date"] <= CUTOFF_DATE, "PRE", "POST")

avail_rows = []
for p in ["PRE", "POST"]:
    sub = shock.loc[shock["period"] == p].copy()
    for c in ["Hybrid", "AD", "Wu", "Kr"]:
        avail_rows.append({
            "period": p,
            "series": c,
            "n_non_missing": int(sub[c].notna().sum()),
            "start": sub.loc[sub[c].notna(), "Date"].min(),
            "end": sub.loc[sub[c].notna(), "Date"].max(),
        })
avail = pd.DataFrame(avail_rows)
avail.to_csv(OUT_AVAIL_CSV, index=False)
print("Saved:", OUT_AVAIL_CSV)

# ============================================================
# SECTION 15 — PRE-2008 MULTI-WINDOW CORRELATION TABLE
# ============================================================
# Three start-date windows (1982, 1990, 1995) assess whether Hybrid vs
# Wu-Xia and Hybrid vs Krippner correlations are stable across sub-samples.
# The 1990 window supports Section 6.1 of the thesis (Wu-Xia data is
# unavailable before 1990).

PRE_WINDOWS = [
    ("1982-2008", "1982-01-01", "2008-10-29"),
    ("1990-2008", "1990-01-01", "2008-10-29"),
    ("1995-2008", "1995-01-01", "2008-10-29"),
]

def window_corr_metrics(df_sub: pd.DataFrame, a: str, b: str):
    """
    Compute Pearson and Spearman correlations for two shock columns within
    a date-restricted sub-sample. Returns NaN if fewer than 10 overlapping
    observations are available.
    """
    valid = df_sub[a].notna() & df_sub[b].notna()
    n = int(valid.sum())
    if n < 10:
        return {"n": n, "pearson": np.nan, "spearman": np.nan}
    av = df_sub.loc[valid, a].astype(float)
    bv = df_sub.loc[valid, b].astype(float)
    return {
        "n": n,
        "pearson": float(av.corr(bv, method="pearson")),
        "spearman": float(av.corr(bv, method="spearman")),
    }

rows = []
for label, start_s, end_s in PRE_WINDOWS:
    start = pd.to_datetime(start_s)
    end = pd.to_datetime(end_s)
    pre_w = shock[(shock["Date"] >= start) & (shock["Date"] <= end)].copy()

    m = window_corr_metrics(pre_w, "Hybrid", "Wu")
    rows.append({"window": label, "pair": "Hybrid-Wu", "n": m["n"],
                 "pearson": m["pearson"], "spearman": m["spearman"]})

    m = window_corr_metrics(pre_w, "Hybrid", "Kr")
    rows.append({"window": label, "pair": "Hybrid-Kr", "n": m["n"],
                 "pearson": m["pearson"], "spearman": m["spearman"]})

pre_window_corr = pd.DataFrame(rows)
pearson_table  = pre_window_corr.pivot(index="window", columns="pair", values="pearson")
spearman_table = pre_window_corr.pivot(index="window", columns="pair", values="spearman")
n_table        = pre_window_corr.pivot(index="window", columns="pair", values="n")

final_table = pearson_table.add_prefix("pearson_").join(
    spearman_table.add_prefix("spearman_")
).join(
    n_table.add_prefix("n_")
).reset_index()

OUT_PRE_WINDOW_TABLE = os.path.join(OUT_DIR, "PRE_window_corr_table.csv")
final_table.to_csv(OUT_PRE_WINDOW_TABLE, index=False)
print("\nSaved PRE window correlation table ->", OUT_PRE_WINDOW_TABLE)
print("\n=== PRE WINDOW CORRELATION TABLE (Hybrid vs Shadow) ===")
print(final_table.to_string(index=False))

# ============================================================
# SECTION 16 — STEP 7: CORRELATIONS AND METRICS
# ============================================================
# Separate correlation matrices and pairwise metric tables are computed
# for PRE and POST periods.
# PRE panel: all four series (Hybrid, AD, Wu, Kr).
# POST panel: Wu and Kr only — Hybrid and AD are not defined in ZLB period.

def corr_table(df_sub: pd.DataFrame, cols: list):
    return df_sub[cols].corr(method="pearson", min_periods=20)

pre  = shock[shock["period"] == "PRE"].copy()
post = shock[shock["period"] == "POST"].copy()

pre_cols  = ["Hybrid", "AD", "Wu", "Kr"]
pre_corr  = corr_table(pre, pre_cols)
pre_corr.to_csv(OUT_PRE_CORR_CSV)
print("Saved:", OUT_PRE_CORR_CSV)

post_cols = ["Wu", "Kr"]
post_corr = corr_table(post, post_cols)
post_corr.to_csv(OUT_POST_CORR_CSV)
print("Saved:", OUT_POST_CORR_CSV)

# PRE: Hybrid / Wu / Kr vs AD (benchmark comparison)
rows = []
for v in ["Hybrid", "Wu", "Kr"]:
    m = corr_metrics(pre[v], pre["AD"])
    m.update({"variant": v, "ref": "AD"})
    rows.append(m)
pre_vs_ad = pd.DataFrame(rows).sort_values("pearson", ascending=False)
pre_vs_ad.to_csv(OUT_PRE_METRICS_VS_AD, index=False)
print("Saved:", OUT_PRE_METRICS_VS_AD)

# PRE: AD / Wu / Kr vs Hybrid (replicated series as reference)
rows = []
for v in ["AD", "Wu", "Kr"]:
    m = corr_metrics(pre[v], pre["Hybrid"])
    m.update({"variant": v, "ref": "Hybrid"})
    rows.append(m)
pre_vs_hyb = pd.DataFrame(rows).sort_values("pearson", ascending=False)
pre_vs_hyb.to_csv(OUT_PRE_METRICS_VS_HYB, index=False)
print("Saved:", OUT_PRE_METRICS_VS_HYB)

# POST: Wu vs Kr (instrument selection validation for Section 6.2)
m = corr_metrics(post["Wu"], post["Kr"])
post_wu_vs_kr = pd.DataFrame([{"variant": "Wu", "ref": "Kr", **m}])
post_wu_vs_kr.to_csv(OUT_POST_METRICS_WU_VS_KR, index=False)
print("Saved:", OUT_POST_METRICS_WU_VS_KR)

if not pre_vs_ad.empty:
    print("\nPRE vs AD — BEST (Pearson):")
    print(pre_vs_ad.iloc[0][["variant", "n", "pearson", "spearman", "mae", "rmse"]].to_string())
    print("\nPRE vs AD — WORST (Pearson):")
    print(pre_vs_ad.iloc[-1][["variant", "n", "pearson", "spearman", "mae", "rmse"]].to_string())

# ============================================================
# SECTION 17 — STEP 8: GRAPHS
# ============================================================
# Standardised overlay plots and correlation heatmaps for PRE and POST.
# Pairwise scatter plots for each non-AD series vs AD (PRE) and Wu vs Kr (POST).

pre_plot = pre[["Date", "Hybrid", "AD", "Wu", "Kr"]].copy()
for c in ["Hybrid", "AD", "Wu", "Kr"]:
    pre_plot[c] = standardize_series(pre_plot[c])

plot_overlay(pre_plot, ["Hybrid", "AD", "Wu", "Kr"], "PRE Shocks (Std)",
             os.path.join(OUT_DIR, "PRE_overlay.png"))
plot_heatmap(pre_corr, "PRE Corr", os.path.join(OUT_DIR, "PRE_corr.png"))

post_plot = post[["Date", "Wu", "Kr"]].copy()
post_plot["Wu"] = standardize_series(post_plot["Wu"])
post_plot["Kr"] = standardize_series(post_plot["Kr"])

plot_overlay(post_plot, ["Wu", "Kr"], "POST Shocks (Std)",
             os.path.join(OUT_DIR, "POST_overlay.png"))
plot_heatmap(post_corr, "POST Corr", os.path.join(OUT_DIR, "POST_corr.png"))

# PRE scatter plots: each series vs AD
for v in ["Hybrid", "Wu", "Kr"]:
    x = pre[v]; y = pre["AD"]
    valid = x.notna() & y.notna()
    if valid.sum() < 30:
        continue
    pear = float(x[valid].corr(y[valid], method="pearson"))
    plt.figure(figsize=(6, 6))
    plt.scatter(x[valid], y[valid], alpha=0.5)
    plt.xlabel(v); plt.ylabel("AD")
    plt.title(f"PRE {v} vs AD ({pear:.3f})")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"PRE_scatter_{v}_AD.png"), dpi=200)
    plt.close()

# POST scatter plot: Wu vs Kr
x = post["Wu"]; y = post["Kr"]
valid = x.notna() & y.notna()
if valid.sum() >= 30:
    pear = float(x[valid].corr(y[valid], method="pearson"))
    plt.figure(figsize=(6, 6))
    plt.scatter(x[valid], y[valid], alpha=0.5)
    plt.xlabel("Wu"); plt.ylabel("Kr")
    plt.title(f"POST Wu vs Kr ({pear:.3f})")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "POST_scatter_Wu_Kr.png"), dpi=200)
    plt.close()

print("\nAll PRE/POST outputs saved under:")
print(OUT_DIR)

# ============================================================
# SECTION 18 — STEP 9: EXTRA PRE-WINDOW GRAPHS (1982/1990/1995)
# ============================================================
# Overlay plots, correlation heatmaps, and pairwise scatter plots for each
# start-date window. Supports the sample-controlled robustness exercise in
# Section 6.1 of the thesis.

PRE_END = CUTOFF_DATE

pre_windows = [
    ("1982_2008", pd.to_datetime("1982-01-01"), PRE_END),
    ("1990_2008", pd.to_datetime("1990-01-01"), PRE_END),
    ("1995_2008", pd.to_datetime("1995-01-01"), PRE_END),
]

win_cols = ["Hybrid", "Wu", "Kr"]

def safe_corr(df_sub, cols):
    return df_sub[cols].corr(method="pearson", min_periods=20)

for tag, start_dt, end_dt in pre_windows:
    win = shock[(shock["Date"] >= start_dt) & (shock["Date"] <= end_dt)].copy()
    win = win[["Date"] + win_cols].copy()
    win = win.dropna(how="all", subset=win_cols)

    win_std = win.copy()
    for c in win_cols:
        win_std[c] = standardize_series(win_std[c])

    plot_overlay(win_std, win_cols, f"PRE {tag} (Std)",
                 os.path.join(OUT_DIR, f"PRE_{tag}_overlay.png"))

    corr_df = safe_corr(win, win_cols)
    plot_heatmap(corr_df, f"PRE {tag} Corr",
                 os.path.join(OUT_DIR, f"PRE_{tag}_corr.png"))

    for a_col, b_col in [("Hybrid", "Wu"), ("Hybrid", "Kr")]:
        if a_col in win.columns and b_col in win.columns:
            x = win[a_col]; y = win[b_col]
            valid = x.notna() & y.notna()
            if valid.sum() >= 30:
                pear = float(x[valid].corr(y[valid], method="pearson"))
                plt.figure(figsize=(6, 6))
                plt.scatter(x[valid], y[valid], alpha=0.5)
                plt.xlabel(a_col); plt.ylabel(b_col)
                plt.title(f"PRE {tag} {a_col} vs {b_col} ({pear:.3f})")
                plt.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig(os.path.join(OUT_DIR,
                            f"PRE_{tag}_scatter_{a_col[0]}_{b_col}.png"), dpi=200)
                plt.close()

print("Extra PRE-window graphs saved (no overwrites).")