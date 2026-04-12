"""
================================================================================
bvar_dataset_builder.py
================================================================================
Thesis:  "When Rates Hit Zero: Identifying Monetary Shocks with Shadow Rates"
Author:  Burak Tanir
Date:    April 2026

Purpose
-------
Constructs the monthly macroeconomic dataset used as input to the Bayesian
Vector Autoregression (BVAR) described in Section 4.4 of the thesis. The
dataset combines monthly FRED macroeconomic series, S&P 500 equity prices,
the Gilchrist-Zakrajsek (2012) excess bond premium, and meeting-level
monetary policy shocks aggregated to monthly frequency.

Three output files are produced covering the full sample and the pre- and
post-2008 subsamples used in the main results and robustness sections.

BVAR variable set (Section 4.4)
--------------------------------
    y1          — 1-year Treasury yield (GS1, FRED)
    log_sp500   — log S&P 500 index (SPX.csv)
    log_rgdp    — log real GDP, interpolated from quarterly to monthly
                  via Kalman smoother (GDPC1, FRED)
    log_pgdp    — log GDP deflator, interpolated from quarterly to monthly
                  via Kalman smoother (GDPDEF, FRED)
    unemp       — unemployment rate (UNRATE, FRED)
    EBP         — excess bond premium (Gilchrist and Zakrajsek, 2012)
    shock_*     — meeting-level shocks aggregated to monthly by summing
                  all FOMC meetings falling within each calendar month

Quarterly to monthly interpolation
------------------------------------
Real GDP and the GDP deflator are published quarterly. They are interpolated
to monthly frequency using a Kalman smoother (UnobservedComponents local
level model with seasonal component). This preserves the quarterly totals
while providing smooth monthly estimates consistent with the BVAR's monthly
timing.

Shock aggregation
-----------------
Meeting-level shocks (Date-indexed, one row per FOMC meeting) are mapped to
calendar months by summing all meetings within each month. Months with no
FOMC meeting receive a shock value of zero. All shock columns in the input
file are aggregated; the AD and Hybrid columns are additionally guaranteed
to be included if present under any common naming variant.

Sample periods
--------------
    Full sample  : October 1982 – December 2019
    Pre-2008     : October 1982 – October 2008  (conventional policy era)
    Post-2008    : November 2008 – December 2019 (zero lower bound era)

The October 2008 cutoff corresponds to the final FOMC meeting at which the
Federal Funds Rate was still positive, consistent with the pre/post split
used throughout the thesis.

Input
-----
    FRED API (live download):
        GS1     — 1-year Treasury constant maturity rate (monthly)
        UNRATE  — Civilian unemployment rate (monthly)
        GDPC1   — Real GDP, seasonally adjusted (quarterly)
        GDPDEF  — GDP deflator, seasonally adjusted (quarterly)

    inputs/SPX.csv
        S&P 500 index with columns: Date (or dt/time), Close (or Adj Close).
        Monthly value: first daily observation of the month.

    inputs/EBP.csv
        Gilchrist-Zakrajsek excess bond premium with columns:
        Date, ebp. Monthly value: mean within month.

    inputs/shock_series_pre_post_2008.csv
        Meeting-level shocks produced by shock_comparison.py.
        Contains Date and one column per shock series
        (Hybrid, AD, Wu, Kr at minimum).

Output
------
    outputs/bvar_monthly_dataset_FULL_with_EBP_with_ALL_meeting_shockSUM.csv
        Full sample: October 1982 – December 2019

    outputs/bvar_monthly_dataset_PRE2008_with_EBP_with_ALL_meeting_shockSUM.csv
        Pre-2008 subsample: October 1982 – October 2008

    outputs/bvar_monthly_dataset_POST2008_with_EBP_with_ALL_meeting_shockSUM.csv
        Post-2008 subsample: November 2008 – December 2019

    All files are indexed by DATE (monthly, month-start format).

Dependencies
------------
    os, typing  (standard library)
    pandas, numpy         — pip install pandas numpy
    fredapi               — pip install fredapi
    statsmodels           — pip install statsmodels
    Python 3.7+ compatible

Usage
-----
    python bvar_dataset_builder.py

    Requires an active internet connection for FRED data download.
    Place SPX.csv, EBP.csv, and shock_series_pre_post_2008.csv in the
    inputs/ folder before running. The outputs/ folder is created
    automatically.
================================================================================
"""

import pandas as pd
import numpy as np
from fredapi import Fred
from statsmodels.tsa.statespace.structural import UnobservedComponents
import os
from typing import Optional, List


# ============================================================
# SECTION 1 — CONFIGURATION
# ============================================================
# All paths are resolved relative to the current working directory.
# The FRED API key is required for live data download from FRED.

BASE_DIR   = os.getcwd()
INPUT_DIR  = os.path.join(BASE_DIR, "inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FRED API key — required for downloading macroeconomic series
FRED_API_KEY = #insertyourapikey
fred = Fred(api_key=FRED_API_KEY)


# ============================================================
# SECTION 2 — SAMPLE PERIOD AND SPLIT DATES
# ============================================================
# Full sample spans October 1982 to December 2019 (monthly, month-start).
# The pre/post split at October 2008 is consistent with the CUTOFF_DATE
# used in shock_comparison.py: the last FOMC meeting with a positive FFR.

START_FULL = "1982-10-01"   # first FOMC meeting in the thesis sample
END_FULL   = "2019-12-01"   # last month of the thesis sample

PRE_END_MS    = "2008-10-01"   # last month of the pre-2008 (conventional) era
POST_START_MS = "2008-11-01"   # first month of the post-2008 (ZLB) era


# ============================================================
# SECTION 3 — INPUT FILE PATHS
# ============================================================

# S&P 500 daily prices — aggregated to monthly first observation
SPX_PATH = os.path.join(INPUT_DIR, "SPX.csv")

# Gilchrist-Zakrajsek (2012) excess bond premium — aggregated to monthly mean
EBP_PATH = os.path.join(INPUT_DIR, "EBP.csv")

# Meeting-level shock series produced by shock_comparison.py
# Contains one row per FOMC meeting with columns: Date, Hybrid, AD, Wu, Kr
MEETING_SHOCK_PATH = os.path.join(INPUT_DIR, "shock_series_pre_post_2008.csv")

# Preferred date column name in the meeting shocks file
# Set to None to trigger auto-detection
MEETING_DATE_COL: Optional[str] = "Date"


# ============================================================
# SECTION 4 — OUTPUT FILE PATHS
# ============================================================
# Three datasets are saved: full sample, pre-2008, and post-2008.
# File names encode the sample period and the shock aggregation method
# (SUM within month) for unambiguous identification.

OUT_FULL = os.path.join(OUTPUT_DIR, "bvar_monthly_dataset_FULL_with_EBP_with_ALL_meeting_shockSUM.csv")
OUT_PRE  = os.path.join(OUTPUT_DIR, "bvar_monthly_dataset_PRE2008_with_EBP_with_ALL_meeting_shockSUM.csv")
OUT_POST = os.path.join(OUTPUT_DIR, "bvar_monthly_dataset_POST2008_with_EBP_with_ALL_meeting_shockSUM.csv")


# ============================================================
# SECTION 5 — HELPER FUNCTIONS: DATE UTILITIES
# ============================================================

def to_month_start(x):
    """
    Convert dates or a date Series to month-start timestamps.

    Used throughout to align all data to a common monthly index in
    month-start (MS) format, regardless of the original day within the month.

    Parameters
    ----------
    x : pd.Series or pd.DatetimeIndex
        Input dates to convert.

    Returns
    -------
    pd.Series or pd.DatetimeIndex
        Dates normalised to the first day of their respective months.
    """
    if isinstance(x, pd.Series):
        dt = pd.to_datetime(x, errors="coerce")
        return dt.dt.to_period("M").dt.to_timestamp()
    else:
        idx = pd.to_datetime(x, errors="coerce")
        return idx.to_period("M").to_timestamp()


def robust_to_datetime(s: pd.Series) -> pd.Series:
    """
    Robustly parse a date Series, with a dayfirst fallback.

    Attempts standard date parsing first. If more than 5% of values
    fail to parse (NaT), retries with dayfirst=True. This handles
    date formats such as DD/MM/YYYY that are common in some financial
    data sources.

    Parameters
    ----------
    s : pd.Series
        Raw date strings to parse.

    Returns
    -------
    pd.Series
        Parsed datetime Series with the fewest NaT values.
    """
    dt = pd.to_datetime(s, errors="coerce")
    nat_share = dt.isna().mean()
    if nat_share > 0.05:
        dt2 = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if dt2.isna().mean() < nat_share:
            return dt2
    return dt


# ============================================================
# SECTION 6 — HELPER FUNCTIONS: FRED DATA LOADERS
# ============================================================

def fred_monthly(series_id: str, start: str, end: str) -> pd.Series:
    """
    Download a monthly FRED series and align to the month-start index.

    For series with multiple observations per month (e.g. daily data stored
    in FRED), the last observation within each month is used.

    Parameters
    ----------
    series_id : str   FRED series identifier (e.g. 'GS1', 'UNRATE').
    start     : str   Start date string (e.g. '1982-10-01').
    end       : str   End date string (e.g. '2019-12-01').

    Returns
    -------
    pd.Series  Monthly series indexed by month-start timestamps.
    """
    s = fred.get_series(series_id)
    s.index = pd.to_datetime(s.index)
    s = s.loc[pd.to_datetime(start):pd.to_datetime(end)]
    s.index = to_month_start(s.index)
    s = s.groupby(s.index).last()
    return s


def fred_quarterly(series_id: str) -> pd.Series:
    """
    Download a quarterly FRED series without frequency conversion.

    The raw quarterly series is returned for subsequent interpolation
    to monthly frequency via the Kalman smoother.

    Parameters
    ----------
    series_id : str   FRED series identifier (e.g. 'GDPC1', 'GDPDEF').

    Returns
    -------
    pd.Series  Quarterly series with DatetimeIndex.
    """
    s = fred.get_series(series_id)
    s.index = pd.to_datetime(s.index)
    return s


# ============================================================
# SECTION 7 — HELPER FUNCTIONS: QUARTERLY TO MONTHLY INTERPOLATION
# ============================================================

def kalman_quarterly_to_monthly(
    q_series: pd.Series,
    monthly_index: pd.DatetimeIndex,
    add_seasonal: bool = True
) -> pd.Series:
    """
    Interpolate a quarterly series to monthly frequency using a Kalman smoother.

    Fits a local level model (with optional seasonal component) via the
    UnobservedComponents framework. Quarterly observations are placed at
    their quarter-start dates and treated as observed; all intermediate
    monthly dates are treated as missing. The Kalman smoother fills in
    the missing monthly values in a way that is consistent with the
    observed quarterly pattern.

    This approach is used for real GDP (GDPC1) and the GDP deflator
    (GDPDEF), both of which are published at quarterly frequency but
    required at monthly frequency for the BVAR.

    Parameters
    ----------
    q_series      : pd.Series         Raw quarterly FRED series.
    monthly_index : pd.DatetimeIndex  Target monthly index (month-start).
    add_seasonal  : bool              Whether to include a 12-month seasonal
                                      component (default: True).

    Returns
    -------
    pd.Series  Monthly series on monthly_index, smoothed by the Kalman filter.
    """
    s = q_series.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s.index = s.index.to_period("M").to_timestamp()

    s = s.loc[(s.index >= monthly_index.min()) & (s.index <= monthly_index.max())]

    # Build a monthly series with NaN at all non-quarter months
    y = pd.Series(index=monthly_index, dtype="float64")
    y.loc[s.index] = pd.to_numeric(s.values, errors="coerce")

    model = UnobservedComponents(
        y,
        level="local level",
        seasonal=12 if add_seasonal else None
    )
    res = model.fit(disp=False)

    return pd.Series(res.smoothed_state[0], index=monthly_index)


# ============================================================
# SECTION 8 — HELPER FUNCTIONS: MARKET DATA LOADERS
# ============================================================

def load_spx_monthly(path: str) -> pd.Series:
    """
    Load S&P 500 daily prices and aggregate to monthly frequency.

    Auto-detects the date and close price columns by matching common
    column name variants (case-insensitive). The monthly value is the
    first daily observation within each calendar month, consistent with
    the beginning-of-month timing convention used for other BVAR variables.

    Parameters
    ----------
    path : str  Path to SPX.csv. Expected columns: Date, Close (or Adj Close).

    Returns
    -------
    pd.Series  Monthly S&P 500 index level, indexed by month-start dates.
               Series name: 'spx'.
    """
    spx = pd.read_csv(path)

    date_col = next((c for c in spx.columns if c.lower() in ["date", "dt", "time"]), spx.columns[0])
    val_col  = next((c for c in spx.columns if c.lower() in ["close", "adj close", "adj_close"]), spx.columns[-1])

    spx[date_col] = robust_to_datetime(spx[date_col])
    spx = spx.dropna(subset=[date_col])

    spx["MS"] = to_month_start(spx[date_col])

    spx_m = spx.sort_values(date_col).groupby("MS")[val_col].first()
    spx_m = pd.to_numeric(spx_m, errors="coerce")
    spx_m.name = "spx"
    return spx_m


def load_ebp_monthly(path: str, ebp_col: str = "ebp") -> pd.Series:
    """
    Load the Gilchrist-Zakrajsek (2012) excess bond premium and aggregate
    to monthly frequency.

    The monthly value is the mean of all daily observations within each
    calendar month. The EBP column is matched case-insensitively to
    handle minor naming differences across file versions.

    Parameters
    ----------
    path    : str  Path to EBP.csv. Expected columns: Date, ebp.
    ebp_col : str  Name of the EBP column (default: 'ebp').

    Returns
    -------
    pd.Series  Monthly excess bond premium, indexed by month-start dates.
               Series name: 'EBP'.

    Raises
    ------
    KeyError  If ebp_col cannot be found in the file.
    """
    ebp = pd.read_csv(path)

    date_col = next((c for c in ebp.columns if c.lower() in ["date", "dt", "time"]), ebp.columns[0])

    colmap = {c.lower().strip(): c for c in ebp.columns}
    key    = ebp_col.lower().strip()
    if key not in colmap:
        raise KeyError(f"Could not find EBP column '{ebp_col}' in file. Columns: {list(ebp.columns)}")

    value_col = colmap[key]

    ebp[date_col] = robust_to_datetime(ebp[date_col])
    ebp = ebp.dropna(subset=[date_col])

    ebp["MS"] = to_month_start(ebp[date_col])

    ebp_m = ebp.sort_values(date_col).groupby("MS")[value_col].mean()
    ebp_m = pd.to_numeric(ebp_m, errors="coerce")
    ebp_m.name = "EBP"
    return ebp_m


# ============================================================
# SECTION 9 — HELPER FUNCTIONS: SHOCK AGGREGATION UTILITIES
# ============================================================

def pick_columns_case_insensitive(raw_cols, candidates):
    """
    Return column names from raw_cols that match any candidate (case-insensitive).

    Used to guarantee that AD and Hybrid shock columns are included in the
    aggregated output even if their exact column names differ slightly from
    the expected values (e.g. 'ad' vs 'AD' vs 'Aruoba').

    Parameters
    ----------
    raw_cols   : list of str  Column names from the input DataFrame.
    candidates : list of str  Target names to match against.

    Returns
    -------
    list of str  Matched column names using their original capitalisation.
    """
    lower_map = {c.lower().strip(): c for c in raw_cols}
    found = []
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            found.append(lower_map[key])
    return found


def detect_meeting_date_col(raw, preferred=None):
    """
    Robustly identify the date column in the meeting shocks file.

    Detection priority:
        1. Use the preferred column name if it exists in the DataFrame.
        2. Search for common date column name variants (case-insensitive).
        3. Fall back to the first column if no match is found.

    Parameters
    ----------
    raw       : pd.DataFrame  Meeting shocks file loaded as a DataFrame.
    preferred : str or None   Expected date column name (from MEETING_DATE_COL).

    Returns
    -------
    str  Name of the detected date column.
    """
    if preferred is not None and preferred in raw.columns:
        return preferred
    for c in raw.columns:
        if c.lower().strip() in ["date", "meeting_date", "meetingdate", "dt", "time"]:
            return c
    return raw.columns[0]


def load_meeting_shocks_to_monthly_sum_all(
    path: str,
    monthly_index: pd.DatetimeIndex,
    date_col: Optional[str] = None,
    shock_cols: Optional[List[str]] = None,
    prefix: str = "shock_",
    suffix: str = "_meeting_sum"
) -> pd.DataFrame:
    """
    Aggregate meeting-level monetary policy shocks to monthly frequency by summing.

    For each calendar month, all FOMC meeting shocks falling within that month
    are summed into a single monthly observation. Months with no FOMC meeting
    receive a value of zero. This preserves the full shock information for months
    with two meetings (which occasionally occur) while keeping the dataset on a
    regular monthly grid for the BVAR.

    The aggregation uses min_count=1 in groupby().sum() to distinguish between
    months with meetings (sum of shocks, possibly zero) and months without
    meetings (truly missing). The reindex().fillna(0.0) step then assigns zero
    to the latter, consistent with the no-meeting-no-shock convention in the BVAR.

    Column naming convention for output:
        {prefix}{original_shock_col}{suffix}
        e.g. 'shock_Hybrid_meeting_sum', 'shock_AD_meeting_sum'

    Parameters
    ----------
    path          : str                Path to the meeting shocks CSV file.
    monthly_index : pd.DatetimeIndex   Target monthly index (month-start).
    date_col      : str or None        Preferred date column name; auto-detected
                                       if None or not found.
    shock_cols    : list of str or None  Specific shock columns to include;
                                         if None, all non-date columns are used.
    prefix        : str                Prefix for output column names (default: 'shock_').
    suffix        : str                Suffix for output column names (default: '_meeting_sum').

    Returns
    -------
    pd.DataFrame  Monthly shock DataFrame on monthly_index, one column per shock series.
    """
    raw = pd.read_csv(path)

    # Step 1: robustly detect the date column
    date_col = detect_meeting_date_col(raw, preferred=date_col)
    if date_col not in raw.columns:
        raise KeyError(f"Meeting shock date column '{date_col}' not found. Columns: {list(raw.columns)}")

    # Step 2: guarantee AD and Hybrid are included if present under any common name
    AD_CANDS = ["AD", "aruoba", "aruoba_drechsel", "ad_shock", "shock_ad"]
    HY_CANDS = ["Hybrid", "HYBRID", "hybrid", "hybrid_shock", "shock_hybrid"]

    ad_found = pick_columns_case_insensitive(raw.columns, AD_CANDS)
    hy_found = pick_columns_case_insensitive(raw.columns, HY_CANDS)

    # Step 3: select shock columns (all non-date columns if not specified)
    if shock_cols is None:
        shock_cols = [c for c in raw.columns if c != date_col]

    # Remove any date-like column names that slipped into the shock list
    shock_cols_clean = []
    for c in shock_cols:
        cl = c.lower().strip()
        if cl in ["date", "meeting_date", "meetingdate", "dt", "time"]:
            continue
        shock_cols_clean.append(c)

    # Ensure AD and Hybrid are in the final list even if not auto-selected
    for c in ad_found + hy_found:
        if c != date_col and c not in shock_cols_clean:
            shock_cols_clean.append(c)

    if len(shock_cols_clean) == 0:
        raise ValueError("No shock columns detected after cleaning.")

    # Step 4: parse meeting dates and map to month-start
    raw[date_col] = robust_to_datetime(raw[date_col])
    raw = raw.dropna(subset=[date_col])
    raw["MS"] = to_month_start(raw[date_col])

    out = pd.DataFrame(index=monthly_index)

    # Step 5: aggregate each shock column to monthly by summing within month
    for c in shock_cols_clean:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
        m = raw.groupby("MS")[c].sum(min_count=1)  # min_count=1: all-NaN months stay NaN
        m = m.reindex(monthly_index).fillna(0.0)   # no-meeting months receive zero
        out[prefix + str(c) + suffix] = m

    # Step 6: diagnostic print confirming which columns were included
    print("\n[Meeting shocks] date_col =", date_col)
    print("[Meeting shocks] shock columns used:", shock_cols_clean)
    print("[Meeting shocks] AD found:", ad_found if ad_found else "None")
    print("[Meeting shocks] Hybrid found:", hy_found if hy_found else "None")

    return out


# ============================================================
# SECTION 10 — INPUT FILE VALIDATION
# ============================================================
# Verify all required local input files exist before initiating
# any FRED downloads or data processing.

for p in [SPX_PATH, EBP_PATH, MEETING_SHOCK_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required input file:\n{p}")


# ============================================================
# SECTION 11 — BUILD FULL MONTHLY DATASET
# ============================================================
# Construct the master monthly DataFrame on the full sample index.
# Variables are added sequentially: FRED monthly series, Kalman-smoothed
# quarterly series, S&P 500, EBP, and aggregated meeting-level shocks.

monthly_index = pd.date_range(pd.to_datetime(START_FULL), pd.to_datetime(END_FULL), freq="MS")
df = pd.DataFrame(index=monthly_index)
df.index.name = "DATE"

# ── Monthly FRED series (direct download) ────────────────────────────────────
df["y1"]    = fred_monthly("GS1",    start=START_FULL, end=END_FULL)  # 1-year Treasury yield
df["unemp"] = fred_monthly("UNRATE", start=START_FULL, end=END_FULL)  # unemployment rate

# ── Quarterly FRED series → monthly via Kalman smoother ─────────────────────
# Real GDP and GDP deflator are published quarterly and must be interpolated
# to monthly frequency for the BVAR. The Kalman smoother produces smooth
# monthly estimates consistent with the observed quarterly values.
gdp_q  = fred_quarterly("GDPC1")    # real GDP (chained 2017 dollars)
pgdp_q = fred_quarterly("GDPDEF")   # GDP deflator (index, 2017=100)

df["rgdp_m"] = kalman_quarterly_to_monthly(gdp_q,  df.index)
df["pgdp_m"] = kalman_quarterly_to_monthly(pgdp_q, df.index)

# Log-transform GDP and deflator for stationarity and IRF interpretation
df["log_rgdp"] = np.log(df["rgdp_m"])
df["log_pgdp"] = np.log(df["pgdp_m"])

# ── S&P 500 (local file) ─────────────────────────────────────────────────────
spx_m = load_spx_monthly(SPX_PATH)
df["spx"]       = spx_m.reindex(df.index)
df["log_sp500"] = np.log(df["spx"])   # log-transform for IRF interpretation

# ── Excess bond premium (local file) ─────────────────────────────────────────
ebp_m    = load_ebp_monthly(EBP_PATH, ebp_col="ebp")
df["EBP"] = ebp_m.reindex(df.index)

# ── Meeting-level shocks → monthly SUM ───────────────────────────────────────
# All shock columns in the file are aggregated; AD and Hybrid are guaranteed
# to be included if present under any common naming variant.
shocks_monthly = load_meeting_shocks_to_monthly_sum_all(
    MEETING_SHOCK_PATH,
    monthly_index = df.index,
    date_col      = MEETING_DATE_COL,   # auto-detected if wrong or None
    shock_cols    = None,               # None → all non-date columns
    prefix        = "shock_",
    suffix        = "_meeting_sum"
)

df = df.join(shocks_monthly, how="left")


# ============================================================
# SECTION 12 — FINAL COLUMN SELECTION AND SAMPLE SPLIT
# ============================================================
# Retain only the BVAR macroeconomic variables and the aggregated shock
# columns. Split into three datasets corresponding to the full sample,
# the conventional policy era (pre-2008), and the ZLB era (post-2008).

macro_cols    = ["y1", "log_sp500", "log_rgdp", "log_pgdp", "unemp", "EBP"]
shock_cols_out = list(shocks_monthly.columns)

keep_cols = macro_cols + shock_cols_out
out_full  = df[keep_cols].copy()

out_pre  = out_full.loc[:pd.to_datetime(PRE_END_MS)].copy()
out_post = out_full.loc[pd.to_datetime(POST_START_MS):].copy()


# ============================================================
# SECTION 13 — DIAGNOSTICS
# ============================================================
# Print sample ranges, NaN counts, and shock coverage statistics to the
# console for verification before saving. Non-zero shock month counts
# confirm that the meeting-to-monthly aggregation worked correctly
# (expected: approximately 8–12 non-zero months per year).

print("\n--- RANGES ---")
print("FULL range:", out_full.index.min(), "→", out_full.index.max(), "| rows:", len(out_full))
print("PRE  range:", out_pre.index.min(),  "→", out_pre.index.max(),  "| rows:", len(out_pre))
print("POST range:", out_post.index.min(), "→", out_post.index.max(), "| rows:", len(out_post))

print("\n--- NaN COUNTS (FULL) ---\n", out_full.isna().sum())
print("\n--- NaN COUNTS (PRE)  ---\n", out_pre.isna().sum())
print("\n--- NaN COUNTS (POST) ---\n", out_post.isna().sum())

# Non-zero month counts confirm shock mapping is working
print("\n--- Non-zero shock months (FULL) ---")
for c in shock_cols_out:
    nz = int((out_full[c] != 0).sum())
    print(f"  {c}: {nz} / {len(out_full)}")

# Explicit confirmation that AD and Hybrid columns appear in the output
ad_like = [c for c in shock_cols_out if "ad" in c.lower()]
hy_like = [c for c in shock_cols_out if "hybrid" in c.lower()]
print("\n--- AD/Hybrid columns in OUTPUT ---")
print("AD-like:", ad_like if ad_like else "None")
print("Hybrid-like:", hy_like if hy_like else "None")


# ============================================================
# SECTION 14 — SAVE OUTPUT
# ============================================================

out_full.to_csv(OUT_FULL,  index=True)
out_pre.to_csv(OUT_PRE,    index=True)
out_post.to_csv(OUT_POST,  index=True)

print(f"\nSaved → {OUT_FULL}")
print(f"Saved → {OUT_PRE}")
print(f"Saved → {OUT_POST}")
