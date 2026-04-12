"""
================================================================================
dataset_builder.py
================================================================================
Thesis:  "When Rates Hit Zero: Identifying Monetary Shocks with Shadow Rates"
Author:  Burak Tanir
Date:    April 2026

Purpose
-------
Constructs the full analysis-ready regressor dataset for the ridge regression
shock identification procedure described in Section 4.3 of the thesis. The
dataset combines sentiment indicators, staff forecasts, lagged variables, and
quadratic terms into a single wide-format CSV that is ready for direct input
into the ridge regression.

The sentiment dates serve as the base calendar for the merge. This ensures
that the dataset covers exactly the FOMC meetings for which sentiment scores
are available, without being constrained by the date coverage of the forecast
or Aruoba-Drechsel datasets.

Pipeline
--------
    Step 1 — Load sentiment data (base calendar)
        Reads the concept-level sentiment scores from
        fomc_concept_sentiments_ADstyle.csv. The Date column defines the
        base set of meeting dates. Only columns ending with
        '_score_per_10k_words_z' are retained as sentiment regressors.

    Step 2 — Load and merge staff forecast data
        Reads the merged forecast dataset from merged_forecasts_1982_2019.csv
        and left-merges onto the sentiment dates. Missing forecast values
        (meeting dates absent from the forecast file) are filled with 0.

    Step 3 — Load and merge Aruoba-Drechsel FFR and shock series
        Reads the 'Shocks and FFR by meeting' sheet from the Aruoba-Drechsel
        Excel workbook. The two target columns — FFR_change and Shock — are
        left-merged onto the sentiment dates. Missing values are preserved as
        NaN (not filled with zeros) since these are the dependent variables,
        and imputing them would distort the regression.

    Step 4 — Sentiment lags (1 to 4)
        Lags 1 through 4 of all sentiment columns are computed. These capture
        the persistence of FOMC document tone across consecutive meetings and
        allow the ridge regression to condition on prior sentiment states.
        Initial lagged rows that have no prior meeting are filled with 0.

    Step 5 — FFR lag and FFR lag squared
        A one-meeting lag of FFR_change and its square are computed. These
        capture autocorrelation in the policy instrument and nonlinear
        persistence effects near the zero lower bound.

    Step 6 — Quadratic terms
        Squares of all forecast columns and all sentiment columns are
        computed. These capture threshold effects and asymmetric policy
        reactions as described in Section 4.2 of the thesis.

    Step 7 — Final dataset assembly
        All components are concatenated into a single DataFrame. Regressor
        columns (all columns except Date, FFR_change, Shock) are cast to
        numeric and filled with 0 for any remaining missing values.
        FFR_change and Shock retain NaN where the Aruoba-Drechsel data does
        not cover the base date.

    Step 8 — Save output
        The final dataset is saved as AD_full_dataset_ready.csv to the
        Forecast and Sentiment Results/ folder.

Input
-----
    sentiment_results/fomc_concept_sentiments_ADstyle.csv
        Produced by sentiment_computation.py. Must contain a 'Date' column
        and columns ending with '_score_per_10k_words_z'.

    forecasts/merged_forecasts_1982_2019.csv
        Produced by forecast_merger.py. Must contain a 'Date' column.

    Aruoba Drechsel Data/Aruoba_Drechsel_Data.xlsx
        Sheet: 'Shocks and FFR by meeting'. Must contain columns:
            'FOMC meeting (scheduled)' — meeting date in YYYY_MM_DD format
            'FFR_change'               — change in Federal Funds Rate target
            'Shock'                    — original Aruoba-Drechsel shock series

Output
------
    Forecast and Sentiment Results/AD_full_dataset_ready.csv
        Wide-format dataset with one row per FOMC meeting date. Columns:
            Date              — FOMC meeting date
            FFR_change        — dependent variable (NaN if not covered by AD)
            Shock             — original AD shock (NaN if not covered by AD)
            {forecast cols}   — staff forecast levels and first differences
            {sentiment cols}  — z-scored concept sentiment scores
            {sentiment}_lag{1-4} — sentiment lags 1 through 4
            FFR_lag1          — one-meeting lag of FFR_change
            FFR_lag1_sq       — square of FFR_lag1
            {forecast}_sq     — squares of forecast columns
            {sentiment}_sq    — squares of sentiment columns

Column conventions
------------------
    Regressor columns (X)  : all columns except Date, FFR_change, Shock
    Dependent variable (Y) : FFR_change  (pre-2008) or shadow rate change
                             (post-2008, substituted externally)
    Benchmark Y            : Shock  (original Aruoba-Drechsel shock series)

    Missing values in X columns are filled with 0.
    Missing values in FFR_change and Shock are preserved as NaN.

Dependencies
------------
    os  (standard library)
    pandas  — pip install pandas
    numpy   — pip install numpy openpyxl

Usage
-----
    python dataset_builder.py

    Run from the directory containing the sentiment_results/, forecasts/,
    and Aruoba Drechsel Data/ folders. The output folder is created
    automatically if absent.
================================================================================
"""

import os
import pandas as pd
import numpy as np


# ============================================================
# SECTION 1 — DIRECTORY CONFIGURATION
# ============================================================
# All folders are resolved relative to the current working directory.
# The output folder is created automatically if it does not exist.

BASE_DIR = os.getcwd()

SENTIMENT_DIR = os.path.join(BASE_DIR, "sentiment_results")             # replicated sentiment scores
FORECAST_DIR  = os.path.join(BASE_DIR, "forecasts")                     # merged staff forecasts
ARUOBA_DIR    = os.path.join(BASE_DIR, "Aruoba Drechsel Data")          # original AD data
OUT_DIR       = os.path.join(BASE_DIR, "Forecast and Sentiment Results") # output folder

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# SECTION 2 — FILE NAMES
# ============================================================

SENTIMENT_FILE = "fomc_concept_sentiments_ADstyle.csv"  # from sentiment_computation.py
FORECAST_FILE  = "merged_forecasts_1982_2019.csv"       # from forecast_merger.py
ARUOBA_FILE    = "Aruoba_Drechsel_Data.xlsx"            # original Aruoba-Drechsel workbook

OUT_FILE = "AD_full_dataset_ready.csv"                  # final output


# ============================================================
# SECTION 3 — FULL FILE PATHS
# ============================================================

SENTIMENT_PATH = os.path.join(SENTIMENT_DIR, SENTIMENT_FILE)
FORECAST_PATH  = os.path.join(FORECAST_DIR,  FORECAST_FILE)
ARUOBA_PATH    = os.path.join(ARUOBA_DIR,    ARUOBA_FILE)
OUT_PATH       = os.path.join(OUT_DIR,       OUT_FILE)


# ============================================================
# SECTION 4 — STEP 1: LOAD SENTIMENT DATA (BASE CALENDAR)
# ============================================================
# The sentiment file defines the base set of meeting dates used throughout
# the merge. Using sentiment dates as the base ensures the dataset covers
# exactly the meetings for which concept-level sentiment scores are available.
#
# Only columns ending with '_score_per_10k_words_z' are retained as
# sentiment regressors. These are the z-standardised, length-normalised
# concept sentiment scores described in Section 4.2 of the thesis.

sent = pd.read_csv(SENTIMENT_PATH)
if "Date" not in sent.columns:
    raise ValueError("Sentiment file must contain a 'Date' column.")

sent["Date"] = pd.to_datetime(sent["Date"], errors="coerce")
sent = sent.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

# Identify sentiment regressor columns by their naming convention
sent_cols = [c for c in sent.columns if c.endswith("_score_per_10k_words_z")]
if not sent_cols:
    raise ValueError("No sentiment columns found ending with '_score_per_10k_words_z'.")

print("Sentiment base shape:", sent.shape)
print("Sentiment base date range:", sent["Date"].min(), "→", sent["Date"].max())
print("Sentiment columns:", len(sent_cols))

# Base dataset: sentiment dates and sentiment regressors only
df = sent[["Date"] + sent_cols].copy()


# ============================================================
# SECTION 5 — STEP 2: LOAD AND MERGE STAFF FORECAST DATA
# ============================================================
# Staff forecast data is left-merged onto the sentiment date base.
# Missing forecast values (meeting dates not covered by the forecast file)
# are filled with 0 rather than dropped, preserving the full sentiment
# date calendar and allowing the ridge regression to run on all meetings.

fc = pd.read_csv(FORECAST_PATH)
if "Date" not in fc.columns:
    raise ValueError("Forecast file must contain a 'Date' column.")

fc["Date"] = pd.to_datetime(fc["Date"], errors="coerce")
fc = fc.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

forecast_cols = [c for c in fc.columns if c != "Date"]
print("Forecast shape:", fc.shape)
print("Forecast columns:", len(forecast_cols))

# Left merge: retain all sentiment dates; fill unmatched forecasts with 0
df = df.merge(fc, on="Date", how="left")

for c in forecast_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)


# ============================================================
# SECTION 6 — STEP 3: LOAD AND MERGE ARUOBA FFR AND SHOCK
# ============================================================
# The dependent variable (FFR_change) and the benchmark shock series (Shock)
# are loaded from the Aruoba-Drechsel Excel workbook. Unlike forecasts,
# missing values in these columns are NOT filled with zeros — they are
# preserved as NaN because these are dependent variables, and imputing
# them would distort the ridge regression and the shock validation.

ffr = pd.read_excel(ARUOBA_PATH, sheet_name="Shocks and FFR by meeting")

date_source_col = "FOMC meeting (scheduled)"
if date_source_col not in ffr.columns:
    raise ValueError(f"Aruoba sheet must contain '{date_source_col}' column.")

ffr["Date"] = pd.to_datetime(
    ffr[date_source_col].astype(str).str.replace("_", "-"),
    errors="coerce"
)
ffr = ffr.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

# Validate that the required dependent variable columns are present
required_y = ["FFR_change", "Shock"]
missing_y  = [c for c in required_y if c not in ffr.columns]
if missing_y:
    raise ValueError(f"Missing required columns in Aruoba sheet: {missing_y}")

ffr = ffr[["Date"] + required_y].copy()

print("Aruoba FFR/Shock shape:", ffr.shape)

# Left merge: do NOT fill FFR_change or Shock with zeros
df = df.merge(ffr, on="Date", how="left").sort_values("Date").reset_index(drop=True)


# ============================================================
# SECTION 7 — STEP 4: SENTIMENT LAGS (1 TO 4)
# ============================================================
# Lags 1 through 4 of all sentiment columns are computed. These capture the
# persistence of FOMC document tone across consecutive meetings and allow the
# ridge regression to condition on prior sentiment states. Rows with no prior
# meeting (the first L rows for lag L) are filled with 0 by construction,
# consistent with the convention used for missing forecast values.
#
# Column naming convention:  {concept}_score_per_10k_words_z_lag{L}

S_base = df[sent_cols].copy()

lag_frames = []
for L in range(1, 5):
    lag = S_base.shift(L).fillna(0.0)   # initial lagged rows → 0
    lag.columns = [f"{c}_lag{L}" for c in sent_cols]
    lag_frames.append(lag)

S_lags = pd.concat(lag_frames, axis=1)


# ============================================================
# SECTION 8 — STEP 5: FFR LAG AND FFR LAG SQUARED
# ============================================================
# A one-meeting lag of FFR_change and its square are included as regressors.
# FFR_lag1 captures autocorrelation in the policy instrument; FFR_lag1_sq
# captures nonlinear persistence effects, particularly near the zero lower
# bound where the policy rate's predictability changes.

FFR_lag1    = pd.to_numeric(df["FFR_change"], errors="coerce").shift(1).fillna(0.0)
FFR_lag1_sq = FFR_lag1 ** 2

FFR_extra = pd.DataFrame({
    "FFR_lag1":    FFR_lag1,
    "FFR_lag1_sq": FFR_lag1_sq
})


# ============================================================
# SECTION 9 — STEP 6: QUADRATIC TERMS
# ============================================================
# Squares of all forecast columns and all sentiment columns are computed.
# These nonlinear transformations capture threshold effects and asymmetric
# policy reactions as described in Section 4.2 of the thesis — for example,
# aggressive policy easing in response to a strongly negative economic outlook
# that a linear specification would understate.
#
# Column naming convention:
#   {forecast_col}_sq   — square of each forecast level or first difference
#   {sentiment_col}_sq  — square of each z-scored concept sentiment score

F_sq = df[forecast_cols].copy()
for c in forecast_cols:
    F_sq[c] = pd.to_numeric(F_sq[c], errors="coerce").fillna(0.0)
F_sq = F_sq ** 2
F_sq.columns = [f"{c}_sq" for c in forecast_cols]

S_sq = S_base ** 2
S_sq.columns = [f"{c}_sq" for c in sent_cols]


# ============================================================
# SECTION 10 — STEP 7: FINAL DATASET ASSEMBLY
# ============================================================
# All components are concatenated into a single wide-format DataFrame.
# Column order:
#   1. Date, FFR_change, Shock   (identifiers and dependent variables)
#   2. Forecast levels and diffs (X_forecast)
#   3. Sentiment scores          (X_sentiment)
#   4. Sentiment lags 1–4        (X_sentiment_lags)
#   5. FFR lag and FFR lag sq    (X_ffr_nonlinear)
#   6. Forecast squares          (X_forecast_sq)
#   7. Sentiment squares         (X_sentiment_sq)
#
# All regressor columns (X) are cast to numeric and filled with 0 for any
# remaining missing values. FFR_change and Shock retain NaN where the
# Aruoba-Drechsel data does not cover the base sentiment date.

df_full = pd.concat(
    [
        df[["Date", "FFR_change", "Shock"]],
        df[forecast_cols],
        S_base,
        S_lags,
        FFR_extra,
        F_sq,
        S_sq,
    ],
    axis=1
)

# Cast all regressor columns to numeric and fill residual NaNs with 0
exclude = ["Date", "FFR_change", "Shock"]
X_cols  = [c for c in df_full.columns if c not in exclude]

for c in X_cols:
    df_full[c] = pd.to_numeric(df_full[c], errors="coerce").fillna(0.0)

# Preserve NaN in dependent variable columns (do not impute)
df_full["FFR_change"] = pd.to_numeric(df_full["FFR_change"], errors="coerce")
df_full["Shock"]      = pd.to_numeric(df_full["Shock"],      errors="coerce")


# ============================================================
# SECTION 11 — STEP 8: SAVE OUTPUT
# ============================================================

df_full.to_csv(OUT_PATH, index=False)

print("Saved →", OUT_PATH)
print("Base observations (sentiment dates):", len(df_full))
print("Regressors:", len(X_cols))
print("Missing FFR_change on base dates:", int(df_full["FFR_change"].isna().sum()))
print("Missing Shock on base dates:",      int(df_full["Shock"].isna().sum()))