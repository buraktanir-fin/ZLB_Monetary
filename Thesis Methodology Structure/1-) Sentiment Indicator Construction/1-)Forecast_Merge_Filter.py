"""
================================================================================
forecast_merger.py
================================================================================
Thesis:  "When Rates Hit Zero: Identifying Monetary Shocks with Shadow Rates"
Author:  Burak Tanir
Date:    April 2026

Purpose
-------
Merges all FOMC staff forecast sheets from the Tealbook/Greenbook Excel workbook
into a single analysis-ready dataset spanning January 1982 to December 2019.

For each forecast variable (e.g. PGDP, RCONSUM, ROUTPUT), the script selects
all available horizon columns (backward horizons B1, B2, ... and forward horizons
F0, F1, F2, ...) and renames them with a sheet-level prefix to ensure uniqueness
across variables. First differences of all numeric columns are computed to capture
forecast revisions, which serve as key inputs to the ridge regression shock
identification procedure.

Input
-----
    input/forecasts.xlsx
        Multi-sheet Excel workbook downloaded from the Philadelphia Fed's
        Real-Time Data Research Center. Each sheet corresponds to one
        macroeconomic forecast variable. A "Documentation" sheet is ignored.
        Required column in each data sheet: GBdate (YYYYMMDD integer format).

Output
------
    forecasts/merged_forecasts_1982_2019.csv
        Wide-format CSV with one row per FOMC meeting date and one column per
        (variable, horizon) combination, plus first-difference columns (_d1
        suffix). The Date column is in datetime format.

Column naming convention
------------------------
    {SHEET}_{ORIG_COL}       — level of forecast variable at given horizon
    {SHEET}_{ORIG_COL}_d1   — first difference (forecast revision)

    Example:  PGDP_F0   → GDP deflator nowcast level
              PGDP_F0_d1 → revision in GDP deflator nowcast between meetings

Dependencies
------------
    pandas, re, os
    Python >= 3.8

Usage
-----
    python forecast_merger.py

    The script is designed to be run from its own directory. All paths are
    resolved relative to the script location, so no working directory
    configuration is needed.
================================================================================
"""

import pandas as pd
import re
import os

# ============================================================
# SECTION 1 — PATH SETUP (RELATIVE TO SCRIPT LOCATION)
# ============================================================
# All paths are resolved relative to the directory containing this script.
# This ensures the script runs correctly regardless of the working directory
# from which it is invoked.

# Root directory: where this script is saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input directory: contains the raw Tealbook/Greenbook Excel workbook
INPUT_DIR  = os.path.join(BASE_DIR, "input")
INPUT_FILE = os.path.join(INPUT_DIR, "forecasts.xlsx")

# Output directory: receives the merged, analysis-ready CSV
# Created automatically if it does not already exist
OUTPUT_DIR  = os.path.join(BASE_DIR, "forecasts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output file: one row per FOMC meeting, one column per (variable, horizon)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "merged_forecasts_1982_2019.csv")


# ============================================================
# SECTION 2 — LOAD EXCEL WORKBOOK
# ============================================================
# Open the workbook and collect the names of all data sheets.
# The "Documentation" sheet (case-insensitive) is excluded because it
# contains metadata rather than forecast observations.

excel = pd.ExcelFile(INPUT_FILE)

# Filter out the documentation sheet; retain all variable-level data sheets
sheets = [s for s in excel.sheet_names if s.lower() != "documentation"]

# Initialise the merged dataset as empty; sheets are joined iteratively
merged = None

# ── Horizon column pattern ────────────────────────────────────────────────────
# Tealbook horizon columns follow the convention:
#   B1, B2, ...  → backward (historical) horizons
#   F0, F1, F2, ... → forward (forecast) horizons
# This regex matches any column name containing B or F followed by one or
# more digits, allowing the script to generalise across all variable sheets.
horizon_pattern = re.compile(r"[BF]\d+")


# ============================================================
# SECTION 3 — SHEET ITERATION AND MERGING
# ============================================================
# For each forecast variable sheet:
#   1. Load the sheet into a DataFrame.
#   2. Validate that it contains the required GBdate key column.
#   3. Retain only the date column and the horizon-level columns.
#   4. Prefix each horizon column with the sheet name to avoid collisions
#      when multiple variables share the same horizon labels (F0, F1, etc.).
#   5. Outer-merge onto the growing dataset on GBdate so that no meeting
#      date is dropped even if a variable is missing for some periods.

for sheet in sheets:

    # Load sheet
    df = pd.read_excel(INPUT_FILE, sheet_name=sheet)

    # Skip sheets that lack the GBdate identifier column
    # (e.g. auxiliary lookup tables occasionally included in the workbook)
    if "GBdate" not in df.columns:
        continue

    # ── Date column cleaning ──────────────────────────────────────────────
    # Drop rows where GBdate is missing (header overflow rows sometimes appear)
    # and cast to integer to standardise the YYYYMMDD format.
    df = df.dropna(subset=["GBdate"])
    df["GBdate"] = df["GBdate"].astype(int)

    # ── Horizon column selection ──────────────────────────────────────────
    # Identify all columns that match the B*/F* horizon naming convention.
    # Only these columns, together with GBdate, are retained.
    horizon_cols = [c for c in df.columns if horizon_pattern.search(c)]
    keep_cols    = ["GBdate"] + horizon_cols
    df           = df[keep_cols]

    # ── Column renaming ───────────────────────────────────────────────────
    # Prefix each horizon column with the sheet (variable) name to produce
    # unique column identifiers in the merged dataset.
    # Example: sheet "PGDP", column "F0" → "PGDP_F0"
    rename_map = {c: f"{sheet}_{c}" for c in df.columns if c != "GBdate"}
    df = df.rename(columns=rename_map)

    # ── Outer merge ───────────────────────────────────────────────────────
    # Use outer join so that all FOMC meeting dates across all variables
    # are preserved, even when a variable starts or ends mid-sample.
    if merged is None:
        merged = df                                          # first sheet seeds the dataset
    else:
        merged = pd.merge(merged, df, on="GBdate", how="outer")


# ============================================================
# SECTION 4 — DATE HANDLING AND SAMPLE RESTRICTION
# ============================================================
# Convert the YYYYMMDD integer to a proper datetime object.
# Rows that cannot be parsed (errors="coerce") become NaT and are implicitly
# dropped when the year filter is applied.

merged["Date"] = pd.to_datetime(
    merged["GBdate"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

# Drop the original integer date column — Date replaces it
merged = merged.drop(columns=["GBdate"])

# ── Sample restriction ────────────────────────────────────────────────────
# Restrict to January 1982 onwards to match the thesis sample period
# (January 1982 – December 2019). The upper bound is imposed implicitly
# by the source data, which runs through 2019.
merged = merged[merged["Date"].dt.year >= 1982]

# Sort chronologically and reset the row index
merged = merged.sort_values("Date").reset_index(drop=True)

# Move the Date column to the first position for readability
merged = merged[["Date"] + [c for c in merged.columns if c != "Date"]]


# ============================================================
# SECTION 5 — FIRST DIFFERENCES (FORECAST REVISIONS)
# ============================================================
# Compute the first difference of every numeric column.
# These revision series capture how policymakers updated their expectations
# between consecutive FOMC meetings and serve as an additional regressor
# block in the ridge regression shock identification.
#
# Column naming convention:
#   {original_col}_d1   → first difference of {original_col}

numeric_cols = merged.select_dtypes(include="number").columns

for col in numeric_cols:
    merged[f"{col}_d1"] = merged[col].diff()


# ============================================================
# SECTION 6 — SAVE OUTPUT
# ============================================================
merged.to_csv(OUTPUT_FILE, index=False)

print(f"Merged forecast dataset saved successfully.")
print(f"  Rows    : {merged.shape[0]:,}  (one per FOMC meeting date)")
print(f"  Columns : {merged.shape[1]:,}  (levels + first differences)")
print(f"  File    : {OUTPUT_FILE}")


# ============================================================
# SECTION 7 — PREVIEW
# ============================================================
# Display all columns and a five-row preview for a quick sanity check.
# This is useful when running the script interactively during development.

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)

print("\n=== Data Preview (first 5 rows) ===")
print(merged.head())