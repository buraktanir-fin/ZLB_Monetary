import pandas as pd
import re
import os

# ============================================================
# PATH SETUP (RELATIVE TO SCRIPT LOCATION)
# ============================================================

# Directory where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input folder (Excel forecasts)
INPUT_DIR = os.path.join(BASE_DIR, "input")
INPUT_FILE = os.path.join(INPUT_DIR, "forecasts.xlsx")

# Output folder (merged forecasts)
OUTPUT_DIR = os.path.join(BASE_DIR, "forecasts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "merged_forecasts_1982_2019.csv")

# ============================================================
# LOAD EXCEL WORKBOOK
# ============================================================

excel = pd.ExcelFile(INPUT_FILE)

# Ignore Documentation sheet
sheets = [s for s in excel.sheet_names if s.lower() != "documentation"]

merged = None

# Regex to capture all B* and F* horizons (B1, F0, F1, F2, ...)
horizon_pattern = re.compile(r"[BF]\d+")

# ============================================================
# LOOP THROUGH SHEETS
# ============================================================

for sheet in sheets:
    df = pd.read_excel(INPUT_FILE, sheet_name=sheet)

    # Skip invalid sheets
    if "GBdate" not in df.columns:
        continue

    # Clean dates
    df = df.dropna(subset=["GBdate"])
    df["GBdate"] = df["GBdate"].astype(int)

    # Select all horizon columns
    horizon_cols = [c for c in df.columns if horizon_pattern.search(c)]
    keep_cols = ["GBdate"] + horizon_cols
    df = df[keep_cols]

    # Rename columns to ensure uniqueness
    rename_map = {c: f"{sheet}_{c}" for c in df.columns if c != "GBdate"}
    df = df.rename(columns=rename_map)

    # Merge
    if merged is None:
        merged = df
    else:
        merged = pd.merge(merged, df, on="GBdate", how="outer")

# ============================================================
# DATE HANDLING
# ============================================================

merged["Date"] = pd.to_datetime(
    merged["GBdate"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

merged = merged.drop(columns=["GBdate"])
merged = merged[merged["Date"].dt.year >= 1982]
merged = merged.sort_values("Date").reset_index(drop=True)

# Reorder columns
merged = merged[["Date"] + [c for c in merged.columns if c != "Date"]]

# ============================================================
# FIRST DIFFERENCES
# ============================================================

numeric_cols = merged.select_dtypes(include="number").columns

for col in numeric_cols:
    merged[f"{col}_d1"] = merged[col].diff()

# ============================================================
# SAVE OUTPUT
# ============================================================

merged.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Merged forecast dataset created")
print(f"   Rows: {merged.shape[0]}")
print(f"   Columns: {merged.shape[1]}")
print(f"📁 Saved to: {OUTPUT_FILE}")

# ============================================================
# DISPLAY SETTINGS + PREVIEW
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)

print("\n=== Data Preview ===")
print(merged.head())
