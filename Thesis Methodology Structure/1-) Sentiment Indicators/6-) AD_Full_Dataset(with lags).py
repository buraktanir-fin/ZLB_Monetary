import os
import pandas as pd
import numpy as np

# ============================================================
# FULL MERGE USING *SENTIMENT DATES* AS BASE (NOT ARUOBA)
# - Base calendar = dates present in your sentiment file
# - Forecasts merged in (missing -> 0)
# - Aruoba (FFR_change, Shock) merged in (kept as NaN if missing)
# - No time restriction
# - Lags and squares created
# - Output saved to: ./Control Results/AD_full_dataset_ready.csv
# ============================================================

# -------------------- DIRECTORIES --------------------
BASE_DIR = os.getcwd()

SENTIMENT_DIR = os.path.join(BASE_DIR, "sentiment_results")
FORECAST_DIR  = os.path.join(BASE_DIR, "forecasts")
ARUOBA_DIR    = os.path.join(BASE_DIR, "Aruoba Drechsel Data")
OUT_DIR       = os.path.join(BASE_DIR, "Forecast and Sentiment Results")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------- FILE NAMES --------------------
SENTIMENT_FILE = "fomc_concept_sentiments_ADstyle.csv"
FORECAST_FILE  = "merged_forecasts_1982_2019.csv"
ARUOBA_FILE    = "Aruoba_Drechsel_Data.xlsx"

OUT_FILE = "AD_full_dataset_ready.csv"

# -------------------- FULL PATHS --------------------
SENTIMENT_PATH = os.path.join(SENTIMENT_DIR, SENTIMENT_FILE)
FORECAST_PATH  = os.path.join(FORECAST_DIR, FORECAST_FILE)
ARUOBA_PATH    = os.path.join(ARUOBA_DIR, ARUOBA_FILE)
OUT_PATH       = os.path.join(OUT_DIR, OUT_FILE)

# ============================
# 1) LOAD SENTIMENT DATA (BASE)
# ============================
sent = pd.read_csv(SENTIMENT_PATH)
if "Date" not in sent.columns:
    raise ValueError("Sentiment file must contain a 'Date' column.")

sent["Date"] = pd.to_datetime(sent["Date"], errors="coerce")
sent = sent.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

# Keep your standard sentiment columns
sent_cols = [c for c in sent.columns if c.endswith("_score_per_10k_words_z")]
if not sent_cols:
    raise ValueError("No sentiment columns found ending with '_score_per_10k_words_z'.")

print("Sentiment base shape:", sent.shape)
print("Sentiment base date range:", sent["Date"].min(), "→", sent["Date"].max())
print("Sentiment columns:", len(sent_cols))

# Base dataset = sentiment dates only
df = sent[["Date"] + sent_cols].copy()

# ============================
# 2) LOAD FORECAST DATA
# ============================
fc = pd.read_csv(FORECAST_PATH)
if "Date" not in fc.columns:
    raise ValueError("Forecast file must contain a 'Date' column.")

fc["Date"] = pd.to_datetime(fc["Date"], errors="coerce")
fc = fc.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

forecast_cols = [c for c in fc.columns if c != "Date"]
print("Forecast shape:", fc.shape)
print("Forecast columns:", len(forecast_cols))

# Merge forecasts onto sentiment dates (missing -> 0)
df = df.merge(fc, on="Date", how="left")

for c in forecast_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

# ============================
# 3) LOAD ARUOBA FFR & SHOCK
# ============================
ffr = pd.read_excel(ARUOBA_PATH, sheet_name="Shocks and FFR by meeting")

date_source_col = "FOMC meeting (scheduled)"
if date_source_col not in ffr.columns:
    raise ValueError(f"Aruoba sheet must contain '{date_source_col}' column.")

ffr["Date"] = pd.to_datetime(
    ffr[date_source_col].astype(str).str.replace("_", "-"),
    errors="coerce"
)
ffr = ffr.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

required_y = ["FFR_change", "Shock"]
missing_y = [c for c in required_y if c not in ffr.columns]
if missing_y:
    raise ValueError(f"Missing required columns in Aruoba sheet: {missing_y}")

ffr = ffr[["Date"] + required_y].copy()

print("Aruoba FFR/Shock shape:", ffr.shape)

# Merge Aruoba onto sentiment dates (do NOT fill FFR_change/Shock with zeros)
df = df.merge(ffr, on="Date", how="left").sort_values("Date").reset_index(drop=True)

# ============================
# 4) SENTIMENT LAGS (1–4)
# ============================
S_base = df[sent_cols].copy()

lag_frames = []
for L in range(1, 5):
    lag = S_base.shift(L).fillna(0.0)  # initial lagged rows -> 0 by construction
    lag.columns = [f"{c}_lag{L}" for c in sent_cols]
    lag_frames.append(lag)

S_lags = pd.concat(lag_frames, axis=1)

# ============================
# 5) FFR LAG + FFR LAG^2
# ============================
FFR_lag1 = pd.to_numeric(df["FFR_change"], errors="coerce").shift(1).fillna(0.0)
FFR_lag1_sq = FFR_lag1 ** 2

FFR_extra = pd.DataFrame({
    "FFR_lag1": FFR_lag1,
    "FFR_lag1_sq": FFR_lag1_sq
})

# ============================
# 6) QUADRATIC TERMS
# ============================
F_sq = df[forecast_cols].copy()
for c in forecast_cols:
    F_sq[c] = pd.to_numeric(F_sq[c], errors="coerce").fillna(0.0)
F_sq = F_sq ** 2
F_sq.columns = [f"{c}_sq" for c in forecast_cols]

S_sq = S_base ** 2
S_sq.columns = [f"{c}_sq" for c in sent_cols]

# ============================
# 7) FINAL DATASET
# ============================
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

# Ensure regressor columns are numeric and have no NA (fill with 0)
exclude = ["Date", "FFR_change", "Shock"]
X_cols = [c for c in df_full.columns if c not in exclude]

for c in X_cols:
    df_full[c] = pd.to_numeric(df_full[c], errors="coerce").fillna(0.0)

# Keep Y columns numeric but do not fill missing shocks/FFR with zeros
df_full["FFR_change"] = pd.to_numeric(df_full["FFR_change"], errors="coerce")
df_full["Shock"] = pd.to_numeric(df_full["Shock"], errors="coerce")

# ============================
# 8) SAVE
# ============================
df_full.to_csv(OUT_PATH, index=False)

print("✅ Saved →", OUT_PATH)
print("Base observations (sentiment dates):", len(df_full))
print("Regressors:", len(X_cols))
print("Missing FFR_change on base dates:", int(df_full["FFR_change"].isna().sum()))
print("Missing Shock on base dates:", int(df_full["Shock"].isna().sum()))
