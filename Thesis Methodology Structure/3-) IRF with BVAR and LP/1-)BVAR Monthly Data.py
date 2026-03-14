import pandas as pd
import numpy as np
from fredapi import Fred
from statsmodels.tsa.statespace.structural import UnobservedComponents
import os
from typing import Optional, List

# ============================================================
# MONTHLY BVAR DATASET + MEETING-LEVEL SHOCKS (ALL TYPES)
# - Macro data: monthly (FRED + SPX + EBP)
# - Shocks: meeting-level -> aggregated to monthly via SUM within month
# - Aggregates ALL shock columns in the meeting shocks file
# - Additionally GUARANTEES inclusion of AD + Hybrid if present (case-insensitive)
# - Outputs: all-in-one FULL / PRE / POST in outputs/
# - Python 3.7+ compatible
# ============================================================

# ================== CONFIG ==================
BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRED_API_KEY = "28700a952f6f8933e07bb46cec616fef"
fred = Fred(api_key=FRED_API_KEY)

# Full monthly span
START_FULL = "1982-10-01"
END_FULL   = "2019-12-01"

# Split around Oct 2008
PRE_END_MS    = "2008-10-01"
POST_START_MS = "2008-11-01"

# Input files
SPX_PATH = os.path.join(INPUT_DIR, "SPX.csv")
EBP_PATH = os.path.join(INPUT_DIR, "EBP.csv")

# Meeting-level shocks file (contains Date + multiple shock columns)
MEETING_SHOCK_PATH = os.path.join(INPUT_DIR, "shock_series_pre_post_2008.csv")

# If wrong/missing, code will auto-detect anyway
MEETING_DATE_COL: Optional[str] = "Date"  # set to None to auto-detect

# Output files (3 outputs)
OUT_FULL = os.path.join(OUTPUT_DIR, "bvar_monthly_dataset_FULL_with_EBP_with_ALL_meeting_shockSUM.csv")
OUT_PRE  = os.path.join(OUTPUT_DIR, "bvar_monthly_dataset_PRE2008_with_EBP_with_ALL_meeting_shockSUM.csv")
OUT_POST = os.path.join(OUTPUT_DIR, "bvar_monthly_dataset_POST2008_with_EBP_with_ALL_meeting_shockSUM.csv")

# ================== HELPERS ==================
def to_month_start(x):
    if isinstance(x, pd.Series):
        dt = pd.to_datetime(x, errors="coerce")
        return dt.dt.to_period("M").dt.to_timestamp()
    else:
        idx = pd.to_datetime(x, errors="coerce")
        return idx.to_period("M").to_timestamp()

def robust_to_datetime(s: pd.Series) -> pd.Series:
    """
    Robust date parsing:
    1) try default parsing
    2) if many NaT, try dayfirst=True fallback
    """
    dt = pd.to_datetime(s, errors="coerce")
    nat_share = dt.isna().mean()
    if nat_share > 0.05:
        dt2 = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if dt2.isna().mean() < nat_share:
            return dt2
    return dt

def fred_monthly(series_id: str, start: str, end: str) -> pd.Series:
    s = fred.get_series(series_id)
    s.index = pd.to_datetime(s.index)
    s = s.loc[pd.to_datetime(start):pd.to_datetime(end)]
    s.index = to_month_start(s.index)
    s = s.groupby(s.index).last()
    return s

def fred_quarterly(series_id: str) -> pd.Series:
    s = fred.get_series(series_id)
    s.index = pd.to_datetime(s.index)
    return s

def kalman_quarterly_to_monthly(
    q_series: pd.Series,
    monthly_index: pd.DatetimeIndex,
    add_seasonal: bool = True
) -> pd.Series:
    s = q_series.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s.index = s.index.to_period("M").to_timestamp()

    s = s.loc[(s.index >= monthly_index.min()) & (s.index <= monthly_index.max())]

    y = pd.Series(index=monthly_index, dtype="float64")
    y.loc[s.index] = pd.to_numeric(s.values, errors="coerce")

    model = UnobservedComponents(
        y,
        level="local level",
        seasonal=12 if add_seasonal else None
    )
    res = model.fit(disp=False)

    return pd.Series(res.smoothed_state[0], index=monthly_index)

def load_spx_monthly(path: str) -> pd.Series:
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
    ebp = pd.read_csv(path)

    date_col = next((c for c in ebp.columns if c.lower() in ["date", "dt", "time"]), ebp.columns[0])

    colmap = {c.lower().strip(): c for c in ebp.columns}
    key = ebp_col.lower().strip()
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

def pick_columns_case_insensitive(raw_cols, candidates):
    """
    Return list of actual column names in raw_cols that match candidates (case-insensitive).
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
    Robust detection of meeting date column.
    - If preferred exists, use it.
    - Else search typical names.
    - Else fallback to first column.
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
    Aggregates meeting-level shocks to monthly via SUM within month.

    Behavior:
    - Robustly detects the date column if the provided one is missing/wrong.
    - If shock_cols is None: uses ALL non-date columns.
    - Additionally: guarantees AD + Hybrid are included if present under
      common alternative names (case-insensitive).
    - Drops accidental 'date-like' columns from shock list.
    """
    raw = pd.read_csv(path)

    # (1) robust date column detection
    date_col = detect_meeting_date_col(raw, preferred=date_col)
    if date_col not in raw.columns:
        raise KeyError(f"Meeting shock date column '{date_col}' not found. Columns: {list(raw.columns)}")

    # (2) candidate discovery for AD / Hybrid (guarantee inclusion if present)
    AD_CANDS = ["AD", "aruoba", "aruoba_drechsel", "ad_shock", "shock_ad"]
    HY_CANDS = ["Hybrid", "HYBRID", "hybrid", "hybrid_shock", "shock_hybrid"]

    ad_found = pick_columns_case_insensitive(raw.columns, AD_CANDS)
    hy_found = pick_columns_case_insensitive(raw.columns, HY_CANDS)

    # (3) choose shock columns
    if shock_cols is None:
        shock_cols = [c for c in raw.columns if c != date_col]

    # remove any date-like columns from shock list (defensive)
    shock_cols_clean = []
    for c in shock_cols:
        cl = c.lower().strip()
        if cl in ["date", "meeting_date", "meetingdate", "dt", "time"]:
            continue
        shock_cols_clean.append(c)

    # ensure AD/hybrid included if found
    for c in ad_found + hy_found:
        if c != date_col and c not in shock_cols_clean:
            shock_cols_clean.append(c)

    if len(shock_cols_clean) == 0:
        raise ValueError("No shock columns detected after cleaning.")

    # (4) parse meeting dates
    raw[date_col] = robust_to_datetime(raw[date_col])
    raw = raw.dropna(subset=[date_col])
    raw["MS"] = to_month_start(raw[date_col])

    out = pd.DataFrame(index=monthly_index)

    # (5) aggregate each shock col
    for c in shock_cols_clean:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
        m = raw.groupby("MS")[c].sum(min_count=1)  # min_count avoids all-NaN -> 0 bias
        m = m.reindex(monthly_index).fillna(0.0)
        out[prefix + str(c) + suffix] = m

    # (6) print what we found (so you see AD/Hybrid explicitly)
    print("\n[Meeting shocks] date_col =", date_col)
    print("[Meeting shocks] shock columns used:", shock_cols_clean)
    print("[Meeting shocks] AD found:", ad_found if ad_found else "None")
    print("[Meeting shocks] Hybrid found:", hy_found if hy_found else "None")

    return out

# ================== CHECK INPUT FILES ==================
for p in [SPX_PATH, EBP_PATH, MEETING_SHOCK_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required input file:\n{p}")

# ================== BUILD FULL MONTHLY DATASET ==================
monthly_index = pd.date_range(pd.to_datetime(START_FULL), pd.to_datetime(END_FULL), freq="MS")
df = pd.DataFrame(index=monthly_index)
df.index.name = "DATE"

# Monthly FRED series
df["y1"]    = fred_monthly("GS1", start=START_FULL, end=END_FULL)
df["unemp"] = fred_monthly("UNRATE", start=START_FULL, end=END_FULL)

# Quarterly → Monthly (Kalman smoothing)
gdp_q  = fred_quarterly("GDPC1")
pgdp_q = fred_quarterly("GDPDEF")

df["rgdp_m"] = kalman_quarterly_to_monthly(gdp_q, df.index)
df["pgdp_m"] = kalman_quarterly_to_monthly(pgdp_q, df.index)

df["log_rgdp"] = np.log(df["rgdp_m"])
df["log_pgdp"] = np.log(df["pgdp_m"])

# SP500
spx_m = load_spx_monthly(SPX_PATH)
df["spx"] = spx_m.reindex(df.index)
df["log_sp500"] = np.log(df["spx"])

# EBP
ebp_m = load_ebp_monthly(EBP_PATH, ebp_col="ebp")
df["EBP"] = ebp_m.reindex(df.index)

# Meeting shocks -> monthly SUM (ALL columns + guaranteed AD/Hybrid if present)
shocks_monthly = load_meeting_shocks_to_monthly_sum_all(
    MEETING_SHOCK_PATH,
    monthly_index=df.index,
    date_col=MEETING_DATE_COL,    # even if wrong, auto-detection fixes it
    shock_cols=None,              # None => all non-date columns
    prefix="shock_",
    suffix="_meeting_sum"
)

# Merge into main df
df = df.join(shocks_monthly, how="left")

# ================== FINAL SELECT + SPLIT ==================
macro_cols = ["y1", "log_sp500", "log_rgdp", "log_pgdp", "unemp", "EBP"]
shock_cols_out = list(shocks_monthly.columns)

keep_cols = macro_cols + shock_cols_out
out_full = df[keep_cols].copy()

out_pre  = out_full.loc[:pd.to_datetime(PRE_END_MS)].copy()
out_post = out_full.loc[pd.to_datetime(POST_START_MS):].copy()

# ================== DIAGNOSTICS ==================
print("\n--- RANGES ---")
print("FULL range:", out_full.index.min(), "→", out_full.index.max(), "| rows:", len(out_full))
print("PRE  range:", out_pre.index.min(),  "→", out_pre.index.max(),  "| rows:", len(out_pre))
print("POST range:", out_post.index.min(), "→", out_post.index.max(), "| rows:", len(out_post))

print("\n--- NaN COUNTS (FULL) ---\n", out_full.isna().sum())
print("\n--- NaN COUNTS (PRE)  ---\n", out_pre.isna().sum())
print("\n--- NaN COUNTS (POST) ---\n", out_post.isna().sum())

# Non-zero month counts for each shock
print("\n--- Non-zero shock months (FULL) ---")
for c in shock_cols_out:
    nz = int((out_full[c] != 0).sum())
    print(f"  {c}: {nz} / {len(out_full)}")

# Explicit check for AD/Hybrid outputs
ad_like = [c for c in shock_cols_out if "ad" in c.lower()]
hy_like = [c for c in shock_cols_out if "hybrid" in c.lower()]
print("\n--- AD/Hybrid columns in OUTPUT ---")
print("AD-like:", ad_like if ad_like else "None")
print("Hybrid-like:", hy_like if hy_like else "None")

# ================== SAVE (3 outputs) ==================
out_full.to_csv(OUT_FULL, index=True)
out_pre.to_csv(OUT_PRE, index=True)
out_post.to_csv(OUT_POST, index=True)

print(f"\n✅ Saved → {OUT_FULL}")
print(f"✅ Saved → {OUT_PRE}")
print(f"✅ Saved → {OUT_POST}")
