import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

# ============================================================
# SHOCKS (RIDGE / AD-HYBRID) + SHADOW-RATE SHOCKS (Δ shadow)
# - Hybrid shock (FFR_change target)
# - Aruoba original shock
# - Wu-Xia shadow-rate shock (target = Δ shadow aligned to meeting dates)
# - Krippner shadow-rate shock (target = Δ shadow aligned to meeting dates)
#
# ALL outputs are saved into:
#   Shock Results / Shadow Rate Results
# ============================================================

# -------------------------
# DIRECTORIES
# -------------------------
BASE_DIR = os.getcwd()

CONTROL_RESULTS_DIR = os.path.join(BASE_DIR, "Forecast and Sentiment Results")
ARUOBA_DIR = os.path.join(BASE_DIR, "Aruoba Drechsel Data")
SHADOW_DIR = os.path.join(BASE_DIR, "Shadow Rates")

OUT_DIR = os.path.join(BASE_DIR, "Shock Results")
SHADOW_OUT_DIR = os.path.join(OUT_DIR, "Shadow Rate Results")
os.makedirs(SHADOW_OUT_DIR, exist_ok=True)

# -------------------------
# INPUT FILES
# -------------------------
FULL_DATASET_PATH = os.path.join(CONTROL_RESULTS_DIR, "AD_full_dataset_ready.csv")

ARUOBA_XLSX = os.path.join(ARUOBA_DIR, "Aruoba_Drechsel_Data.xlsx")
ARUOBA_SHEET = "Shocks and FFR by meeting"
ARUOBA_DATE_COL = "FOMC meeting (scheduled)"
ARUOBA_SHOCK_COL = "Shock"

WU_XIA_PATH = os.path.join(SHADOW_DIR, "WuXiaShadowRate.xlsx")
KRIPPNER_PATH = os.path.join(SHADOW_DIR, "krippner_shadow_rate.xlsx")

# -------------------------
# OUTPUT FILES (ALL IN SHADOW_OUT_DIR)
# -------------------------
OUT_SHOCKS_CSV = os.path.join(SHADOW_OUT_DIR, "shock_series_shadow_only.csv")
OUT_DETAILS_CSV = os.path.join(SHADOW_OUT_DIR, "shock_model_details_shadow_only.csv")
OUT_METRICS_CSV = os.path.join(SHADOW_OUT_DIR, "shock_corr_vs_aruoba_shadow_only.csv")

# Graph outputs (short names)
TS_PATH = os.path.join(SHADOW_OUT_DIR, "shocks_overlay_std.png")
HM_PATH = os.path.join(SHADOW_OUT_DIR, "shocks_corr_heatmap.png")

# -------------------------
# RIDGE SETTINGS
# -------------------------
ALPHAS = np.logspace(-3, 3, 30)
CV_FOLDS = 10


# ============================================================
# HELPERS
# ============================================================
def ensure_datetime(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=[col]).sort_values(col)
    df = df.drop_duplicates(subset=[col]).reset_index(drop=True)
    return df


def standardize_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=0)
    if sd and sd > 0:
        return (x - mu) / sd
    return x


def corr_metrics(a: pd.Series, b: pd.Series):
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
        "rmse": float(np.sqrt((diff**2).mean())),
    }


def ridge_shock(df: pd.DataFrame, y_col: str, x_cols: list, model_name: str):
    """
    Exact logic:
      - X numeric
      - drop all-NaN columns
      - fill remaining NaN with column mean
      - standardize
      - RidgeCV
      - shock = y - y_hat
    Returns: shock, y_hat, alpha, final_x_cols
    """
    work = df[["Date", y_col] + x_cols].copy()

    y = pd.to_numeric(work[y_col], errors="coerce").astype(float).to_numpy()
    X = work[x_cols].apply(pd.to_numeric, errors="coerce").to_numpy()

    # drop columns that are all NaN
    col_nan_all = np.isnan(X).all(axis=0)
    if col_nan_all.any():
        X = X[:, ~col_nan_all]
        x_cols = [c for i, c in enumerate(x_cols) if not col_nan_all[i]]

    # fill NaN with column mean
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

    # y NaNs: fit on valid subset, predict all
    valid_y = ~np.isnan(y)
    if valid_y.sum() < max(30, CV_FOLDS + 5):
        shock = np.full_like(y, np.nan, dtype=float)
        y_hat = np.full_like(y, np.nan, dtype=float)
        return shock, y_hat, np.nan, x_cols

    X_train = X[valid_y]
    y_train = y[valid_y]

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_all_std = scaler.transform(X)

    ridge = RidgeCV(alphas=ALPHAS, cv=CV_FOLDS)
    ridge.fit(X_train_std, y_train)

    y_hat = ridge.predict(X_all_std)
    shock = y - y_hat

    return shock, y_hat, float(ridge.alpha_), x_cols


def add_target_lags(df: pd.DataFrame, y_col: str, prefix: str):
    """
    Builds lag1 and lag1^2 from the TARGET series (e.g., WuXia_change).
    """
    y = pd.to_numeric(df[y_col], errors="coerce")
    df[f"{prefix}_lag1"] = y.shift(1)
    df[f"{prefix}_lag1_sq"] = df[f"{prefix}_lag1"] ** 2
    return [f"{prefix}_lag1", f"{prefix}_lag1_sq"]


# ============================================================
# SHADOW RATE LOADERS (REFLECTING YOUR FILE STRUCTURES)
# ============================================================
# --- REPLACE your two shadow-rate loader functions with these versions ---

def load_wu_xia_levels(path: str) -> pd.DataFrame:
    """
    WuXiaShadowRate.xlsx
      - MUST use sheet: "Data"
      - Date column: first datetime-like col (fallback: first col)
      - Level column: best match containing 'wu' + 'shadow' (fallback: best numeric)
    Returns: Date, level
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    raw = pd.read_excel(path, sheet_name="Data")

    # date col
    date_col = None
    for c in raw.columns:
        if pd.api.types.is_datetime64_any_dtype(raw[c]):
            date_col = c
            break
    if date_col is None:
        date_col = raw.columns[0]

    # level col
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
        candidates = [c for c in raw.columns if c != date_col]
        best, best_n = None, -1
        for c in candidates:
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


def load_krippner_us_levels(path: str) -> pd.DataFrame:
    """
    krippner_shadow_rate.xlsx
      - MUST use sheet: "B. Month end SSR series"
      - The sheet contains some header/disclaimer rows.
      - We locate the header row containing "US SSR" then read:
          col0 = Date, col1 = US SSR (level)
    Returns: Date, level
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    raw = pd.read_excel(path, sheet_name="D. Month-end SSR series", header=None)

    # Find the header row containing "US SSR"
    hit = np.where(raw.astype(str).values == "US SSR")
    if len(hit[0]) == 0:
        raise ValueError("Could not locate 'US SSR' header cell in Krippner month-end sheet.")
    header_row = int(hit[0][0])

    data = raw.iloc[header_row + 1 :, [0, 1]].copy()
    data.columns = ["Date", "level"]

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["level"] = pd.to_numeric(data["level"], errors="coerce")
    data = data.dropna(subset=["Date", "level"]).sort_values("Date")
    data = data.drop_duplicates(subset=["Date"]).reset_index(drop=True)
    return data[["Date", "level"]].copy()


def align_levels_to_meetings_and_diff(meeting_dates: pd.Series, levels: pd.DataFrame, out_col: str) -> pd.DataFrame:
    """
    Aligns shadow LEVELS to meeting dates using merge_asof (backward),
    then computes meeting-to-meeting Δ on the aligned series:
      Δ_t = level_aligned_t - level_aligned_{t-1}
    """
    base = pd.DataFrame({"Date": pd.to_datetime(meeting_dates)}).sort_values("Date").reset_index(drop=True)

    lv = levels.copy()
    lv = ensure_datetime(lv, "Date")
    lv = lv.sort_values("Date").reset_index(drop=True)

    aligned = pd.merge_asof(base, lv, on="Date", direction="backward", allow_exact_matches=True)
    aligned["level_aligned"] = aligned["level"]

    aligned[out_col] = aligned["level_aligned"] - aligned["level_aligned"].shift(1)
    return aligned[["Date", out_col]].copy()


# ============================================================
# 1) LOAD FULL DATASET (NO DATE RESTRICTION)
# ============================================================
if not os.path.exists(FULL_DATASET_PATH):
    raise FileNotFoundError(f"Full dataset not found: {FULL_DATASET_PATH}")

df = pd.read_csv(FULL_DATASET_PATH)
if "Date" not in df.columns:
    raise ValueError("Dataset must have a 'Date' column.")
df = ensure_datetime(df, "Date")

if "FFR_change" not in df.columns:
    raise ValueError("Dataset must contain 'FFR_change'.")

# ============================================================
# 2) IDENTIFY COLUMN GROUPS (MATCHING YOUR RULES)
# ============================================================
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

extra_cols_ffr = [c for c in ["FFR_lag1", "FFR_lag1_sq"] if c in df.columns]

print("Forecast cols:", len(forecast_cols))
print("Sentiment cols:", len(sentiment_cols))
print("FFR lag cols:", extra_cols_ffr)

# ============================================================
# 3) LOAD ARUOBA ORIGINAL SHOCKS
# ============================================================
if not os.path.exists(ARUOBA_XLSX):
    raise FileNotFoundError(f"Aruoba file not found: {ARUOBA_XLSX}")

aru = pd.read_excel(ARUOBA_XLSX, sheet_name=ARUOBA_SHEET)

if ARUOBA_DATE_COL not in aru.columns or ARUOBA_SHOCK_COL not in aru.columns:
    raise ValueError(f"Aruoba sheet must contain '{ARUOBA_DATE_COL}' and '{ARUOBA_SHOCK_COL}'.")

aru["Date"] = pd.to_datetime(aru[ARUOBA_DATE_COL].astype(str).str.replace("_", "-"), errors="coerce")
aru = ensure_datetime(aru, "Date")
aru["aruoba_original_shock"] = pd.to_numeric(aru[ARUOBA_SHOCK_COL], errors="coerce")
aru = aru.dropna(subset=["aruoba_original_shock"])[["Date", "aruoba_original_shock"]].copy()

df = df.merge(aru, on="Date", how="left")

# ============================================================
# 4) LOAD + ALIGN SHADOW LEVELS, THEN COMPUTE Δ ON MEETING GRID
# ============================================================
if not os.path.exists(WU_XIA_PATH):
    raise FileNotFoundError(f"Wu-Xia shadow file not found: {WU_XIA_PATH}")
if not os.path.exists(KRIPPNER_PATH):
    raise FileNotFoundError(f"Krippner shadow file not found: {KRIPPNER_PATH}")

wu_levels = load_wu_xia_levels(WU_XIA_PATH)
kr_levels = load_krippner_us_levels(KRIPPNER_PATH)

wu_change_df = align_levels_to_meetings_and_diff(df["Date"], wu_levels, "WuXia_change")
kr_change_df = align_levels_to_meetings_and_diff(df["Date"], kr_levels, "Krippner_change")

df = df.merge(wu_change_df, on="Date", how="left")
df = df.merge(kr_change_df, on="Date", how="left")

# ============================================================
# 5) BUILD SHOCK VARIANTS (HYBRID + ARUOBA + SHADOWS)
# ============================================================
shock_out = pd.DataFrame({"Date": df["Date"]})
details_rows = []

# (A) Hybrid (AD replication): y=FFR_change, X=forecast+sentiment+FFR lags
X_hybrid = forecast_cols + sentiment_cols + extra_cols_ffr
shock, yhat, alpha, used_cols = ridge_shock(df, "FFR_change", X_hybrid, "hybrid_ffr")
shock_out["hybrid_shock"] = shock
shock_out["FFR_pred_hybrid"] = yhat
details_rows.append({"model": "hybrid_shock", "y": "FFR_change", "alpha": alpha, "k_regressors": len(used_cols)})

# (B) Aruoba original shocks
shock_out["aruoba_original_shock"] = df["aruoba_original_shock"]
details_rows.append({"model": "aruoba_original_shock", "y": "Aruoba", "alpha": np.nan, "k_regressors": np.nan})

# (C) Wu-Xia shadow shock: y=ΔWuXia, X=forecast+sentiment+lags(ΔWuXia)
wu_lags = add_target_lags(df, "WuXia_change", "WuXia")
X_wu = forecast_cols + sentiment_cols + wu_lags
shock, yhat, alpha, used_cols = ridge_shock(df, "WuXia_change", X_wu, "wuxia_shadow")
shock_out["wuxia_shock"] = shock
shock_out["WuXia_pred"] = yhat
details_rows.append({"model": "wuxia_shock", "y": "WuXia_change", "alpha": alpha, "k_regressors": len(used_cols)})

# (D) Krippner shadow shock: y=ΔKrippner, X=forecast+sentiment+lags(ΔKrippner)
kr_lags = add_target_lags(df, "Krippner_change", "Krippner")
X_kr = forecast_cols + sentiment_cols + kr_lags
shock, yhat, alpha, used_cols = ridge_shock(df, "Krippner_change", X_kr, "krippner_shadow")
shock_out["krippner_shock"] = shock
shock_out["Krippner_pred"] = yhat
details_rows.append({"model": "krippner_shock", "y": "Krippner_change", "alpha": alpha, "k_regressors": len(used_cols)})

# Save outputs (ALL IN SHADOW_OUT_DIR)
shock_out.to_csv(OUT_SHOCKS_CSV, index=False)
pd.DataFrame(details_rows).to_csv(OUT_DETAILS_CSV, index=False)
print("✅ Saved shock series →", OUT_SHOCKS_CSV)
print("✅ Saved model details →", OUT_DETAILS_CSV)

# ============================================================
# 6) COMPARISON METRICS VS ARUOBA ORIGINAL (WHERE OVERLAP EXISTS)
# ============================================================
ref = "aruoba_original_shock"
variants = [c for c in shock_out.columns if c not in ("Date", ref) and not c.endswith("_pred")]

metrics_rows = []
for v in variants:
    m = corr_metrics(shock_out[v], shock_out[ref])
    m.update({"variant": v, "reference": ref})
    metrics_rows.append(m)

metrics = pd.DataFrame(metrics_rows).sort_values("pearson", ascending=False)
metrics.to_csv(OUT_METRICS_CSV, index=False)
print("✅ Saved comparison metrics →", OUT_METRICS_CSV)

if not metrics.empty:
    print("\nBEST (Pearson):")
    print(metrics.iloc[0][["variant", "n", "pearson", "spearman", "mae", "rmse"]].to_string())
    print("\nWORST (Pearson):")
    print(metrics.iloc[-1][["variant", "n", "pearson", "spearman", "mae", "rmse"]].to_string())

# ============================================================
# 7) GRAPHS (SHORT TITLES) — SAVED INTO SHADOW_OUT_DIR
# ============================================================
# Overlay (standardized)
plot_cols = [ref] + variants
plot_df = shock_out[["Date"] + plot_cols].copy()
for c in plot_cols:
    plot_df[c] = standardize_series(plot_df[c])

plt.figure(figsize=(12, 5))
for c in plot_cols:
    plt.plot(plot_df["Date"], plot_df[c], label=c, linewidth=1)

plt.title("Shocks (Std)")
plt.xlabel("Date")
plt.ylabel("Std shock")
plt.legend(ncol=2, fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(TS_PATH, dpi=200)
plt.close()
print("✅ Saved:", TS_PATH)

# Correlation heatmap (Pearson)
corr_mat = shock_out[plot_cols].corr(method="pearson", min_periods=20)

plt.figure(figsize=(9, 7))
plt.imshow(corr_mat.values, aspect="auto")
plt.xticks(range(len(corr_mat.columns)), corr_mat.columns, rotation=90, fontsize=7)
plt.yticks(range(len(corr_mat.index)), corr_mat.index, fontsize=7)
plt.title("Corr (Pearson)")
plt.colorbar()
plt.tight_layout()
plt.savefig(HM_PATH, dpi=200)
plt.close()
print("✅ Saved:", HM_PATH)

# Scatter vs AD
for v in variants:
    x = shock_out[v]
    y = shock_out[ref]
    valid = x.notna() & y.notna()
    if valid.sum() < 30:
        continue

    pear = x[valid].corr(y[valid], method="pearson")

    plt.figure(figsize=(6, 6))
    plt.scatter(x[valid], y[valid], alpha=0.5)
    plt.xlabel(v)
    plt.ylabel(ref)
    plt.title(f"{v} vs AD ({pear:.3f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    sp_path = os.path.join(SHADOW_OUT_DIR, f"sc_{v}_vs_AD.png")
    plt.savefig(sp_path, dpi=200)
    plt.close()

print("✅ Saved scatterplots in:", SHADOW_OUT_DIR)
