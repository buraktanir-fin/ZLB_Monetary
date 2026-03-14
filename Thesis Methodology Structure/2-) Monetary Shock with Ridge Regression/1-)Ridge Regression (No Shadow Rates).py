import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

# ============================================================
# BUILD SHOCK SERIES (RIDGE / AD-HYBRID STYLE) + GRAPHS
# (NO SHADOW RATES)
# ============================================================

# -------------------------
# DIRECTORIES
# -------------------------
BASE_DIR = os.getcwd()

CONTROL_RESULTS_DIR = os.path.join(BASE_DIR, "Forecast and Sentiment Results")
ARUOBA_DIR = os.path.join(BASE_DIR, "Aruoba Drechsel Data")

OUT_DIR = os.path.join(BASE_DIR, "Shock Results")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# INPUT FILES
# -------------------------
FULL_DATASET_PATH = os.path.join(CONTROL_RESULTS_DIR, "AD_full_dataset_ready.csv")

ARUOBA_XLSX = os.path.join(ARUOBA_DIR, "Aruoba_Drechsel_Data.xlsx")
ARUOBA_SHEET = "Shocks and FFR by meeting"
ARUOBA_DATE_COL = "FOMC meeting (scheduled)"
ARUOBA_SHOCK_COL = "Shock"

# -------------------------
# OUTPUT FILES
# -------------------------
OUT_SHOCKS_CSV = os.path.join(OUT_DIR, "shock_series_no_shadow.csv")
OUT_DETAILS_CSV = os.path.join(OUT_DIR, "shock_model_details_no_shadow.csv")
OUT_METRICS_CSV = os.path.join(OUT_DIR, "shock_corr_vs_aruoba_no_shadow.csv")

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


def standardize_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=0)
    if sd and sd > 0:
        return (x - mu) / sd
    return x


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
# 2) IDENTIFY COLUMN GROUPS (MATCHING YOUR CODE)
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

if len(extra_cols_ffr) < 1:
    print("[WARN] FFR lag columns not found. Models will run without them.")


# ============================================================
# 3) LOAD ARUOBA ORIGINAL SHOCKS
# ============================================================
aru = pd.read_excel(ARUOBA_XLSX, sheet_name=ARUOBA_SHEET)

if ARUOBA_DATE_COL not in aru.columns or ARUOBA_SHOCK_COL not in aru.columns:
    raise ValueError(f"Aruoba sheet must contain '{ARUOBA_DATE_COL}' and '{ARUOBA_SHOCK_COL}'.")

aru["Date"] = pd.to_datetime(aru[ARUOBA_DATE_COL].astype(str).str.replace("_", "-"), errors="coerce")
aru = ensure_datetime(aru, "Date")

aru["aruoba_original_shock"] = pd.to_numeric(aru[ARUOBA_SHOCK_COL], errors="coerce")
aru = aru.dropna(subset=["aruoba_original_shock"])[["Date", "aruoba_original_shock"]].copy()

df = df.merge(aru, on="Date", how="left")


# ============================================================
# 4) BUILD SHOCK VARIANTS (NO SHADOW)
# ============================================================
shock_out = pd.DataFrame({"Date": df["Date"]})
details_rows = []

# (A) Hybrid: forecast + sentiment + FFR lags
X_hybrid = forecast_cols + sentiment_cols + extra_cols_ffr
shock, yhat, alpha, used_cols = ridge_shock(df, "FFR_change", X_hybrid, "hybrid")
shock_out["hybrid_shock"] = shock
shock_out["FFR_pred_hybrid"] = yhat
details_rows.append({"model": "hybrid_shock", "y": "FFR_change", "alpha": alpha, "k_regressors": len(used_cols)})

# (B) Sentiment-only: sentiment + FFR lags
X_sent_only = sentiment_cols + extra_cols_ffr
shock, yhat, alpha, used_cols = ridge_shock(df, "FFR_change", X_sent_only, "sentiment_only")
shock_out["sentiment_only_shock"] = shock
shock_out["FFR_pred_sent_only"] = yhat
details_rows.append({"model": "sentiment_only_shock", "y": "FFR_change", "alpha": alpha, "k_regressors": len(used_cols)})

# (C) Forecast-only: forecast + FFR lags
X_fcst_only = forecast_cols + extra_cols_ffr
shock, yhat, alpha, used_cols = ridge_shock(df, "FFR_change", X_fcst_only, "forecast_only")
shock_out["forecast_only_shock"] = shock
shock_out["FFR_pred_fcst_only"] = yhat
details_rows.append({"model": "forecast_only_shock", "y": "FFR_change", "alpha": alpha, "k_regressors": len(used_cols)})

# (D) Aruoba original
shock_out["aruoba_original_shock"] = df["aruoba_original_shock"]
details_rows.append({"model": "aruoba_original_shock", "y": "Aruoba", "alpha": np.nan, "k_regressors": np.nan})

# Save outputs
shock_out.to_csv(OUT_SHOCKS_CSV, index=False)
pd.DataFrame(details_rows).to_csv(OUT_DETAILS_CSV, index=False)

print("✅ Saved shock series →", OUT_SHOCKS_CSV)
print("✅ Saved model details →", OUT_DETAILS_CSV)


# ============================================================
# 5) COMPARISON METRICS VS ARUOBA
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
# 6) GRAPHS (SHORT TITLES)
# ============================================================
# 6.1 Overlay (standardized)
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
ts_path = os.path.join(OUT_DIR, "shocks_overlay_std.png")
plt.savefig(ts_path, dpi=200)
plt.close()
print("✅ Saved:", ts_path)

# 6.2 Correlation heatmap (Pearson)
corr_mat = shock_out[plot_cols].corr(method="pearson", min_periods=20)

plt.figure(figsize=(9, 7))
plt.imshow(corr_mat.values, aspect="auto")
plt.xticks(range(len(corr_mat.columns)), corr_mat.columns, rotation=90, fontsize=7)
plt.yticks(range(len(corr_mat.index)), corr_mat.index, fontsize=7)
plt.title("Corr (Pearson)")
plt.colorbar()
plt.tight_layout()
hm_path = os.path.join(OUT_DIR, "shocks_corr_heatmap.png")
plt.savefig(hm_path, dpi=200)
plt.close()
print("✅ Saved:", hm_path)

# 6.3 Scatterplots vs Aruoba
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
    sp_path = os.path.join(OUT_DIR, f"scatter_{v}_vs_AD.png")
    plt.savefig(sp_path, dpi=200)
    plt.close()

print("✅ Saved scatterplots in:", OUT_DIR)
