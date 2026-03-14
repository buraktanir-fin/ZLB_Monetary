import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# ============================================================
# ANALYSIS 2: SAMPLE-CONTROLLED COMPARISON — KRIPPNER (1995–2008)
#
# Generates ONE figure with three series all estimated over
# the identical 1995–2008 window:
#   1. AD shocks        (1995–2008)
#   2. FFR shocks       (1995–2008)
#   3. Krippner shocks  (1995–2008)
#
# Purpose: by holding the sample constant across all three
# instruments, any remaining differences between Krippner and
# the FFR-based series reflect instrument design characteristics
# rather than sample length differences.
#
# Uses dataset:
#   outputs/bvar_monthly_dataset_FULL_with_EBP_with_ALL_meeting_shockSUM.csv
# ============================================================

# ================== PATHS ==================
BASE_DIR    = os.getcwd()
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

FULL_DATA_FILENAME = "bvar_monthly_dataset_FULL_with_EBP_with_ALL_meeting_shockSUM.csv"
DATA_PATH   = os.path.join(OUTPUTS_DIR, FULL_DATA_FILENAME)

OUTDIR = os.path.join(BASE_DIR, "irf_outputs_EBP_final")
os.makedirs(OUTDIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find source data file:\n{DATA_PATH}\n"
        f"Check FULL_DATA_FILENAME or that the file is in outputs/."
    )

print(f"✅ Using source data: {DATA_PATH}")
print(f"✅ Outputs folder    : {OUTDIR}")

# ================== CONFIG ==================
DATE_COL_CANDIDATES = ["MS", "DATE", "Date", "date"]

SHOCK_WX_COL  = "shock_Wu_meeting_sum"
SHOCK_KR_COL  = "shock_Kr_meeting_sum"

AD_CANDIDATES     = ["AD", "shock_AD_meeting_sum", "ad_meeting_sum",
                     "aruoba", "aruoba_drechsel"]
HYBRID_CANDIDATES = ["hybrid", "shock_hybrid_meeting_sum",
                     "hybrid_meeting_sum", "policy_hybrid"]

# ── Sample window ─────────────────────────────────────────────
SAMPLE_START = "1995-03-01"   # Krippner availability
SAMPLE_END   = "2008-11-01"   # pre-2008 split

# BVAR settings
HORIZON    = 48
DRAWS      = 4000
P_LAGS_PRE = 12

DF0      = 10
S0_SCALE = 0.2
LAMBDA_B = 0.2

CI_LO, CI_HI = 16, 84

CLIP_Q_LO, CLIP_Q_HI = 0.001, 0.999

STANDARDIZE                     = True
NORMALIZE_SHOCK_IMPACT_TO_ONE   = True
BACKTRANSFORM_TO_ORIGINAL_UNITS = True
SIGN_ALIGN_TO_Y1_POSITIVE       = True

SCALE_LOGS_BY_100 = False
LOG_VARS_TO_SCALE = ["log_sp500", "log_rgdp", "log_pgdp"]

DISPLAY_LABELS = {
    "y1":        "1-year bond yield (%)",
    "log_sp500": "S&P500 (100 × log)",
    "log_rgdp":  "Real GDP (100 × log)",
    "unemp":     "Unemployment (%)",
    "log_pgdp":  "GDP deflator (100 × log)",
    "EBP":       "EBP (%)",
}

def pretty_ylabel(v):
    return DISPLAY_LABELS.get(v, v)

SEED = 123
rng  = np.random.default_rng(SEED)

# ================== UTILITIES ==================
def detect_date_col(df):
    for c in DATE_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def to_datetime_index(df):
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
    lo, hi = s.quantile([qlo, qhi])
    return s.clip(lo, hi)

def zscore_with_stats(X):
    X  = np.asarray(X, float)
    m  = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=0)
    sd[sd == 0] = 1.0
    return (X - m) / sd, m, sd

def make_lag_matrix(Y, p):
    T, k = Y.shape
    return np.hstack([Y[p - L : T - L, :] for L in range(1, p + 1)])

def companion_matrix(A_list):
    k   = A_list[0].shape[0]
    p   = len(A_list)
    top = np.hstack(A_list)
    if p == 1:
        return top
    I     = np.eye(k * (p - 1))
    zeros = np.zeros((k * (p - 1), k))
    return np.vstack([top, np.hstack([I, zeros])])

def is_stable(A_list):
    return np.max(np.abs(np.linalg.eigvals(
        companion_matrix(A_list)))) < 0.9999

def wishart_rnd(df, V):
    V = np.asarray(V, float)
    L = np.linalg.cholesky(V)
    p = V.shape[0]
    A = np.zeros((p, p))
    for i in range(p):
        A[i, i] = np.sqrt(rng.chisquare(df - i))
        for j in range(i):
            A[i, j] = rng.normal()
    LA = L @ A
    return LA @ LA.T

def invwishart_rnd(df, S):
    return np.linalg.inv(
        wishart_rnd(df, np.linalg.inv(np.asarray(S, float))))

def sample_matrix_normal(Bbar, V, Sigma):
    Bbar, V, Sigma = [np.asarray(x, float) for x in (Bbar, V, Sigma)]
    Lv = np.linalg.cholesky(V)
    Ls = np.linalg.cholesky(Sigma)
    m, k = Bbar.shape
    Z = rng.normal(size=(m, k))
    return Bbar + (Lv @ Z @ Ls.T)

def bvar_draws_irf(Y, p, H, draws, lambda_b=LAMBDA_B,
                   df0=DF0, s0_scale=S0_SCALE, tag=""):
    t0   = time.time()
    T, k = Y.shape

    Xlags   = make_lag_matrix(Y, p)
    Ydep    = Y[p:, :]
    X       = np.hstack([np.ones((T - p, 1)), Xlags])
    ncoef   = X.shape[1]

    B0     = np.zeros((ncoef, k))
    V0_inv = np.eye(ncoef) / (lambda_b ** 2)
    S0     = (s0_scale ** 2) * np.eye(k)

    XtX    = X.T @ X
    XtY    = X.T @ Ydep
    V_post = np.linalg.inv(V0_inv + XtX)
    B_post = V_post @ (V0_inv @ B0 + XtY)

    U      = Ydep - X @ B_post
    S_post = (S0 + U.T @ U
              + (B_post - B0).T @ V0_inv @ (B_post - B0))
    df_post = df0 + (T - p)

    def parse_A_list(Bmat):
        A_list, start = [], 1
        for L in range(p):
            A_list.append(
                Bmat[start + L*k : start + (L+1)*k, :].T)
        return A_list

    keep_irfs  = []
    milestones = {250, 500, 1000, 2000, 3000, 4000}

    for d in range(draws):
        Sigma  = invwishart_rnd(df_post, S_post)
        Bdraw  = sample_matrix_normal(B_post, V_post, Sigma)
        A_list = parse_A_list(Bdraw)
        if not is_stable(A_list):
            continue

        P      = np.linalg.cholesky(Sigma)
        impact = P[:, 0].copy()
        F      = companion_matrix(A_list)
        J      = np.hstack([np.eye(k), np.zeros((k, k*(p-1)))])

        irf    = np.zeros((H + 1, k))
        irf[0] = impact
        Fh     = np.eye(k * p)
        for hh in range(1, H + 1):
            Fh      = Fh @ F
            irf[hh] = (J @ Fh @ J.T) @ impact

        keep_irfs.append(irf)
        if len(keep_irfs) in milestones:
            print(f"   [{tag}] kept {len(keep_irfs)} draws | "
                  f"attempt {d+1}/{draws} | "
                  f"elapsed {time.time()-t0:.1f}s")

    if not keep_irfs:
        raise RuntimeError(f"[{tag}] No stable posterior draws kept.")

    print(f"   [{tag}] DONE — kept {len(keep_irfs)} / {draws} | "
          f"elapsed {time.time()-t0:.1f}s")
    return np.array(keep_irfs)

def normalize_and_backtransform(irfs, sd_vec,
                                 normalize_shock=True,
                                 backtransform=True):
    irfs2 = irfs.copy()
    if normalize_shock:
        impacts = irfs2[:, 0, 0].copy()
        impacts[np.abs(impacts) < 1e-12] = np.nan
        irfs2 = irfs2 * (1.0 / impacts)[:, None, None]
    if backtransform:
        for j in range(1, irfs2.shape[2]):
            irfs2[:, :, j] *= sd_vec[j]
    return irfs2

def sign_align_to_y1(irfs, y1_index=1):
    if np.median(irfs[:, 0, y1_index]) < 0:
        return -irfs, True
    return irfs, False

def detect_col_fuzzy(df, candidates):
    cols      = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for cand in candidates:
        for lc, orig in lower_map.items():
            if cand.lower() in lc:
                return orig
    return None

# ================== CORE PIPELINE ==================
def run_bvar_irf(df_sub, shock_col, var_order, tag, p_lags):
    miss = [c for c in var_order if c not in df_sub.columns]
    if miss:
        raise ValueError(f"[{tag}] Missing columns: {miss}")

    d = df_sub[var_order].copy()
    for c in var_order:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan)
    d[shock_col] = d[shock_col].fillna(0.0)

    macro_cols = [c for c in var_order if c != shock_col]
    d = d.dropna(subset=macro_cols).copy()

    print(f"\n[{tag}] p={p_lags} | T={len(d)} "
          f"{d.index.min().date()} → {d.index.max().date()} | "
          f"nonzero shock months={(d[shock_col] != 0).sum()}")

    for c in var_order:
        d[c] = clip_series(d[c])

    Y_raw = d[var_order].values.astype(float)
    if STANDARDIZE:
        Y, _, sd_vec = zscore_with_stats(Y_raw)
    else:
        Y, sd_vec = Y_raw, np.ones(Y_raw.shape[1])

    irfs = bvar_draws_irf(Y, p=p_lags, H=HORIZON,
                          draws=DRAWS, tag=tag)
    irfs = normalize_and_backtransform(
        irfs, sd_vec,
        normalize_shock=NORMALIZE_SHOCK_IMPACT_TO_ONE,
        backtransform=BACKTRANSFORM_TO_ORIGINAL_UNITS,
    )
    if SIGN_ALIGN_TO_Y1_POSITIVE:
        irfs, flipped = sign_align_to_y1(irfs)
        if flipped:
            print(f"   [{tag}] Sign-aligned (flipped).")

    h   = np.arange(HORIZON + 1)
    med = np.median(irfs, axis=0)
    lo  = np.percentile(irfs, CI_LO, axis=0)
    hi  = np.percentile(irfs, CI_HI, axis=0)

    return h, (med, lo, hi), var_order[1:]

# ================== PLOT ==================
def plot_three_lines_1995(h, irf_dict, macro_vars, out_fig):
    """
    Three series all estimated over the same 1995-2008 window.
    Median lines + 68% credible bands.
    """
    styles = {
        "AD (1995–2008)":       dict(color="#1f77b4", ls="-",  lw=2.2),
        "FFR (1995–2008)":      dict(color="#ff7f0e", ls="--", lw=2.2),
        "Krippner (1995–2008)": dict(color="#d62728", ls="-.", lw=2.2),
    }

    n     = len(macro_vars)
    nrows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, 2,
                             figsize=(13, 4.5 * nrows),
                             sharex=True)
    axes = np.array(axes).reshape(-1)

    fig.suptitle(
        "Sample-Controlled Comparison: AD vs. FFR vs. Krippner (1995–2008)\n"
        "Pre-2008 | BVAR with EBP | Median IRFs with 68% credible intervals",
        fontsize=11, y=1.01
    )

    for i, v in enumerate(macro_vars):
        ax = axes[i]
        j  = i + 1
        for lab, (med, lo, hi) in irf_dict.items():
            st = styles[lab]
            ax.plot(h, med[:, j], label=lab,
                    color=st["color"], ls=st["ls"], lw=st["lw"])
            ax.fill_between(h, lo[:, j], hi[:, j],
                            color=st["color"], alpha=0.12)
        ax.axhline(0, color="steelblue", lw=0.9)
        ax.set_ylabel(pretty_ylabel(v), fontsize=9)
        ax.set_title(v, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, loc="best")

    for k in range(n, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved figure → {out_fig}")

# ================== MAIN ==================
def main():
    df = pd.read_csv(DATA_PATH)
    df = to_datetime_index(df)

    if "EBP" not in df.columns:
        raise ValueError(
            f"EBP column not found. Available: {list(df.columns)}")

    if SCALE_LOGS_BY_100:
        for c in LOG_VARS_TO_SCALE:
            if c in df.columns:
                df[c] = 100.0 * pd.to_numeric(df[c], errors="coerce")

    macro_vars     = ["y1", "log_sp500", "log_rgdp",
                      "log_pgdp", "unemp", "EBP"]
    baseline_order = ["SHOCK"] + macro_vars

    def build_order(sc):
        return [sc if x == "SHOCK" else x for x in baseline_order]

    # ── detect AD and FFR columns ─────────────────────────────
    ad_col = detect_col_fuzzy(df, AD_CANDIDATES)
    hy_col = detect_col_fuzzy(df, HYBRID_CANDIDATES)

    if ad_col is None:
        raise ValueError(
            f"Could not detect AD shock column.\n"
            f"Tried: {AD_CANDIDATES}\n"
            f"Available: {list(df.columns)}")
    if hy_col is None:
        raise ValueError(
            f"Could not detect FFR/Hybrid shock column.\n"
            f"Tried: {HYBRID_CANDIDATES}\n"
            f"Available: {list(df.columns)}")

    print(f"✅ AD column detected       : '{ad_col}'")
    print(f"✅ FFR column detected      : '{hy_col}'")
    print(f"✅ Krippner column          : '{SHOCK_KR_COL}'")

    # ── restrict to 1995–2008 ─────────────────────────────────
    start  = pd.to_datetime(SAMPLE_START)
    end    = pd.to_datetime(SAMPLE_END)
    df_sub = df.loc[(df.index >= start) & (df.index < end)].copy()

    print(f"\n✅ Subsample: {df_sub.index.min().date()} → "
          f"{df_sub.index.max().date()} "
          f"({len(df_sub)} months)")

    # ── run 1: AD (1995–2008) ─────────────────────────────────
    print("\n" + "="*60)
    print("RUN 1 of 3 — AD shocks | 1995–2008")
    print("="*60)
    h, irf_ad, mv = run_bvar_irf(
        df_sub, ad_col, build_order(ad_col),
        tag="AD 1995-2008", p_lags=P_LAGS_PRE,
    )

    # ── run 2: FFR/Hybrid (1995–2008) ────────────────────────
    print("\n" + "="*60)
    print("RUN 2 of 3 — FFR (Hybrid) shocks | 1995–2008")
    print("="*60)
    _, irf_ffr, _ = run_bvar_irf(
        df_sub, hy_col, build_order(hy_col),
        tag="FFR 1995-2008", p_lags=P_LAGS_PRE,
    )

    # ── run 3: Krippner (1995–2008) ──────────────────────────
    print("\n" + "="*60)
    print("RUN 3 of 3 — Krippner shocks | 1995–2008")
    print("="*60)
    _, irf_kr, _ = run_bvar_irf(
        df_sub, SHOCK_KR_COL, build_order(SHOCK_KR_COL),
        tag="Krippner 1995-2008", p_lags=P_LAGS_PRE,
    )

    # ── plot ──────────────────────────────────────────────────
    out_fig = os.path.join(
        OUTDIR,
        f"Analysis2_1995_AD_FFR_Krippner_"
        f"p{P_LAGS_PRE}_H{HORIZON}_EBP.png"
    )

    plot_three_lines_1995(
        h,
        irf_dict={
            "AD (1995–2008)":       irf_ad,
            "FFR (1995–2008)":      irf_ffr,
            "Krippner (1995–2008)": irf_kr,
        },
        macro_vars=mv,
        out_fig=out_fig,
    )

    print("\n✅ Analysis 2 complete.")
    print(f"   Figure saved: {out_fig}")

if __name__ == "__main__":
    main()