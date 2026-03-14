import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# THESIS FIGURES FROM SAVED IRF CSVs (NO RE-ESTIMATION)
#
# Required figures:
#   1) PRE-2008: AD vs FFR   (FFR is the Hybrid shock series; label as "FFR")
#   2) POST-2008: Wu–Xia ONLY
#   3) Cross-regime: PRE-2008 FFR vs POST-2008 Wu–Xia
#
# Reads the IRF CSV outputs produced by your estimation code:
#   Table_IRF_<Shock>_<Sample>_p<...>_H<...>_EBP.csv
# ============================================================

# ================== PATHS ==================
BASE_DIR = os.getcwd()

# Folder where your estimation script saved the IRF CSVs:
IRF_DIR = os.path.join(BASE_DIR, "irf_outputs_EBP_final")
OUTDIR = os.path.join(IRF_DIR, "thesis_figures_from_csv")
os.makedirs(OUTDIR, exist_ok=True)

# ================== DISPLAY (THESIS UNITS) ==================
DISPLAY_LABELS = {
    "y1": "1-year bond yield (%)",
    "log_sp500": "S&P500 (100 × log)",
    "log_rgdp": "Real GDP (100 × log)",
    "unemp": "Unemployment (%)",
    "log_pgdp": "GDP deflator (100 × log)",
    "EBP": "EBP (%)",
}

PREFERRED_ORDER = ["y1", "log_sp500", "log_rgdp", "unemp", "log_pgdp", "EBP"]

def pretty_ylabel(varname: str) -> str:
    return DISPLAY_LABELS.get(varname, varname)

# ================== FILE HELPERS ==================
def find_first(pattern: str) -> str:
    """Find the first file in IRF_DIR matching a regex pattern (case-sensitive)."""
    rx = re.compile(pattern)
    for fn in sorted(os.listdir(IRF_DIR)):
        if rx.fullmatch(fn):
            return os.path.join(IRF_DIR, fn)
    raise FileNotFoundError(f"Could not find a file matching:\n  {pattern}\nIn:\n  {IRF_DIR}")

def read_irf_csv(path: str):
    """
    Reads a saved IRF CSV (h + <var>_med/_lo/_hi columns).
    Returns: h (np.array), var_list (list), med/lo/hi dicts: var -> np.array
    """
    df = pd.read_csv(path)
    if "h" not in df.columns:
        raise ValueError(f"'h' column missing in {path}. Columns: {list(df.columns)}")

    h = df["h"].to_numpy()

    vars_found = []
    for c in df.columns:
        if c.endswith("_med"):
            vars_found.append(c[:-4])
    vars_found = sorted(set(vars_found))

    ordered = [v for v in PREFERRED_ORDER if v in vars_found] + [v for v in vars_found if v not in PREFERRED_ORDER]

    med, lo, hi = {}, {}, {}
    for v in ordered:
        for suf in ["_med", "_lo", "_hi"]:
            col = v + suf
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {path}.")
        med[v] = df[v + "_med"].to_numpy()
        lo[v]  = df[v + "_lo"].to_numpy()
        hi[v]  = df[v + "_hi"].to_numpy()

    return h, ordered, med, lo, hi

# ================== PLOTTING ==================
def plot_overlay_two_shocks(h, vars_list, A, B, labelA, labelB, title, out_png):
    """
    A/B: tuples (med_dict, lo_dict, hi_dict)
    """
    medA, loA, hiA = A
    medB, loB, hiB = B

    n = len(vars_list)
    nrows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for i, v in enumerate(vars_list):
        ax = axes[i]
        ax.plot(h, medA[v], linewidth=2, label=labelA)
        ax.fill_between(h, loA[v], hiA[v], alpha=0.20)
        ax.plot(h, medB[v], linewidth=2, linestyle="--", label=labelB)
        ax.fill_between(h, loB[v], hiB[v], alpha=0.20)

        ax.axhline(0, linewidth=1)
        ax.set_ylabel(pretty_ylabel(v))
        ax.grid(True)
        ax.legend()

    for k in range(n, len(axes)):
        axes[k].axis("off")

    fig.suptitle(title, y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved: {out_png}")

def plot_single_shock(h, vars_list, med, lo, hi, title, out_png):
    n = len(vars_list)
    nrows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for i, v in enumerate(vars_list):
        ax = axes[i]
        ax.plot(h, med[v], linewidth=2)
        ax.fill_between(h, lo[v], hi[v], alpha=0.25)
        ax.axhline(0, linewidth=1)
        ax.set_ylabel(pretty_ylabel(v))
        ax.grid(True)

    for k in range(n, len(axes)):
        axes[k].axis("off")

    fig.suptitle(title, y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved: {out_png}")

# ================== MAIN ==================
def main():
    # ------------------------------------------------------------
    # Locate needed CSVs (robust to exact p/H by regex)
    #   PRE: AD + Hybrid (but labeled FFR in plots)
    #   POST: WuXia
    # ------------------------------------------------------------
    ad_pre_path = find_first(r"Table_IRF_AruobaDrechsel_Pre2008_p\d+_H\d+_EBP\.csv")
    hy_pre_path = find_first(r"Table_IRF_Hybrid_Pre2008_p\d+_H\d+_EBP\.csv")
    wx_post_path = find_first(r"Table_IRF_WuXia_Post2008_p\d+_H\d+_EBP\.csv")

    # Read
    h_ad, vars_ad, med_ad, lo_ad, hi_ad = read_irf_csv(ad_pre_path)
    h_ffr, vars_ffr, med_ffr, lo_ffr, hi_ffr = read_irf_csv(hy_pre_path)  # Hybrid series, displayed as FFR
    h_wx, vars_wx, med_wx, lo_wx, hi_wx = read_irf_csv(wx_post_path)

    # Variables intersection for overlays
    common_vars_pre = [v for v in PREFERRED_ORDER if (v in vars_ad and v in vars_ffr)]
    if not common_vars_pre:
        common_vars_pre = sorted(set(vars_ad).intersection(vars_ffr))

    common_vars_ffr_wx = [v for v in PREFERRED_ORDER if (v in vars_ffr and v in vars_wx)]
    if not common_vars_ffr_wx:
        common_vars_ffr_wx = sorted(set(vars_ffr).intersection(vars_wx))

    # Horizon alignment warnings
    if len(h_ad) != len(h_ffr) or not np.allclose(h_ad, h_ffr):
        print("⚠️ PRE horizons differ between AD and FFR; using AD horizon for plotting.")
    if len(h_ffr) != len(h_wx) or not np.allclose(h_ffr, h_wx):
        print("⚠️ Horizons differ between PRE FFR and POST Wu–Xia; using PRE horizon for the cross-regime overlay.")

    # ------------------------------------------------------------
    # Figure 1: PRE-2008 AD vs FFR  (Hybrid labeled as FFR)
    # ------------------------------------------------------------
    out1 = os.path.join(OUTDIR, "Figure_Pre2008_Compare_AD_vs_FFR.png")
    plot_overlay_two_shocks(
        h_ad,
        common_vars_pre,
        (med_ad, lo_ad, hi_ad),
        (med_ffr, lo_ffr, hi_ffr),
        labelA="AD",
        labelB="FFR",
        title="Pre-2008: IRFs to AD vs FFR Monetary Policy Shocks",
        out_png=out1,
    )

    # ------------------------------------------------------------
    # Figure 2: POST-2008 Wu–Xia only
    # ------------------------------------------------------------
    out2 = os.path.join(OUTDIR, "Figure_Post2008_IRF_WuXia.png")
    vars_wx_plot = [v for v in PREFERRED_ORDER if v in vars_wx] + [v for v in vars_wx if v not in PREFERRED_ORDER]
    plot_single_shock(
        h_wx,
        vars_wx_plot,
        med_wx, lo_wx, hi_wx,
        title="Post-2008: IRFs to Wu–Xia Monetary Policy Shock",
        out_png=out2,
    )

    # ------------------------------------------------------------
    # Figure 3: Cross-regime overlay PRE-2008 FFR vs POST-2008 Wu–Xia
    # ------------------------------------------------------------
    # Use PRE horizon by default (matches your request wording)
    out3 = os.path.join(OUTDIR, "Figure_CrossRegime_Compare_PreFFR_vs_PostWuXia.png")
    plot_overlay_two_shocks(
        h_ffr,
        common_vars_ffr_wx,
        (med_ffr, lo_ffr, hi_ffr),
        (med_wx, lo_wx, hi_wx),
        labelA="FFR (Pre-2008)",
        labelB="Wu–Xia (Post-2008)",
        title="Cross-regime: Pre-2008 FFR vs Post-2008 Wu–Xia",
        out_png=out3,
    )

    print("\n✅ Done.")
    print("Inputs:")
    print("  AD PRE    :", ad_pre_path)
    print("  FFR PRE   :", hy_pre_path, " (Hybrid series labeled as FFR)")
    print("  WX POST   :", wx_post_path)
    print("Outputs:")
    print(" ", OUTDIR)

if __name__ == "__main__":
    main()