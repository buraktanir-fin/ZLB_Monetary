import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# THESIS FIGURE FROM SAVED IRF CSVs (NO RE-ESTIMATION)
#
# Figure created:
#   PRE-2008: 4-shock comparison
#       - AD
#       - FFR   (read from Hybrid CSV, displayed as FFR)
#       - Wu–Xia
#       - Krippner
#
# Reads existing CSV outputs such as:
#   Table_IRF_AruobaDrechsel_Pre2008_p12_H48_EBP.csv
#   Table_IRF_Hybrid_Pre2008_p12_H48_EBP.csv
#   Table_IRF_WuXia_Pre2008_p12_H48_EBP.csv
#   Table_IRF_Krippner_Pre2008_p12_H48_EBP.csv
# ============================================================

# ================== PATHS ==================
BASE_DIR = os.getcwd()
IRF_DIR = os.path.join(BASE_DIR, "irf_outputs_EBP_final")
OUTDIR = os.path.join(IRF_DIR, "thesis_figures_from_csv")
os.makedirs(OUTDIR, exist_ok=True)

# ================== DISPLAY LABELS ==================
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
    rx = re.compile(pattern)
    for fn in sorted(os.listdir(IRF_DIR)):
        if rx.fullmatch(fn):
            return os.path.join(IRF_DIR, fn)
    raise FileNotFoundError(
        f"Could not find a file matching:\n  {pattern}\nIn:\n  {IRF_DIR}"
    )

def read_irf_csv(path: str):
    """
    Reads a saved IRF CSV with columns:
      h, <var>_med, <var>_lo, <var>_hi
    Returns:
      h, ordered_vars, med_dict, lo_dict, hi_dict
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
    ordered_vars = [v for v in PREFERRED_ORDER if v in vars_found] + [v for v in vars_found if v not in PREFERRED_ORDER]

    med, lo, hi = {}, {}, {}
    for v in ordered_vars:
        med_col = f"{v}_med"
        lo_col = f"{v}_lo"
        hi_col = f"{v}_hi"

        if med_col not in df.columns or lo_col not in df.columns or hi_col not in df.columns:
            raise ValueError(f"Missing one of {med_col}, {lo_col}, {hi_col} in {path}")

        med[v] = df[med_col].to_numpy()
        lo[v] = df[lo_col].to_numpy()
        hi[v] = df[hi_col].to_numpy()

    return h, ordered_vars, med, lo, hi

# ================== PLOTTING ==================
def plot_four_shock_pre_comparison(h, vars_list, irf_dict, title, out_png):
    """
    irf_dict format:
      {
        "AD": (med, lo, hi),
        "FFR": (med, lo, hi),
        "Wu–Xia": (med, lo, hi),
        "Krippner": (med, lo, hi),
      }

    Plots median IRFs only, to match a clean comparison figure.
    """
    n = len(vars_list)
    nrows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    line_styles = {
        "AD": "-",
        "FFR": "--",
        "Wu–Xia": ":",
        "Krippner": "-."
    }

    for i, v in enumerate(vars_list):
        ax = axes[i]

        for label, (med, _, _) in irf_dict.items():
            ax.plot(
                h,
                med[v],
                linewidth=2,
                linestyle=line_styles.get(label, "-"),
                label=label
            )

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

# ================== MAIN ==================
def main():
    # ------------------------------------------------------------
    # Locate PRE-2008 CSVs
    # ------------------------------------------------------------
    ad_pre_path = find_first(r"Table_IRF_AruobaDrechsel_Pre2008_p\d+_H\d+_EBP\.csv")
    ffr_pre_path = find_first(r"Table_IRF_Hybrid_Pre2008_p\d+_H\d+_EBP\.csv")   # displayed as FFR
    wx_pre_path = find_first(r"Table_IRF_WuXia_Pre2008_p\d+_H\d+_EBP\.csv")
    kr_pre_path = find_first(r"Table_IRF_Krippner_Pre2008_p\d+_H\d+_EBP\.csv")

    # Read CSVs
    h_ad, vars_ad, med_ad, lo_ad, hi_ad = read_irf_csv(ad_pre_path)
    h_ffr, vars_ffr, med_ffr, lo_ffr, hi_ffr = read_irf_csv(ffr_pre_path)
    h_wx, vars_wx, med_wx, lo_wx, hi_wx = read_irf_csv(wx_pre_path)
    h_kr, vars_kr, med_kr, lo_kr, hi_kr = read_irf_csv(kr_pre_path)

    # Common variable set across all four
    common_vars = [
        v for v in PREFERRED_ORDER
        if (v in vars_ad and v in vars_ffr and v in vars_wx and v in vars_kr)
    ]
    if not common_vars:
        common_vars = sorted(set(vars_ad).intersection(vars_ffr).intersection(vars_wx).intersection(vars_kr))

    if not common_vars:
        raise ValueError("No common variables found across AD, FFR, Wu–Xia, and Krippner CSVs.")

    # Horizon consistency check
    if not (len(h_ad) == len(h_ffr) == len(h_wx) == len(h_kr)):
        raise ValueError("IRF horizons differ across the four CSV files.")
    if not (np.allclose(h_ad, h_ffr) and np.allclose(h_ad, h_wx) and np.allclose(h_ad, h_kr)):
        raise ValueError("IRF horizon values differ across the four CSV files.")

    h = h_ad

    # Thesis-ready figure name
    out_png = os.path.join(
        OUTDIR,
        "Figure_Pre2008_Compare_AD_FFR_WuXia_Krippner_.png"
    )

    plot_four_shock_pre_comparison(
        h=h,
        vars_list=common_vars,
        irf_dict={
            "AD": (med_ad, lo_ad, hi_ad),
            "FFR": (med_ffr, lo_ffr, hi_ffr),
            "Wu–Xia": (med_wx, lo_wx, hi_wx),
            "Krippner": (med_kr, lo_kr, hi_kr),
        },
        title="Pre-2008: Comparison of IRFs to AD, FFR, Wu–Xia, and Krippner Monetary Policy Shocks",
        out_png=out_png,
    )

    print("\n✅ Done.")
    print("Inputs:")
    print("  AD        :", ad_pre_path)
    print("  FFR       :", ffr_pre_path, " (Hybrid CSV displayed as FFR)")
    print("  Wu–Xia    :", wx_pre_path)
    print("  Krippner  :", kr_pre_path)
    print("Output:")
    print(" ", out_png)

if __name__ == "__main__":
    main()