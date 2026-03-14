import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# THESIS FIGURE FROM SAVED IRF CSVs (NO RE-ESTIMATION)
#
# Figure created:
#   POST-2008: 2-shock comparison
#       - Wu–Xia
#       - Krippner
#
# Reads existing CSV outputs such as:
#   Table_IRF_WuXia_Post2008_p12_H48_EBP.csv
#   Table_IRF_Krippner_Post2008_p12_H48_EBP.csv
# ============================================================

# ================== PATHS ==================
BASE_DIR = os.getcwd()
IRF_DIR  = os.path.join(BASE_DIR, "irf_outputs_EBP_final")
OUTDIR   = os.path.join(IRF_DIR, "thesis_figures_from_csv")
os.makedirs(OUTDIR, exist_ok=True)

# ================== DISPLAY LABELS ==================
DISPLAY_LABELS = {
    "y1":        "1-year bond yield (%)",
    "log_sp500": "S&P500 (100 × log)",
    "log_rgdp":  "Real GDP (100 × log)",
    "unemp":     "Unemployment (%)",
    "log_pgdp":  "GDP deflator (100 × log)",
    "EBP":       "EBP (%)",
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
        raise ValueError(
            f"'h' column missing in {path}. Columns: {list(df.columns)}"
        )

    h = df["h"].to_numpy()

    vars_found = []
    for c in df.columns:
        if c.endswith("_med"):
            vars_found.append(c[:-4])

    vars_found   = sorted(set(vars_found))
    ordered_vars = (
        [v for v in PREFERRED_ORDER if v in vars_found] +
        [v for v in vars_found if v not in PREFERRED_ORDER]
    )

    med, lo, hi = {}, {}, {}
    for v in ordered_vars:
        med_col = f"{v}_med"
        lo_col  = f"{v}_lo"
        hi_col  = f"{v}_hi"

        if med_col not in df.columns or lo_col not in df.columns or hi_col not in df.columns:
            raise ValueError(
                f"Missing one of {med_col}, {lo_col}, {hi_col} in {path}"
            )

        med[v] = df[med_col].to_numpy()
        lo[v]  = df[lo_col].to_numpy()
        hi[v]  = df[hi_col].to_numpy()

    return h, ordered_vars, med, lo, hi

# ================== PLOTTING ==================
def plot_two_shock_post_comparison(h, vars_list, irf_dict, title, out_png):
    """
    irf_dict format:
      {
        "Wu–Xia":   (med, lo, hi),
        "Krippner": (med, lo, hi),
      }

    Plots median IRFs with 68% credible bands for both instruments.
    """
    styles = {
        "Wu–Xia":   dict(color="#2ca02c", ls=":",  lw=2.2),
        "Krippner": dict(color="#d62728", ls="-.", lw=2.2),
    }

    n     = len(vars_list)
    nrows = int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrows, 2,
                             figsize=(12, 4 * nrows),
                             sharex=True)
    axes = np.array(axes).reshape(-1)

    for i, v in enumerate(vars_list):
        ax = axes[i]

        for label, (med, lo, hi) in irf_dict.items():
            st = styles[label]
            ax.plot(h, med[v], label=label,
                    color=st["color"], ls=st["ls"], lw=st["lw"])
            ax.fill_between(h, lo[v], hi[v],
                            color=st["color"], alpha=0.15)

        ax.axhline(0, color="steelblue", linewidth=0.9)
        ax.set_ylabel(pretty_ylabel(v))
        ax.set_title(v, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    for k in range(n, len(axes)):
        axes[k].axis("off")

    fig.suptitle(title, y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved: {out_png}")

# ================== MAIN ==================
def main():
    # ── Locate post-2008 CSVs ────────────────────────────────
    wx_post_path = find_first(
        r"Table_IRF_WuXia_Post2008_p\d+_H\d+_EBP\.csv"
    )
    kr_post_path = find_first(
        r"Table_IRF_Krippner_Post2008_p\d+_H\d+_EBP\.csv"
    )

    # ── Read CSVs ────────────────────────────────────────────
    h_wx, vars_wx, med_wx, lo_wx, hi_wx = read_irf_csv(wx_post_path)
    h_kr, vars_kr, med_kr, lo_kr, hi_kr = read_irf_csv(kr_post_path)

    # ── Common variable set ──────────────────────────────────
    common_vars = [
        v for v in PREFERRED_ORDER
        if v in vars_wx and v in vars_kr
    ]
    if not common_vars:
        common_vars = sorted(
            set(vars_wx).intersection(vars_kr)
        )
    if not common_vars:
        raise ValueError(
            "No common variables found across Wu–Xia and Krippner CSVs."
        )

    # ── Horizon consistency check ────────────────────────────
    if len(h_wx) != len(h_kr):
        raise ValueError("IRF horizons differ between Wu–Xia and Krippner CSVs.")
    if not np.allclose(h_wx, h_kr):
        raise ValueError("IRF horizon values differ between Wu–Xia and Krippner CSVs.")

    h = h_wx

    # ── Output path ──────────────────────────────────────────
    out_png = os.path.join(
        OUTDIR,
        "Figure_Post2008_Compare_WuXia_Krippner.png"
    )

    # ── Plot ─────────────────────────────────────────────────
    plot_two_shock_post_comparison(
        h        = h,
        vars_list= common_vars,
        irf_dict = {
            "Wu–Xia":   (med_wx, lo_wx, hi_wx),
            "Krippner": (med_kr, lo_kr, hi_kr),
        },
        title    = (
            "Post-2008: Comparison of IRFs to Wu–Xia and Krippner "
            "Shadow Rate Monetary Policy Shocks\n"
            "ZLB Period | BVAR with EBP | Median IRFs with 68% credible intervals"
        ),
        out_png  = out_png,
    )

    print("\n✅ Done.")
    print("Inputs:")
    print("  Wu–Xia   :", wx_post_path)
    print("  Krippner :", kr_post_path)
    print("Output:")
    print(" ", out_png)

if __name__ == "__main__":
    main()