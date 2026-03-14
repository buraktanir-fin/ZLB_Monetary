import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams

# ============================================================
# THESIS LP FIGURES FROM SAVED CSV OUTPUTS
#
# Figure 1 (Pre-2008):
#   Four-shock comparison — AD, FFR, Wu-Xia, Krippner
#   with ±1 SE credible bands
#
# Figure 2 (Post-2008):
#   Wu-Xia vs Krippner comparison
#   with ±1 SE credible bands
#
# Reads from:
#   lp_outputs_EBP_linear/lp_AD_PRE.csv
#   lp_outputs_EBP_linear/lp_HYBRID_PRE.csv
#   lp_outputs_EBP_linear/lp_WX_PRE.csv       (unused in Fig 2)
#   lp_outputs_EBP_linear/lp_KR_PRE.csv       (unused in Fig 2)
#   lp_outputs_EBP_linear/lp_WX_POST.csv
#   lp_outputs_EBP_linear/lp_KR_POST.csv
# ============================================================

# ─── PATHS ───────────────────────────────────────────────────
BASE_DIR   = os.getcwd()
LP_DIR     = os.path.join(BASE_DIR, "lp_outputs_EBP_linear")
OUTDIR     = os.path.join(LP_DIR, "thesis_figures")
os.makedirs(OUTDIR, exist_ok=True)

# ─── VARIABLE ORDER & LABELS (match BVAR thesis figures) ─────
VAR_ORDER = ["y1", "log_sp500", "log_rgdp", "log_pgdp", "unemp", "EBP"]

PANEL_LABELS = {
    "y1":        "1-Year Bond Yield (%)",
    "log_sp500": "S&P 500 (log)",
    "log_rgdp":  "Real GDP (log)",
    "unemp":     "Unemployment Rate (%)",
    "log_pgdp":  "GDP Deflator (log)",
    "EBP":       "Excess Bond Premium (%)",
}

# ─── TYPOGRAPHY & STYLE ──────────────────────────────────────
rcParams["font.family"]      = "serif"
rcParams["font.serif"]       = ["Times New Roman", "Times", "DejaVu Serif"]
rcParams["mathtext.fontset"] = "stix"
rcParams["axes.spines.top"]  = False
rcParams["axes.spines.right"]= False

# ─── COLOUR / LINE PALETTE ───────────────────────────────────
# Colours match the BVAR sample-controlled comparison figures exactly
# Figure 1 — four shocks
STYLES_PRE = {
    "AD":      dict(color="#1f77b4", ls="-",  lw=2.0, zorder=4),  # matplotlib blue
    "FFR":     dict(color="#ff7f0e", ls="--", lw=2.0, zorder=3),  # matplotlib orange
    "Wu-Xia":  dict(color="#2ca02c", ls=":",  lw=2.2, zorder=2),  # matplotlib green
    "Krippner":dict(color="#d62728", ls="-.", lw=2.0, zorder=1),  # matplotlib red
}
FILL_ALPHA_PRE = 0.12

# Figure 2 — two shocks post
STYLES_POST = {
    "Wu-Xia":  dict(color="#2ca02c", ls=":",  lw=2.2, zorder=3),  # matplotlib green
    "Krippner":dict(color="#d62728", ls="-.", lw=2.0, zorder=2),  # matplotlib red
}
FILL_ALPHA_POST = 0.12

# ─── HELPERS ─────────────────────────────────────────────────
def read_lp_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"LP CSV not found:\n  {path}")
    return pd.read_csv(path)

def get_arrays(df: pd.DataFrame, var: str):
    """Return (h, b, lo, hi) for a variable from a linear LP CSV."""
    h  = df["h"].to_numpy()
    b  = df[f"{var}_b"].to_numpy()
    lo = df[f"{var}_lo"].to_numpy()
    hi = df[f"{var}_hi"].to_numpy()
    return h, b, lo, hi

def find_csv(directory: str, pattern: str) -> str:
    """Find first file matching regex pattern in directory."""
    rx = re.compile(pattern, re.IGNORECASE)
    for fn in sorted(os.listdir(directory)):
        if rx.fullmatch(fn):
            return os.path.join(directory, fn)
    raise FileNotFoundError(
        f"No file matching '{pattern}' in:\n  {directory}"
    )

def add_panel_label(ax, letter, x=-0.12, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold",
            va="top", ha="left", fontfamily="serif")

def format_ax(ax, ylabel, show_xlabel=False):
    ax.axhline(0, color="#999999", lw=0.8, ls="-", zorder=0)
    ax.set_ylabel(ylabel, fontsize=8.5, labelpad=4)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4g"))
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    if show_xlabel:
        ax.set_xlabel("Months after shock", fontsize=8.5)

# ─── FIGURE 1: PRE-2008 FOUR-SHOCK ───────────────────────────
def make_figure_pre(out_png: str):
    # load CSVs
    df_ad  = read_lp_csv(find_csv(LP_DIR, r"lp_AD_PRE\.csv"))
    df_ffr = read_lp_csv(find_csv(LP_DIR, r"lp_HYBRID_PRE\.csv"))
    df_wx  = read_lp_csv(find_csv(LP_DIR, r"lp_WX_PRE\.csv"))
    df_kr  = read_lp_csv(find_csv(LP_DIR, r"lp_KR_PRE\.csv"))

    data = {
        "AD":       df_ad,
        "FFR":      df_ffr,
        "Wu-Xia":   df_wx,
        "Krippner": df_kr,
    }

    # filter to vars present in all CSVs
    available = [v for v in VAR_ORDER
                 if all(f"{v}_b" in df.columns for df in data.values())]

    n     = len(available)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.5 * ncols, 3.2 * nrows),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.array(axes).reshape(-1)

    panel_letters = "ABCDEF"

    for i, var in enumerate(available):
        ax = axes[i]
        show_x = (i >= n - ncols)

        for label, df in data.items():
            h, b, lo, hi = get_arrays(df, var)
            st = STYLES_PRE[label]
            ax.plot(h, b, label=label,
                    color=st["color"], ls=st["ls"],
                    lw=st["lw"], zorder=st["zorder"])
            ax.fill_between(h, lo, hi,
                            color=st["color"],
                            alpha=FILL_ALPHA_PRE,
                            zorder=st["zorder"] - 1)

        format_ax(ax, PANEL_LABELS.get(var, var), show_xlabel=show_x)
        add_panel_label(ax, panel_letters[i])

        if i == 0:
            ax.legend(
                fontsize=8, framealpha=0.9,
                edgecolor="#cccccc", loc="upper right",
                handlelength=2.2, labelspacing=0.3,
            )

    # turn off unused panels
    for k in range(n, len(axes)):
        axes[k].axis("off")

    fig.suptitle(
        "Figure LP-1  |  Pre-2008 Local Projection Impulse Responses\n"
        "AD, FFR, Wu-Xia and Krippner Monetary Policy Shocks  "
        r"$\pm$1 S.E. bands  |  Newey-West HAC",
        fontsize=10, y=1.01, ha="center",
    )

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved Figure 1 → {out_png}")

# ─── FIGURE 2: POST-2008 WX vs KR ────────────────────────────
def make_figure_post(out_png: str):
    df_wx = read_lp_csv(find_csv(LP_DIR, r"lp_WX_POST\.csv"))
    df_kr = read_lp_csv(find_csv(LP_DIR, r"lp_KR_POST\.csv"))

    data = {
        "Wu-Xia":   df_wx,
        "Krippner": df_kr,
    }

    available = [v for v in VAR_ORDER
                 if all(f"{v}_b" in df.columns for df in data.values())]

    n     = len(available)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.5 * ncols, 3.2 * nrows),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.array(axes).reshape(-1)

    panel_letters = "ABCDEF"

    for i, var in enumerate(available):
        ax = axes[i]
        show_x = (i >= n - ncols)

        for label, df in data.items():
            h, b, lo, hi = get_arrays(df, var)
            st = STYLES_POST[label]
            ax.plot(h, b, label=label,
                    color=st["color"], ls=st["ls"],
                    lw=st["lw"], zorder=st["zorder"])
            ax.fill_between(h, lo, hi,
                            color=st["color"],
                            alpha=FILL_ALPHA_POST,
                            zorder=st["zorder"] - 1)

        format_ax(ax, PANEL_LABELS.get(var, var), show_xlabel=show_x)
        add_panel_label(ax, panel_letters[i])

        if i == 0:
            ax.legend(
                fontsize=8, framealpha=0.9,
                edgecolor="#cccccc", loc="upper right",
                handlelength=2.2, labelspacing=0.3,
            )

    for k in range(n, len(axes)):
        axes[k].axis("off")

    fig.suptitle(
        "Figure LP-2  |  Post-2008 Local Projection Impulse Responses\n"
        "Wu-Xia and Krippner Shadow Rate Shocks  "
        r"$\pm$1 S.E. bands  |  Newey-West HAC  |  ZLB Period",
        fontsize=10, y=1.01, ha="center",
    )

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved Figure 2 → {out_png}")

# ─── MAIN ─────────────────────────────────────────────────────
def main():
    out1 = os.path.join(OUTDIR, "LP_Figure1_Pre2008_FourShock.png")
    out2 = os.path.join(OUTDIR, "LP_Figure2_Post2008_WX_vs_KR.png")

    print("Generating Figure 1 — Pre-2008 four-shock comparison...")
    make_figure_pre(out1)

    print("Generating Figure 2 — Post-2008 Wu-Xia vs Krippner...")
    make_figure_post(out2)

    print("\n✅ Done. Figures saved in:")
    print(f"   {OUTDIR}")

if __name__ == "__main__":
    main()