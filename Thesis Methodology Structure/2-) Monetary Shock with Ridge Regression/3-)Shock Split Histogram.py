"""
Shock Distribution Histograms
==============================
Reads the existing output file:
    shock_series_pre_post_2008.csv
    (columns: Date, Hybrid, AD, Wu, Kr, period)

Produces two panels:
  Left  — Pre-2008 FFR shock (Hybrid column, period == PRE)
  Right — Post-2008 Wu-Xia shock (Wu column, period == POST)

No ridge regression or model fitting is performed.
All computation is descriptive (mean, std, skewness, kurtosis).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from scipy import stats

# ── PATHS ────────────────────────────────────────────────────────────────
# Adjust BASE_DIR to point at your "Shadow Rate Results/Pre-Post 2008/" folder
BASE_DIR   = os.path.join(os.getcwd(),
                          "Shock Results",
                          "Shadow Rate Results",
                          "Pre-Post 2008")
SHOCKS_CSV = os.path.join(BASE_DIR, "shock_series_pre_post_2008.csv")
OUT_FIG    = os.path.join(BASE_DIR, "shock_distributions_pre_post.png")

# ── STYLE ────────────────────────────────────────────────────────────────
rcParams["font.family"]       = "serif"
rcParams["font.serif"]        = ["Times New Roman", "Times", "DejaVu Serif"]
rcParams["mathtext.fontset"]  = "stix"
rcParams["axes.spines.top"]   = False
rcParams["axes.spines.right"] = False

# Colours consistent with BVAR/LP figures
COL_PRE  = "#1f77b4"   # blue  — FFR / Hybrid (pre-2008)
COL_POST = "#2ca02c"   # green — Wu-Xia (post-2008)
COL_KDE  = "black"
ALPHA_HIST = 0.55

# ── LOAD ─────────────────────────────────────────────────────────────────
if not os.path.exists(SHOCKS_CSV):
    raise FileNotFoundError(
        f"Shock CSV not found:\n  {SHOCKS_CSV}\n"
        "Run the pre/post 2008 comparison script first."
    )

df = pd.read_csv(SHOCKS_CSV, parse_dates=["Date"])

# "period" is not saved to CSV — assigned after save in original script.
# Reconstruct from Date using the identical cutoff.
CUTOFF_DATE = pd.to_datetime("2008-10-29")
df["period"] = df["Date"].apply(lambda d: "PRE" if d <= CUTOFF_DATE else "POST")
pre  = df[df["period"] == "PRE"]["Hybrid"].dropna().astype(float)
post = df[df["period"] == "POST"]["Wu"].dropna().astype(float)


# ── DESCRIPTIVE STATS ────────────────────────────────────────────────────
def desc(s):
    return {
        "N":        len(s),
        "Mean":     s.mean(),
        "Std":      s.std(ddof=1),
        "Skewness": float(stats.skew(s)),
        "Kurtosis": float(stats.kurtosis(s)),   # excess kurtosis
        "Min":      s.min(),
        "Max":      s.max(),
    }

pre_stats  = desc(pre)
post_stats = desc(post)

# Jarque-Bera normality test
jb_pre,  p_pre  = stats.jarque_bera(pre)
jb_post, p_post = stats.jarque_bera(post)

# ── FIGURE ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 5.5))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

def make_panel(ax, series, color, title, period_label, stats_d, jb, p_jb):

    # --- histogram ---
    n_bins = min(30, max(15, int(np.sqrt(len(series)) * 1.8)))
    counts, edges, patches = ax.hist(
        series, bins=n_bins,
        color=color, alpha=ALPHA_HIST,
        edgecolor="white", linewidth=0.4,
        density=True, label="Empirical distribution"
    )

    # --- KDE overlay ---
    kde   = stats.gaussian_kde(series, bw_method="scott")
    x_kde = np.linspace(series.min() - 0.5 * series.std(),
                        series.max() + 0.5 * series.std(), 400)
    ax.plot(x_kde, kde(x_kde),
            color=COL_KDE, lw=1.6, ls="-", label="KDE", zorder=5)

    # --- normal reference ---
    mu, sd = series.mean(), series.std(ddof=1)
    x_norm = np.linspace(series.min() - 0.5 * sd,
                         series.max() + 0.5 * sd, 400)
    ax.plot(x_norm, stats.norm.pdf(x_norm, mu, sd),
            color=color, lw=1.4, ls="--", alpha=0.85,
            label=r"$\mathcal{N}(\hat{\mu},\,\hat{\sigma}^2)$", zorder=4)

    # --- zero line ---
    ax.axvline(0, color="grey", lw=0.8, ls=":", alpha=0.7, zorder=3)

    # --- mean line ---
    ax.axvline(mu, color=color, lw=1.0, ls="-.", alpha=0.9, zorder=3,
               label=f"Mean = {mu:.4f}")

    # --- stats box ---
    jb_str = f"JB = {jb:.2f}  (p = {p_jb:.3f})"
    box_text = (
        f"$N$ = {stats_d['N']}\n"
        f"Mean = {stats_d['Mean']:.4f}\n"
        f"Std = {stats_d['Std']:.4f}\n"
        f"Skewness = {stats_d['Skewness']:.3f}\n"
        f"Ex. Kurtosis = {stats_d['Kurtosis']:.3f}\n"
        f"{jb_str}"
    )
    ax.text(0.97, 0.97, box_text,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            horizontalalignment="right",
            family="serif",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="#cccccc",
                      alpha=0.88))

    # --- labels ---
    ax.set_title(title, fontsize=11.5, pad=9, fontfamily="serif")
    ax.set_xlabel("Shock size (std. units)", fontsize=9.5, fontfamily="serif")
    ax.set_ylabel("Density", fontsize=9.5, fontfamily="serif")
    ax.tick_params(labelsize=8.5)
    ax.legend(fontsize=8, framealpha=0.7, loc="upper left")

    return ax.get_ylim()[1]

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

ymax1 = make_panel(
    ax1, pre, COL_PRE,
    title="Pre-2008 FFR Shock\n(Hybrid / Federal Funds Rate)",
    period_label="Oct 1982 – Oct 2008",
    stats_d=pre_stats, jb=jb_pre, p_jb=p_pre
)

ymax2 = make_panel(
    ax2, post, COL_POST,
    title="Post-2008 Wu-Xia Shadow Rate Shock\n(ZLB Period)",
    period_label="Nov 2008 – Dec 2019",
    stats_d=post_stats, jb=jb_post, p_jb=p_post
)

# ── Shared y-axis: same density scale for direct comparison ──────────────
shared_ymax = max(ymax1, ymax2) * 1.08
ax1.set_ylim(0, shared_ymax)
ax2.set_ylim(0, shared_ymax)
ax2.set_ylabel("")  # remove duplicate y-label on right panel

# --- supertitle ---
fig.suptitle(
    "Distribution of Identified Monetary Policy Shocks\n"
    "Pre-2008 FFR (Oct 1982–Oct 2008)  and  Post-2008 Wu-Xia (Nov 2008–Dec 2019)",
    fontsize=10.5, y=1.01, fontfamily="serif", linespacing=1.5
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
plt.close()

print(f"\n✅  Figure saved → {OUT_FIG}")
print("\n=== Pre-2008 FFR Shock ===")
for k, v in pre_stats.items():
    print(f"  {k:<14}: {v:.4f}" if isinstance(v, float) else f"  {k:<14}: {v}")
print(f"  {'JB stat':<14}: {jb_pre:.4f}  (p = {p_pre:.4f})")

print("\n=== Post-2008 Wu-Xia Shock ===")
for k, v in post_stats.items():
    print(f"  {k:<14}: {v:.4f}" if isinstance(v, float) else f"  {k:<14}: {v}")
print(f"  {'JB stat':<14}: {jb_post:.4f}  (p = {p_post:.4f})")