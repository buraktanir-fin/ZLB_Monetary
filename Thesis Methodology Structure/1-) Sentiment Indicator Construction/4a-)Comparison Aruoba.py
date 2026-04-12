"""
================================================================================
sentiment_validation.py
================================================================================
Thesis:  "When Rates Hit Zero: Identifying Monetary Shocks with Shadow Rates"
Author:  Burak Tanir
Date:    April 2026

Purpose
-------
Validates the replicated concept-level sentiment scores against the original
Aruoba and Drechsel (2024) sentiment series by computing Pearson and Spearman
correlations for each of the 296 economic concepts across all matched FOMC
meeting dates.

This script is the primary quantitative evidence for the claim made in
Section 4 of the thesis that the replicated sentiment construction achieves
broad alignment with the original Aruoba-Drechsel scores (median Pearson
correlation of 0.711, pooled correlation of 0.647). The concept-level
correlation results identify which concepts replicate well (core monetary
concepts such as natural rate and inflation compensation) and which diverge
(peripheral agricultural and regional concepts carrying low ridge regression
weights).

Pipeline
--------
    Step 1 — Load replicated sentiment scores
        Reads fomc_concept_sentiments_ADstyle.csv produced by
        sentiment_computation.py. Each column is of the form:
            {concept}_score_per_10k_words_z

    Step 2 — Load original Aruoba-Drechsel sentiment data
        Reads the "Sentiments by meeting" sheet from the Aruoba-Drechsel
        Excel workbook. The date column (format YYYY_MM_DD) is auto-detected
        and converted to a datetime index.

    Step 3 — Inner merge on meeting date
        Both datasets are aligned on the Date column. Only meeting dates
        present in both datasets are retained (inner join), ensuring a
        balanced comparison sample.

    Step 4 — Per-concept correlation and error metrics
        For each concept in the 296-concept list, the following statistics
        are computed across all matched meeting dates:
            Pearson r       — linear correlation
            Spearman rho    — rank correlation (robust to outliers)
            MAE             — mean absolute error
            RMSE            — root mean squared error
            Max |diff|      — worst-case meeting-level discrepancy
            Mean diff       — signed bias (replicated minus original)

    Step 5 — Pooled correlation
        All concept-date pairs are stacked into a single vector and a
        pooled Pearson and Spearman correlation is computed across the
        full distribution of concept-meeting observations.

    Step 6 — Output
        concept_correlation_summary.csv    — full 296-concept table
        best_correlations_top20.csv        — top 20 concepts by Pearson r
        worst_correlations_bottom20.csv    — bottom 20 concepts by Pearson r
        general_correlation_summary.txt    — pooled statistics summary
        scatter_best_concept.png           — scatter plot for best concept
        scatter_worst_concept.png          — scatter plot for worst concept

Input
-----
    sentiment_results/fomc_concept_sentiments_ADstyle.csv
        Produced by sentiment_computation.py. Must contain a 'Date' column
        and columns named {concept}_score_per_10k_words_z for each concept.

    Aruoba Drechsel Data/Aruoba_Drechsel_Data.xlsx
        Original sentiment data provided by Aruoba and Drechsel (2024).
        Sheet: "Sentiments by meeting". Date column format: YYYY_MM_DD.

Output
------
    Control Results/concept_correlation_summary.csv
    Control Results/best_correlations_top20.csv
    Control Results/worst_correlations_bottom20.csv
    Control Results/general_correlation_summary.txt
    Control Results/scatter_best_concept.png
    Control Results/scatter_worst_concept.png

Key settings
------------
    MY_METRIC_SUFFIX : column suffix used to identify replicated sentiment
                       columns in the CSV (default: _score_per_10k_words_z)
    MIN_N            : minimum number of overlapping observations required
                       to compute a correlation for a concept (default: 5)
    START_DATE       : optional start date for restricting the comparison
                       sample (set to None to use all available dates)
    END_DATE         : optional end date for restricting the comparison
                       sample (set to None to use all available dates)

Dependencies
------------
    os, re  (standard library)
    numpy, pandas, matplotlib  — pip install numpy pandas matplotlib openpyxl

Usage
-----
    python sentiment_validation.py

    Run from the directory containing the sentiment_results/ and
    Aruoba Drechsel Data/ folders. The Control Results/ folder is
    created automatically.
================================================================================
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SECTION 1 — FOLDER CONFIGURATION
# ============================================================
# All folders are resolved relative to the current working directory.
# The Control Results/ folder is created automatically if absent.

BASE_DIR = os.getcwd()

# Input folders
SENTIMENT_DIR = os.path.join(BASE_DIR, "sentiment_results")        # replicated sentiments
ARUOBA_DIR    = os.path.join(BASE_DIR, "Aruoba Drechsel Data")     # original AD data

# Output folder for all validation results and figures
CONTROL_DIR   = os.path.join(BASE_DIR, "Control Results")

os.makedirs(CONTROL_DIR, exist_ok=True)


# ============================================================
# SECTION 2 — FILE PATHS
# ============================================================
# Input files
MY_SENT_PATH  = os.path.join(SENTIMENT_DIR, "fomc_concept_sentiments_ADstyle.csv")
ARUOBA_PATH   = os.path.join(ARUOBA_DIR,    "Aruoba_Drechsel_Data.xlsx")
ARUOBA_SHEET  = "Sentiments by meeting"   # sheet name in the Aruoba-Drechsel workbook

# Output files
OUT_CORR_CSV     = os.path.join(CONTROL_DIR, "concept_correlation_summary.csv")
OUT_GENERAL_TXT  = os.path.join(CONTROL_DIR, "general_correlation_summary.txt")
OUT_BEST_CSV     = os.path.join(CONTROL_DIR, "best_correlations_top20.csv")
OUT_WORST_CSV    = os.path.join(CONTROL_DIR, "worst_correlations_bottom20.csv")
OUT_SCATTER_BEST  = os.path.join(CONTROL_DIR, "scatter_best_concept.png")
OUT_SCATTER_WORST = os.path.join(CONTROL_DIR, "scatter_worst_concept.png")


# ============================================================
# SECTION 3 — KEY SETTINGS
# ============================================================

# Column suffix identifying the replicated sentiment metric to compare.
# Must match the column naming convention in fomc_concept_sentiments_ADstyle.csv.
# Format: {concept}{MY_METRIC_SUFFIX}  e.g. labor_market_score_per_10k_words_z
MY_METRIC_SUFFIX = "_score_per_10k_words_z"

# Minimum number of overlapping meeting-date observations required to
# compute a valid correlation for a concept. Concepts with fewer overlapping
# dates are skipped to avoid spurious correlations from very small samples.
MIN_N = 5

# Optional date restriction for the comparison sample.
# Set both to None to use all available overlapping meeting dates.
# Example to restrict to pre-ZLB period: START_DATE = "1982-01-01", END_DATE = "2008-10-01"
START_DATE = None
END_DATE   = None


# ============================================================
# SECTION 4 — CONCEPT LIST (296 ECONOMIC CONCEPTS)
# ============================================================
# Must match the concept list in sentiment_computation.py exactly.
# Concepts not present in both datasets are skipped automatically.

concepts = [
    "borrowing","brazil","banks","canada","credit","china","consumption","construction",
    "currencies","deposits","employment","employment_cost","equipment","euro","exports",
    "germany","hiring","hours","housing","imports","inflation","inventories","investment",
    "japan","liquidity","loans","leasing","lending","machinery","mexico","mortgage","output",
    "productivity","profits","recovery","reserves","savings","spread","structures","tourism",
    "unemployment","utilization","wages","weather","yield","aggregate_demand","auto_sales",
    "bond_issuance","budget_deficit","economic_activity","business_confidence",
    "business_spending","capital_expenditures","commodity_prices","consumer_confidence",
    "current_account","debt_growth","defense_spending","delinquency_rates",
    "developing_countries","domestic_demand","drilling_activity","durable_goods",
    "economic_growth","energy_prices","equity_issuance","equity_prices","euro_area",
    "exchange_rate","federal_debt","financial_conditions","financial_developments",
    "fiscal_policy","fiscal_stimulus","food_prices","foreign_economies","gas_prices",
    "gasoline_prices","government_purchases","home_prices","home_sales","hourly_compensation",
    "household_debt","household_spending","import_prices","income","industrial_production",
    "industrial_supplies","inflation_compensation","inflation_expectations","initial_claims",
    "input_prices","intermediate_materials","international_developments","labor_market",
    "manufacturing_activity","manufacturing_firms","monetary_aggregates","mortgage_interest",
    "natural_rate","net_exports","new_orders","nondefense_capital","oil_prices","output_gap",
    "potential_output","price_pressures","producer_prices","refinancing_activity",
    "residential_investment","retail_prices","retail_sales","retail_trade","share_prices",
    "social_security","stock_market","trade_balance","trade_deficit","trade_surplus",
    "treasury_securities","treasury_yield","vacancy_rates","wholesale_prices","wholesale_trade",
    "yield_curve","advanced_foreign_economies","commercial_real_estate",
    "compensation_per_hour","domestic_final_purchases","domestic_financial_developments",
    "emerging_market_economies","foreign_exchange","foreign_industrial_countries",
    "gross_domestic_purchases","household_net_worth","international_financial_transactions",
    "labor_force_participation","major_industrial_countries","market_interest_rates",
    "nondefense_capital_goods","output_per_hour","real_estate_activity","real_estate_market",
    "real_interest_rate","residential_real_estate","unit_labor_cost","money_market_mutual",
    "gdp","nominal_gdp","cpi","nairu","services","core_inflation","bonds","economy",
    "motor_vehicles","outlays","financing","financial_institutions","depository_institutions",
    "assets","finance","credit_standards","shipments","capacity","office","computers",
    "industries","producers","supply","homes","sectors","agriculture","merchandise","investors",
    "aircraft","stocks","buildings","cash","consumer_prices","trucks","semiconductors",
    "crude_oil","loan_demand","united_kingdom","farm","uncertainty","households","crop",
    "apparel","steel","money_market","automotive","metals","market_participants","permits",
    "commerce","commercial_paper","housing_starts","housing_activity","transportation",
    "natural_gas","consumer_goods","municipal","commodities","corporations","liabilities",
    "consumers","balance_sheet","firms","trading","financial_markets","corn",
    "economic_indicators","asia","taxes","software","mining","losses","jobs","cars",
    "depreciation","recession","france","korea","italy","lumber","volatility","wheat",
    "final_sales","credit_quality","international_transactions","livestock","rents",
    "finished_goods","petroleum","latin_america","traffic","fuel","plants","economic_outlook",
    "technology","argentina","cattle","crisis","utilities","travel","payrolls","factory",
    "transfers","drought","domestic_developments","gold","salaries","oil_imports","cotton",
    "home_equity","coal","philippines","singapore","taiwan","thailand","soybean","swaps",
    "harvest","environment","deflator","delinquencies","chemicals","mergers","rigs","indonesia",
    "political","peso","headline_inflation","retirement","raw_materials","holiday_season",
    "inflationary_pressures","tobacco","loan_officer","hurricane","health_care",
    "foreign_net_purchases","loan_rates","equities","russia","workers","economic_expansion",
    "economic_data","canadian_dollar","contractors","corporate_profits","insurance_companies",
    "wage_pressures","market_expectations",
]


# ============================================================
# SECTION 5 — HELPER FUNCTIONS
# ============================================================

def normalize_name(s: str) -> str:
    """
    Normalise a concept or column name to a canonical lowercase string
    for fuzzy matching between the replicated and original column names.

    Converts underscores to spaces, removes non-alphanumeric characters,
    and collapses multiple spaces. This allows matching across minor
    formatting differences between the two datasets (e.g. 'labor_market'
    matches 'Labor Market' in the Aruoba-Drechsel Excel column headers).

    Parameters
    ----------
    s : str
        Raw concept name or column header.

    Returns
    -------
    str
        Normalised lowercase string with single spaces and no punctuation.
    """
    s = str(s).lower()
    s = s.replace("_", " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def my_col_name(concept: str) -> str:
    """
    Construct the full column name for a concept in the replicated
    sentiment CSV by appending MY_METRIC_SUFFIX.

    Parameters
    ----------
    concept : str
        Concept key (e.g. 'labor_market').

    Returns
    -------
    str
        Full column name (e.g. 'labor_market_score_per_10k_words_z').
    """
    return f"{concept}{MY_METRIC_SUFFIX}"


def save_scatter(merged: pd.DataFrame, my_col: str, ad_col: str, title: str, out_path: str):
    """
    Generate and save a scatter plot comparing the replicated and original
    sentiment scores for a single concept across all meeting dates.

    The plot shows one point per meeting date. The Pearson correlation and
    observation count are displayed in the title. Plots are saved as 200 dpi
    PNG files to the Control Results/ folder.

    Parameters
    ----------
    merged : pd.DataFrame
        Inner-merged dataset containing both replicated and original scores.
    my_col : str
        Column name of the replicated sentiment score in `merged`.
    ad_col : str
        Column name of the original Aruoba-Drechsel score in `merged`.
    title : str
        Chart title prefix (concept name and rank are prepended).
    out_path : str
        Absolute path for the output PNG file.

    Returns
    -------
    None
        The figure is saved to disk and the plot object is closed.
    """
    x = pd.to_numeric(merged[my_col], errors="coerce")
    y = pd.to_numeric(merged[ad_col], errors="coerce")
    valid = x.notna() & y.notna()
    if int(valid.sum()) < MIN_N:
        print(f"Not enough observations to plot: {title}")
        return

    corr = x[valid].corr(y[valid])

    plt.figure(figsize=(7, 6))
    plt.scatter(x[valid], y[valid], alpha=0.6)
    plt.xlabel(f"My: {my_col}")
    plt.ylabel(f"Aruoba: {ad_col}")
    plt.title(f"{title} | Pearson={corr:.3f} (n={int(valid.sum())})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved scatter:", out_path)


# ============================================================
# SECTION 6 — STEP 1: LOAD REPLICATED SENTIMENT SCORES
# ============================================================
# Read the concept-level sentiment CSV produced by sentiment_computation.py.
# The Date column is parsed to datetime and rows with missing dates are dropped.

my_df = pd.read_csv(MY_SENT_PATH)
if "Date" not in my_df.columns:
    raise ValueError("Your sentiment CSV must contain a 'Date' column.")

my_df["Date"] = pd.to_datetime(my_df["Date"], errors="coerce")
my_df = my_df.dropna(subset=["Date"])


# ============================================================
# SECTION 7 — STEP 2: LOAD ORIGINAL ARUOBA-DRECHSEL DATA
# ============================================================
# Load the "Sentiments by meeting" sheet from the Aruoba-Drechsel workbook.
# The date column is auto-detected by searching for cells matching the
# YYYY_MM_DD format used in the original data. The detected column is
# converted to a standard datetime and replaced with a 'Date' column.

ad_df = pd.read_excel(ARUOBA_PATH, sheet_name=ARUOBA_SHEET)

# Auto-detect the date column by matching the YYYY_MM_DD pattern
date_col = None
for col in ad_df.columns:
    if ad_df[col].astype(str).str.match(r"^\d{4}_\d{2}_\d{2}$").sum() > 0:
        date_col = col
        break

if date_col is None:
    raise ValueError("Could not detect YYYY_MM_DD date column in Aruoba Excel.")

print("Aruoba date column detected:", date_col)

ad_df["Date"] = pd.to_datetime(ad_df[date_col].astype(str).str.replace("_", "-"), errors="coerce")
ad_df = ad_df.drop(columns=[date_col]).dropna(subset=["Date"])

# Build a normalised name → original column name lookup for fuzzy matching
ad_name_map = {normalize_name(c): c for c in ad_df.columns if c != "Date"}


# ============================================================
# SECTION 8 — STEP 3: INNER MERGE ON MEETING DATE
# ============================================================
# Align both datasets on the Date column using an inner join.
# Only meeting dates present in both the replicated and original datasets
# are retained. An optional date restriction can be applied via START_DATE
# and END_DATE if the comparison should be limited to a sub-period.

merged = my_df.merge(ad_df, on="Date", how="inner").sort_values("Date")

if START_DATE and END_DATE:
    merged = merged[
        (merged["Date"] >= pd.to_datetime(START_DATE)) &
        (merged["Date"] <= pd.to_datetime(END_DATE))
    ]

print("Merged observations:", len(merged))


# ============================================================
# SECTION 9 — STEP 4: PER-CONCEPT CORRELATIONS AND ERROR METRICS
# ============================================================
# For each concept, the replicated and original sentiment columns are
# paired, valid (non-NaN) observations are identified, and the following
# statistics are computed:
#   pearson              — linear correlation coefficient
#   spearman             — rank correlation coefficient
#   mae                  — mean absolute error between the two series
#   rmse                 — root mean squared error
#   max_abs_diff         — largest single-meeting discrepancy
#   mean_diff_my_minus_aruoba — signed bias of the replicated series
#
# Concepts that cannot be matched across datasets or have fewer than MIN_N
# valid observations are skipped.

rows     = []
pooled_x = []   # stacked replicated scores for pooled correlation
pooled_y = []   # stacked original scores for pooled correlation

for c in concepts:
    my_col = my_col_name(c)
    key    = normalize_name(c)

    # Skip if the concept is absent from either dataset
    if my_col not in merged.columns or key not in ad_name_map:
        continue

    ad_col = ad_name_map[key]

    x = pd.to_numeric(merged[my_col], errors="coerce")
    y = pd.to_numeric(merged[ad_col], errors="coerce")

    valid = x.notna() & y.notna()
    n     = int(valid.sum())

    # Skip concepts with insufficient overlapping observations
    if n < MIN_N:
        continue

    pearson  = x[valid].corr(y[valid], method="pearson")
    spearman = x[valid].corr(y[valid], method="spearman")

    diff = x[valid] - y[valid]

    rows.append({
        "concept":                  c,
        "aruoba_col":               ad_col,
        "my_col":                   my_col,
        "n_obs":                    n,
        "pearson":                  pearson,
        "spearman":                 spearman,
        "mae":                      float(diff.abs().mean()),
        "rmse":                     float(np.sqrt((diff ** 2).mean())),
        "max_abs_diff":             float(diff.abs().max()),
        "mean_diff_my_minus_aruoba":float(diff.mean())
    })

    # Accumulate concept-date score pairs for the pooled correlation
    pooled_x.append(x[valid].to_numpy(dtype=float))
    pooled_y.append(y[valid].to_numpy(dtype=float))

summary = pd.DataFrame(rows)

if summary.empty:
    raise ValueError(
        "No comparable concept columns were found. "
        "Check MY_METRIC_SUFFIX and whether your CSV columns match concept+suffix."
    )

# Sort by Pearson then Spearman, both descending
summary = summary.sort_values(["pearson", "spearman"], ascending=[False, False])
summary.to_csv(OUT_CORR_CSV, index=False)
print(f"\nSaved concept-level summary to:\n{OUT_CORR_CSV}")


# ============================================================
# SECTION 10 — STEP 5: BEST AND WORST CORRELATION TABLES
# ============================================================
# Extract the top 20 and bottom 20 concepts by Pearson correlation.
# These tables are saved as CSVs and printed to the console for
# immediate inspection. They form the basis of the validation table
# reported in Section 4 of the thesis.

best20  = summary.head(20).copy()
worst20 = summary.tail(20).copy()

best20.to_csv(OUT_BEST_CSV,   index=False)
worst20.to_csv(OUT_WORST_CSV, index=False)

print("\n========== BEST 20 (highest Pearson) ==========\n")
print(best20[["concept", "n_obs", "pearson", "spearman", "mae", "rmse"]].to_string(index=False))

print("\n========== WORST 20 (lowest Pearson) ==========\n")
print(worst20[["concept", "n_obs", "pearson", "spearman", "mae", "rmse"]].to_string(index=False))

print(f"\nSaved BEST 20 to:  {OUT_BEST_CSV}")
print(f"Saved WORST 20 to: {OUT_WORST_CSV}")


# ============================================================
# SECTION 11 — STEP 5 (CONT.): POOLED CORRELATION SUMMARY
# ============================================================
# All concept-date score pairs accumulated in the per-concept loop are
# stacked into flat vectors and a single pooled Pearson and Spearman
# correlation is computed. This pooled statistic summarises the overall
# alignment between the replicated and original sentiment series across
# all concepts and all meeting dates simultaneously.
#
# The pooled Pearson correlation (0.647) and the median per-concept
# Pearson correlation (0.711) are the two headline validation statistics
# reported in the thesis.

X_all = np.concatenate(pooled_x) if pooled_x else np.array([])
Y_all = np.concatenate(pooled_y) if pooled_y else np.array([])

if X_all.size > 10 and Y_all.size == X_all.size:
    pooled_pearson  = float(pd.Series(X_all).corr(pd.Series(Y_all), method="pearson"))
    pooled_spearman = float(pd.Series(X_all).corr(pd.Series(Y_all), method="spearman"))
else:
    pooled_pearson  = np.nan
    pooled_spearman = np.nan

general_lines = [
    "GENERAL (POOLED) CORRELATION SUMMARY",
    f"Merged observations (dates): {len(merged)}",
    f"Concepts compared (rows in summary): {len(summary)}",
    f"Metric compared: {MY_METRIC_SUFFIX}",
    "",
    f"Pooled Pearson (all concept-date pairs):  {pooled_pearson}",
    f"Pooled Spearman (all concept-date pairs): {pooled_spearman}",
    "",
    f"Median Pearson across concepts: {float(summary['pearson'].median())}",
    f"Mean Pearson across concepts:   {float(summary['pearson'].mean())}",
    f"Min Pearson across concepts:    {float(summary['pearson'].min())}",
    f"Max Pearson across concepts:    {float(summary['pearson'].max())}",
]

with open(OUT_GENERAL_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(general_lines))

print("\n=== GENERAL SUMMARY ===")
print("\n".join(general_lines))
print(f"\nSaved general summary to:\n{OUT_GENERAL_TXT}")


# ============================================================
# SECTION 12 — STEP 6: SCATTER PLOTS FOR BEST AND WORST CONCEPTS
# ============================================================
# Generate one scatter plot for the best-correlated concept and one for
# the worst-correlated concept. Each plot shows the replicated score
# (x-axis) against the original Aruoba-Drechsel score (y-axis) across
# all matched meeting dates, with the Pearson correlation in the title.
# These figures are saved to Control Results/ for inclusion in the thesis
# appendix or supervisory review.

best_row  = best20.iloc[0]
worst_row = worst20.iloc[0]

save_scatter(
    merged   = merged,
    my_col   = best_row["my_col"],
    ad_col   = best_row["aruoba_col"],
    title    = f"Best concept: {best_row['concept']}",
    out_path = OUT_SCATTER_BEST
)

save_scatter(
    merged   = merged,
    my_col   = worst_row["my_col"],
    ad_col   = worst_row["aruoba_col"],
    title    = f"Worst concept: {worst_row['concept']}",
    out_path = OUT_SCATTER_WORST
)