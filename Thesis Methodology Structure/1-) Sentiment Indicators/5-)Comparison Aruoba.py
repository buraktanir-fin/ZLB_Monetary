import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# ARUOBA vs YOUR SENTIMENTS — STANDALONE COMPARISON (ENHANCED)
# Adds:
#  - General (overall) pooled correlation results
#  - Best/worst correlation tables (printed + saved)
#  - One correlation graph (scatter) for:
#       (a) best-correlated concept and
#       (b) worst-correlated concept
#     (saved as PNG in Control Results)
# ============================================================

# -------------------- FOLDERS --------------------
BASE_DIR = os.getcwd()

SENTIMENT_DIR = os.path.join(BASE_DIR, "sentiment_results")
ARUOBA_DIR = os.path.join(BASE_DIR, "Aruoba Drechsel Data")
CONTROL_DIR = os.path.join(BASE_DIR, "Control Results")

os.makedirs(CONTROL_DIR, exist_ok=True)

# -------------------- FILE PATHS --------------------
MY_SENT_PATH = os.path.join(SENTIMENT_DIR, "fomc_concept_sentiments_ADstyle.csv")
ARUOBA_PATH = os.path.join(ARUOBA_DIR, "Aruoba_Drechsel_Data.xlsx")
ARUOBA_SHEET = "Sentiments by meeting"

OUT_CORR_CSV = os.path.join(CONTROL_DIR, "concept_correlation_summary.csv")
OUT_GENERAL_TXT = os.path.join(CONTROL_DIR, "general_correlation_summary.txt")
OUT_BEST_CSV = os.path.join(CONTROL_DIR, "best_correlations_top20.csv")
OUT_WORST_CSV = os.path.join(CONTROL_DIR, "worst_correlations_bottom20.csv")

OUT_SCATTER_BEST = os.path.join(CONTROL_DIR, "scatter_best_concept.png")
OUT_SCATTER_WORST = os.path.join(CONTROL_DIR, "scatter_worst_concept.png")

# Metric to compare (must match your columns)
MY_METRIC_SUFFIX = "_score_per_10k_words_z"

# Minimum observations required to compute correlations for a concept
MIN_N = 5

# Optional date restriction (disabled)
START_DATE = None
END_DATE = None


# -------------------- CONCEPT LIST --------------------
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


# -------------------- HELPERS --------------------
def normalize_name(s: str) -> str:
    s = str(s).lower()
    s = s.replace("_", " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def my_col_name(concept: str) -> str:
    return f"{concept}{MY_METRIC_SUFFIX}"

def save_scatter(merged: pd.DataFrame, my_col: str, ad_col: str, title: str, out_path: str):
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
    print("✅ Saved scatter:", out_path)


# -------------------- LOAD YOUR SENTIMENTS --------------------
my_df = pd.read_csv(MY_SENT_PATH)
if "Date" not in my_df.columns:
    raise ValueError("Your sentiment CSV must contain a 'Date' column.")

my_df["Date"] = pd.to_datetime(my_df["Date"], errors="coerce")
my_df = my_df.dropna(subset=["Date"])

# -------------------- LOAD ARUOBA DATA --------------------
ad_df = pd.read_excel(ARUOBA_PATH, sheet_name=ARUOBA_SHEET)

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

ad_name_map = {normalize_name(c): c for c in ad_df.columns if c != "Date"}

# -------------------- MERGE --------------------
merged = my_df.merge(ad_df, on="Date", how="inner").sort_values("Date")

if START_DATE and END_DATE:
    merged = merged[
        (merged["Date"] >= pd.to_datetime(START_DATE)) &
        (merged["Date"] <= pd.to_datetime(END_DATE))
    ]

print("Merged observations:", len(merged))

# -------------------- PER-CONCEPT CORRELATIONS + METRICS --------------------
rows = []
pooled_x = []
pooled_y = []

for c in concepts:
    my_col = my_col_name(c)
    key = normalize_name(c)

    if my_col not in merged.columns or key not in ad_name_map:
        continue

    ad_col = ad_name_map[key]

    x = pd.to_numeric(merged[my_col], errors="coerce")
    y = pd.to_numeric(merged[ad_col], errors="coerce")

    valid = x.notna() & y.notna()
    n = int(valid.sum())

    if n < MIN_N:
        continue

    pearson = x[valid].corr(y[valid], method="pearson")
    spearman = x[valid].corr(y[valid], method="spearman")

    diff = x[valid] - y[valid]

    rows.append({
        "concept": c,
        "aruoba_col": ad_col,
        "my_col": my_col,
        "n_obs": n,
        "pearson": pearson,
        "spearman": spearman,
        "mae": float(diff.abs().mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "max_abs_diff": float(diff.abs().max()),
        "mean_diff_my_minus_aruoba": float(diff.mean())
    })

    # pooled (overall) correlation vectors
    pooled_x.append(x[valid].to_numpy(dtype=float))
    pooled_y.append(y[valid].to_numpy(dtype=float))

summary = pd.DataFrame(rows)

if summary.empty:
    raise ValueError(
        "No comparable concept columns were found. "
        "Check MY_METRIC_SUFFIX and whether your CSV columns match concept+suffix."
    )

summary = summary.sort_values(["pearson", "spearman"], ascending=[False, False])
summary.to_csv(OUT_CORR_CSV, index=False)
print(f"\n✅ Saved concept-level summary to:\n{OUT_CORR_CSV}")

# -------------------- BEST / WORST TABLES --------------------
best20 = summary.head(20).copy()
worst20 = summary.tail(20).copy()

best20.to_csv(OUT_BEST_CSV, index=False)
worst20.to_csv(OUT_WORST_CSV, index=False)

print("\n========== BEST 20 (highest Pearson) ==========\n")
print(best20[["concept", "n_obs", "pearson", "spearman", "mae", "rmse"]].to_string(index=False))

print("\n========== WORST 20 (lowest Pearson) ==========\n")
print(worst20[["concept", "n_obs", "pearson", "spearman", "mae", "rmse"]].to_string(index=False))

print(f"\n✅ Saved BEST 20 to:  {OUT_BEST_CSV}")
print(f"✅ Saved WORST 20 to: {OUT_WORST_CSV}")

# -------------------- GENERAL (POOLED) CORRELATION --------------------
# Pooled over all concept-date pairs (stacked vectors)
X_all = np.concatenate(pooled_x) if pooled_x else np.array([])
Y_all = np.concatenate(pooled_y) if pooled_y else np.array([])

if X_all.size > 10 and Y_all.size == X_all.size:
    pooled_pearson = float(pd.Series(X_all).corr(pd.Series(Y_all), method="pearson"))
    pooled_spearman = float(pd.Series(X_all).corr(pd.Series(Y_all), method="spearman"))
else:
    pooled_pearson = np.nan
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
print(f"\n✅ Saved general summary to:\n{OUT_GENERAL_TXT}")

# -------------------- ONE CORRELATION GRAPH (BEST + WORST) --------------------
best_row = best20.iloc[0]
worst_row = worst20.iloc[0]

save_scatter(
    merged=merged,
    my_col=best_row["my_col"],
    ad_col=best_row["aruoba_col"],
    title=f"Best concept: {best_row['concept']}",
    out_path=OUT_SCATTER_BEST
)

save_scatter(
    merged=merged,
    my_col=worst_row["my_col"],
    ad_col=worst_row["aruoba_col"],
    title=f"Worst concept: {worst_row['concept']}",
    out_path=OUT_SCATTER_WORST
)
