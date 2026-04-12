"""
================================================================================
Sentiment Results.py
================================================================================
Thesis:  "When Rates Hit Zero: Identifying Monetary Shocks with Shadow Rates"
Author:  Burak Tanir
Date:    April 2026

Purpose
-------
Constructs concept-level sentiment indicators from FOMC meeting documents
following the Aruoba and Drechsel (2024) methodology. The resulting sentiment
time series serve as inputs to the ridge regression shock identification
procedure described in Section 4 of the thesis.

The script operates in two sequential parts:

    Part 1 — LM Dictionary Construction
        Loads the Loughran-McDonald (2011) master sentiment dictionary,
        retains only words classified as positive or negative, removes
        economically ambiguous terms from the base dictionary, and adds
        Federal Reserve-specific terminology. The resulting customised
        dictionary is saved as a CSV for use in Part 2.

    Part 2 — Concept Sentiment Computation
        For each FOMC meeting text file (produced by fomc_text_merger.py),
        and for each of the 296 economic concepts in the concept list:
            1. Locates all occurrences of the concept phrase in the tokenised text
            2. Extracts a ±10 word window around each occurrence
            3. Scores the window using the customised LM dictionary
               (+1 per positive word, -1 per negative word)
            4. Normalises the raw score per 10,000 words to remove
               document-length effects
            5. Computes the cross-meeting z-score for each concept
               (Aruoba-Drechsel style standardisation)

Methodology notes
-----------------
    Tokenisation  : lowercase, remove numbers, filter stopwords and
                    single-character tokens (AD Step 1)
    Window size   : ±10 words around each concept occurrence (AD default)
    Concepts      : 296 uni-, bi-, and tri-gram economic concepts covering
                    macroeconomic aggregates, financial conditions, sectors,
                    international economies, and commodity markets
    Standardisation: z-score per concept across all meeting dates (ddof=0)

Input
-----
    input/
        Loughran-McDonald master dictionary CSV (auto-detected by filename
        keywords: Loughran, McDonald, MasterDictionary). Download from:
        https://sraf.nd.edu/loughranmcdonald-master-dictionary/

    fomc_merged_AD/
        One merged text file per FOMC meeting (YYYY-MM-DD.txt), produced
        by fomc_text_merger.py. Contains Greenbook and Tealbook A content.

Output
------
    sentiment_results/lm_dictionary_for_concepts.csv
        Customised LM dictionary: columns [word, sentiment].
        Sentiment values: 'pos' or 'neg'.

    sentiment_results/fomc_concept_sentiments_ADstyle.csv
        Panel dataset: one row per FOMC meeting, columns per concept:
            {concept}_score              — raw window sentiment score
            {concept}_hits               — number of sentiment word matches
            {concept}_score_per_10k_words — length-normalised score
            {concept}_score_per_10k_words_z — z-score standardised score

LM dictionary modifications
----------------------------
    Removed (economically ambiguous in FOMC context):
        unemployment, unemployed, employment, inflation, deflation,
        recession, depression, productivity, consumption, investment,
        unforeseen

    Added (Federal Reserve-specific terminology):
        tightening, tighten, tightened → neg
        easing, ease, eased           → pos
        accommodative                 → pos
        restrictive                   → neg
        headwinds                     → neg
        tailwinds                     → pos
        sluggish                      → neg
        resilient                     → pos
        buoyant                       → pos
        deterioration, deteriorated   → neg
        booms, boomed                 → pos

Dependencies
------------
    os, re  (standard library)
    nltk         — pip install nltk
    pandas       — pip install pandas
    tqdm         — pip install tqdm

Usage
-----

    Place the LM master dictionary CSV in the input/ folder before running.
    The sentiment_results/ folder is created automatically.
================================================================================
"""

import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from tqdm import tqdm


# ============================================================
# SECTION 1 — OUTPUT FOLDER SETUP
# ============================================================
# All output CSVs are saved to sentiment_results/ to keep results
# separate from raw inputs and intermediate files.

RESULTS_DIR = "sentiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# SECTION 2 — LM DICTIONARY AUTO-DETECTION
# ============================================================
# The Loughran-McDonald master dictionary CSV is auto-detected from the
# input/ folder by scoring each filename against known keywords.
# This avoids hardcoding the exact filename, which varies across
# annual releases of the dictionary (e.g. 2018, 2022, 2024 versions).

LM_INPUT_FOLDER = "input"   # place LM master dictionary CSV in this folder


def pick_lm_master_csv(folder: str) -> str:
    """
    Auto-detect the Loughran-McDonald master dictionary CSV in `folder`.

    Files are ranked by how closely their names match known LM dictionary
    naming conventions. The highest-scoring file is selected automatically.

    Scoring rules (cumulative):
        +6  filename contains 'masterdictionary'
        +5  filename contains 'loughran'
        +5  filename contains 'mcdonald'
        +2  filename contains 'dictionary'
        +1  filename contains a known release year ('1993', '2024', etc.)

    Parameters
    ----------
    folder : str
        Path to the folder containing the LM dictionary CSV.

    Returns
    -------
    str
        Absolute path to the best-matching CSV file.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist or contains no CSV files.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"LM input folder not found: {folder}")

    csv_files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in folder: {folder}")

    def score(fname: str) -> int:
        n = fname.lower()
        s = 0
        if "loughran" in n:          s += 5
        if "mcdonald" in n:          s += 5
        if "masterdictionary" in n:  s += 6
        if "dictionary" in n:        s += 2
        if "1993" in n:              s += 1
        if "2024" in n:              s += 1
        return s

    csv_files = sorted(csv_files, key=score, reverse=True)
    return os.path.join(folder, csv_files[0])


# ============================================================
# SECTION 3 — PART 1: LM DICTIONARY CONSTRUCTION
# ============================================================

# ── File paths ────────────────────────────────────────────────────────────────
LM_PATH  = pick_lm_master_csv(LM_INPUT_FOLDER)          # dynamic LM path
OUT_PATH = os.path.join(RESULTS_DIR, "lm_dictionary_for_concepts.csv")  # output

print(f"Using LM master dictionary: {LM_PATH}")

# ── Load the LM master dictionary ────────────────────────────────────────────
# Expected columns: 'Word', 'Positive', 'Negative'
lm = pd.read_csv(LM_PATH)

required_cols = {"Word", "Positive", "Negative"}
if not required_cols.issubset(lm.columns):
    raise ValueError(f"LM file must contain columns: {required_cols}, but has {lm.columns.tolist()}")

# ── Step 3a: Retain only positive or negative words ──────────────────────────
# Words with both positive and negative scores are ambiguous and are removed.
mask     = (lm["Positive"] > 0) | (lm["Negative"] > 0)
df       = lm.loc[mask, ["Word", "Positive", "Negative"]].copy()

df["word"] = df["Word"].str.lower()

both_mask = (df["Positive"] > 0) & (df["Negative"] > 0)
df        = df.loc[~both_mask].copy()

df["sentiment"] = df.apply(lambda row: "pos" if row["Positive"] > 0 else "neg", axis=1)

base_dict = df[["word", "sentiment"]].drop_duplicates().reset_index(drop=True)
print(f"Base LM positive/negative word count: {len(base_dict)}")

# ── Step 3b: Remove economically ambiguous terms ──────────────────────────────
# These terms carry inherently neutral or context-dependent meaning in FOMC
# documents and are removed following the Aruoba-Drechsel (2024) approach.
remove_words = {
    "unemployment", "unemployed", "employment",
    "inflation", "deflation",
    "recession", "depression",
    "productivity", "consumption", "investment",
    "unforeseen",
}

base_dict = base_dict[~base_dict["word"].isin(remove_words)].reset_index(drop=True)
print(f"Word count after removals: {len(base_dict)}")

# ── Step 3c: Add Federal Reserve-specific terminology ─────────────────────────
# Terms frequently used in FOMC documents that carry clear directional
# sentiment but are absent or mis-scored in the general-purpose LM dictionary.
manual_additions = {
    "tightening":    "neg",
    "tighten":       "neg",
    "tightened":     "neg",
    "easing":        "pos",
    "ease":          "pos",
    "eased":         "pos",
    "accommodative": "pos",
    "restrictive":   "neg",
    "headwinds":     "neg",
    "tailwinds":     "pos",
    "sluggish":      "neg",
    "resilient":     "pos",
    "buoyant":       "pos",
    "deterioration": "neg",
    "deteriorated":  "neg",
    "booms":         "pos",
    "boomed":        "pos",
}

manual_df = pd.DataFrame([{"word": w.lower(), "sentiment": s} for w, s in manual_additions.items()])

combined  = pd.concat([base_dict, manual_df], ignore_index=True)
combined  = combined.drop_duplicates(subset=["word"], keep="last").reset_index(drop=True)

print(f"Total word count after manual additions: {len(combined)}")

# ── Save the customised dictionary ────────────────────────────────────────────
combined.to_csv(OUT_PATH, index=False)
print(f"Saved customised LM dictionary: {OUT_PATH}")
print(combined.head())


# ============================================================
# SECTION 4 — PART 2: CONCEPT SENTIMENT COMPUTATION
# ============================================================

# ── Settings ──────────────────────────────────────────────────────────────────
# MERGED_FOLDER : meeting-level merged text files from fomc_text_merger.py
# LM_DICT_PATH  : customised dictionary produced in Part 1
# OUT_CSV       : output panel dataset with concept-level sentiment scores
# WINDOW        : number of tokens on each side of a concept occurrence
#                 to include in the sentiment scoring window (AD default: 10)

MERGED_FOLDER = "fomc_merged_AD"
LM_DICT_PATH  = os.path.join(RESULTS_DIR, "lm_dictionary_for_concepts.csv")
OUT_CSV       = os.path.join(RESULTS_DIR, "fomc_concept_sentiments_ADstyle.csv")

WINDOW = 10   # ±10 word window around each concept occurrence (Aruoba and Drechsel, 2024)

# ── Load the customised LM dictionary ────────────────────────────────────────
lm = pd.read_csv(LM_DICT_PATH)
lm["word"] = lm["word"].str.lower()

pos_words = set(lm[lm["sentiment"] == "pos"]["word"])
neg_words = set(lm[lm["sentiment"] == "neg"]["word"])
print(f"Loaded {len(pos_words)} positive and {len(neg_words)} negative words.")


# ============================================================
# SECTION 5 — TEXT CLEANING AND TOKENISATION (AD STEP 1)
# ============================================================

# Download NLTK stopwords if not already present
nltk.download("stopwords", quiet=True)
STOP = set(stopwords.words("english"))

# Compiled regex patterns for efficient repeated application
number_re = re.compile(r"\d+")        # matches numeric sequences
token_re  = re.compile(r"[a-zA-Z]+")  # matches alphabetic tokens only


def clean_and_tokenize_AD(text: str):
    """
    Clean and tokenise a meeting-level text string following the
    Aruoba-Drechsel (2024) preprocessing specification.

    Steps applied:
        1. Convert to lowercase
        2. Remove all numeric characters
        3. Extract alphabetic tokens only (strips punctuation)
        4. Remove English stopwords (NLTK list)
        5. Remove single-character tokens

    Parameters
    ----------
    text : str
        Raw meeting-level text content.

    Returns
    -------
    list of str
        Cleaned token list ready for concept matching.
    """
    text   = text.lower()
    text   = number_re.sub(" ", text)
    tokens = token_re.findall(text)

    cleaned = []
    for t in tokens:
        if t in STOP:
            continue
        if len(t) <= 1:
            continue
        cleaned.append(t)
    return cleaned


# ============================================================
# SECTION 6 — CONCEPT LIST (296 ECONOMIC CONCEPTS)
# ============================================================
# Each concept is defined by one or more phrase variants, where each variant
# is a list of tokens to match sequentially in the cleaned token stream.
# Uni-gram, bi-gram, and tri-gram concepts are included.
#
# Concept coverage spans:
#   Macroeconomic aggregates  (GDP, inflation, unemployment, output)
#   Financial conditions      (credit, spreads, equity prices, yield curve)
#   Labour market             (employment, wages, hiring, payrolls)
#   Sectors                   (manufacturing, housing, agriculture, energy)
#   International economies   (euro area, Japan, China, emerging markets)
#   Commodity markets         (oil, gas, wheat, metals, lumber)

concepts = {
    "borrowing":                          [["borrowing"]],
    "brazil":                             [["brazil"]],
    "banks":                              [["banks"]],
    "canada":                             [["canada"]],
    "credit":                             [["credit"]],
    "china":                              [["china"]],
    "consumption":                        [["consumption"]],
    "construction":                       [["construction"]],
    "currencies":                         [["currencies"]],
    "deposits":                           [["deposits"]],
    "employment":                         [["employment"]],
    "employment_cost":                    [["employment", "cost"]],
    "equipment":                          [["equipment"]],
    "euro":                               [["euro"]],
    "exports":                            [["exports"]],
    "germany":                            [["germany"]],
    "hiring":                             [["hiring"]],
    "hours":                              [["hours"]],
    "housing":                            [["housing"]],
    "imports":                            [["imports"]],
    "inflation":                          [["inflation"]],
    "inventories":                        [["inventories"]],
    "investment":                         [["investment"]],
    "japan":                              [["japan"]],
    "liquidity":                          [["liquidity"]],
    "loans":                              [["loans"]],
    "leasing":                            [["leasing"]],
    "lending":                            [["lending"]],
    "machinery":                          [["machinery"]],
    "mexico":                             [["mexico"]],
    "mortgage":                           [["mortgage"]],
    "output":                             [["output"]],
    "productivity":                       [["productivity"]],
    "profits":                            [["profits"]],
    "recovery":                           [["recovery"]],
    "reserves":                           [["reserves"]],
    "savings":                            [["savings"]],
    "spread":                             [["spread"]],
    "structures":                         [["structures"]],
    "tourism":                            [["tourism"]],
    "unemployment":                       [["unemployment"]],
    "utilization":                        [["utilization"]],
    "wages":                              [["wages"]],
    "weather":                            [["weather"]],
    "yield":                              [["yield"]],
    "aggregate_demand":                   [["aggregate", "demand"]],
    "auto_sales":                         [["auto", "sales"]],
    "bond_issuance":                      [["bond", "issuance"]],
    "budget_deficit":                     [["budget", "deficit"]],
    "economic_activity":                  [["economic", "activity"]],
    "business_confidence":                [["business", "confidence"]],
    "business_spending":                  [["business", "spending"]],
    "capital_expenditures":               [["capital", "expenditures"]],
    "commodity_prices":                   [["commodity", "prices"]],
    "consumer_confidence":                [["consumer", "confidence"]],
    "current_account":                    [["current", "account"]],
    "debt_growth":                        [["debt", "growth"]],
    "defense_spending":                   [["defense", "spending"]],
    "delinquency_rates":                  [["delinquency", "rates"]],
    "developing_countries":               [["developing", "countries"]],
    "domestic_demand":                    [["domestic", "demand"]],
    "drilling_activity":                  [["drilling", "activity"]],
    "durable_goods":                      [["durable", "goods"]],
    "economic_growth":                    [["economic", "growth"]],
    "energy_prices":                      [["energy", "prices"]],
    "equity_issuance":                    [["equity", "issuance"]],
    "equity_prices":                      [["equity", "prices"]],
    "euro_area":                          [["euro", "area"]],
    "exchange_rate":                      [["exchange", "rate"]],
    "federal_debt":                       [["federal", "debt"]],
    "financial_conditions":               [["financial", "conditions"]],
    "financial_developments":             [["financial", "developments"]],
    "fiscal_policy":                      [["fiscal", "policy"]],
    "fiscal_stimulus":                    [["fiscal", "stimulus"]],
    "food_prices":                        [["food", "prices"]],
    "foreign_economies":                  [["foreign", "economies"]],
    "gas_prices":                         [["gas", "prices"]],
    "gasoline_prices":                    [["gasoline", "prices"]],
    "government_purchases":               [["government", "purchases"]],
    "home_prices":                        [["home", "prices"]],
    "home_sales":                         [["home", "sales"]],
    "hourly_compensation":                [["hourly", "compensation"]],
    "household_debt":                     [["household", "debt"]],
    "household_spending":                 [["household", "spending"]],
    "import_prices":                      [["import", "prices"]],
    "income":                             [["income"]],
    "industrial_production":              [["industrial", "production"]],
    "industrial_supplies":                [["industrial", "supplies"]],
    "inflation_compensation":             [["inflation", "compensation"]],
    "inflation_expectations":             [["inflation", "expectations"]],
    "initial_claims":                     [["initial", "claims"]],
    "input_prices":                       [["input", "prices"]],
    "intermediate_materials":             [["intermediate", "materials"]],
    "international_developments":         [["international", "developments"]],
    "labor_market":                       [["labor", "market"]],
    "manufacturing_activity":             [["manufacturing", "activity"]],
    "manufacturing_firms":                [["manufacturing", "firms"]],
    "monetary_aggregates":                [["monetary", "aggregates"]],
    "mortgage_interest":                  [["mortgage", "interest"]],
    "natural_rate":                       [["natural", "rate"]],
    "net_exports":                        [["net", "exports"]],
    "new_orders":                         [["new", "orders"]],
    "nondefense_capital":                 [["nondefense", "capital"]],
    "oil_prices":                         [["oil", "prices"]],
    "output_gap":                         [["output", "gap"]],
    "potential_output":                   [["potential", "output"]],
    "price_pressures":                    [["price", "pressures"]],
    "producer_prices":                    [["producer", "prices"]],
    "refinancing_activity":               [["refinancing", "activity"]],
    "residential_investment":             [["residential", "investment"]],
    "retail_prices":                      [["retail", "prices"]],
    "retail_sales":                       [["retail", "sales"]],
    "retail_trade":                       [["retail", "trade"]],
    "share_prices":                       [["share", "prices"]],
    "social_security":                    [["social", "security"]],
    "stock_market":                       [["stock", "market"]],
    "trade_balance":                      [["trade", "balance"]],
    "trade_deficit":                      [["trade", "deficit"]],
    "trade_surplus":                      [["trade", "surplus"]],
    "treasury_securities":                [["treasury", "securities"]],
    "treasury_yield":                     [["treasury", "yield"]],
    "vacancy_rates":                      [["vacancy", "rates"]],
    "wholesale_prices":                   [["wholesale", "prices"]],
    "wholesale_trade":                    [["wholesale", "trade"]],
    "yield_curve":                        [["yield", "curve"]],
    "advanced_foreign_economies":         [["advanced", "foreign", "economies"]],
    "commercial_real_estate":             [["commercial", "real", "estate"]],
    "compensation_per_hour":              [["compensation", "per", "hour"]],
    "domestic_final_purchases":           [["domestic", "final", "purchases"]],
    "domestic_financial_developments":    [["domestic", "financial", "developments"]],
    "emerging_market_economies":          [["emerging", "market", "economies"]],
    "foreign_exchange":                   [["foreign", "exchange"]],
    "foreign_industrial_countries":       [["foreign", "industrial", "countries"]],
    "gross_domestic_purchases":           [["gross", "domestic", "purchases"]],
    "household_net_worth":                [["household", "net", "worth"]],
    "international_financial_transactions":[["international", "financial", "transactions"]],
    "labor_force_participation":          [["labor", "force", "participation"]],
    "major_industrial_countries":         [["major", "industrial", "countries"]],
    "market_interest_rates":              [["market", "interest", "rates"]],
    "nondefense_capital_goods":           [["nondefense", "capital", "goods"]],
    "output_per_hour":                    [["output", "per", "hour"]],
    "real_estate_activity":               [["real", "estate", "activity"]],
    "real_estate_market":                 [["real", "estate", "market"]],
    "real_interest_rate":                 [["real", "interest", "rate"]],
    "residential_real_estate":            [["residential", "real", "estate"]],
    "unit_labor_cost":                    [["unit", "labor", "cost"]],
    "money_market_mutual":                [["money", "market", "mutual"]],
    "gdp":                                [["gdp"]],
    "nominal_gdp":                        [["nominal", "gdp"]],
    "cpi":                                [["cpi"]],
    "nairu":                              [["nairu"]],
    "services":                           [["services"]],
    "core_inflation":                     [["core", "inflation"]],
    "bonds":                              [["bonds"]],
    "economy":                            [["economy"]],
    "motor_vehicles":                     [["motor", "vehicles"]],
    "outlays":                            [["outlays"]],
    "financing":                          [["financing"]],
    "financial_institutions":             [["financial", "institutions"]],
    "depository_institutions":            [["depository", "institutions"]],
    "assets":                             [["assets"]],
    "finance":                            [["finance"]],
    "credit_standards":                   [["credit", "standards"]],
    "shipments":                          [["shipments"]],
    "capacity":                           [["capacity"]],
    "office":                             [["office"]],
    "computers":                          [["computers"]],
    "industries":                         [["industries"]],
    "producers":                          [["producers"]],
    "supply":                             [["supply"]],
    "homes":                              [["homes"]],
    "sectors":                            [["sectors"]],
    "agriculture":                        [["agriculture"]],
    "merchandise":                        [["merchandise"]],
    "investors":                          [["investors"]],
    "aircraft":                           [["aircraft"]],
    "stocks":                             [["stocks"]],
    "buildings":                          [["buildings"]],
    "cash":                               [["cash"]],
    "consumer_prices":                    [["consumer", "prices"]],
    "trucks":                             [["trucks"]],
    "semiconductors":                     [["semiconductors"]],
    "crude_oil":                          [["crude", "oil"]],
    "loan_demand":                        [["loan", "demand"]],
    "united_kingdom":                     [["united", "kingdom"]],
    "farm":                               [["farm"]],
    "uncertainty":                        [["uncertainty"]],
    "households":                         [["households"]],
    "crop":                               [["crop"]],
    "apparel":                            [["apparel"]],
    "steel":                              [["steel"]],
    "money_market":                       [["money", "market"]],
    "automotive":                         [["automotive"]],
    "metals":                             [["metals"]],
    "market_participants":                [["market", "participants"]],
    "permits":                            [["permits"]],
    "commerce":                           [["commerce"]],
    "commercial_paper":                   [["commercial", "paper"]],
    "housing_starts":                     [["housing", "starts"]],
    "housing_activity":                   [["housing", "activity"]],
    "transportation":                     [["transportation"]],
    "natural_gas":                        [["natural", "gas"]],
    "consumer_goods":                     [["consumer", "goods"]],
    "municipal":                          [["municipal"]],
    "commodities":                        [["commodities"]],
    "corporations":                       [["corporations"]],
    "liabilities":                        [["liabilities"]],
    "consumers":                          [["consumers"]],
    "balance_sheet":                      [["balance", "sheet"]],
    "firms":                              [["firms"]],
    "trading":                            [["trading"]],
    "financial_markets":                  [["financial", "markets"]],
    "corn":                               [["corn"]],
    "economic_indicators":                [["economic", "indicators"]],
    "asia":                               [["asia"]],
    "taxes":                              [["taxes"]],
    "software":                           [["software"]],
    "mining":                             [["mining"]],
    "losses":                             [["losses"]],
    "jobs":                               [["jobs"]],
    "cars":                               [["cars"]],
    "depreciation":                       [["depreciation"]],
    "recession":                          [["recession"]],
    "france":                             [["france"]],
    "korea":                              [["korea"]],
    "italy":                              [["italy"]],
    "lumber":                             [["lumber"]],
    "volatility":                         [["volatility"]],
    "wheat":                              [["wheat"]],
    "final_sales":                        [["final", "sales"]],
    "credit_quality":                     [["credit", "quality"]],
    "international_transactions":         [["international", "transactions"]],
    "livestock":                          [["livestock"]],
    "rents":                              [["rents"]],
    "finished_goods":                     [["finished", "goods"]],
    "petroleum":                          [["petroleum"]],
    "latin_america":                      [["latin", "america"]],
    "traffic":                            [["traffic"]],
    "fuel":                               [["fuel"]],
    "plants":                             [["plants"]],
    "economic_outlook":                   [["economic", "outlook"]],
    "technology":                         [["technology"]],
    "argentina":                          [["argentina"]],
    "cattle":                             [["cattle"]],
    "crisis":                             [["crisis"]],
    "utilities":                          [["utilities"]],
    "travel":                             [["travel"]],
    "payrolls":                           [["payrolls"]],
    "factory":                            [["factory"]],
    "transfers":                          [["transfers"]],
    "drought":                            [["drought"]],
    "domestic_developments":              [["domestic", "developments"]],
    "gold":                               [["gold"]],
    "salaries":                           [["salaries"]],
    "oil_imports":                        [["oil", "imports"]],
    "cotton":                             [["cotton"]],
    "home_equity":                        [["home", "equity"]],
    "coal":                               [["coal"]],
    "philippines":                        [["philippines"]],
    "singapore":                          [["singapore"]],
    "taiwan":                             [["taiwan"]],
    "thailand":                           [["thailand"]],
    "soybean":                            [["soybean"]],
    "swaps":                              [["swaps"]],
    "harvest":                            [["harvest"]],
    "environment":                        [["environment"]],
    "deflator":                           [["deflator"]],
    "delinquencies":                      [["delinquencies"]],
    "chemicals":                          [["chemicals"]],
    "mergers":                            [["mergers"]],
    "rigs":                               [["rigs"]],
    "indonesia":                          [["indonesia"]],
    "political":                          [["political"]],
    "peso":                               [["peso"]],
    "headline_inflation":                 [["headline", "inflation"]],
    "retirement":                         [["retirement"]],
    "raw_materials":                      [["raw", "materials"]],
    "holiday_season":                     [["holiday", "season"]],
    "inflationary_pressures":             [["inflationary", "pressures"]],
    "tobacco":                            [["tobacco"]],
    "loan_officer":                       [["loan", "officer"]],
    "hurricane":                          [["hurricane"]],
    "health_care":                        [["health", "care"]],
    "foreign_net_purchases":              [["foreign", "net", "purchases"]],
    "loan_rates":                         [["loan", "rates"]],
    "equities":                           [["equities"]],
    "russia":                             [["russia"]],
    "workers":                            [["workers"]],
    "economic_expansion":                 [["economic", "expansion"]],
    "economic_data":                      [["economic", "data"]],
    "canadian_dollar":                    [["canadian", "dollar"]],
    "contractors":                        [["contractors"]],
    "corporate_profits":                  [["corporate", "profits"]],
    "insurance_companies":                [["insurance", "companies"]],
    "wage_pressures":                     [["wage", "pressures"]],
    "market_expectations":                [["market", "expectations"]],
}


# ============================================================
# SECTION 7 — HELPER FUNCTIONS
# ============================================================

def find_phrase_positions(tokens, phrase_tokens):
    """
    Find all starting positions of a phrase in a token list.

    Performs exact sequential matching: the phrase must appear as a
    contiguous subsequence of tokens at the correct positions.

    Parameters
    ----------
    tokens : list of str
        Full cleaned token list for a meeting document.
    phrase_tokens : list of str
        Ordered list of tokens forming the target phrase.

    Returns
    -------
    list of int
        Zero-based starting indices of each occurrence of the phrase.
    """
    positions = []
    L = len(phrase_tokens)
    for i in range(len(tokens) - L + 1):
        if tokens[i:i+L] == phrase_tokens:
            positions.append(i)
    return positions


def sentiment_for_concept(tokens, phrase_variants, window=WINDOW):
    """
    Compute the aggregate sentiment score for one concept in a meeting document.

    For each phrase variant and each occurrence of that variant in the token
    list, a window of ±WINDOW tokens is extracted. Each token in the window
    is matched against the positive and negative word sets:
        +1 if the token is in pos_words
        -1 if the token is in neg_words
         0 otherwise

    Parameters
    ----------
    tokens : list of str
        Full cleaned token list for a meeting document.
    phrase_variants : list of list of str
        All phrase variants for this concept (supports multiple surface forms).
    window : int
        Number of tokens on each side of the concept phrase to include.
        Default: WINDOW (10), following Aruoba and Drechsel (2024).

    Returns
    -------
    total_score : int
        Net sentiment score summed across all occurrences and all variants.
    hits : int
        Total number of sentiment word matches across all windows.
    """
    total_score = 0
    hits        = 0
    for phrase in phrase_variants:
        positions = find_phrase_positions(tokens, phrase)
        for pos in positions:
            left           = max(0, pos - window)
            right          = min(len(tokens) - 1, pos + window)
            window_tokens  = tokens[left:right+1]
            for w in window_tokens:
                if w in pos_words:
                    total_score += 1
                    hits        += 1
                elif w in neg_words:
                    total_score -= 1
                    hits        += 1
    return total_score, hits


# ============================================================
# SECTION 8 — MAIN LOOP: COMPUTE CONCEPT SENTIMENTS
# ============================================================
# Iterates over every FOMC meeting text file, tokenises the content,
# and computes three metrics for each of the 296 concepts:
#   {concept}_score              — raw window sentiment score
#   {concept}_hits               — count of sentiment word matches
#   {concept}_score_per_10k_words — length-normalised score
# A z-score standardisation is applied after all meetings are processed.

files = [f for f in os.listdir(MERGED_FOLDER) if f.lower().endswith(".txt")]
files.sort()

rows = []

for fname in tqdm(files, desc="Computing concept-based AD-style sentiments"):

    date  = fname.replace(".txt", "")
    fpath = os.path.join(MERGED_FOLDER, fname)

    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    tokens = clean_and_tokenize_AD(raw)
    wc     = len(tokens)

    row = {"Date": date, "word_count": wc}

    for cname, variants in concepts.items():
        score, hits = sentiment_for_concept(tokens, variants)
        row[f"{cname}_score"]                = score
        row[f"{cname}_hits"]                 = hits
        row[f"{cname}_score_per_10k_words"]  = score / wc * 10000 if wc > 0 else 0.0

    rows.append(row)

df = pd.DataFrame(rows).sort_values("Date")


# ============================================================
# SECTION 9 — Z-SCORE STANDARDISATION (ARUOBA-DRECHSEL STYLE)
# ============================================================
# Each concept's length-normalised score is standardised across all meeting
# dates by subtracting the cross-meeting mean and dividing by the population
# standard deviation (ddof=0). Concepts with zero variance receive a score
# of 0.0. The z-scores are the inputs to the ridge regression.

for cname in concepts.keys():
    col  = f"{cname}_score_per_10k_words"
    mean = df[col].mean()
    std  = df[col].std(ddof=0)
    if std > 0:
        df[f"{col}_z"] = (df[col] - mean) / std
    else:
        df[f"{col}_z"] = 0.0

df.to_csv(OUT_CSV, index=False)
print("Saved AD-style concept sentiments to:", OUT_CSV)