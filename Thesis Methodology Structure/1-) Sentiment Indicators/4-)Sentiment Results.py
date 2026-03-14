import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from tqdm import tqdm

# ============================================================
# FULL CODE (ONLY REQUESTED CHANGES APPLIED):
# 1) LM_PATH is selected dynamically from an input folder
# 2) Create folder "sentiment_results" and save all created CSVs there
# ============================================================

# =======================
# OUTPUT FOLDER
# =======================
RESULTS_DIR = "sentiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =======================
# LM INPUT FOLDER (DYNAMIC)
# =======================
LM_INPUT_FOLDER = "input"  # put LM master dictionary CSV in this folder

def pick_lm_master_csv(folder: str) -> str:
    """
    Auto-detect LM master dictionary CSV inside `folder`.
    Preference order: file names with Loughran/McDonald/MasterDictionary keywords.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"LM input folder not found: {folder}")

    csv_files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in folder: {folder}")

    def score(fname: str) -> int:
        n = fname.lower()
        s = 0
        if "loughran" in n: s += 5
        if "mcdonald" in n: s += 5
        if "masterdictionary" in n: s += 6
        if "dictionary" in n: s += 2
        if "1993" in n: s += 1
        if "2024" in n: s += 1
        return s

    csv_files = sorted(csv_files, key=score, reverse=True)
    return os.path.join(folder, csv_files[0])

# === 1) GİRDİ / ÇIKTI DOSYA İSİMLERİ ===
LM_PATH = pick_lm_master_csv(LM_INPUT_FOLDER)  # ✅ dynamic LM path
OUT_PATH = os.path.join(RESULTS_DIR, "lm_dictionary_for_concepts.csv")  # ✅ saved to results folder

print(f"✅ Using LM master dictionary: {LM_PATH}")

# === 2) LM MASTER DICTIONARY'Yİ YÜKLE ===
lm = pd.read_csv(LM_PATH)

# Beklenen kolon isimleri: 'Word', 'Positive', 'Negative'
required_cols = {"Word", "Positive", "Negative"}
if not required_cols.issubset(lm.columns):
    raise ValueError(f"LM file must contain columns: {required_cols}, but has {lm.columns.tolist()}")

# === 3) SADELEŞTİRME: SADECE POS/NEG KELİMELERİ SEÇ ===
mask = (lm["Positive"] > 0) | (lm["Negative"] > 0)
df = lm.loc[mask, ["Word", "Positive", "Negative"]].copy()

df["word"] = df["Word"].str.lower()

both_mask = (df["Positive"] > 0) & (df["Negative"] > 0)
df = df.loc[~both_mask].copy()

df["sentiment"] = df.apply(lambda row: "pos" if row["Positive"] > 0 else "neg", axis=1)

base_dict = df[["word", "sentiment"]].drop_duplicates().reset_index(drop=True)
print(f"Base LM pos/neg kelime sayısı: {len(base_dict)}")

# === 4) ARUOBA-STYLED ÇIKARMALAR (REMOVE) ===
remove_words = {
    "unemployment", "unemployed", "employment",
    "inflation", "deflation",
    "recession", "depression",
    "productivity", "consumption", "investment",
    "unforeseen",
}

base_dict = base_dict[~base_dict["word"].isin(remove_words)].reset_index(drop=True)
print(f"Remove sonrası kelime sayısı: {len(base_dict)}")

# === 5) FED-SPESİFİK KELİME EKLEMELERİ (ADD) ===
manual_additions = {
    "tightening": "neg",
    "tighten": "neg",
    "tightened": "neg",
    "easing": "pos",
    "ease": "pos",
    "eased": "pos",
    "accommodative": "pos",
    "restrictive": "neg",
    "headwinds": "neg",
    "tailwinds": "pos",
    "sluggish": "neg",
    "resilient": "pos",
    "buoyant": "pos",
    "deterioration": "neg",
    "deteriorated": "neg",
    "booms": "pos",
    "boomed": "pos",
}

manual_df = pd.DataFrame([{"word": w.lower(), "sentiment": s} for w, s in manual_additions.items()])

combined = pd.concat([base_dict, manual_df], ignore_index=True)
combined = combined.drop_duplicates(subset=["word"], keep="last").reset_index(drop=True)

print(f"Manual eklemeler sonrası toplam kelime sayısı: {len(combined)}")

# === 6) SON HALİNİ KAYDET ===
combined.to_csv(OUT_PATH, index=False)
print(f"✅ Kaydedildi: {OUT_PATH}")
print(combined.head())

# ============================================================
# SECOND PART: CONCEPT SENTIMENT COMPUTATION
# ============================================================

# =============== SETTINGS ===============
MERGED_FOLDER = "fomc_merged_AD"  # meeting-level merged txt files
LM_DICT_PATH = os.path.join(RESULTS_DIR, "lm_dictionary_for_concepts.csv")  # ✅ read from results folder
OUT_CSV = os.path.join(RESULTS_DIR, "fomc_concept_sentiments_ADstyle.csv")  # ✅ save to results folder

WINDOW = 10  # ±10 kelime window (Aruoba & Drechsel)

# =============== LOAD LM DICTIONARY ===============
lm = pd.read_csv(LM_DICT_PATH)
lm["word"] = lm["word"].str.lower()

pos_words = set(lm[lm["sentiment"] == "pos"]["word"])
neg_words = set(lm[lm["sentiment"] == "neg"]["word"])
print(f"Loaded {len(pos_words)} positive and {len(neg_words)} negative words.")

# =============== CLEANING SETUP (AD STEP 1) ===============
nltk.download("stopwords", quiet=True)
STOP = set(stopwords.words("english"))

number_re = re.compile(r"\d+")
token_re = re.compile(r"[a-zA-Z]+")

def clean_and_tokenize_AD(text: str):
    text = text.lower()
    text = number_re.sub(" ", text)
    tokens = token_re.findall(text)

    cleaned = []
    for t in tokens:
        if t in STOP:
            continue
        if len(t) <= 1:
            continue
        cleaned.append(t)
    return cleaned

# =============== CONCEPT LIST (YOUR LIST) ===============
concepts = {
    "borrowing": [["borrowing"]],
    "brazil": [["brazil"]],
    "banks": [["banks"]],
    "canada": [["canada"]],
    "credit": [["credit"]],
    "china": [["china"]],
    "consumption": [["consumption"]],
    "construction": [["construction"]],
    "currencies": [["currencies"]],
    "deposits": [["deposits"]],
    "employment": [["employment"]],
    "employment_cost": [["employment", "cost"]],
    "equipment": [["equipment"]],
    "euro": [["euro"]],
    "exports": [["exports"]],
    "germany": [["germany"]],
    "hiring": [["hiring"]],
    "hours": [["hours"]],
    "housing": [["housing"]],
    "imports": [["imports"]],
    "inflation": [["inflation"]],
    "inventories": [["inventories"]],
    "investment": [["investment"]],
    "japan": [["japan"]],
    "liquidity": [["liquidity"]],
    "loans": [["loans"]],
    "leasing": [["leasing"]],
    "lending": [["lending"]],
    "machinery": [["machinery"]],
    "mexico": [["mexico"]],
    "mortgage": [["mortgage"]],
    "output": [["output"]],
    "productivity": [["productivity"]],
    "profits": [["profits"]],
    "recovery": [["recovery"]],
    "reserves": [["reserves"]],
    "savings": [["savings"]],
    "spread": [["spread"]],
    "structures": [["structures"]],
    "tourism": [["tourism"]],
    "unemployment": [["unemployment"]],
    "utilization": [["utilization"]],
    "wages": [["wages"]],
    "weather": [["weather"]],
    "yield": [["yield"]],
    "aggregate_demand": [["aggregate", "demand"]],
    "auto_sales": [["auto", "sales"]],
    "bond_issuance": [["bond", "issuance"]],
    "budget_deficit": [["budget", "deficit"]],
    "economic_activity": [["economic", "activity"]],
    "business_confidence": [["business", "confidence"]],
    "business_spending": [["business", "spending"]],
    "capital_expenditures": [["capital", "expenditures"]],
    "commodity_prices": [["commodity", "prices"]],
    "consumer_confidence": [["consumer", "confidence"]],
    "current_account": [["current", "account"]],
    "debt_growth": [["debt", "growth"]],
    "defense_spending": [["defense", "spending"]],
    "delinquency_rates": [["delinquency", "rates"]],
    "developing_countries": [["developing", "countries"]],
    "domestic_demand": [["domestic", "demand"]],
    "drilling_activity": [["drilling", "activity"]],
    "durable_goods": [["durable", "goods"]],
    "economic_growth": [["economic", "growth"]],
    "energy_prices": [["energy", "prices"]],
    "equity_issuance": [["equity", "issuance"]],
    "equity_prices": [["equity", "prices"]],
    "euro_area": [["euro", "area"]],
    "exchange_rate": [["exchange", "rate"]],
    "federal_debt": [["federal", "debt"]],
    "financial_conditions": [["financial", "conditions"]],
    "financial_developments": [["financial", "developments"]],
    "fiscal_policy": [["fiscal", "policy"]],
    "fiscal_stimulus": [["fiscal", "stimulus"]],
    "food_prices": [["food", "prices"]],
    "foreign_economies": [["foreign", "economies"]],
    "gas_prices": [["gas", "prices"]],
    "gasoline_prices": [["gasoline", "prices"]],
    "government_purchases": [["government", "purchases"]],
    "home_prices": [["home", "prices"]],
    "home_sales": [["home", "sales"]],
    "hourly_compensation": [["hourly", "compensation"]],
    "household_debt": [["household", "debt"]],
    "household_spending": [["household", "spending"]],
    "import_prices": [["import", "prices"]],
    "income": [["income"]],
    "industrial_production": [["industrial", "production"]],
    "industrial_supplies": [["industrial", "supplies"]],
    "inflation_compensation": [["inflation", "compensation"]],
    "inflation_expectations": [["inflation", "expectations"]],
    "initial_claims": [["initial", "claims"]],
    "input_prices": [["input", "prices"]],
    "intermediate_materials": [["intermediate", "materials"]],
    "international_developments": [["international", "developments"]],
    "labor_market": [["labor", "market"]],
    "manufacturing_activity": [["manufacturing", "activity"]],
    "manufacturing_firms": [["manufacturing", "firms"]],
    "monetary_aggregates": [["monetary", "aggregates"]],
    "mortgage_interest": [["mortgage", "interest"]],
    "natural_rate": [["natural", "rate"]],
    "net_exports": [["net", "exports"]],
    "new_orders": [["new", "orders"]],
    "nondefense_capital": [["nondefense", "capital"]],
    "oil_prices": [["oil", "prices"]],
    "output_gap": [["output", "gap"]],
    "potential_output": [["potential", "output"]],
    "price_pressures": [["price", "pressures"]],
    "producer_prices": [["producer", "prices"]],
    "refinancing_activity": [["refinancing", "activity"]],
    "residential_investment": [["residential", "investment"]],
    "retail_prices": [["retail", "prices"]],
    "retail_sales": [["retail", "sales"]],
    "retail_trade": [["retail", "trade"]],
    "share_prices": [["share", "prices"]],
    "social_security": [["social", "security"]],
    "stock_market": [["stock", "market"]],
    "trade_balance": [["trade", "balance"]],
    "trade_deficit": [["trade", "deficit"]],
    "trade_surplus": [["trade", "surplus"]],
    "treasury_securities": [["treasury", "securities"]],
    "treasury_yield": [["treasury", "yield"]],
    "vacancy_rates": [["vacancy", "rates"]],
    "wholesale_prices": [["wholesale", "prices"]],
    "wholesale_trade": [["wholesale", "trade"]],
    "yield_curve": [["yield", "curve"]],
    "advanced_foreign_economies": [["advanced", "foreign", "economies"]],
    "commercial_real_estate": [["commercial", "real", "estate"]],
    "compensation_per_hour": [["compensation", "per", "hour"]],
    "domestic_final_purchases": [["domestic", "final", "purchases"]],
    "domestic_financial_developments": [["domestic", "financial", "developments"]],
    "emerging_market_economies": [["emerging", "market", "economies"]],
    "foreign_exchange": [["foreign", "exchange"]],
    "foreign_industrial_countries": [["foreign", "industrial", "countries"]],
    "gross_domestic_purchases": [["gross", "domestic", "purchases"]],
    "household_net_worth": [["household", "net", "worth"]],
    "international_financial_transactions": [["international", "financial", "transactions"]],
    "labor_force_participation": [["labor", "force", "participation"]],
    "major_industrial_countries": [["major", "industrial", "countries"]],
    "market_interest_rates": [["market", "interest", "rates"]],
    "nondefense_capital_goods": [["nondefense", "capital", "goods"]],
    "output_per_hour": [["output", "per", "hour"]],
    "real_estate_activity": [["real", "estate", "activity"]],
    "real_estate_market": [["real", "estate", "market"]],
    "real_interest_rate": [["real", "interest", "rate"]],
    "residential_real_estate": [["residential", "real", "estate"]],
    "unit_labor_cost": [["unit", "labor", "cost"]],
    "money_market_mutual": [["money", "market", "mutual"]],
    "gdp": [["gdp"]],
    "nominal_gdp": [["nominal", "gdp"]],
    "cpi": [["cpi"]],
    "nairu": [["nairu"]],
    "services": [["services"]],
    "core_inflation": [["core", "inflation"]],
    "bonds": [["bonds"]],
    "economy": [["economy"]],
    "motor_vehicles": [["motor", "vehicles"]],
    "outlays": [["outlays"]],
    "financing": [["financing"]],
    "financial_institutions": [["financial", "institutions"]],
    "depository_institutions": [["depository", "institutions"]],
    "assets": [["assets"]],
    "finance": [["finance"]],
    "credit_standards": [["credit", "standards"]],
    "shipments": [["shipments"]],
    "capacity": [["capacity"]],
    "office": [["office"]],
    "computers": [["computers"]],
    "industries": [["industries"]],
    "producers": [["producers"]],
    "supply": [["supply"]],
    "homes": [["homes"]],
    "sectors": [["sectors"]],
    "agriculture": [["agriculture"]],
    "merchandise": [["merchandise"]],
    "investors": [["investors"]],
    "aircraft": [["aircraft"]],
    "stocks": [["stocks"]],
    "buildings": [["buildings"]],
    "cash": [["cash"]],
    "consumer_prices": [["consumer", "prices"]],
    "trucks": [["trucks"]],
    "semiconductors": [["semiconductors"]],
    "crude_oil": [["crude", "oil"]],
    "loan_demand": [["loan", "demand"]],
    "united_kingdom": [["united", "kingdom"]],
    "farm": [["farm"]],
    "uncertainty": [["uncertainty"]],
    "households": [["households"]],
    "crop": [["crop"]],
    "apparel": [["apparel"]],
    "steel": [["steel"]],
    "money_market": [["money", "market"]],
    "automotive": [["automotive"]],
    "metals": [["metals"]],
    "market_participants": [["market", "participants"]],
    "permits": [["permits"]],
    "commerce": [["commerce"]],
    "commercial_paper": [["commercial", "paper"]],
    "housing_starts": [["housing", "starts"]],
    "housing_activity": [["housing", "activity"]],
    "transportation": [["transportation"]],
    "natural_gas": [["natural", "gas"]],
    "consumer_goods": [["consumer", "goods"]],
    "municipal": [["municipal"]],
    "commodities": [["commodities"]],
    "corporations": [["corporations"]],
    "liabilities": [["liabilities"]],
    "consumers": [["consumers"]],
    "balance_sheet": [["balance", "sheet"]],
    "firms": [["firms"]],
    "trading": [["trading"]],
    "financial_markets": [["financial", "markets"]],
    "corn": [["corn"]],
    "economic_indicators": [["economic", "indicators"]],
    "asia": [["asia"]],
    "taxes": [["taxes"]],
    "software": [["software"]],
    "mining": [["mining"]],
    "losses": [["losses"]],
    "jobs": [["jobs"]],
    "cars": [["cars"]],
    "depreciation": [["depreciation"]],
    "recession": [["recession"]],
    "france": [["france"]],
    "korea": [["korea"]],
    "italy": [["italy"]],
    "lumber": [["lumber"]],
    "volatility": [["volatility"]],
    "wheat": [["wheat"]],
    "final_sales": [["final", "sales"]],
    "credit_quality": [["credit", "quality"]],
    "international_transactions": [["international", "transactions"]],
    "livestock": [["livestock"]],
    "rents": [["rents"]],
    "finished_goods": [["finished", "goods"]],
    "petroleum": [["petroleum"]],
    "latin_america": [["latin", "america"]],
    "traffic": [["traffic"]],
    "fuel": [["fuel"]],
    "plants": [["plants"]],
    "economic_outlook": [["economic", "outlook"]],
    "technology": [["technology"]],
    "argentina": [["argentina"]],
    "cattle": [["cattle"]],
    "crisis": [["crisis"]],
    "utilities": [["utilities"]],
    "travel": [["travel"]],
    "payrolls": [["payrolls"]],
    "factory": [["factory"]],
    "transfers": [["transfers"]],
    "drought": [["drought"]],
    "domestic_developments": [["domestic", "developments"]],
    "gold": [["gold"]],
    "salaries": [["salaries"]],
    "oil_imports": [["oil", "imports"]],
    "cotton": [["cotton"]],
    "home_equity": [["home", "equity"]],
    "coal": [["coal"]],
    "philippines": [["philippines"]],
    "singapore": [["singapore"]],
    "taiwan": [["taiwan"]],
    "thailand": [["thailand"]],
    "soybean": [["soybean"]],
    "swaps": [["swaps"]],
    "harvest": [["harvest"]],
    "environment": [["environment"]],
    "deflator": [["deflator"]],
    "delinquencies": [["delinquencies"]],
    "chemicals": [["chemicals"]],
    "mergers": [["mergers"]],
    "rigs": [["rigs"]],
    "indonesia": [["indonesia"]],
    "political": [["political"]],
    "peso": [["peso"]],
    "headline_inflation": [["headline", "inflation"]],
    "retirement": [["retirement"]],
    "raw_materials": [["raw", "materials"]],
    "holiday_season": [["holiday", "season"]],
    "inflationary_pressures": [["inflationary", "pressures"]],
    "tobacco": [["tobacco"]],
    "loan_officer": [["loan", "officer"]],
    "hurricane": [["hurricane"]],
    "health_care": [["health", "care"]],
    "foreign_net_purchases": [["foreign", "net", "purchases"]],
    "loan_rates": [["loan", "rates"]],
    "equities": [["equities"]],
    "russia": [["russia"]],
    "workers": [["workers"]],
    "economic_expansion": [["economic", "expansion"]],
    "economic_data": [["economic", "data"]],
    "canadian_dollar": [["canadian", "dollar"]],
    "contractors": [["contractors"]],
    "corporate_profits": [["corporate", "profits"]],
    "insurance_companies": [["insurance", "companies"]],
    "wage_pressures": [["wage", "pressures"]],
    "market_expectations": [["market", "expectations"]],
}

# =============== HELPER FUNCTIONS ===============
def find_phrase_positions(tokens, phrase_tokens):
    positions = []
    L = len(phrase_tokens)
    for i in range(len(tokens) - L + 1):
        if tokens[i:i+L] == phrase_tokens:
            positions.append(i)
    return positions

def sentiment_for_concept(tokens, phrase_variants, window=WINDOW):
    total_score = 0
    hits = 0
    for phrase in phrase_variants:
        positions = find_phrase_positions(tokens, phrase)
        for pos in positions:
            left = max(0, pos - window)
            right = min(len(tokens) - 1, pos + window)
            window_tokens = tokens[left:right+1]
            for w in window_tokens:
                if w in pos_words:
                    total_score += 1
                    hits += 1
                elif w in neg_words:
                    total_score -= 1
                    hits += 1
    return total_score, hits

# =============== MAIN LOOP ===============
files = [f for f in os.listdir(MERGED_FOLDER) if f.lower().endswith(".txt")]
files.sort()

rows = []

for fname in tqdm(files, desc="Computing concept-based AD-style sentiments"):
    date = fname.replace(".txt", "")
    fpath = os.path.join(MERGED_FOLDER, fname)

    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    tokens = clean_and_tokenize_AD(raw)
    wc = len(tokens)

    row = {"Date": date, "word_count": wc}

    for cname, variants in concepts.items():
        score, hits = sentiment_for_concept(tokens, variants)
        row[f"{cname}_score"] = score
        row[f"{cname}_hits"] = hits
        row[f"{cname}_score_per_10k_words"] = score / wc * 10000 if wc > 0 else 0.0

    rows.append(row)

df = pd.DataFrame(rows).sort_values("Date")

# =============== Z-SCORE STANDARDIZATION (OPTIONAL BUT A&D STYLE) ===============
for cname in concepts.keys():
    col = f"{cname}_score_per_10k_words"
    mean = df[col].mean()
    std = df[col].std(ddof=0)
    if std > 0:
        df[f"{col}_z"] = (df[col] - mean) / std
    else:
        df[f"{col}_z"] = 0.0

df.to_csv(OUT_CSV, index=False)
print("✅ Saved AD-style concept sentiments to:", OUT_CSV)
