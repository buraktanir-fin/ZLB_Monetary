import os
import re
import datetime
from collections import defaultdict

from pdfminer.high_level import extract_text

# ============================================================
#  FOMC PDF -> TXT -> MEETING MERGE (STRICT AD SET)
#
#  INCLUDED:
#       Greenbook (incl. parts + supplements)
#       Tealbook A
#
#  EXCLUDED:
#       Redbook           ← REMOVED from previous version
#       Beige Book
#       Minutes / transcripts
#       Bluebook
#       Tealbook B
#       Press materials
#
#  Merges meetings starting from 1982-10-05
#
#  CHANGES FROM PREVIOUS VERSION:
#       1. Redbook removed from ALLOWED_TYPES and all detection logic
#       2. Content detection window expanded from 4000 to 12000 characters
#       3. Fuzzy matching added for OCR artifacts in early sample documents
#       4. Silent failure diagnostic added — unclassified files are logged
# ============================================================

# =========================
# PATHS
# =========================
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_FOLDER = os.path.join(CODE_DIR, "fomc_pdfs")
TXT_FOLDER = os.path.join(CODE_DIR, "fomc_txt")
MERGED_FOLDER = os.path.join(CODE_DIR, "fomc_merged_AD")
LOG_FOLDER = os.path.join(CODE_DIR, "fomc_logs")

os.makedirs(TXT_FOLDER, exist_ok=True)
os.makedirs(MERGED_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# =========================
# SETTINGS
# =========================
START_MEETING_DATE = datetime.date(1982, 10, 5)

# ← Redbook keywords removed from SKIP_KEYWORDS
SKIP_KEYWORDS = [
    "tealbookb", "tealbook_b", "tealbook-b",
    "material", "materials", "briefing", "briefings",
]

DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})", re.IGNORECASE)

# Expanded window: 12000 characters instead of 4000
# Covers cases where title page is preceded by long OCR preamble
CONTENT_DETECTION_WINDOW = 12000

# ============================================================
# UTILITIES
# ============================================================

def parse_date_yyyy_mm_dd(s: str):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

def should_skip_filename(fname: str) -> bool:
    s = fname.lower()
    return any(kw in s for kw in SKIP_KEYWORDS)

# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text_fallback(pdf_path: str) -> str:
    try:
        return extract_text(pdf_path)
    except Exception:
        import fitz
        text = []
        doc = fitz.open(pdf_path)
        for page in doc:
            text.append(page.get_text())
        doc.close()
        return "\n".join(text)

# ============================================================
# DOCUMENT TYPE DETECTION (STRICT AD SET — NO REDBOOK)
# ============================================================

def detect_doc_type_from_filename(fname: str):

    s = fname.lower()

    # Hard exclusions
    excluded = [
        "minutes", "transcript", "transcripts",
        "bluebook", "blue book",
        "statement", "press", "announcement",
        "tealbookb", "tealbook b", "tealbook_b", "tealbook-b",
        "record of policy actions",
        "material", "materials",
        "beigebook", "beige book",
        "redbook", "red book", "redbk",   # ← EXPLICITLY EXCLUDED
    ]

    if any(x in s for x in excluded):
        return None

    # GREENBOOK
    green_patterns = [
        "greenbook", "grnbk", "grnbook",
        "gbpt1", "gbpt_1", "gbpt-1",
        "gbpt2", "gbpt_2", "gbpt-2",
        "gbpt3", "gbpt_3", "gbpt-3",
        "gbsup", "gb_sup", "gb-sup",
        "gbsupp", "gbsupplement",
        "gbk",
    ]
    if any(p in s for p in green_patterns):
        return "greenbook"

    # TEALBOOK A
    teal_patterns = [
        "tealbooka", "tealbook_a", "tealbook-a",
        "tealbook a",
        "tb_a",
    ]
    if any(p in s for p in teal_patterns):
        return "tealbook_a"

    # ← Redbook detection block entirely removed

    return None


def detect_doc_type_from_content(txt_path: str):
    """
    Fallback detection when filename is ambiguous.

    Reads first CONTENT_DETECTION_WINDOW characters (expanded to 12000)
    to handle early-sample documents where OCR preamble or table data
    precedes the document title.

    Fuzzy matching added for common OCR artifacts in 1980s-era scans.
    Redbook signatures entirely removed.
    """
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(CONTENT_DETECTION_WINDOW).lower()
    except Exception:
        return None

    # ← Beige Book signatures absent (removed in previous version, kept absent)
    # ← Redbook signatures absent (new removal)

    # TEALBOOK A — check first as it is more specific
    if "tealbook a" in head:
        return "tealbook_a"

    # GREENBOOK — exact match
    if "greenbook" in head or "green book" in head:
        return "greenbook"

    # GREENBOOK — fuzzy match for OCR artifacts in early sample (1982-1990)
    # Covers: Gr3enbook, G reenbook, Gre3nbook, GreenBo0k etc.
    if re.search(r'gr[e3][e3]n\s?b[o0][o0]k', head):
        return "greenbook"

    # TEALBOOK A — fuzzy match
    if re.search(r't[e3]albook\s?a', head):
        return "tealbook_a"

    return None


def detect_doc_type(txt_path: str):
    fname = os.path.basename(txt_path)
    t = detect_doc_type_from_filename(fname)
    if t is not None:
        return t
    return detect_doc_type_from_content(txt_path)


# ============================================================
# STEP 1: PDF → TXT
# ============================================================

def convert_pdfs_to_txt():

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    pdf_files = sorted(pdf_files)

    for fname in pdf_files:

        if should_skip_filename(fname):
            continue

        pdf_path = os.path.join(PDF_FOLDER, fname)
        txt_path = os.path.join(TXT_FOLDER, fname[:-4] + ".txt")

        if os.path.exists(txt_path):
            continue

        try:
            text = extract_text_fallback(pdf_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text if text is not None else "")
        except Exception as e:
            print("ERROR:", fname, e)


# ============================================================
# STEP 2: MERGE (STRICT AD SET — GREENBOOK + TEALBOOK A ONLY)
# ============================================================

# ← Redbook removed from ALLOWED_TYPES
ALLOWED_TYPES = {"greenbook", "tealbook_a"}


def merge_txts_by_meeting_starting_from(start_date: datetime.date):

    txt_files = sorted(
        [f for f in os.listdir(TXT_FOLDER) if f.lower().endswith(".txt")]
    )

    files_by_date = defaultdict(list)

    # ── Diagnostic tracker ──────────────────────────────────
    unclassified = []   # files that returned None from both detectors
    excluded_known = [] # files that were explicitly excluded (non-AD types)
    # ────────────────────────────────────────────────────────

    for fname in txt_files:

        m = DATE_PATTERN.search(fname)
        if not m:
            continue

        d = parse_date_yyyy_mm_dd(m.group(1))
        if d is None or d < start_date:
            continue

        full_path = os.path.join(TXT_FOLDER, fname)
        doc_type = detect_doc_type(full_path)

        # STRICT FILTER
        if doc_type not in ALLOWED_TYPES:
            if doc_type is None:
                unclassified.append(fname)        # silent failure candidate
            else:
                excluded_known.append((fname, doc_type))  # known non-AD type
            continue

        files_by_date[m.group(1)].append((full_path, doc_type))

    # ── Write merged meeting files ───────────────────────────
    for date_str, file_type_list in sorted(files_by_date.items()):

        merged_path = os.path.join(MERGED_FOLDER, f"{date_str}.txt")

        file_type_list = sorted(file_type_list, key=lambda z: z[1])

        with open(merged_path, "w", encoding="utf-8") as out:

            for fpath, doc_type in file_type_list:

                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

                out.write(f"==== {doc_type.upper()} | {os.path.basename(fpath)} ====\n\n")
                out.write(text)
                out.write("\n\n\n")

    # ── Diagnostic report ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"MERGE COMPLETE")
    print(f"{'='*60}")
    print(f"Meetings merged:           {len(files_by_date)}")
    print(f"Known excluded files:      {len(excluded_known)}")
    print(f"Unclassified files (None): {len(unclassified)}")

    if unclassified:
        print(f"\n⚠ WARNING: The following files could not be classified")
        print(f"  and were SILENTLY EXCLUDED. Review manually:\n")
        for fname in unclassified:
            print(f"  {fname}")

    # ── Write diagnostic logs ────────────────────────────────
    log_path = os.path.join(LOG_FOLDER, "unclassified_files.txt")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("UNCLASSIFIED FILES — MANUAL REVIEW REQUIRED\n")
        log.write(f"Generated: {datetime.datetime.now()}\n")
        log.write("="*60 + "\n\n")
        if unclassified:
            for fname in unclassified:
                log.write(fname + "\n")
        else:
            log.write("No unclassified files detected.\n")

    excluded_log_path = os.path.join(LOG_FOLDER, "excluded_known_files.txt")
    with open(excluded_log_path, "w", encoding="utf-8") as log:
        log.write("KNOWN EXCLUDED FILES (non-AD document types)\n")
        log.write(f"Generated: {datetime.datetime.now()}\n")
        log.write("="*60 + "\n\n")
        for fname, dtype in excluded_known:
            log.write(f"{fname}  →  {dtype}\n")

    print(f"\nDiagnostic logs written to: {LOG_FOLDER}")
    print(f"{'='*60}\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    convert_pdfs_to_txt()
    merge_txts_by_meeting_starting_from(START_MEETING_DATE)