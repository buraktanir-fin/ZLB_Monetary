"""
================================================================================
fomc_text_merger.py
================================================================================
Thesis:  "When Rates Hit Zero: Identifying Monetary Shocks with Shadow Rates"
Author:  Burak Tanir
Date:    April 2026

Purpose
-------
Converts FOMC staff documents from PDF to plain text and merges all documents
belonging to the same FOMC meeting into a single text file. The merged files
serve as the direct input to the sentiment analysis pipeline described in
Section 4 of the thesis.

This script implements the strict Aruoba-Drechsel (2024) document set,
which includes only Greenbook (including parts and supplements) and
Tealbook A. All other document types are explicitly excluded.

Document scope
--------------
    INCLUDED:
        Greenbook (including parts: Part 1, Part 2, Part 3, supplements)
        Tealbook A

    EXCLUDED:
        Redbook           (removed — contains regional anecdotal content)
        Beige Book        (regional summary; not part of pre-decision set)
        Minutes           (released after the policy decision)
        Transcripts       (released with a five-year lag)
        Bluebook          (replaced by Tealbook B after 2010)
        Tealbook B        (monetary policy alternatives; not staff outlook)
        Press materials   (post-decision announcements)

    Sample start: October 5, 1982  (first meeting in the thesis sample)

Pipeline
--------
    Step 1 — PDF to TXT conversion
        Each PDF in fomc_pdfs/ is converted to plain text using pdfminer.
        If pdfminer fails (common for early scanned documents), PyMuPDF
        (fitz) is used as a fallback extractor.

    Step 2 — Document type detection
        Each text file is classified as greenbook, tealbook_a, or None
        using a two-stage detection approach:
            (a) Filename-based detection  — fast; covers most cases
            (b) Content-based detection   — fallback for ambiguous filenames;
                reads the first 12,000 characters; includes fuzzy matching
                for OCR artifacts in 1980s-era scanned documents

    Step 3 — Meeting-level merge
        All classified documents belonging to the same meeting date are
        concatenated into a single file named {YYYY-MM-DD}.txt and saved
        to fomc_merged_AD/. Only greenbook and tealbook_a documents are
        included in the merged output.

    Step 4 — Diagnostic logging
        Two log files are written to fomc_logs/:
            unclassified_files.txt   — files that could not be classified
            excluded_known_files.txt — files classified but excluded (non-AD)

Input
-----
    fomc_pdfs/
        Raw FOMC PDFs downloaded by fomc_pdf_downloader.py.
        Filenames must include a YYYY-MM-DD date prefix.

Output
------
    fomc_txt/
        One plain-text file per PDF (intermediate; preserved for inspection).

    fomc_merged_AD/
        One merged text file per FOMC meeting date: {YYYY-MM-DD}.txt
        Contains only Greenbook and Tealbook A content, separated by
        document-type headers.

    fomc_logs/
        unclassified_files.txt    — files returning None from both detectors
        excluded_known_files.txt  — files of known non-AD document types



Dependencies
------------
    os, re, datetime, collections  (standard library)
    pdfminer.six                   — pip install pdfminer.six
    PyMuPDF (fitz)                 — pip install pymupdf  (fallback extractor)

Usage
-----
    python fomc_text_merger.py

    All folders are created automatically next to this script.
    The script is safe to re-run; already-converted TXT files are skipped.
================================================================================
"""

import os
import re
import datetime
from collections import defaultdict

from pdfminer.high_level import extract_text

# ============================================================
# SECTION 1 — PATH CONFIGURATION
# ============================================================
# All folders are resolved relative to the directory containing this script.
# This ensures consistent behaviour regardless of the working directory.

CODE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input: raw PDFs downloaded by fomc_pdf_downloader.py
PDF_FOLDER    = os.path.join(CODE_DIR, "fomc_pdfs")

# Intermediate: one plain-text file per PDF
TXT_FOLDER    = os.path.join(CODE_DIR, "fomc_txt")

# Output: one merged text file per FOMC meeting (Greenbook + Tealbook A only)
MERGED_FOLDER = os.path.join(CODE_DIR, "fomc_merged_AD")

# Logs: diagnostic reports for unclassified and excluded files
LOG_FOLDER    = os.path.join(CODE_DIR, "fomc_logs")

# Create all output folders if they do not already exist
os.makedirs(TXT_FOLDER, exist_ok=True)
os.makedirs(MERGED_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)


# ============================================================
# SECTION 2 — SETTINGS
# ============================================================

# First FOMC meeting included in the thesis sample period
START_MEETING_DATE = datetime.date(1982, 10, 5)

# Filename substrings that unconditionally exclude a file from processing.
# Redbook keywords removed in this version (see Changes section in docstring).
SKIP_KEYWORDS = [
    "tealbookb", "tealbook_b", "tealbook-b",
    "material", "materials", "briefing", "briefings",
]

# Regex to extract the YYYY-MM-DD meeting date from a filename
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})", re.IGNORECASE)

# Number of characters read from the beginning of a text file during
# content-based document type detection. Expanded from 4,000 to 12,000
# to handle early-sample documents where OCR preamble or table data
# precedes the document title.
CONTENT_DETECTION_WINDOW = 12000


# ============================================================
# SECTION 3 — UTILITY FUNCTIONS
# ============================================================

def parse_date_yyyy_mm_dd(s: str):
    """
    Convert a YYYY-MM-DD string to a datetime.date object.

    Returns None if the string cannot be parsed, rather than raising
    an exception, so the caller can handle missing dates gracefully.
    """
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def should_skip_filename(fname: str) -> bool:
    """
    Return True if the filename contains any keyword in SKIP_KEYWORDS.

    Used as a fast pre-filter before PDF-to-TXT conversion to avoid
    processing documents that are unconditionally excluded from the
    Aruoba-Drechsel document set.
    """
    s = fname.lower()
    return any(kw in s for kw in SKIP_KEYWORDS)


# ============================================================
# SECTION 4 — TEXT EXTRACTION
# ============================================================

def extract_text_fallback(pdf_path: str) -> str:
    """
    Extract plain text from a PDF file.

    Primary extractor  : pdfminer (handles digitally-created PDFs well)
    Fallback extractor : PyMuPDF / fitz (handles scanned PDFs from the
                         early sample period 1982-1990 more robustly)

    Parameters
    ----------
    pdf_path : str
        Absolute path to the PDF file.

    Returns
    -------
    str
        Extracted plain text. Returns an empty string if both extractors fail.
    """
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
# SECTION 5 — DOCUMENT TYPE DETECTION
# ============================================================

def detect_doc_type_from_filename(fname: str):
    """
    Classify a document as 'greenbook', 'tealbook_a', or None based on
    its filename alone.

    The Federal Reserve uses consistent filename conventions for its staff
    documents, so filename-based detection correctly classifies the majority
    of files. Content-based detection is used only when the filename is
    ambiguous (returns None here).

    Hard exclusions are checked first to short-circuit classification of
    documents that are definitively outside the Aruoba-Drechsel set.
    Redbook is explicitly excluded in this version.

    Parameters
    ----------
    fname : str
        Filename (basename only, not full path).

    Returns
    -------
    str or None
        'greenbook', 'tealbook_a', or None if the file should be excluded
        or cannot be classified from the filename.
    """
    s = fname.lower()

    # ── Hard exclusions ───────────────────────────────────────────────────────
    # Files matching any of these strings are excluded regardless of other
    # patterns. Redbook explicitly added to exclusion list in this version.
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

    # ── Greenbook detection ───────────────────────────────────────────────────
    # Covers the full Greenbook family: main document, parts 1-3, supplements
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

    # ── Tealbook A detection ──────────────────────────────────────────────────
    # Tealbook A replaced the Greenbook from June 2010 onwards
    teal_patterns = [
        "tealbooka", "tealbook_a", "tealbook-a",
        "tealbook a",
        "tb_a",
    ]
    if any(p in s for p in teal_patterns):
        return "tealbook_a"

    # ← Redbook detection block entirely removed in this version

    return None


def detect_doc_type_from_content(txt_path: str):
    """
    Fallback document type detection using the file's text content.

    Reads the first CONTENT_DETECTION_WINDOW characters (12,000) of the
    extracted text file. This larger window is necessary for early-sample
    documents (1982-1990) where OCR preamble or multi-column table data
    precedes the document title on the opening pages.

    Fuzzy regex patterns are included to handle common OCR substitutions
    in scanned documents (e.g. 'e' ↔ '3', 'o' ↔ '0', missing spaces).

    Redbook signatures are absent — removed in this version.
    Beige Book signatures are absent — removed in previous version.

    Parameters
    ----------
    txt_path : str
        Absolute path to the extracted plain-text file.

    Returns
    -------
    str or None
        'greenbook', 'tealbook_a', or None if the document type cannot
        be determined from the content.
    """
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(CONTENT_DETECTION_WINDOW).lower()
    except Exception:
        return None

    # ── Tealbook A — exact match (checked first; more specific) ──────────────
    if "tealbook a" in head:
        return "tealbook_a"

    # ── Greenbook — exact match ───────────────────────────────────────────────
    if "greenbook" in head or "green book" in head:
        return "greenbook"

    # ── Greenbook — fuzzy match for OCR artifacts (1982–1990 scans) ──────────
    # Covers variants: Gr3enbook, G reenbook, Gre3nbook, GreenBo0k etc.
    if re.search(r'gr[e3][e3]n\s?b[o0][o0]k', head):
        return "greenbook"

    # ── Tealbook A — fuzzy match ──────────────────────────────────────────────
    if re.search(r't[e3]albook\s?a', head):
        return "tealbook_a"

    return None


def detect_doc_type(txt_path: str):
    """
    Classify a document using a two-stage detection strategy.

    Stage 1: Filename-based detection (fast; covers most cases)
    Stage 2: Content-based detection (fallback for ambiguous filenames)

    Parameters
    ----------
    txt_path : str
        Absolute path to the extracted plain-text file.

    Returns
    -------
    str or None
        'greenbook', 'tealbook_a', or None.
    """
    fname = os.path.basename(txt_path)
    t = detect_doc_type_from_filename(fname)
    if t is not None:
        return t
    return detect_doc_type_from_content(txt_path)


# ============================================================
# SECTION 6 — STEP 1: PDF TO TXT CONVERSION
# ============================================================

def convert_pdfs_to_txt():
    """
    Convert all FOMC PDFs in fomc_pdfs/ to plain-text files in fomc_txt/.

    Processing rules:
        - Files matching SKIP_KEYWORDS are skipped without conversion.
        - Files whose corresponding TXT already exists are skipped (safe
          to re-run after interruptions).
        - Extraction errors are printed but do not halt processing.

    The text files produced here are consumed by merge_txts_by_meeting_starting_from()
    in Step 2.
    """
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    pdf_files = sorted(pdf_files)

    for fname in pdf_files:

        # Skip files that are unconditionally excluded
        if should_skip_filename(fname):
            continue

        pdf_path = os.path.join(PDF_FOLDER, fname)
        txt_path = os.path.join(TXT_FOLDER, fname[:-4] + ".txt")

        # Skip if TXT already exists (idempotent behaviour)
        if os.path.exists(txt_path):
            continue

        try:
            text = extract_text_fallback(pdf_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text if text is not None else "")
        except Exception as e:
            print("ERROR:", fname, e)


# ============================================================
# SECTION 7 — STEP 2: MERGE BY MEETING DATE
# ============================================================

# Document types included in the merged output (strict AD set)
# Redbook removed from ALLOWED_TYPES in this version
ALLOWED_TYPES = {"greenbook", "tealbook_a"}


def merge_txts_by_meeting_starting_from(start_date: datetime.date):
    """
    Merge classified text files into one file per FOMC meeting date.

    Only documents whose detected type is in ALLOWED_TYPES (greenbook,
    tealbook_a) are included. All other documents are logged for review.

    For each meeting date, constituent documents are sorted by document type
    before concatenation, ensuring a consistent ordering across all meetings.
    Each document is prefixed with a header line identifying its type and
    original filename.

    Diagnostic output
    -----------------
    Two log files are written to fomc_logs/:
        unclassified_files.txt    — files returning None from both detectors;
                                    these were silently excluded and may
                                    represent misnamed documents requiring
                                    manual inspection
        excluded_known_files.txt  — files of known non-AD document types
                                    (e.g. minutes, Tealbook B); excluded
                                    as expected

    Parameters
    ----------
    start_date : datetime.date
        First meeting date to include. Files dated before this date are
        skipped. For this thesis: datetime.date(1982, 10, 5).
    """
    txt_files = sorted(
        [f for f in os.listdir(TXT_FOLDER) if f.lower().endswith(".txt")]
    )

    files_by_date = defaultdict(list)

    # ── Diagnostic trackers ───────────────────────────────────────────────────
    unclassified   = []   # files that returned None from both detectors
    excluded_known = []   # files of known but excluded non-AD document types

    for fname in txt_files:

        # Extract the meeting date from the filename
        m = DATE_PATTERN.search(fname)
        if not m:
            continue

        d = parse_date_yyyy_mm_dd(m.group(1))
        if d is None or d < start_date:
            continue

        full_path = os.path.join(TXT_FOLDER, fname)
        doc_type  = detect_doc_type(full_path)

        # ── Strict filter: only AD document types proceed to merge ────────────
        if doc_type not in ALLOWED_TYPES:
            if doc_type is None:
                unclassified.append(fname)              # silent failure candidate
            else:
                excluded_known.append((fname, doc_type))  # known non-AD type
            continue

        files_by_date[m.group(1)].append((full_path, doc_type))

    # ── Write merged meeting files ────────────────────────────────────────────
    for date_str, file_type_list in sorted(files_by_date.items()):

        merged_path = os.path.join(MERGED_FOLDER, f"{date_str}.txt")

        # Sort documents within a meeting by type for consistent ordering
        file_type_list = sorted(file_type_list, key=lambda z: z[1])

        with open(merged_path, "w", encoding="utf-8") as out:

            for fpath, doc_type in file_type_list:

                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

                # Write a document header before each constituent document
                out.write(f"==== {doc_type.upper()} | {os.path.basename(fpath)} ====\n\n")
                out.write(text)
                out.write("\n\n\n")

    # ── Console diagnostic report ─────────────────────────────────────────────
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

    # ── Write diagnostic log files ────────────────────────────────────────────

    # Log 1: files that could not be classified by either detector
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

    # Log 2: files of known non-AD document types (excluded as expected)
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
# SECTION 8 — MAIN EXECUTION
# ============================================================
# Runs the two-step pipeline sequentially:
#   Step 1 — Convert all qualifying PDFs to plain text
#   Step 2 — Merge classified text files by FOMC meeting date

if __name__ == "__main__":
    convert_pdfs_to_txt()
    merge_txts_by_meeting_starting_from(START_MEETING_DATE)