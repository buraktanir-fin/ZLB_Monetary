import os
import re
import time
import requests
from bs4 import BeautifulSoup
import datetime

BASE_URL = "https://www.federalreserve.gov"

# Folder will be created in the SAME directory as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FOLDER = os.path.join(SCRIPT_DIR, "fomc_pdfs")

os.makedirs(OUT_FOLDER, exist_ok=True)


def parse_meeting_date_from_heading(text):
    """
    Example heading: 'January 30-31 Meeting - 1996'
    Converts to '1996-01-31' (uses last day if a range).
    """
    text = " ".join(str(text).split())  # clean whitespace

    m = re.search(r"([A-Za-z]+)\s+(\d{1,2})(?:-(\d{1,2}))?\s+Meeting\s*-\s*(\d{4})", text)
    if not m:
        return None

    month_name, day1, day2, year = m.groups()
    day = int(day2 or day1)

    try:
        month_num = datetime.datetime.strptime(month_name[:3], "%b").month
    except ValueError:
        return None

    return f"{year}-{month_num:02d}-{day:02d}"


def download_fomc_year(year):
    url = f"{BASE_URL}/monetarypolicy/fomchistorical{year}.htm"
    print(f"\n=== Scanning {year} ===")
    print("URL:", url)

    try:
        resp = requests.get(url, timeout=30)
    except Exception as e:
        print(f"!! Connection error: {e}")
        return

    if resp.status_code != 200:
        print(f"-- Page not found for {year} (HTTP {resp.status_code}).")
        return

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find ALL PDF links on the page
    pdf_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            pdf_links.append(a)

    if not pdf_links:
        print(f"-- No PDF links found for {year}.")
        return

    print(f"PDF links found for {year}: {len(pdf_links)}")

    for a in pdf_links:
        href = a["href"]
        pdf_url = href if href.startswith("http") else BASE_URL + href

        # Find nearest previous heading containing "Meeting - YYYY"
        heading_str = a.find_previous(string=re.compile(r"Meeting\s*-\s*\d{4}"))
        if heading_str:
            date_str = parse_meeting_date_from_heading(heading_str)
        else:
            date_str = None

        if not date_str:
            # Fallback if date cannot be parsed
            date_str = f"{year}-01-01"

        raw_name = pdf_url.split("/")[-1].split("?")[0]
        safe_name = re.sub(r"[^A-Za-z0-9\-_.]", "_", raw_name)

        final_name = f"{date_str}_{safe_name}"
        out_path = os.path.join(OUT_FOLDER, final_name)

        if os.path.exists(out_path):
            print("   (Exists)    ", final_name)
            continue

        try:
            r = requests.get(pdf_url, timeout=60)
            r.raise_for_status()
        except Exception as e:
            print("   !! Download error:", e)
            continue

        with open(out_path, "wb") as f:
            f.write(r.content)

        print("   Downloaded →", final_name)


if __name__ == "__main__":
    for year in range(1982, 2019 + 1):  # 1982–2019 inclusive
        download_fomc_year(year)
        time.sleep(1)  # be polite to the Fed servers

    print(f"\n🎉 Completed: 1982–2019 FOMC PDF download attempts finished.")
