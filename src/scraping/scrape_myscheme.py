"""Step 3: polite scraper for myscheme.gov.in scheme pages.

Scheme pages are rendered with JavaScript and the public sitemap does not list
them, so this uses a headless browser (Playwright) on known scheme URLs:
  * --refresh N   re-scrape N schemes already in the dataset (to keep data current
                  and measure how much the text changed since the dataset was built)
  * --slugs a b   scrape specific scheme slugs (the part after /schemes/ in the URL)

robots.txt allows all paths; we still wait 2.5 s between pages.

Setup:  pip install playwright && python -m playwright install chromium
Run:    python -m src.scraping.scrape_myscheme --refresh 20
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import pandas as pd
from rapidfuzz import fuzz

from src import config
from src.data_prep import clean_text

BASE = "https://www.myscheme.gov.in/schemes/"
OUT = config.DATA_RAW / "myscheme_scrape.jsonl"
HEADINGS = {"Details": "description", "Benefits": "benefits", "Eligibility": "eligibility_text",
            "Exclusions": "exclusions", "Application Process": "application_process",
            "Documents Required": "documents_required"}
STOP = {"Frequently Asked Questions", "Sources And References", "Feedback", "Was this helpful?"}


def parse_page(text: str) -> dict:
    """Split the rendered page text into sections using its headings."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    try:
        start = lines.index("Check Eligibility")
    except ValueError:
        return {}
    # The line above the name is the state for state schemes, the ministry for central ones.
    rec = {"state_or_ministry": lines[start + 1] if start + 1 < len(lines) else "",
           "name": lines[start + 2] if start + 2 < len(lines) else ""}
    current, buf = None, {}
    for line in lines[start + 3:]:
        if line in STOP:
            current = None
            if line == "Sources And References":
                break
            continue
        if line in HEADINGS:
            current = HEADINGS[line]
            buf.setdefault(current, [])
            continue
        if current:
            buf[current].append(line)
    rec.update({k: "\n".join(v) for k, v in buf.items()})
    return rec


def scrape(slugs: list[str], delay: float = 2.5) -> list[dict]:
    from playwright.sync_api import sync_playwright
    out = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent="Mozilla/5.0 (college NLP project; polite scraper)")
        for n, slug in enumerate(slugs, 1):
            try:
                page.goto(BASE + slug, wait_until="networkidle", timeout=60000)
                page.wait_for_selector("text=Eligibility", timeout=15000)
                rec = parse_page(page.inner_text("body"))
                rec.update({"id": slug, "official_url": BASE + slug,
                            "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "ok": bool(rec.get("eligibility_text"))})
            except Exception as e:  # keep going; record the failure
                rec = {"id": slug, "ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}
            out.append(rec)
            print(f"[{n}/{len(slugs)}] {slug}: {'ok' if rec['ok'] else rec.get('error', 'no eligibility section')}", flush=True)
            time.sleep(delay)
        browser.close()
    return out


def compare_with_dataset(recs: list[dict]) -> pd.DataFrame:
    ds = pd.read_parquet(config.SCHEMES_PARQUET).set_index("id")
    rows = []
    for r in recs:
        if not r.get("ok") or r["id"] not in ds.index:
            continue
        old = ds.loc[r["id"], "eligibility_text"]
        new = clean_text(r["eligibility_text"])
        rows.append({"id": r["id"], "name": r.get("name", ""),
                     "eligibility_similarity": round(fuzz.ratio(old, new) / 100, 3),
                     "changed": fuzz.ratio(old, new) < 95})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", type=int, default=0, help="re-scrape N random schemes from the dataset")
    ap.add_argument("--slugs", nargs="*", default=[])
    ap.add_argument("--delay", type=float, default=2.5)
    args = ap.parse_args()
    slugs = list(args.slugs)
    if args.refresh:
        ds = pd.read_parquet(config.SCHEMES_PARQUET)
        slugs += ds.id.sample(args.refresh, random_state=int(time.time()) % 10_000).tolist()
    if not slugs:
        ap.error("give --refresh N or --slugs")
    recs = scrape(slugs, args.delay)
    with OUT.open("a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(r["ok"] for r in recs)
    print(f"Scraped {ok}/{len(recs)} pages -> {OUT}")
    diff = compare_with_dataset(recs)
    if len(diff):
        print(f"Eligibility text changed since the dataset was built: {int(diff.changed.sum())}/{len(diff)}")
        print(diff.to_string(index=False))


if __name__ == "__main__":
    main()
