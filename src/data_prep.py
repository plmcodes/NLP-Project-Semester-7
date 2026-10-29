"""Step 3/4a: load the raw scheme CSV, clean the text and save a tidy parquet file.

Run:  python -m src.data_prep
"""
import html
import json
import re

import ftfy
import pandas as pd

from src.config import CATEGORIES, RAW_CSV, SCHEMES_PARQUET

TEXT_COLS = ["name", "description", "benefits", "eligibility_text",
             "application_process", "documents_required"]


def clean_text(s) -> str:
    """Remove HTML/markdown noise but keep one item per line."""
    if not isinstance(s, str) or s.strip().lower() == "null":
        return ""
    s = ftfy.fix_text(html.unescape(s))
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)                 # other tags
    s = s.replace("**", "").replace("__", "")
    lines = []
    for line in s.splitlines():
        line = re.sub(r"^\s*(>+|[-*•]|\d+[.)])\s*", "", line)  # quote / bullet / "1."
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def parse_categories(raw) -> list[str]:
    if not isinstance(raw, str):
        return []
    return [c for c in CATEGORIES if c in raw]


def parse_json_list(raw) -> list[str]:
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else []
    except (TypeError, ValueError):
        return []


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV, na_values=["null"], keep_default_na=True, encoding="utf-8")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["id"] = df["slug"]
    for c in TEXT_COLS:
        out[c] = df[c].map(clean_text)
    out["name"] = out["name"].str.strip(' "')
    out["ministry"] = df["ministry"].fillna("")
    out["department"] = df["department"].fillna("")
    out["state"] = df["state"].fillna("Central")
    out["level"] = out["state"].map(lambda s: "Central" if s == "Central" else "State")
    out["categories"] = df["category"].map(parse_categories)
    out["beneficiary_type"] = df["beneficiary_type"].fillna("")
    out["apply_url"] = df["apply_url"].fillna("")
    out["official_url"] = df["official_url"].fillna("")
    out["scraped_at"] = df["scraped_at"].astype(str).str[:10]

    # Machine-parsed fields that ship with the dataset. We keep them as
    # "silver" labels to compare our own extractor against (never as truth).
    out["silver_age_min"] = pd.to_numeric(df["eligibility_age_min"], errors="coerce")
    out["silver_age_max"] = pd.to_numeric(df["eligibility_age_max"], errors="coerce")
    out["silver_income_max"] = pd.to_numeric(df["eligibility_income_max"], errors="coerce")
    out["silver_gender"] = df["eligibility_gender"].fillna("all")
    out["silver_caste"] = df["eligibility_caste"].map(parse_json_list)
    out["silver_residence"] = df["eligibility_residence"].fillna("both")
    out["silver_states"] = df["eligibility_state"].map(parse_json_list)
    out["silver_disability"] = df["eligibility_disability"].astype(str).str.lower().eq("true")
    out["silver_bpl"] = df["eligibility_bpl"].astype(str).str.lower().eq("true")

    # Full text for keyword search (BM25 / TF-IDF) and re-ranking in the app.
    out["search_text"] = (out["name"] + ". " + out["description"] + "\n"
                          + out["benefits"] + "\n" + out["eligibility_text"])
    # Text without the description. It is embedded once and reused by the app, the
    # classifier and the known-item test (whose queries come from descriptions, so
    # descriptions must not be indexed there). Categories are left out of both
    # texts because they are the classifier's labels.
    out["search_text_nodesc"] = (out["name"] + ".\n" + out["benefits"] + "\n" + out["eligibility_text"])

    before = len(out)
    out = out.drop_duplicates("id").reset_index(drop=True)
    out.attrs["dropped_duplicates"] = before - len(out)
    return out


def main():
    df = prepare(load_raw())
    df.to_parquet(SCHEMES_PARQUET, index=False)
    print(f"Saved {len(df)} schemes to {SCHEMES_PARQUET} "
          f"(dropped {df.attrs['dropped_duplicates']} duplicate ids)")
    print(f"Empty eligibility text: {(df.eligibility_text == '').sum()}")
    print(f"No category: {(df.categories.map(len) == 0).sum()}")


if __name__ == "__main__":
    main()
