"""Apply the rule extractor to every scheme and save the structured fields.

Run:  python -m src.extraction.run_extraction
"""
import json

import pandas as pd

from src.config import EXTRACTED_PARQUET, SCHEMES_PARQUET
from src.extraction.rules import extract


def main():
    df = pd.read_parquet(SCHEMES_PARQUET)
    rows = [extract(t, s, n).to_dict() for t, s, n in zip(df.eligibility_text, df.state, df.name)]
    ext = pd.DataFrame(rows)
    ext["evidence"] = ext["evidence"].map(json.dumps)
    ext.columns = ["ext_" + c for c in ext.columns]
    out = pd.concat([df, ext], axis=1)
    out.to_parquet(EXTRACTED_PARQUET, index=False)
    print(f"Extracted fields for {len(out)} schemes -> {EXTRACTED_PARQUET}")
    for c in ["ext_age_min", "ext_age_max", "ext_income_max"]:
        print(f"  {c}: found in {out[c].notna().sum()} schemes")
    print(f"  ext_gender female: {(out.ext_gender == 'female').sum()}")
    print(f"  ext_castes non-empty: {(out.ext_castes.map(len) > 0).sum()}")


if __name__ == "__main__":
    main()
