"""Step 5: corpus statistics for the Insights page and the report.

Run:  python -m src.analysis.insights      (after run_extraction; uses cached embeddings if present)
"""
import json
import re

import matplotlib
import numpy as np
import pandas as pd
import textstat

from src import config

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DOCS = {
    "Aadhaar card": r"aadh?aa?r",
    "Bank account details": r"bank (?:account|passbook|details)|passbook|cancelled cheque",
    "Photograph": r"photo",
    "Income certificate": r"income certificate|income proof",
    "Caste certificate": r"caste certificate",
    "Residence / domicile proof": r"domicile|residen(?:ce|tial) (?:certificate|proof)|address proof",
    "Ration card": r"ration card",
    "Age proof / birth certificate": r"birth certificate|age proof|proof of age|date of birth",
    "Educational certificates": r"mark ?sheet|marks card|educational (?:certificate|qualification)|degree certificate",
    "Disability certificate": r"disability certificate|udid",
    "PAN card": r"\bpan\b",
    "Mobile number": r"mobile number",
}


def barh(series, title, path, xlabel="Schemes"):
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(series))))
    ax.barh(series.index[::-1], series.values[::-1], color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    for i, v in enumerate(series.values[::-1]):
        ax.text(v, i, f" {v}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    df = pd.read_parquet(config.EXTRACTED_PARQUET)
    n = len(df)
    cats = df.categories.explode().value_counts()
    states = df.state.value_counts()
    docs = pd.Series({k: int(df.documents_required.str.contains(p, flags=re.I, regex=True).sum()) for k, p in DOCS.items()}).sort_values(ascending=False)
    occ = df.ext_occupations.explode().value_counts()

    elig = df.eligibility_text[df.eligibility_text.str.len() > 100]
    flesch = elig.map(textstat.flesch_reading_ease)
    grade = elig.map(textstat.flesch_kincaid_grade)
    words = df.eligibility_text.str.split().map(len)
    income = df.ext_income_max.dropna()

    stats = {
        "n_schemes": n,
        "central": int((df.level == "Central").sum()), "state": int((df.level == "State").sum()),
        "n_states_uts": int(df.state[df.state != "Central"].nunique()),
        "by_category": cats.to_dict(), "by_state_top15": states.head(15).to_dict(),
        "targets": {
            "women_only": int((df.ext_gender == "female").sum()),
            "sc_st": int(df.ext_castes.map(lambda c: "SC" in c or "ST" in c).sum()),
            "obc": int(df.ext_castes.map(lambda c: "OBC" in c).sum()),
            "disability": int(df.ext_disability.sum()), "bpl": int(df.ext_bpl.sum()),
            "has_income_cap": int(len(income)),
            "has_age_rule": int((df.ext_age_min.notna() | df.ext_age_max.notna()).sum()),
        },
        "income_cap_median": int(income.median()) if len(income) else None,
        "income_cap_quartiles": [int(x) for x in income.quantile([0.25, 0.75])] if len(income) else None,
        "occupations_mentioned": occ.head(12).to_dict(),
        "documents": docs.to_dict(),
        "eligibility_words_median": int(words.median()),
        "readability": {"flesch_reading_ease_median": round(float(flesch.median()), 1),
                        "fk_grade_median": round(float(grade.median()), 1),
                        "share_difficult_below_50": round(float((flesch < 50).mean()), 3),
                        "n_texts": int(len(elig))},
    }

    # Near-duplicate schemes (cosine > 0.97 on cached scheme embeddings)
    tag = config.EMBED_MODEL.replace("/", "__")
    emb_path = config.INDEX_DIR / f"schemes_{tag}.npy"
    if emb_path.exists():
        emb = np.load(emb_path)
        ids = json.loads((config.INDEX_DIR / f"schemes_{tag}.ids.json").read_text())
        if ids == df.id.tolist():
            sim = emb @ emb.T
            np.fill_diagonal(sim, 0)
            i, j = np.where(np.triu(sim) > 0.97)
            pairs = sorted(zip(sim[i, j], i, j), reverse=True)
            stats["near_duplicates"] = {"threshold": 0.97, "n_pairs": len(pairs), "examples": [
                {"a": df.name.iat[a], "a_state": df.state.iat[a], "b": df.name.iat[b], "b_state": df.state.iat[b],
                 "cosine": round(float(s), 3)} for s, a, b in pairs[:8]]}

    (config.REPORTS / "insights.json").write_text(json.dumps(stats, indent=2, default=int))

    barh(cats, "Schemes per category", config.FIGURES / "categories.png")
    barh(states.head(15), "Schemes per state (top 15; 'Central' = central schemes)", config.FIGURES / "states.png")
    barh(docs, "Most requested documents", config.FIGURES / "documents.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(flesch.clip(-50, 100), bins=30, color="#4C72B0")
    ax.axvline(50, color="k", ls="--", lw=1)
    ax.set_xlabel("Flesch reading ease (below 50 = difficult, college level)")
    ax.set_ylabel("Schemes")
    ax.set_title("Eligibility text is hard to read")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "readability.png", dpi=150)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("by_category", "by_state_top15")}, indent=1, default=int)[:2500])


if __name__ == "__main__":
    main()
