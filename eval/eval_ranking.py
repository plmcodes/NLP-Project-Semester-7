"""Step 8: evaluate retrieval / recommendation.

Tests that need no hand labels (run now):
  1. Known-item search: 150 schemes; the query is the scheme's description with
     its name removed; the index does NOT contain descriptions. Which system
     ranks the right scheme highest?  (ablation: keyword -> TF-IDF -> BM25 ->
     dense -> hybrid -> hybrid + re-rank)
  2. Hindi/English consistency: 20 paired queries; overlap of top-10 results.
  3. Eligibility filter: 200 random personas; agreement of our filter with a
     filter built from the dataset's machine-parsed fields.

Test that needs the team's hand labels:
  4. data/test_sets/queries.csv (query, relevant_ids separated by ';').
     `python -m eval.eval_ranking --pool` writes candidate lists to judge.

Run:  python -m eval.eval_ranking
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src import config
from src.recommend.eligibility import NOT_ELIGIBLE, Profile, check
from src.recommend.engine import scheme_embeddings
from src.recommend.retrieval import HybridIndex, embed_queries, tokenize

QUERIES = config.TEST_SETS / "queries.csv"
QUERY_TEMPLATE = config.TEST_SETS / "queries_template.csv"
POOL = config.TEST_SETS / "queries_pool.csv"

# Paired English / Hindi needs (written for this project; not labelled).
PAIRS = [
    ("scholarship for SC students in college", "कॉलेज में अनुसूचित जाति के छात्रों के लिए छात्रवृत्ति"),
    ("pension for widows", "विधवा महिलाओं के लिए पेंशन"),
    ("loan to start a small business", "छोटा व्यवसाय शुरू करने के लिए ऋण"),
    ("financial help for farmers to buy seeds", "किसानों को बीज खरीदने के लिए आर्थिक सहायता"),
    ("free house for poor families", "गरीब परिवारों के लिए मुफ्त घर"),
    ("money for girl child education", "बालिका की शिक्षा के लिए पैसे"),
    ("pension for old age people", "बुजुर्ग लोगों के लिए पेंशन"),
    ("help for persons with disability", "दिव्यांग व्यक्तियों के लिए सहायता"),
    ("marriage assistance for daughters", "बेटी की शादी के लिए सहायता"),
    ("health insurance for poor families", "गरीब परिवारों के लिए स्वास्थ्य बीमा"),
    ("skill training for unemployed youth", "बेरोजगार युवाओं के लिए कौशल प्रशिक्षण"),
    ("subsidy for solar pump for irrigation", "सिंचाई के लिए सोलर पंप पर सब्सिडी"),
    ("support for pregnant women", "गर्भवती महिलाओं के लिए सहायता"),
    ("accident insurance for workers", "मजदूरों के लिए दुर्घटना बीमा"),
    ("fellowship for PhD research", "पीएचडी शोध के लिए फेलोशिप"),
    ("assistance for fishermen during ban period", "मछली पकड़ने पर रोक के दौरान मछुआरों को सहायता"),
    ("sports training for athletes", "खिलाड़ियों के लिए खेल प्रशिक्षण"),
    ("cow and buffalo purchase subsidy", "गाय और भैंस खरीदने पर सब्सिडी"),
    ("coaching for competitive exams for minority students", "अल्पसंख्यक छात्रों के लिए प्रतियोगी परीक्षा कोचिंग"),
    ("e-rickshaw or vehicle loan for drivers", "ड्राइवरों के लिए ई-रिक्शा या वाहन ऋण"),
]

EXAMPLE_QUERIES = [p[0] for p in PAIRS] + [
    "I am a 45 year old farmer in Maharashtra and need money for a tube well",
    "scholarship for engineering student whose family earns less than 2 lakh",
    "my mother is a widow in Uttar Pradesh, what pension can she get",
    "startup funding for women entrepreneurs",
    "help for street vendors to get working capital",
    "free coaching for UPSC for OBC students",
    "subsidy for building a toilet in a village",
    "support for artisans and weavers to sell products",
    "insurance for crop loss due to flood",
    "reward for students who score high marks in class 10",
]


def metrics(ranks: list[int | None]) -> dict:
    rr = np.array([1 / r if r else 0.0 for r in ranks])
    nd = np.array([1 / np.log2(r + 1) if r and r <= 10 else 0.0 for r in ranks])
    return {"MRR": round(float(rr.mean()), 4),
            "Recall@1": round(float(np.mean([bool(r) and r <= 1 for r in ranks])), 4),
            "Recall@10": round(float(np.mean([bool(r) and r <= 10 for r in ranks])), 4),
            "nDCG@10": round(float(nd.mean()), 4)}


def rank_of(target: int, ranking: list[int]) -> int | None:
    try:
        return ranking.index(target) + 1
    except ValueError:
        return None


def keyword_name_search(names_tok, query, k=100):
    q = set(tokenize(query))
    scores = np.array([len(q & n) for n in names_tok], dtype=float)
    top = np.argsort(-scores, kind="stable")[:k]
    return [int(i) for i in top if scores[i] > 0]


def systems(index: HybridIndex, names_tok, query, qvec):
    hyb = index.search_hybrid(query, 100, qvec)
    return {
        "1 keyword (scheme names)": keyword_name_search(names_tok, query),
        "2 TF-IDF": index.search_tfidf(query),
        "3 BM25": index.search_bm25(query),
        "4 dense (e5)": index.search_dense(query, 100, qvec),
        "5 BM25 + dense (RRF)": hyb,
        "6 hybrid + cross-encoder re-rank": [i for i, _ in index.rerank(query, hyb, top_n=10)],
    }


def known_item(df: pd.DataFrame, n=150, seed=7) -> dict:
    texts = df.search_text_nodesc.tolist()
    index = HybridIndex(texts, scheme_embeddings(df))
    names_tok = [set(tokenize(x)) for x in df.name]
    ok = df[df.description.str.split().map(len) >= 15]
    sample = ok.sample(n=n, random_state=seed)
    queries = []
    for pos, row in sample.iterrows():
        q = re.sub(re.escape(row["name"]), " ", row.description, flags=re.I)
        q = re.sub(r'"[^"]{3,80}"', " ", q)            # quoted scheme names
        q = " ".join(q.split()[:25])
        queries.append((pos, q))
    qvecs = embed_queries([q for _, q in queries])
    ranks: dict[str, list] = {}
    for (pos, q), qv in zip(queries, qvecs):
        for name, ranking in systems(index, names_tok, q, qv).items():
            ranks.setdefault(name, []).append(rank_of(pos, ranking))
    table = {name: metrics(r) for name, r in ranks.items()}
    rr = {name: np.array([1 / x if x else 0.0 for x in r]) for name, r in ranks.items()}
    tests = {}
    for a, b in [("3 BM25", "5 BM25 + dense (RRF)"), ("4 dense (e5)", "5 BM25 + dense (RRF)"),
                 ("5 BM25 + dense (RRF)", "6 hybrid + cross-encoder re-rank"), ("2 TF-IDF", "3 BM25")]:
        d = rr[a] - rr[b]
        tests[f"{a} vs {b}"] = ({"p_value": float(wilcoxon(rr[a], rr[b]).pvalue),
                                  "mean_rr_diff": round(float(d.mean()), 4)} if np.any(d) else {"note": "identical"})
    examples = [{"query": q, "target": df.name.iat[pos]} for pos, q in queries[:3]]
    return {"n_queries": n, "results": table, "wilcoxon_reciprocal_rank": tests, "example_queries": examples}


def hindi_consistency(df: pd.DataFrame) -> dict:
    texts = df.search_text.tolist()
    index = HybridIndex(texts, scheme_embeddings(df))
    out = {"BM25": [], "dense (e5)": [], "BM25 + dense (RRF)": []}
    for en, hi in PAIRS:
        ve, vh = embed_queries([en, hi])
        for name, fn in (("BM25", lambda q, v: index.search_bm25(q, 10)),
                         ("dense (e5)", lambda q, v: index.search_dense(q, 10, v)),
                         ("BM25 + dense (RRF)", lambda q, v: index.search_hybrid(q, 10, v))):
            a, b = set(fn(en, ve)[:10]), set(fn(hi, vh)[:10])
            out[name].append(len(a & b) / 10)
    return {"n_pairs": len(PAIRS), "mean_overlap_at_10": {k: round(float(np.mean(v)), 3) for k, v in out.items()}}


def silver_rules(r: dict) -> dict:
    nz = lambda v: None if pd.isna(v) else float(v)
    return {"age_min": nz(r["silver_age_min"]), "age_max": nz(r["silver_age_max"]),
            "income_max": nz(r["silver_income_max"]), "gender": r["silver_gender"],
            "castes": list(r["silver_caste"]), "states": list(r["silver_states"]) or ["All"],
            "residence": r["silver_residence"], "disability": bool(r["silver_disability"]),
            "bpl": bool(r["silver_bpl"])}


def ext_rules(r: dict) -> dict:
    nz = lambda v: None if pd.isna(v) else float(v)
    return {"age_min": nz(r["ext_age_min"]), "age_max": nz(r["ext_age_max"]),
            "income_max": nz(r["ext_income_max"]), "gender": r["ext_gender"],
            "castes": list(r["ext_castes"]), "states": list(r["ext_states"]),
            "residence": r["ext_residence"], "disability": bool(r["ext_disability"]), "bpl": bool(r["ext_bpl"])}


def random_personas(n=200, seed=11):
    rng = np.random.default_rng(seed)
    from src.extraction.rules import STATES
    out = []
    for _ in range(n):
        out.append(Profile(
            age=int(rng.integers(5, 81)), gender=str(rng.choice(["female", "male"])),
            state=str(rng.choice(STATES)), residence=str(rng.choice(["rural", "urban"], p=[0.65, 0.35])),
            income=int(rng.choice([40_000, 80_000, 150_000, 250_000, 400_000, 700_000, 1_200_000])),
            caste=str(rng.choice(["SC", "ST", "OBC", "General"], p=[0.17, 0.09, 0.41, 0.33])),
            disability=bool(rng.random() < 0.05), bpl=bool(rng.random() < 0.2)))
    return out


def filter_agreement(df: pd.DataFrame) -> dict:
    recs = df.to_dict("records")
    ours = [ext_rules(r) for r in recs]
    silver = [silver_rules(r) for r in recs]
    P, R, J, kept = [], [], [], []
    for p in random_personas():
        a = {i for i, s in enumerate(ours) if check(s, p).status != NOT_ELIGIBLE}
        b = {i for i, s in enumerate(silver) if check(s, p).status != NOT_ELIGIBLE}
        inter = len(a & b)
        P.append(inter / len(a) if a else 1.0)
        R.append(inter / len(b) if b else 1.0)
        J.append(inter / len(a | b) if a | b else 1.0)
        kept.append(len(a) / len(df))
    return {"n_personas": 200,
            "precision_vs_silver": round(float(np.mean(P)), 3), "recall_vs_silver": round(float(np.mean(R)), 3),
            "jaccard": round(float(np.mean(J)), 3), "share_of_schemes_kept": round(float(np.mean(kept)), 3)}


def labelled_queries(df: pd.DataFrame) -> dict | None:
    if not QUERIES.exists():
        return None
    q = pd.read_csv(QUERIES).dropna(subset=["relevant_ids"])
    texts = df.search_text.tolist()
    index = HybridIndex(texts, scheme_embeddings(df))
    pos = {sid: i for i, sid in enumerate(df.id)}
    names_tok = [set(tokenize(x)) for x in df.name]
    per = {}
    for query, rel in zip(q["query"], q["relevant_ids"]):
        relevant = {pos[x.strip()] for x in str(rel).split(";") if x.strip() in pos}
        if not relevant:
            continue
        qv = embed_queries([query])[0]
        for name, ranking in systems(index, names_tok, query, qv).items():
            top5, top10 = ranking[:5], ranking[:10]
            first = next((k + 1 for k, d in enumerate(ranking) if d in relevant), None)
            dcg = sum(1 / np.log2(k + 2) for k, d in enumerate(top10) if d in relevant)
            idcg = sum(1 / np.log2(k + 2) for k in range(min(len(relevant), 10)))
            m = per.setdefault(name, {"P@5": [], "Recall@10": [], "MRR": [], "nDCG@10": []})
            m["P@5"].append(len(set(top5) & relevant) / 5)
            m["Recall@10"].append(len(set(top10) & relevant) / len(relevant))
            m["MRR"].append(1 / first if first else 0)
            m["nDCG@10"].append(dcg / idcg)
    return {"n_queries": len(next(iter(per.values()))["MRR"]) if per else 0,
            "results": {k: {m: round(float(np.mean(v)), 4) for m, v in d.items()} for k, d in per.items()}}


def write_pool(df: pd.DataFrame):
    """Top-10 candidates from three systems per query, for the team to judge."""
    src = pd.read_csv(QUERIES) if QUERIES.exists() else pd.read_csv(QUERY_TEMPLATE)
    texts = df.search_text.tolist()
    index = HybridIndex(texts, scheme_embeddings(df))
    rows = []
    for query in src["query"]:
        qv = embed_queries([query])[0]
        pool = dict.fromkeys(index.search_bm25(query, 10) + index.search_dense(query, 10, qv) + index.search_hybrid(query, 10, qv))
        for i in pool:
            rows.append({"query": query, "id": df.id.iat[i], "name": df.name.iat[i], "state": df.state.iat[i],
                         "summary": df.description.iat[i][:300], "relevant (1/0)": ""})
    pd.DataFrame(rows).to_csv(POOL, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(rows)} query-scheme pairs to judge: {POOL}")


def to_md(report: dict) -> str:
    ki = report["known_item"]
    lines = [f"### Known-item search ({ki['n_queries']} queries, description -> scheme)", "",
             "| System | MRR | Recall@1 | Recall@10 | nDCG@10 |", "|---|---|---|---|---|"]
    for name, m in ki["results"].items():
        lines.append(f"| {name} | {m['MRR']} | {m['Recall@1']} | {m['Recall@10']} | {m['nDCG@10']} |")
    lines += ["", "Wilcoxon signed-rank on reciprocal rank:", ""]
    for k, v in ki["wilcoxon_reciprocal_rank"].items():
        lines.append(f"- {k}: {v}")
    hc = report["hindi_consistency"]
    lines += ["", f"### Hindi vs English top-10 overlap ({hc['n_pairs']} paired queries)", "",
              "| System | Mean overlap@10 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in hc["mean_overlap_at_10"].items()]
    fa = report["filter_agreement"]
    lines += ["", f"### Eligibility filter vs dataset's parsed fields ({fa['n_personas']} random personas)", "",
              f"- Precision {fa['precision_vs_silver']}, recall {fa['recall_vs_silver']}, Jaccard {fa['jaccard']}",
              f"- Share of schemes kept per persona: {fa['share_of_schemes_kept']}"]
    if report.get("labelled_queries"):
        lq = report["labelled_queries"]
        lines += ["", f"### Hand-labelled queries ({lq['n_queries']})", "",
                  "| System | P@5 | Recall@10 | MRR | nDCG@10 |", "|---|---|---|---|---|"]
        for name, m in lq["results"].items():
            lines.append(f"| {name} | {m['P@5']} | {m['Recall@10']} | {m['MRR']} | {m['nDCG@10']} |")
    else:
        lines += ["", "_No hand-labelled queries yet (data/test_sets/queries.csv)._"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="store_true", help="write candidate lists for hand labelling")
    args = ap.parse_args()
    df = pd.read_parquet(config.EXTRACTED_PARQUET).reset_index(drop=True)
    if not QUERY_TEMPLATE.exists():
        pd.DataFrame({"query": EXAMPLE_QUERIES, "relevant_ids": ""}).to_csv(QUERY_TEMPLATE, index=False, encoding="utf-8-sig")
    if args.pool:
        write_pool(df)
        return
    report = {"known_item": known_item(df), "hindi_consistency": hindi_consistency(df),
              "filter_agreement": filter_agreement(df), "labelled_queries": labelled_queries(df)}
    (config.REPORTS / "ranking_eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = to_md(report)
    (config.REPORTS / "ranking_eval.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
