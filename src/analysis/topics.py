"""Step 5: topic modelling of scheme text.

Fits LDA (counts) and NMF (TF-IDF) for several topic counts, scores each
with UMass coherence, keeps the best NMF model and reports its topics.

Run:  python -m src.analysis.topics
"""
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src import config

DOMAIN_STOP = """scheme schemes applicant applicants beneficiary beneficiaries government state central
department shall eligible eligibility rs lakh year years month per apply application provided provide
amount financial assistance benefit benefits under shall must should also following etc available""".split()


def umass(top_words, X_bin, vocab_index):
    """Mean UMass coherence over word pairs, using document co-occurrence."""
    scores = []
    for words in top_words:
        ids = [vocab_index[w] for w in words if w in vocab_index]
        for i in range(1, len(ids)):
            for j in range(i):
                wi, wj = X_bin[:, ids[i]], X_bin[:, ids[j]]
                co = wi.multiply(wj).sum()
                scores.append(np.log((co + 1) / max(wj.sum(), 1)))
    return float(np.mean(scores))


def top_words(components, vocab, n=10):
    return [[vocab[i] for i in comp.argsort()[::-1][:n]] for comp in components]


def main():
    df = pd.read_parquet(config.EXTRACTED_PARQUET)
    texts = (df.description + " " + df.eligibility_text + " " + df.benefits).tolist()
    stop = list(set(TfidfVectorizer(stop_words="english").get_stop_words()) | set(DOMAIN_STOP))
    common = dict(max_df=0.5, min_df=5, stop_words=stop, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b")

    tfidf = TfidfVectorizer(**common, sublinear_tf=True)
    Xt = tfidf.fit_transform(texts)
    counts = CountVectorizer(**common)
    Xc = counts.fit_transform(texts)
    vocab_t, vocab_c = tfidf.get_feature_names_out(), counts.get_feature_names_out()
    Xbin = (Xc > 0).astype(np.int32).tocsc()
    vidx = {w: i for i, w in enumerate(vocab_c)}

    results, best = [], None
    for k in (8, 12, 15, 20):
        nmf = NMF(n_components=k, random_state=42, init="nndsvda", max_iter=400).fit(Xt)
        lda = LatentDirichletAllocation(n_components=k, random_state=42, learning_method="batch", max_iter=30).fit(Xc)
        tn, tl = top_words(nmf.components_, vocab_t), top_words(lda.components_, vocab_c)
        row = {"k": k, "nmf_umass": round(umass(tn, Xbin, vidx), 3), "lda_umass": round(umass(tl, Xbin, vidx), 3)}
        results.append(row)
        print(row)
        if best is None or row["nmf_umass"] > best[0]:
            best = (row["nmf_umass"], k, nmf, tn)

    _, k, nmf, words = best
    W = nmf.transform(Xt)
    dominant = W.argmax(axis=1)
    topics = []
    for t in range(k):
        mask = dominant == t
        topics.append({
            "topic": t, "top_words": words[t], "n_schemes": int(mask.sum()),
            "share_central": round(float((df.level[mask] == "Central").mean()), 3) if mask.any() else 0,
            "example_schemes": df.name[mask].head(3).tolist(),
            "top_states": df.state[mask].value_counts().head(3).to_dict(),
        })
    pd.DataFrame({"id": df.id, "topic": dominant}).to_parquet(config.DATA_PROC / "topics.parquet", index=False)
    out = {"model_selection": results, "chosen": {"model": "NMF", "k": k}, "topics": topics}
    (config.REPORTS / "topics.json").write_text(json.dumps(out, indent=2))
    print(f"Chose NMF with k={k}")
    for t in topics:
        print(t["topic"], t["n_schemes"], ", ".join(t["top_words"][:8]))


if __name__ == "__main__":
    main()
