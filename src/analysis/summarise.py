"""Step 5: short plain-language summaries with TextRank (extractive).

Each scheme's description + benefits are split into sentences, linked by
TF-IDF cosine similarity, ranked with PageRank; the top 2 sentences are kept
in their original order.

Run:  python -m src.analysis.summarise
"""
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import config
from src.extraction.rules import split_sentences


def textrank(text: str, n: int = 2, max_chars: int = 400) -> str:
    sents = [s for s in split_sentences(text) if len(s) > 30]
    if len(sents) <= n:
        return " ".join(sents)[:max_chars]
    try:
        X = TfidfVectorizer(stop_words="english").fit_transform(sents)
    except ValueError:
        return " ".join(sents[:n])[:max_chars]
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0)
    scores = nx.pagerank(nx.from_numpy_array(sim), max_iter=200)
    top = sorted(sorted(scores, key=scores.get, reverse=True)[:n])
    return " ".join(sents[i] for i in top)[:max_chars]


def main():
    df = pd.read_parquet(config.SCHEMES_PARQUET)
    summaries = [textrank(d + "\n" + b) for d, b in zip(df.description, df.benefits)]
    out = pd.DataFrame({"id": df.id, "summary": summaries})
    out.to_parquet(config.DATA_PROC / "summaries.parquet", index=False)
    src_words = (df.description + " " + df.benefits).str.split().map(len)
    sum_words = out.summary.str.split().map(len)
    print(f"Summaries for {len(out)} schemes; median length {int(sum_words.median())} words "
          f"vs {int(src_words.median())} in the source text")


if __name__ == "__main__":
    main()
