"""Step 6b (2-3): BM25, TF-IDF and dense retrieval, fused with Reciprocal Rank Fusion,
plus an optional cross-encoder re-ranker.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config

STOP = set("""a an the of to in for and or on at by with from is are be been being this that these those
it its as into under over per any all each such which who whom whose can may shall should must will
not no only also than then there their they them he she his her you your i we our us was were has
have had do does did up out about above below between scheme schemes applicant applicants
""".split())
TOKEN = re.compile(r"[a-z0-9]+|[ऀ-ॿ]+")  # Latin words or Devanagari runs


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return scores


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
    model.max_seq_length = config.EMBED_MAX_LEN
    return model


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANK_MODEL, device="cpu", max_length=config.RERANK_MAX_LEN)


def embed_passages(texts: list[str], batch_size: int = 32) -> np.ndarray:
    model = get_embedder()
    texts = [config.PASSAGE_PREFIX + t[:1500] for t in texts]
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=len(texts) > 200).astype("float32")


def embed_queries(texts: list[str]) -> np.ndarray:
    model = get_embedder()
    return model.encode([config.QUERY_PREFIX + t for t in texts], normalize_embeddings=True).astype("float32")


class HybridIndex:
    """Keeps BM25, TF-IDF and dense indexes over the same list of documents."""

    def __init__(self, texts: list[str], embeddings: np.ndarray | None = None):
        import faiss
        self.texts = texts
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                     ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.tfidf_matrix = self.tfidf.fit_transform(texts)
        self.emb = embeddings if embeddings is not None else embed_passages(texts)
        self.faiss = faiss.IndexFlatIP(self.emb.shape[1])
        self.faiss.add(self.emb)

    # Each search returns document positions, best first.
    def search_bm25(self, query: str, k: int = 100) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [int(i) for i in top if scores[i] > 0]

    def search_tfidf(self, query: str, k: int = 100) -> list[int]:
        q = self.tfidf.transform([query])
        scores = (self.tfidf_matrix @ q.T).toarray().ravel()
        top = np.argsort(-scores)[:k]
        return [int(i) for i in top if scores[i] > 0]

    def search_dense(self, query: str, k: int = 100, qvec: np.ndarray | None = None) -> list[int]:
        q = qvec if qvec is not None else embed_queries([query])
        _, ids = self.faiss.search(q.reshape(1, -1), k)
        return [int(i) for i in ids[0] if i >= 0]

    def search_hybrid(self, query: str, k: int = 100, qvec: np.ndarray | None = None) -> list[int]:
        fused = rrf([self.search_bm25(query, k), self.search_dense(query, k, qvec)])
        return [d for d, _ in sorted(fused.items(), key=lambda x: -x[1])][:k]

    def rerank(self, query: str, candidates: list[int], top_n: int | None = None) -> list[tuple[int, float]]:
        top_n = top_n or config.RERANK_TOP_N
        cand = candidates[:top_n]
        if not cand:
            return []
        pairs = [(query, self.texts[i][:1200]) for i in cand]
        scores = get_reranker().predict(pairs, batch_size=16, show_progress_bar=False)
        order = np.argsort(-scores)
        reranked = [(cand[j], float(scores[j])) for j in order]
        rest = [(i, float("-inf")) for i in candidates[top_n:]]
        return reranked + rest
