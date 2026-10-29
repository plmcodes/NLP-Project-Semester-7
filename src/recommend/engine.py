"""Step 6: the hybrid recommender.

score(s, u) = alpha * relevance + beta * (rules matched / rules known) + gamma * boost
"""
from __future__ import annotations

import json
from dataclasses import asdict
from functools import lru_cache

import numpy as np
import pandas as pd

from src import config
from src.recommend.eligibility import ELIGIBLE, NOT_ELIGIBLE, POSSIBLE, Profile, check
from src.recommend.profile_parser import parse_need
from src.recommend.retrieval import HybridIndex, embed_passages

ALPHA, BETA, GAMMA = 0.6, 0.3, 0.1
SUMMARIES = config.DATA_PROC / "summaries.parquet"
LABEL_SECTORS = config.DATA_PROC / "predicted_sectors.parquet"


def _clean(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def load_embeddings(ids: list[str], texts: list[str], name: str) -> np.ndarray:
    """Cache embeddings on disk so the app starts in seconds after the first run."""
    tag = config.EMBED_MODEL.replace("/", "__")
    path = config.INDEX_DIR / f"{name}_{tag}.npy"
    ids_path = config.INDEX_DIR / f"{name}_{tag}.ids.json"
    if path.exists() and ids_path.exists() and json.loads(ids_path.read_text()) == ids:
        return np.load(path)
    emb = embed_passages(texts)
    np.save(path, emb)
    ids_path.write_text(json.dumps(ids))
    return emb


def scheme_embeddings(df: pd.DataFrame) -> np.ndarray:
    """One embedding per scheme (name + benefits + eligibility), shared by every step."""
    return load_embeddings(df.id.tolist(), df.search_text_nodesc.tolist(), "schemes")


class Recommender:
    def __init__(self, df: pd.DataFrame | None = None):
        self.df = (df if df is not None else pd.read_parquet(config.EXTRACTED_PARQUET)).reset_index(drop=True)
        if SUMMARIES.exists():
            self.df = self.df.merge(pd.read_parquet(SUMMARIES), on="id", how="left")
        else:
            self.df["summary"] = self.df["description"].str.slice(0, 300)
        self.df["summary"] = self.df["summary"].fillna("")
        self.index = HybridIndex(self.df.search_text.tolist(), scheme_embeddings(self.df))
        self.rules = [self._rules(r) for r in self.df.to_dict("records")]
        self.pos = {sid: i for i, sid in enumerate(self.df.id)}

    @staticmethod
    def _rules(r: dict) -> dict:
        return {
            "age_min": _clean(r["ext_age_min"]), "age_max": _clean(r["ext_age_max"]),
            "income_max": _clean(r["ext_income_max"]), "gender": r["ext_gender"],
            "castes": list(r["ext_castes"]), "states": list(r["ext_states"]),
            "residence": r["ext_residence"], "disability": bool(r["ext_disability"]),
            "bpl": bool(r["ext_bpl"]), "occupations": list(r["ext_occupations"]),
        }

    def scheme(self, sid: str) -> dict | None:
        i = self.pos.get(sid)
        if i is None:
            return None
        r = self.df.iloc[i]
        keep = ["id", "name", "state", "level", "ministry", "department", "description", "benefits",
                "eligibility_text", "application_process", "documents_required", "official_url",
                "apply_url", "summary", "scraped_at"]
        out = {k: _clean(r[k]) for k in keep}
        out["categories"] = list(r["categories"])
        out["rules"] = self.rules[i]
        out["evidence"] = json.loads(r["ext_evidence"] or "{}")
        return out

    def recommend(self, profile: Profile | dict, need: str = "", k: int = 10, rerank: bool = True,
                  include_possible: bool = True, parse_text: bool = True) -> dict:
        if isinstance(profile, dict):
            profile = Profile.from_dict(profile)
        need = need or profile.need
        parsed = {}
        if parse_text and need:
            parsed = {k2: v for k2, v in parse_need(need).items() if getattr(profile, k2, None) is None}
            for k2, v in parsed.items():
                setattr(profile, k2, v)

        verdicts = [check(r, profile) for r in self.rules]
        allowed = {i for i, v in enumerate(verdicts)
                   if v.status == ELIGIBLE or (include_possible and v.status == POSSIBLE)}

        rel = np.zeros(len(self.df))
        if need.strip():
            cands = [i for i in self.index.search_hybrid(need, k=400) if i in allowed]
            ranked = self.index.rerank(need, cands) if rerank else [(i, 0.0) for i in cands]
            # Relevance decays quickly with rank (1, 0.83, 0.71, ... 0.33 at rank 11) so the
            # best text matches stay on top; a linear scale was too flat and let the
            # rules-matched term dominate.
            for pos, (i, _) in enumerate(ranked):
                rel[i] = 1.0 / (1 + pos / 5)
            pool = [i for i, _ in ranked]
        else:
            pool = list(allowed)

        results = []
        for i in pool:
            v = verdicts[i]
            match = v.matched / v.known if v.known else 0.0
            r = self.rules[i]
            boost = 0.0
            if profile.state and self.df.state.iat[i] == profile.state:
                boost += 0.5
            if profile.occupation and profile.occupation in r["occupations"]:
                boost += 0.5
            score = ALPHA * rel[i] + BETA * match + GAMMA * boost
            results.append((score, i, v, match))
        results.sort(key=lambda x: (-x[0], x[2].status != ELIGIBLE))

        items = []
        for score, i, v, match in results[:k]:
            row = self.df.iloc[i]
            items.append({
                "id": row.id, "name": row["name"], "state": row.state, "level": row.level,
                "categories": list(row.categories), "status": v.status,
                "checks": [asdict(c) for c in v.checks], "score": round(float(score), 4),
                "relevance": round(float(rel[i]), 4), "rule_match": round(match, 3),
                "summary": row.summary, "benefits": row.benefits[:400],
                "official_url": row.official_url, "apply_url": row.apply_url,
            })
        counts = {s: sum(v.status == s for v in verdicts) for s in (ELIGIBLE, POSSIBLE, NOT_ELIGIBLE)}
        return {"profile": {k2: v for k2, v in asdict(profile).items() if v not in (None, "")},
                "parsed_from_text": parsed, "counts": counts, "results": items}


@lru_cache(maxsize=1)
def get_recommender() -> Recommender:
    return Recommender()
