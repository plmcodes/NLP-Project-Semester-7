"""Step 9: REST API.

Run:  uvicorn api.main:app --port 8000      then open http://localhost:8000/docs
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config
from src.recommend import rag
from src.recommend.engine import get_recommender

app = FastAPI(title="Government Scheme Recommender",
              description="NLP system that extracts eligibility rules from Indian government schemes "
                          "and recommends schemes for a citizen. Always verify on the official page.",
              version="1.0")


class ProfileIn(BaseModel):
    age: int | None = Field(None, ge=0, le=120)
    gender: str | None = Field(None, examples=["female"])
    state: str | None = Field(None, examples=["Bihar"])
    residence: str | None = Field(None, examples=["rural"])
    income: int | None = Field(None, ge=0, description="Annual family income in rupees")
    caste: str | None = Field(None, examples=["OBC"])
    disability: bool | None = None
    bpl: bool | None = None
    occupation: str | None = Field(None, examples=["farmer"])


class RecommendIn(BaseModel):
    profile: ProfileIn = ProfileIn()
    need: str = Field("", examples=["I need money to dig a tube well for my farm"])
    k: int = Field(10, ge=1, le=50)
    rerank: bool = False  # did not improve the known-item test; adds ~5 s on a CPU
    include_possible: bool = True


class AskIn(BaseModel):
    question: str = Field(..., min_length=3, examples=["What documents do I need for PM Kisan?"])
    profile: ProfileIn | None = None


@app.on_event("startup")
def warm_up():
    get_recommender()


@app.get("/health")
def health():
    return {"status": "ok", "schemes": len(get_recommender().df), "llm": bool(rag.llm_config())}


@app.post("/recommend")
def recommend(body: RecommendIn):
    return get_recommender().recommend(body.profile.model_dump(), body.need, k=body.k,
                                       rerank=body.rerank, include_possible=body.include_possible)


@app.post("/ask")
def ask(body: AskIn):
    profile = body.profile.model_dump() if body.profile else None
    return rag.answer(get_recommender(), body.question, profile)


@app.get("/scheme/{scheme_id}")
def scheme(scheme_id: str):
    s = get_recommender().scheme(scheme_id)
    if s is None:
        raise HTTPException(404, "Scheme not found")
    return s


@app.get("/stats")
def stats():
    path = config.REPORTS / "insights.json"
    if not path.exists():
        raise HTTPException(503, "Run `python -m src.analysis.insights` first")
    return json.loads(path.read_text())
