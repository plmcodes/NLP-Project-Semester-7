"""Step 6c: question answering grounded in scheme text (RAG).

1. Find the 3 most relevant schemes (hybrid search + re-rank, filtered by eligibility
   when a profile is given).
2. Split their sections into sentences and pick the ones closest to the question.
3. If an LLM is configured, it writes the answer from those passages only.
   Otherwise the passages are returned directly (extractive answer), so the app
   works offline with no API key.

LLM config (any OpenAI-compatible endpoint: Groq, Ollama, OpenRouter, ...):
  LLM_BASE_URL   e.g. https://api.groq.com/openai/v1  or  http://localhost:11434/v1
  LLM_MODEL      e.g. llama-3.1-8b-instant            or  llama3.1:8b
  LLM_API_KEY    (not needed for Ollama)
"""
from __future__ import annotations

import os
import re

import numpy as np
import requests

from src.extraction.rules import split_sentences
from src.recommend.eligibility import NOT_ELIGIBLE, Profile, check
from src.recommend.retrieval import embed_passages, embed_queries, tokenize

# Checked in this order: "who can apply" is an eligibility question, not an application one.
SECTIONS = {
    "documents": ("documents_required", r"document|certificate|proof|papers|दस्तावेज|कागज"),
    "eligibility": ("eligibility_text", r"eligib|who can|who is|qualify|criteria|\bage\b|\bincome\b|पात्र|योग्य|कौन"),
    "apply": ("application_process", r"apply|application|register|how to|process|kaise|आवेदन|कैसे"),
    "benefits": ("benefits", r"benefit|amount|money|how much|assistance|subsidy|grant|लाभ|राशि|पैसे"),
}
DEFAULT_SECTIONS = ["description", "benefits", "eligibility_text"]


def detect_language(text: str) -> str:
    dev = len(re.findall(r"[ऀ-ॿ]", text))
    letters = len(re.findall(r"[A-Za-zऀ-ॿ]", text)) or 1
    return "hi" if dev / letters > 0.3 else "en"


GENERIC = set("""what which how much many who whom can get give gives given need needed needs required
benefit benefits document documents apply application eligible eligibility scheme schemes yojana
money amount does is are for the ke liye kaun se chahiye kya hai""".split())


def names_match(question: str, scheme_name: str) -> bool:
    """False when the question names something (e.g. 'Sukanya Samriddhi') that shares no
    word with the matched scheme's name, so the answer should be flagged as a closest match."""
    q = {t for t in tokenize(question) if t not in GENERIC and len(t) > 2}
    return not q or bool(q & set(tokenize(scheme_name)))


def detect_intent(question: str) -> str | None:
    for intent, (_, pat) in SECTIONS.items():
        if re.search(pat, question, re.I):
            return intent
    return None


def llm_config() -> dict | None:
    base, model = os.getenv("LLM_BASE_URL"), os.getenv("LLM_MODEL")
    if not base or not model:
        return None
    return {"base": base.rstrip("/"), "model": model, "key": os.getenv("LLM_API_KEY", "")}


def call_llm(cfg: dict, question: str, passages: list[dict], lang: str, timeout: int = 60) -> str:
    context = "\n\n".join(f"[{i + 1}] {p['scheme']} ({p['section']}): {p['text']}" for i, p in enumerate(passages))
    reply_lang = "Hindi" if lang == "hi" else "English"
    messages = [
        {"role": "system", "content": (
            "You help Indian citizens understand government schemes. Answer ONLY from the numbered "
            "passages. Cite passages like [1]. If the passages do not contain the answer, say you "
            f"don't know and suggest checking the official page. Reply in {reply_lang}, in at most 120 words.")},
        {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
    ]
    headers = {"Content-Type": "application/json"}
    if cfg["key"]:
        headers["Authorization"] = f"Bearer {cfg['key']}"
    r = requests.post(f"{cfg['base']}/chat/completions", headers=headers, timeout=timeout,
                      json={"model": cfg["model"], "messages": messages, "temperature": 0.1})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def answer(rec, question: str, profile: Profile | dict | None = None, n_schemes: int = 3,
           n_passages: int = 5) -> dict:
    if isinstance(profile, dict):
        profile = Profile.from_dict(profile)
    lang = detect_language(question)
    intent = detect_intent(question)

    cands = rec.index.search_hybrid(question, k=100)
    if profile is not None:
        cands = [i for i in cands if check(rec.rules[i], profile).status != NOT_ELIGIBLE]
    top = [i for i, _ in rec.index.rerank(question, cands)[:n_schemes]]
    if not top:
        return {"answer": "I could not find a scheme that answers this. Try rephrasing, or check myscheme.gov.in.",
                "mode": "none", "language": lang, "intent": intent, "sources": [], "related": []}

    # Answer from the best-matching scheme; the next ones are offered as related schemes.
    row = rec.df.iloc[top[0]]
    cols = [SECTIONS[intent][0]] if intent else DEFAULT_SECTIONS
    passages = []
    for col in cols:
        for pos, s in enumerate(split_sentences(row[col] or "")):
            if len(s) > 3:
                passages.append({"scheme_id": row.id, "scheme": row["name"], "section": col,
                                 "url": row.official_url, "text": s[:600], "pos": (cols.index(col), pos)})
    if not passages:  # the asked-for section is empty: fall back to the general sections
        return answer(rec, re.sub(SECTIONS[intent][1], " ", question, flags=re.I), profile, n_schemes, n_passages) \
            if intent else {"answer": "This scheme's page has no text for that. Check the official page.",
                            "mode": "none", "language": lang, "intent": intent, "sources": [], "related": []}

    if intent and len(passages) <= 15:
        best = passages  # short lists (documents, steps): show all of them
    else:
        q = embed_queries([question])[0]
        sims = embed_passages([x["text"] for x in passages], batch_size=64) @ q
        best = [passages[j] for j in np.argsort(-sims)[:n_passages]]
    best.sort(key=lambda x: x["pos"])  # keep the page's own order (steps stay in sequence)

    mode, text = "extractive", None
    cfg = llm_config()
    if cfg:
        try:
            text, mode = call_llm(cfg, question, best, lang), "llm"
        except (requests.RequestException, KeyError, ValueError) as e:
            mode = f"extractive (LLM failed: {type(e).__name__})"
    if text is None:
        section = cols[0].replace("_text", "").replace("_", " ") if intent else "summary"
        lines = [f"**{row['name']}** ({row.state}) - {section}:"]
        if lang == "en" and not names_match(question, row["name"]):
            lines.insert(0, "_Closest match found; the exact scheme you named may not be in the dataset._")
        lines += [f"- {x['text']}" for x in best]
        if lang == "hi":
            lines.insert(0, "(योजना की जानकारी अंग्रेज़ी में उपलब्ध है।)")
        text = "\n".join(lines)

    sources = [{"n": 1, "scheme_id": row.id, "scheme": row["name"], "section": cols[0], "url": row.official_url}]
    related = [{"scheme_id": rec.df.id.iat[i], "scheme": rec.df.name.iat[i], "state": rec.df.state.iat[i],
                "url": rec.df.official_url.iat[i]} for i in top[1:]]
    return {"answer": text, "mode": mode, "language": lang, "intent": intent, "sources": sources, "related": related}
