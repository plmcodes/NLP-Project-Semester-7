"""Step 9: web app.  Run:  streamlit run app/streamlit_app.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.extraction.rules import OCCUPATIONS, STATES  # noqa: E402
from src.recommend import rag  # noqa: E402
from src.recommend.engine import Recommender  # noqa: E402

st.set_page_config(page_title="Yojana Finder", page_icon="🏛️", layout="wide")


@st.cache_resource(show_spinner="Loading schemes and models (first run takes a minute)...")
def load():
    return Recommender()


def read_json(name):
    p = config.REPORTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


rec = load()
st.title("Yojana Finder")
st.caption(f"Find government schemes you may qualify for · {len(rec.df):,} central and state schemes from "
           "myScheme (via the SmartDuke Hugging Face dataset, July 2026). Results are suggestions: "
           "always confirm eligibility on the official page. Your details are not stored.")

UNSET = "Prefer not to say"
with st.sidebar:
    st.header("Your profile")
    st.caption("Fill what you like; unknown fields give 'possibly eligible' results.")
    age = st.number_input("Age", min_value=0, max_value=110, value=None, placeholder=UNSET)
    gender = st.selectbox("Gender", [UNSET, "female", "male", "other"])
    state = st.selectbox("State / UT", [UNSET] + STATES)
    residence = st.selectbox("Area", [UNSET, "rural", "urban"])
    income = st.number_input("Annual family income (₹)", min_value=0, step=10000, value=None, placeholder=UNSET)
    caste = st.selectbox("Category", [UNSET, "General", "OBC", "SC", "ST"])
    occupation = st.selectbox("Occupation", [UNSET] + list(OCCUPATIONS))
    disability = st.selectbox("Person with disability", [UNSET, "No", "Yes"])
    bpl = st.selectbox("BPL card", [UNSET, "No", "Yes"])

yn = {"Yes": True, "No": False}
profile = {
    "age": age, "gender": None if gender == UNSET else gender, "state": None if state == UNSET else state,
    "residence": None if residence == UNSET else residence, "income": income,
    "caste": None if caste == UNSET else caste, "occupation": None if occupation == UNSET else occupation,
    "disability": yn.get(disability), "bpl": yn.get(bpl),
}

find_tab, ask_tab, insights_tab = st.tabs(["🔎 Find schemes", "💬 Ask", "📊 Insights"])

ICON = {"pass": "✅", "fail": "❌", "unknown": "❔"}
with find_tab:
    need = st.text_area("What do you need? (English or Hindi, optional)",
                        placeholder="e.g. I am a 45 year old farmer in Bihar and need money for a tube well / "
                                    "मुझे बेटी की पढ़ाई के लिए छात्रवृत्ति चाहिए")
    c1, c2, c3 = st.columns([1, 1, 2])
    k = c1.slider("Results", 5, 30, 10)
    use_rerank = c2.toggle("Cross-encoder re-rank (slower)", value=False)
    only_eligible = c3.toggle("Hide 'possibly eligible'", value=False)
    if st.button("Find schemes", type="primary"):
        with st.spinner("Searching..."):
            out = rec.recommend(profile, need, k=k, rerank=use_rerank, include_possible=not only_eligible)
        m1, m2, m3 = st.columns(3)
        m1.metric("Eligible", out["counts"]["eligible"])
        m2.metric("Possibly eligible", out["counts"]["possibly eligible"])
        m3.metric("Not eligible", out["counts"]["not eligible"])
        if out["parsed_from_text"]:
            st.info("Understood from your text: " + ", ".join(f"{k2} = {v}" for k2, v in out["parsed_from_text"].items()))
        if not out["results"]:
            st.warning("No matching schemes. Try removing some filters or describing your need differently.")
        for r in out["results"]:
            with st.container(border=True):
                badge = "🟢 Eligible" if r["status"] == "eligible" else "🟡 Possibly eligible"
                st.markdown(f"**{r['name']}**  \n{badge} · {r['state']} · {', '.join(r['categories'])}")
                st.write(r["summary"])
                if r["checks"]:
                    st.caption("  ·  ".join(f"{ICON[c['status']]} {c['detail']}" for c in r["checks"]))
                links = [f"[Official page]({r['official_url']})"] if r["official_url"] else []
                if r["apply_url"]:
                    links.append(f"[Apply]({r['apply_url']})")
                st.markdown(" | ".join(links) + f"  · score {r['score']}")

with ask_tab:
    st.write("Ask about a scheme: benefits, documents, how to apply. Answers come only from scheme text.")
    mode = "LLM: " + rag.llm_config()["model"] if rag.llm_config() else "extractive (no LLM configured)"
    st.caption(f"Answer mode: {mode}")
    use_profile = st.checkbox("Only use schemes my profile could qualify for", value=False)
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    q = st.chat_input("e.g. What documents are needed for a widow pension in Gujarat?")
    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"), st.spinner("Looking it up..."):
            res = rag.answer(rec, q, profile if use_profile else None)
            text = res["answer"]
            if res["sources"]:
                s = res["sources"][0]
                text += f"\n\n**Source:** [{s['scheme']} – official page]({s['url']})"
            if res.get("related"):
                text += "\n\n**Related schemes:** " + " · ".join(f"[{r['scheme']}]({r['url']}) ({r['state']})" for r in res["related"])
            st.markdown(text)
        st.session_state.chat.append({"role": "assistant", "content": text})

with insights_tab:
    ins = read_json("insights.json")
    if not ins:
        st.info("Run `python -m src.analysis.insights` to build this page.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Schemes", f"{ins['n_schemes']:,}")
        b.metric("Central / State", f"{ins['central']:,} / {ins['state']:,}")
        c.metric("Women-only schemes", ins["targets"]["women_only"])
        d.metric("Median income cap", f"₹{ins['income_cap_median']:,}" if ins["income_cap_median"] else "–")
        l, r = st.columns(2)
        l.subheader("Schemes per category")
        l.bar_chart(pd.Series(ins["by_category"]).sort_values(), horizontal=True)
        r.subheader("Most requested documents")
        r.bar_chart(pd.Series(ins["documents"]).sort_values(), horizontal=True)
        rd = ins["readability"]
        st.markdown(f"**Readability:** median Flesch reading ease of eligibility text is "
                    f"**{rd['flesch_reading_ease_median']}** (grade level {rd['fk_grade_median']}); "
                    f"{rd['share_difficult_below_50']:.0%} of texts score below 50 (difficult).")
        topics = read_json("topics.json")
        if topics:
            st.subheader(f"Topics found in scheme text (NMF, k={topics['chosen']['k']})")
            st.dataframe(pd.DataFrame([{"Topic": t["topic"], "Top words": ", ".join(t["top_words"][:8]),
                                        "Schemes": t["n_schemes"], "Example": t["example_schemes"][0] if t["example_schemes"] else ""}
                                       for t in topics["topics"]]), hide_index=True, width="stretch")
        nd = ins.get("near_duplicates")
        if nd:
            st.subheader(f"Near-duplicate schemes ({nd['n_pairs']} pairs with cosine > {nd['threshold']})")
            st.dataframe(pd.DataFrame(nd["examples"]), hide_index=True, width="stretch")
        for name, title in (("ranking_eval.md", "Search evaluation"), ("extraction_eval.md", "Extraction evaluation")):
            p = config.REPORTS / name
            if p.exists():
                with st.expander(title):
                    st.markdown(p.read_text(encoding="utf-8"))
