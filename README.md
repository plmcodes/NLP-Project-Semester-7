# Government Scheme Analysis and Recommendation System

An NLP system that reads 4,693 Indian central and state government schemes, extracts who each
scheme is for (age, income, gender, caste, state, disability, BPL...), analyses the corpus, and
recommends schemes to a citizen from a profile form or a plain-language need in English or Hindi.
A chatbot answers questions using only scheme text and cites its sources.

> Results are suggestions. Always confirm eligibility on the official myScheme page.

## Quick start (Windows, CPU only)

```powershell
cd C:\Users\Admin\Projects\gov-scheme-nlp
.\.venv\Scripts\activate                     # already created; see "Fresh setup" below otherwise
streamlit run app\streamlit_app.py           # web app  -> http://localhost:8501
uvicorn api.main:app --port 8000             # REST API -> http://localhost:8000/docs
```

The first page load takes about a minute (models load once). All processed data, the search
index and the reports are already built. To rebuild everything from the raw CSV (about
40 minutes on this laptop's CPU, most of it computing embeddings once):

```powershell
python run_pipeline.py
```

### Fresh setup on another machine

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# dataset (CC BY 4.0, SmartDuke Technologies, collected July 2026):
curl -L -o data/raw/Schemes.csv https://huggingface.co/datasets/smartduketech/indian-government-schemes-2025/resolve/main/Schemes.csv
python run_pipeline.py
```

Optional scraper: `pip install playwright && python -m playwright install chromium`, then
`python -m src.scraping.scrape_myscheme --refresh 20`.

### Optional: use an LLM for chatbot answers

By default the chatbot is *extractive*: it returns the most relevant sentences from scheme pages
with citations, and needs no API key. To have an LLM write the answer (from those same passages
only), set three environment variables for any OpenAI-compatible endpoint:

```powershell
# Groq (free tier):
$env:LLM_BASE_URL="https://api.groq.com/openai/v1"; $env:LLM_MODEL="llama-3.1-8b-instant"; $env:LLM_API_KEY="<your key>"
# or local Ollama:
$env:LLM_BASE_URL="http://localhost:11434/v1"; $env:LLM_MODEL="llama3.1:8b"
```

## How it works

| Step | Module | What it does |
|---|---|---|
| 3 Data | `src/data_prep.py` | Cleans HTML/markdown, parses categories, keeps the dataset's own parsed fields as *silver* labels |
| 3 Scraping | `src/scraping/scrape_myscheme.py` | Headless-browser scraper for myScheme pages; measures how much text changed since the dataset |
| 4 Extraction | `src/extraction/rules.py` | Regex + keyword rules turn eligibility text into fields; keeps the evidence sentence |
| 5 Analysis | `src/analysis/classify.py` | Sector classifier: TF-IDF + LR vs multilingual-e5 embeddings + LR |
| 5 Analysis | `src/analysis/topics.py` | LDA vs NMF topic models, chosen by UMass coherence |
| 5 Analysis | `src/analysis/summarise.py` | TextRank 2-sentence summaries |
| 5 Analysis | `src/analysis/insights.py` | Corpus statistics, readability, required documents, near-duplicates, figures |
| 6 Recommend | `src/recommend/eligibility.py` | Rule check per profile: eligible / possibly eligible / not eligible, with reasons |
| 6 Recommend | `src/recommend/retrieval.py` | BM25, TF-IDF, dense (FAISS), Reciprocal Rank Fusion, cross-encoder re-ranking |
| 6 Recommend | `src/recommend/engine.py` | score = 0.6 relevance + 0.3 rules matched + 0.1 boost (home state, occupation) |
| 6 Chatbot | `src/recommend/rag.py` | Retrieve schemes, pick passages, extractive or LLM answer with citations |
| 7 Hindi | `src/recommend/profile_parser.py`, e5 model | Hindi needs are parsed and searched with a multilingual embedding model |
| 8 Evaluation | `eval/` | Extraction agreement, known-item search ablation, Hindi consistency, filter agreement |
| 9 App | `api/main.py`, `app/streamlit_app.py` | FastAPI (`/recommend`, `/ask`, `/scheme/{id}`, `/stats`) and a Streamlit UI |

Models (all run on CPU): `intfloat/multilingual-e5-small` (embeddings, 118M parameters) and
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual re-ranker). Change them with the
`EMBED_MODEL` / `RERANK_MODEL` environment variables (for example `BAAI/bge-m3` on a GPU).

## Results (full tables in `reports/RESULTS.md`, figures in `reports/figures/`)

| What | Result |
|---|---|
| Corpus | 4,693 schemes (684 central, 4,009 state, 36 states/UTs). 66% of eligibility texts are "difficult" to read (Flesch < 50; median grade level 10.9) |
| Eligibility extraction (100 labelled schemes) | Our rules, F1: income cap 0.96, age max 0.83, age min 0.74, women-only 0.80, caste 0.82, disability 0.73, BPL 0.86. The dataset's own parsed fields score higher on most fields (caste 0.94, age min 0.97, women-only 1.00) |
| Sector classifier (15 classes) | TF-IDF + LR micro-F1 **0.794** vs multilingual-e5 + LR 0.703; difference significant (Wilcoxon p ≈ 2e-14) |
| Topics | NMF with 8 topics: students, construction workers, business loans, pensions/BPL, farmers, research fellowships, disability, marriage/girl child |
| Known-item search (150 queries) | MRR: keyword 0.19 → TF-IDF 0.50 → BM25 0.56 → dense 0.50 → **hybrid 0.57** (Recall@10 0.78). Hybrid beats dense alone (p = 0.0001) but not BM25 (p = 0.15). Cross-encoder re-rank did not help (0.54, p = 0.21) |
| Labelled citizen queries (29) | P@5 / MRR / nDCG@10: keyword 0.47 / 0.66 / 0.46 → BM25 0.60 / 0.79 / 0.65 → dense 0.61 / 0.82 / 0.68 → **hybrid 0.66 / 0.88 / 0.69**. Re-rank adds nothing (0.63 / 0.88 / 0.69). With realistic queries, dense search catches up with BM25, unlike in the known-item test |
| Hindi queries (20 pairs) | Top-10 overlap with the English version: BM25 0.00, dense 0.46, hybrid 0.38, so multilingual embeddings are what make Hindi search work |
| Eligibility filter (200 personas) | Agrees with a filter built on the dataset's parsed fields: precision 0.94, recall 0.86; keeps about 11% of schemes per person |
| Scraper | 5/5 live myScheme pages scraped; eligibility text unchanged since the dataset (July 2026) |

### Limitations (say these in the viva before the examiner does)

- Extraction accuracy comes from only 100 labelled schemes (`data/test_sets/extraction_gold.csv`),
  so each figure has a wide margin of error. The eligibility-filter numbers still have no
  hand-labelled ground truth.
- Search relevance was judged on a pool of each system's top 10 (`data/test_sets/queries_guidelines.md`),
  by one judge, on 29 queries. Relevant schemes no system found are mostly missing, so recall is
  an upper estimate. The Maharashtra tube-well query has no matching scheme in the dataset.
- The dataset's own parsed fields beat our rules on most fields. Common rule errors: "not more
  than 30 years" read as a minimum age; schemes that only mention a widow, daughter or women's
  group as one option marked women-only; groups mentioned as alternatives ("SC, ST or PwD") read as requirements.
- The known-item test takes queries from scheme descriptions, so it favours keyword overlap
  (BM25) more than real citizen queries would.
- Rules miss unusual phrasing and sometimes read a benefit for a group as a requirement
  ("women get 10% extra" is not women-only). The app labels such schemes "possibly eligible" and
  always links the official page.
- Slow-CPU settings: embeddings read the first 256 tokens of each scheme; re-ranking is off by default.
- Coverage depends on the dataset: for example, few Bihar irrigation schemes, and no Sukanya
  Samriddhi (the chatbot says when it only found a closest match).

## What the team still needs to do

1. **Review the extraction gold set** (done 2026-11-22: `data/test_sets/extraction_gold.csv`,
   rules in `extraction_gold_guidelines.md`). Check the rows with a note; after any change rerun
   `python -m eval.eval_extraction` and `python -m eval.make_results`.
2. **Review the search relevance labels** (done 2026-11-22: `data/test_sets/queries.csv`, every
   judged pair in `queries_pool.csv`, rules in `queries_guidelines.md`). After any change rerun
   `python -m eval.eval_ranking` and `python -m eval.make_results`.
3. **User study**: System Usability Scale with 10-15 users.
4. Optional upgrades: fine-tuned NER on the gold labels, IndicTrans2 translation, voice input.

## Tests

```powershell
python -m pytest -q                      # 13 tests, about 2 minutes (includes app + API smoke tests)
python -m pytest -q -m "not slow"        # 11 fast unit tests
```

## Data and licence notes

- Scheme data: [smartduketech/indian-government-schemes-2025](https://huggingface.co/datasets/smartduketech/indian-government-schemes-2025),
  CC BY 4.0, originally from [myscheme.gov.in](https://www.myscheme.gov.in). Cite both.
- The app stores no personal data; profiles are processed in memory only.
