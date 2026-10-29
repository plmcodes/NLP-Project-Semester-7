"""Paths and model names used across the project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
TEST_SETS = ROOT / "data" / "test_sets"
INDEX_DIR = ROOT / "models" / "index"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

RAW_CSV = DATA_RAW / "Schemes.csv"
SCHEMES_PARQUET = DATA_PROC / "schemes.parquet"

# Small multilingual models so everything runs on a CPU laptop.
# multilingual-e5 expects "query: " / "passage: " prefixes.
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
QUERY_PREFIX = os.getenv("QUERY_PREFIX", "query: ")
PASSAGE_PREFIX = os.getenv("PASSAGE_PREFIX", "passage: ")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

# The 15 myScheme categories. Some contain commas, so the raw
# comma-separated column cannot be split naively.
CATEGORIES = [
    "Agriculture,Rural & Environment",
    "Banking,Financial Services and Insurance",
    "Business & Entrepreneurship",
    "Education & Learning",
    "Health & Wellness",
    "Housing & Shelter",
    "Public Safety,Law & Justice",
    "Science, IT & Communications",
    "Skills & Employment",
    "Social welfare & Empowerment",
    "Sports & Culture",
    "Transport & Infrastructure",
    "Travel & Tourism",
    "Utility & Sanitation",
    "Women and Child",
]

for d in (DATA_PROC, TEST_SETS, INDEX_DIR, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

EXTRACTED_PARQUET = DATA_PROC / "schemes_extracted.parquet"

# Token limits. This laptop's CPU needs ~0.35 s per 256-token passage, so we cap
# lengths; raise them on a faster machine or a GPU.
EMBED_MAX_LEN = int(os.getenv("EMBED_MAX_LEN", "256"))
RERANK_MAX_LEN = int(os.getenv("RERANK_MAX_LEN", "256"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "15"))
