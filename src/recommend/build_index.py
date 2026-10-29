"""Build and cache the scheme embeddings used by the app.  Run: python -m src.recommend.build_index"""
import time

from src.recommend.engine import Recommender

t = time.time()
rec = Recommender()
print(f"Index ready for {len(rec.df)} schemes in {time.time() - t:.0f}s")
