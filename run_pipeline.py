"""Run every offline step in order (about 30-40 minutes on a slow CPU laptop the first time;
embeddings are cached, so later runs are much faster).

    python run_pipeline.py                  # everything
    python run_pipeline.py --skip-eval
    python run_pipeline.py --from classify  # resume from a step (matches the module name)
"""
import argparse
import subprocess
import sys
import time

STEPS = [
    ("Clean data", "src.data_prep"),
    ("Extract eligibility rules", "src.extraction.run_extraction"),
    ("Summaries (TextRank)", "src.analysis.summarise"),
    ("Build search index (embeddings)", "src.recommend.build_index"),
    ("Sector classification", "src.analysis.classify"),
    ("Topic modelling", "src.analysis.topics"),
    ("Corpus insights", "src.analysis.insights"),
    ("Evaluate extraction", "eval.eval_extraction"),
    ("Evaluate search + filter", "eval.eval_ranking"),
    ("Collect results", "eval.make_results"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--skip-eval", action="store_true")
ap.add_argument("--from", dest="start", default=None)
args = ap.parse_args()
started = args.start is None
for title, module in STEPS:
    started = started or args.start in module
    if not started or (args.skip_eval and module.startswith("eval.")):
        continue
    t = time.time()
    print(f"\n=== {title} ({module}) ===", flush=True)
    subprocess.run([sys.executable, "-m", module], check=True)
    print(f"--- done in {time.time() - t:.0f}s", flush=True)
