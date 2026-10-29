"""Step 5: multi-label sector classification (15 myScheme categories).

Compares
  A. TF-IDF + logistic regression (baseline)
  B. multilingual-e5 sentence embeddings + logistic regression
on a fixed 80/20 split. The category column is used as the label and is
never part of the input text.

Run:  python -m src.analysis.classify
"""
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from src import config
from src.recommend.engine import scheme_embeddings

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def input_text(df):
    # Same text the embedding model sees (name + benefits + eligibility), so both
    # classifiers get equal input. Categories (the labels) are never included.
    return df["search_text_nodesc"].tolist()


def tfidf_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True, stop_words="english"),
        OneVsRestClassifier(LogisticRegression(C=8, class_weight="balanced", max_iter=3000)),
    )


def main():
    df = pd.read_parquet(config.EXTRACTED_PARQUET)
    all_emb = scheme_embeddings(df)
    keep = (df.categories.map(len) > 0).to_numpy()
    labelled = df[keep].reset_index(drop=True)
    mlb = MultiLabelBinarizer(classes=config.CATEGORIES)
    Y = mlb.fit_transform(labelled.categories)
    texts = input_text(labelled)
    idx_train, idx_test = train_test_split(np.arange(len(labelled)), test_size=0.2, random_state=42)

    # A. TF-IDF baseline
    a = tfidf_model().fit([texts[i] for i in idx_train], Y[idx_train])
    pa = a.predict([texts[i] for i in idx_test])

    # B. sentence embeddings
    emb = all_emb[keep]
    b = OneVsRestClassifier(LogisticRegression(C=4, class_weight="balanced", max_iter=3000)).fit(emb[idx_train], Y[idx_train])
    pb = b.predict(emb[idx_test])

    yt = Y[idx_test]
    res = {"n_train": len(idx_train), "n_test": len(idx_test), "models": {}}
    for name, p in (("tfidf_logreg", pa), ("e5_embeddings_logreg", pb)):
        res["models"][name] = {
            "micro_f1": round(f1_score(yt, p, average="micro", zero_division=0), 4),
            "macro_f1": round(f1_score(yt, p, average="macro", zero_division=0), 4),
            "samples_f1": round(f1_score(yt, p, average="samples", zero_division=0), 4),
            "per_class_f1": dict(zip(config.CATEGORIES, np.round(f1_score(yt, p, average=None, zero_division=0), 3).tolist())),
        }
    # Paired test on per-scheme F1
    fa = f1_score(yt.T, pa.T, average=None, zero_division=0)
    fb = f1_score(yt.T, pb.T, average=None, zero_division=0)
    diff = fa - fb
    res["wilcoxon_per_scheme_f1"] = (
        {"statistic": float(wilcoxon(fa, fb).statistic), "p_value": float(wilcoxon(fa, fb).pvalue),
         "mean_diff_tfidf_minus_e5": round(float(diff.mean()), 4)} if np.any(diff) else {"note": "identical"})

    # Final model on all labelled data -> predict sectors for every scheme
    final = tfidf_model().fit(texts, Y)
    joblib.dump({"model": final, "classes": config.CATEGORIES}, config.ROOT / "models" / "sector_tfidf.joblib")
    pred = final.predict(input_text(df))
    pd.DataFrame({"id": df.id, "predicted_sectors": [list(mlb.classes_[r.astype(bool)]) for r in pred]}) \
        .to_parquet(config.DATA_PROC / "predicted_sectors.parquet", index=False)

    (config.REPORTS / "classification.json").write_text(json.dumps(res, indent=2))

    # Figure: per-class F1 for both models
    cats = config.CATEGORIES
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(x - 0.2, [res["models"]["tfidf_logreg"]["per_class_f1"][c] for c in cats], 0.4, label="TF-IDF + LR")
    ax.barh(x + 0.2, [res["models"]["e5_embeddings_logreg"]["per_class_f1"][c] for c in cats], 0.4, label="e5 embeddings + LR")
    ax.set_yticks(x, cats)
    ax.invert_yaxis()
    ax.set_xlabel("F1 on test set")
    ax.set_title("Sector classification: per-class F1")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "classification_per_class_f1.png", dpi=150)

    for name, m in res["models"].items():
        print(f"{name}: micro-F1 {m['micro_f1']}, macro-F1 {m['macro_f1']}")
    print("Wilcoxon:", res["wilcoxon_per_scheme_f1"])


if __name__ == "__main__":
    main()
