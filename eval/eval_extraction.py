"""Step 8: evaluate the rule extractor.

1. Agreement with the dataset's machine-parsed ("silver") fields, for all schemes.
2. If data/test_sets/extraction_gold.csv exists (hand-labelled by the team),
   score BOTH our extractor and the silver fields against it.

Also writes extraction_gold_template.csv (100 sampled schemes) for the team to label.

Run:  python -m eval.eval_extraction
"""
import json

import numpy as np
import pandas as pd

from src.config import EXTRACTED_PARQUET, REPORTS, TEST_SETS

GOLD = TEST_SETS / "extraction_gold.csv"
TEMPLATE = TEST_SETS / "extraction_gold_template.csv"


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


def numeric_scores(pred: pd.Series, ref: pd.Series) -> dict:
    """A value counts as correct only if it matches exactly."""
    p_has, r_has = pred.notna(), ref.notna()
    exact = p_has & r_has & (pred == ref)
    tp = int(exact.sum())
    fp = int((p_has & ~exact).sum())
    fn = int((r_has & ~exact).sum())
    both_none = int((~p_has & ~r_has).sum())
    P, R, F = prf(tp, fp, fn)
    return {"precision": P, "recall": R, "f1": F,
            "accuracy": round((tp + both_none) / len(pred), 3),
            "pred_found": int(p_has.sum()), "ref_found": int(r_has.sum())}


def binary_scores(pred: pd.Series, ref: pd.Series) -> dict:
    pred, ref = pred.astype(bool), ref.astype(bool)
    P, R, F = prf(int((pred & ref).sum()), int((pred & ~ref).sum()), int((~pred & ref).sum()))
    return {"precision": P, "recall": R, "f1": F,
            "accuracy": round(float((pred == ref).mean()), 3),
            "pred_pos": int(pred.sum()), "ref_pos": int(ref.sum())}


def multilabel_scores(pred, ref, labels) -> dict:
    out, TP = {}, [0, 0, 0]
    for lab in labels:
        p = pred.map(lambda x: lab in list(x))
        r = ref.map(lambda x: lab in list(x))
        tp, fp, fn = int((p & r).sum()), int((p & ~r).sum()), int((~p & r).sum())
        TP = [TP[0] + tp, TP[1] + fp, TP[2] + fn]
        out[lab] = dict(zip(("precision", "recall", "f1"), prf(tp, fp, fn)))
    out["micro"] = dict(zip(("precision", "recall", "f1"), prf(*TP)))
    return out


def score(df: pd.DataFrame, pred_prefix: str, ref_prefix: str) -> dict:
    g = lambda pre, f: df[pre + f]
    res = {
        "age_min": numeric_scores(g(pred_prefix, "age_min"), g(ref_prefix, "age_min")),
        "age_max": numeric_scores(g(pred_prefix, "age_max"), g(ref_prefix, "age_max")),
        "income_max": numeric_scores(g(pred_prefix, "income_max"), g(ref_prefix, "income_max")),
        "gender_female": binary_scores(g(pred_prefix, "gender") == "female", g(ref_prefix, "gender") == "female"),
        "disability": binary_scores(g(pred_prefix, "disability"), g(ref_prefix, "disability")),
        "bpl": binary_scores(g(pred_prefix, "bpl"), g(ref_prefix, "bpl")),
        "residence_rural": binary_scores(g(pred_prefix, "residence") == "rural", g(ref_prefix, "residence") == "rural"),
        "residence_urban": binary_scores(g(pred_prefix, "residence") == "urban", g(ref_prefix, "residence") == "urban"),
        "castes": multilabel_scores(g(pred_prefix, "castes"), g(ref_prefix, "castes"), ["SC", "ST", "OBC", "General"]),
    }
    return res


def write_template(df: pd.DataFrame, n=100, seed=42):
    if TEMPLATE.exists():
        return
    rng = np.random.default_rng(seed)
    sample = df.iloc[rng.choice(len(df), size=n, replace=False)]
    t = pd.DataFrame({"id": sample.id, "name": sample.name, "state": sample.state,
                      "eligibility_text": sample.eligibility_text})
    for c in ["age_min", "age_max", "income_max", "gender", "castes", "residence", "disability", "bpl"]:
        t[c] = ""
    t.to_csv(TEMPLATE, index=False, encoding="utf-8-sig")
    print(f"Wrote labelling template: {TEMPLATE}")


def load_gold(df: pd.DataFrame) -> pd.DataFrame | None:
    if not GOLD.exists():
        return None
    gold = pd.read_csv(GOLD).dropna(subset=["gender"])
    gold = gold.rename(columns={c: "gold_" + c for c in gold.columns if c != "id"})
    for c in ["gold_age_min", "gold_age_max", "gold_income_max"]:
        gold[c] = pd.to_numeric(gold[c], errors="coerce")
    gold["gold_castes"] = gold["gold_castes"].fillna("").map(lambda s: [x.strip() for x in str(s).split(";") if x.strip()])
    gold["gold_residence"] = gold["gold_residence"].fillna("both")
    for c in ["gold_disability", "gold_bpl"]:
        gold[c] = gold[c].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    return df.merge(gold[[c for c in gold.columns if c == "id" or c.startswith("gold_")]], on="id")


def to_markdown(res: dict, title: str) -> str:
    lines = [f"### {title}", "", "| Field | Precision | Recall | F1 | Accuracy |", "|---|---|---|---|---|"]
    for k, v in res.items():
        if k == "castes":
            for lab, s in v.items():
                lines.append(f"| caste {lab} | {s['precision']} | {s['recall']} | {s['f1']} | |")
        else:
            lines.append(f"| {k} | {v['precision']} | {v['recall']} | {v['f1']} | {v['accuracy']} |")
    return "\n".join(lines) + "\n"


def main():
    df = pd.read_parquet(EXTRACTED_PARQUET)
    df["silver_castes"] = df["silver_caste"]
    write_template(df)

    report = {"n_schemes": len(df), "rules_vs_silver": score(df, "ext_", "silver_")}
    md = [to_markdown(report["rules_vs_silver"],
                      f"Rule extractor vs dataset's machine-parsed fields (n={len(df)}) - agreement, not accuracy")]

    gold = load_gold(df)
    if gold is not None and len(gold):
        report["n_gold"] = len(gold)
        report["rules_vs_gold"] = score(gold, "ext_", "gold_")
        report["silver_vs_gold"] = score(gold, "silver_", "gold_")
        md.append(to_markdown(report["rules_vs_gold"], f"Rule extractor vs hand labels (n={len(gold)})"))
        md.append(to_markdown(report["silver_vs_gold"], f"Dataset's parsed fields vs hand labels (n={len(gold)})"))
    else:
        md.append("_No hand-labelled gold set yet: fill in data/test_sets/extraction_gold_template.csv "
                  "and save it as extraction_gold.csv, then rerun._\n")

    (REPORTS / "extraction_eval.json").write_text(json.dumps(report, indent=2))
    (REPORTS / "extraction_eval.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
