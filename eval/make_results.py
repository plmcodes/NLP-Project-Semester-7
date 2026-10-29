"""Collect every report JSON into reports/RESULTS.md (paste-ready for the project report).

Run:  python -m eval.make_results
"""
import json

from src import config

R = config.REPORTS


def load(name):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    out = ["# Results", "",
           "Generated from the files in `reports/`. Dataset: 4,693 schemes "
           "(smartduketech/indian-government-schemes-2025, collected July 2026).", ""]

    ins = load("insights.json")
    if ins:
        t, rd = ins["targets"], ins["readability"]
        out += ["## 1. Corpus insights", "",
                f"- {ins['n_schemes']:,} schemes: {ins['central']:,} central, {ins['state']:,} state, from {ins['n_states_uts']} states/UTs.",
                f"- Rules found by our extractor: age limit in {t['has_age_rule']:,} schemes, income cap in {t['has_income_cap']:,} "
                f"(median cap ₹{ins['income_cap_median']:,}; middle half ₹{ins['income_cap_quartiles'][0]:,}–₹{ins['income_cap_quartiles'][1]:,}).",
                f"- Women-only {t['women_only']}, SC/ST {t['sc_st']}, OBC {t['obc']}, disability {t['disability']}, BPL {t['bpl']}.",
                f"- Eligibility text: median {ins['eligibility_words_median']} words; median Flesch reading ease "
                f"{rd['flesch_reading_ease_median']} (grade {rd['fk_grade_median']}); "
                f"{rd['share_difficult_below_50']:.0%} of {rd['n_texts']:,} texts are 'difficult' (below 50).",
                "- Most requested documents: " + ", ".join(f"{k} ({v:,})" for k, v in list(ins["documents"].items())[:5]) + ".",
                ]
        nd = ins.get("near_duplicates")
        if nd:
            out.append(f"- {nd['n_pairs']} scheme pairs are near-duplicates (embedding cosine > {nd['threshold']}).")
        out += ["", "Figures: `reports/figures/categories.png`, `states.png`, `documents.png`, `readability.png`.", ""]

    ext = R / "extraction_eval.md"
    if ext.exists():
        out += ["## 2. Eligibility extraction", "",
                "Agreement with the dataset's own machine-parsed fields. Neither side is ground truth: "
                "a hand-labelled set is needed for accuracy (see README).", "",
                ext.read_text(encoding="utf-8"), ""]

    cl = load("classification.json")
    if cl:
        out += ["## 3. Sector classification (15 categories, multi-label)", "",
                f"Train {cl['n_train']:,} / test {cl['n_test']:,} schemes. Input: name + benefits + eligibility text.", "",
                "| Model | Micro-F1 | Macro-F1 | Samples-F1 |", "|---|---|---|---|"]
        for name, m in cl["models"].items():
            out.append(f"| {name} | {m['micro_f1']} | {m['macro_f1']} | {m['samples_f1']} |")
        w = cl["wilcoxon_per_scheme_f1"]
        if "p_value" in w:
            out += ["", f"Paired Wilcoxon test on per-scheme F1: p = {w['p_value']:.3g} "
                        f"(mean difference TF-IDF minus e5 = {w['mean_diff_tfidf_minus_e5']})."]
        out += ["", "Per-class F1: `reports/figures/classification_per_class_f1.png`.", ""]

    tp = load("topics.json")
    if tp:
        out += ["## 4. Topic modelling", "", "| k | NMF UMass | LDA UMass |", "|---|---|---|"]
        out += [f"| {r['k']} | {r['nmf_umass']} | {r['lda_umass']} |" for r in tp["model_selection"]]
        best_lda = max(tp["model_selection"], key=lambda r: r["lda_umass"])
        out += ["", f"UMass closer to 0 is better. The app uses NMF with k = {tp['chosen']['k']} "
                    f"(best NMF setting). LDA's best score was {best_lda['lda_umass']} at k = {best_lda['k']}; "
                    "compare the two topic lists by hand before finalising which one you report.", "",
                "| Topic | Top words | Schemes |", "|---|---|---|"]
        out += [f"| {t['topic']} | {', '.join(t['top_words'][:7])} | {t['n_schemes']} |" for t in tp["topics"]]
        out.append("")

    rk = R / "ranking_eval.md"
    if rk.exists():
        out += ["## 5. Search and recommendation", "", rk.read_text(encoding="utf-8"), ""]

    (R / "RESULTS.md").write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {R / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
