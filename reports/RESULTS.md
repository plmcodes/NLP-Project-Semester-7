# Results

Generated from the files in `reports/`. Dataset: 4,693 schemes (smartduketech/indian-government-schemes-2025, collected July 2026).

## 1. Corpus insights

- 4,693 schemes: 684 central, 4,009 state, from 36 states/UTs.
- Rules found by our extractor: age limit in 1,194 schemes, income cap in 809 (median cap ₹240,000; middle half ₹100,000–₹450,000).
- Women-only 650, SC/ST 691, OBC 195, disability 443, BPL 238.
- Eligibility text: median 70 words; median Flesch reading ease 43.1 (grade 10.9); 66% of 4,506 texts are 'difficult' (below 50).
- Most requested documents: Aadhaar card (2,753), Bank account details (2,287), Photograph (2,100), Residence / domicile proof (1,218), Age proof / birth certificate (1,174).
- 375 scheme pairs are near-duplicates (embedding cosine > 0.97).

Figures: `reports/figures/categories.png`, `states.png`, `documents.png`, `readability.png`.

## 2. Eligibility extraction

The first table is agreement with the dataset's own machine-parsed fields (neither side is ground truth). The other two score both against a labelled gold set of 100 random schemes (`data/test_sets/extraction_gold.csv`, rules in `extraction_gold_guidelines.md`).

### Rule extractor vs dataset's machine-parsed fields (n=4693) - agreement, not accuracy

| Field | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| age_min | 0.811 | 0.809 | 0.81 | 0.93 |
| age_max | 0.834 | 0.8 | 0.817 | 0.945 |
| income_max | 0.923 | 0.959 | 0.941 | 0.981 |
| gender_female | 0.642 | 0.931 | 0.76 | 0.944 |
| disability | 0.637 | 0.916 | 0.751 | 0.96 |
| bpl | 0.597 | 0.904 | 0.719 | 0.976 |
| residence_rural | 0.717 | 0.493 | 0.584 | 0.99 |
| residence_urban | 0.233 | 0.538 | 0.326 | 0.988 |
| caste SC | 0.733 | 0.888 | 0.803 | |
| caste ST | 0.693 | 0.949 | 0.801 | |
| caste OBC | 0.631 | 0.866 | 0.73 | |
| caste General | 0.297 | 0.31 | 0.303 | |
| caste micro | 0.674 | 0.862 | 0.756 | |

### Rule extractor vs hand labels (n=100)

| Field | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| age_min | 0.778 | 0.7 | 0.737 | 0.92 |
| age_max | 1.0 | 0.714 | 0.833 | 0.94 |
| income_max | 0.957 | 0.957 | 0.957 | 0.98 |
| gender_female | 0.667 | 1.0 | 0.8 | 0.95 |
| disability | 0.615 | 0.889 | 0.727 | 0.94 |
| bpl | 1.0 | 0.75 | 0.857 | 0.99 |
| residence_rural | 1.0 | 0.333 | 0.5 | 0.98 |
| residence_urban | 0.0 | 0.0 | 0.0 | 0.97 |
| caste SC | 0.786 | 0.846 | 0.815 | |
| caste ST | 0.818 | 1.0 | 0.9 | |
| caste OBC | 0.857 | 0.857 | 0.857 | |
| caste General | 0.667 | 0.5 | 0.571 | |
| caste micro | 0.8 | 0.848 | 0.824 | |

### Dataset's parsed fields vs hand labels (n=100)

| Field | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| age_min | 1.0 | 0.95 | 0.974 | 0.99 |
| age_max | 0.95 | 0.905 | 0.927 | 0.98 |
| income_max | 1.0 | 0.87 | 0.93 | 0.97 |
| gender_female | 1.0 | 1.0 | 1.0 | 1.0 |
| disability | 1.0 | 0.889 | 0.941 | 0.99 |
| bpl | 1.0 | 0.75 | 0.857 | 0.99 |
| residence_rural | 1.0 | 0.333 | 0.5 | 0.98 |
| residence_urban | 0.0 | 0.0 | 0.0 | 1.0 |
| caste SC | 1.0 | 1.0 | 1.0 | |
| caste ST | 1.0 | 1.0 | 1.0 | |
| caste OBC | 1.0 | 0.857 | 0.923 | |
| caste General | 1.0 | 0.25 | 0.4 | |
| caste micro | 1.0 | 0.879 | 0.935 | |


## 3. Sector classification (15 categories, multi-label)

Train 3,751 / test 938 schemes. Input: name + benefits + eligibility text.

| Model | Micro-F1 | Macro-F1 | Samples-F1 |
|---|---|---|---|
| tfidf_logreg | 0.794 | 0.7181 | 0.7965 |
| e5_embeddings_logreg | 0.7027 | 0.6176 | 0.729 |

Paired Wilcoxon test on per-scheme F1: p = 2.06e-14 (mean difference TF-IDF minus e5 = 0.0675).

Per-class F1: `reports/figures/classification_per_class_f1.png`.

## 4. Topic modelling

| k | NMF UMass | LDA UMass |
|---|---|---|
| 8 | -1.453 | -1.378 |
| 12 | -1.46 | -1.568 |
| 15 | -1.516 | -1.619 |
| 20 | -1.569 | -1.712 |

UMass closer to 0 is better. The app uses NMF with k = 8 (best NMF setting). LDA's best score was -1.378 at k = 8; compare the two topic lists by hand before finalising which one you report.

| Topic | Top words | Schemes |
|---|---|---|
| 0 | students, student, class, scholarship, studying, education, school | 584 |
| 1 | construction, worker, workers, building, board, registered, welfare | 474 |
| 2 | loan, subsidy, unit, units, sector, capital, enterprise | 638 |
| 3 | pension, family, poverty, line, bpl, age, income | 596 |
| 4 | farmers, farmer, subsidy, cost, scheduled, land, agricultural | 808 |
| 5 | research, fellowship, india, training, institutions, award, candidates | 985 |
| 6 | abled, differently, persons, disability, person, tamil, nadu | 265 |
| 7 | marriage, child, girl, children, bride, daughters, women | 343 |

## 5. Search and recommendation

### Known-item search (150 queries, description -> scheme)

| System | MRR | Recall@1 | Recall@10 | nDCG@10 |
|---|---|---|---|---|
| 1 keyword (scheme names) | 0.1892 | 0.1533 | 0.28 | 0.2061 |
| 2 TF-IDF | 0.4999 | 0.4067 | 0.6867 | 0.538 |
| 3 BM25 | 0.5599 | 0.4733 | 0.74 | 0.5981 |
| 4 dense (e5) | 0.4953 | 0.4067 | 0.6867 | 0.5338 |
| 5 BM25 + dense (RRF) | 0.5709 | 0.4667 | 0.78 | 0.6159 |
| 6 hybrid + cross-encoder re-rank | 0.5384 | 0.4067 | 0.78 | 0.5918 |

Wilcoxon signed-rank on reciprocal rank:

- 3 BM25 vs 5 BM25 + dense (RRF): {'p_value': 0.14722828205536992, 'mean_rr_diff': -0.011}
- 4 dense (e5) vs 5 BM25 + dense (RRF): {'p_value': 0.00011808626702271323, 'mean_rr_diff': -0.0756}
- 5 BM25 + dense (RRF) vs 6 hybrid + cross-encoder re-rank: {'p_value': 0.21181997545109632, 'mean_rr_diff': 0.0324}
- 2 TF-IDF vs 3 BM25: {'p_value': 0.017171063773205545, 'mean_rr_diff': -0.06}

### Hindi vs English top-10 overlap (20 paired queries)

| System | Mean overlap@10 |
|---|---|
| BM25 | 0.0 |
| dense (e5) | 0.455 |
| BM25 + dense (RRF) | 0.375 |

### Eligibility filter vs dataset's parsed fields (200 random personas)

- Precision 0.937, recall 0.857, Jaccard 0.809
- Share of schemes kept per persona: 0.109

### Hand-labelled queries (29)

| System | P@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| 1 keyword (scheme names) | 0.469 | 0.3224 | 0.6595 | 0.456 |
| 2 TF-IDF | 0.5586 | 0.4771 | 0.8272 | 0.64 |
| 3 BM25 | 0.6 | 0.4826 | 0.7869 | 0.6452 |
| 4 dense (e5) | 0.6069 | 0.5229 | 0.819 | 0.6848 |
| 5 BM25 + dense (RRF) | 0.6552 | 0.5145 | 0.8807 | 0.6916 |
| 6 hybrid + cross-encoder re-rank | 0.6345 | 0.5145 | 0.8805 | 0.688 |

