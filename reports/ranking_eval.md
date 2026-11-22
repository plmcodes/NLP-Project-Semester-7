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
