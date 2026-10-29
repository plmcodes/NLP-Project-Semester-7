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

_No hand-labelled gold set yet: fill in data/test_sets/extraction_gold_template.csv and save it as extraction_gold.csv, then rerun._
