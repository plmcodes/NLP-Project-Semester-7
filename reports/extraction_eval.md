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
