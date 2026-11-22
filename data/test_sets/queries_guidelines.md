# Search queries: relevance judgements

`queries.csv` lists 30 citizen-style queries and the schemes judged relevant to each
(`relevant_ids`, separated by `;`). `queries_pool.csv` keeps every judged query-scheme pair with
its 1/0 label.

Judged 2026-11-22 by Claude (AI assistant) on the team's behalf. The team should review the
labels before the numbers go into the report.

## How candidates were collected (pooling)

`python -m eval.eval_ranking --pool` takes the top 10 schemes from every system (keyword on
names, TF-IDF, BM25, dense, hybrid; the re-ranker only reorders the hybrid top 10), giving 816
query-scheme pairs (19-34 per query). Candidates were sorted by name before judging, so the
judge could not see which system ranked them first.

Four relevant schemes that no system retrieved were added after searching the whole dataset:
`ignwps` (UP widow query), `fcssos` and `cmacs` (OBC UPSC coaching query), `sbm-g-ii` (village
toilet query). Other relevant schemes outside the pool may still exist, so recall is an upper
estimate.

## What counts as relevant

A scheme is relevant when its main purpose meets the need in the query and nothing in the query
rules the person out:

- **State named** (e.g. "widow in Uttar Pradesh"): the scheme must be for that state or Central.
- **Group named** (SC, OBC, minority, women): the scheme must target that group, alone or among
  others. A general scheme open to everyone is not counted.
- **No group named**: schemes for a narrower group (e.g. widows of construction workers for
  "pension for widows") still count, because they answer the need for some people.
- **Topic**: the benefit must be the one asked for. Examples: a seed *production* subsidy is not
  help to *buy* seeds; an award for coaches is not training for athletes; NGO grants are not
  help for a person with disability; postdoctoral fellowships are not PhD fellowships;
  state crop-loss relief schemes count for "insurance for crop loss due to flood" because they
  replace crop insurance in those states.
- Descriptions that were too short to judge were checked against the full eligibility text.

## Known gaps

- "I am a 45 year old farmer in Maharashtra and need money for a tube well" has **no** relevant
  scheme in the dataset (nothing for tube wells in Maharashtra or at Central level), so the
  evaluation skips it and reports 29 queries.
- One judge, no agreement check. A second team member labelling a few queries would let the
  report quote inter-annotator agreement.
