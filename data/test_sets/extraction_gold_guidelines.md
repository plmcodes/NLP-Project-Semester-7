# Extraction gold set: labelling rules

`extraction_gold.csv` labels the 100 schemes sampled into `extraction_gold_template.csv`
(random, seed 42). Each label was read from the scheme's eligibility text only, without looking
at the rule extractor's output or the dataset's parsed fields. Judgement calls are explained in
the `notes` column.

Labelled 2026-11-22 by Claude (AI assistant) on the team's behalf. The team should review the
labels, especially rows with a note, before the numbers go into the report.

| Field | Values | Rule |
|---|---|---|
| age_min, age_max | whole years, blank if none | The number written in the text ("above 18" gives 18, "under 18" gives 18). Applies to the beneficiary. |
| income_max | annual rupees, blank if none | Family/applicant income cap. Monthly caps are multiplied by 12. |
| gender | `all`, `female`, `male` | `female` only when the scheme is for women/girls only (pregnant women, widows, "women only"). Transgender-only schemes are `all`. |
| castes | `SC`, `ST`, `OBC`, `General`, joined by `;` | Only categories the scheme is restricted to. SEBC / Backward Classes = OBC; EWS / EBC / "General category" = General. |
| residence | `rural`, `urban`, `both` | `rural`/`urban` only when the text requires living in a village/rural area or a town/urban area. |
| disability | `True` / `False` | True only when disability is required. |
| bpl | `True` / `False` | True only when BPL status (or income within the poverty line) is required. |

General rules:

- Only requirements count. Preferences, priorities, reservations and age relaxations for a group
  ("one award reserved for a differently abled teacher") are not requirements.
- If one route out of several alternatives mentions a group (e.g. "SC, ST, OBC, minority or
  PwD"), only fields shared by every route count; the caste list keeps the named castes.
- When the text gives several limits for different groups (rural/urban income, male/female upper
  age), the most lenient limit is used, because anyone outside it is ineligible.
- Schemes with several separate components are labelled for the first one (noted).
