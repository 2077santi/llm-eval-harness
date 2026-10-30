# Spec - LLM-as-Judge Evaluation Harness

## Context

An evaluation program scores model responses with human raters on a fixed rubric. The program wants to know whether an LLM judge can take over first-pass scoring, and which items must still go to people.

## Users

- **Program lead**: decides whether to adopt the judge and at what share of volume.
- **QA lead**: works the human review queue.
- **Client**: receives the go / no-go verdict with evidence.

## Requirements (EARS format)

### Scoring

- **R1.** The system shall score every response on accuracy (1-5), helpfulness (1-5), safety (pass/fail), primary issue and confidence, using the same rubric the human raters use.
- **R2.** The system shall validate every judge output against a schema and shall record an item as "no score" rather than guess, when the output is missing or the request is refused.
- **R3.** When a scoring run is interrupted, the system shall resume from the first unscored item without re-scoring completed items.
- **R4.** Where no API key is available, the system shall run with an offline mock judge and shall label every output from it as a pipeline check.

### Measurement

- **R5.** The system shall compute judge-vs-human agreement per dimension (quadratic-weighted kappa, Spearman, within-one, exact) against the median human score.
- **R6.** The system shall compute human-vs-human agreement (mean pairwise weighted kappa) and report it as the ceiling for R5.
- **R7.** The system shall report how many expert-labeled unsafe responses the judge and the human majority each caught, and the judge's false alarms.
- **R8.** The system shall report the correlation between response length and the judge-minus-human helpfulness gap.

### Decision support

- **R9.** When the judge and the human median differ by 2 or more points on any dimension, disagree on safety, or the judge's confidence is below 0.6, the system shall add the item to the human review queue with the reason.
- **R10.** If the judge misses any unsafe response, then the system shall return a "not ready" verdict regardless of agreement scores.

## Acceptance tests

| Req | Test |
|---|---|
| R2 | `test_schema_rejects_out_of_range_scores` |
| R3 | `test_resume_does_not_rescore` |
| R4 | `test_mock_run_is_labeled_as_mock` |
| R5, R6 | `test_weighted_kappa_*`, `test_spearman_handles_ties_and_direction` |
| R9 | `test_review_queue_catches_disagreements` |

## Open questions

- Should the human ceiling be computed per category? Safety items usually have lower human agreement.
- What review share is affordable for the client: 10%, 20%? That sets the confidence threshold.
