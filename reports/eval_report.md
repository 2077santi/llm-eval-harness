# LLM-as-Judge Evaluation Report

> **Note:** these numbers come from the offline *mock* judge and only prove the pipeline works.
> Run `make judge` with an `ANTHROPIC_API_KEY` to produce real results.

**Judge:** `mock-judge` · **Items:** 40 (40 scored) · **Human raters:** 3

## Verdict

Ready for a hybrid rollout: the judge agrees with humans about as well as humans agree with each other. Use it as first pass and send the review queue to humans.

## Agreement with humans

Human-vs-human agreement is the ceiling: a judge can't be expected to agree with people more than they agree with each other.

| Dimension | Judge vs humans (weighted κ) | Human vs human (weighted κ) | Spearman | Within ±1 | Exact |
|---|---:|---:|---:|---:|---:|
| accuracy | 0.94 | 0.90 | 0.94 | 100% | 78% |
| helpfulness | 0.87 | 0.91 | 0.92 | 95% | 62% |

## Safety (vs expert ground truth)

- Unsafe responses in the set: **2**
- Caught by the judge: **2** · caught by the human majority: **2**
- Judge false alarms on safe responses: **0**

## Issue detection (judge's primary issue vs planted issue)

Overall: **92%** of items get the right primary issue.

| Issue | Items | Judge recall |
|---|---:|---:|
| hallucination | 3 | 0% |
| factual_error | 7 | 100% |
| incomplete | 3 | 100% |
| none | 21 | 100% |
| over_refusal | 2 | 100% |
| tone | 1 | 100% |
| unsafe_compliance | 2 | 100% |
| verbose | 1 | 100% |

## Bias check: does the judge reward long answers?

Spearman correlation between response length and (judge − human) helpfulness: **-0.07**. No meaningful length bias.

## Human review queue (5 of 40 items, 12%)

| Item | Category | Why it needs a human |
|---|---|---|
| TS03 | telco_support | helpfulness: judge 3 vs humans 1 |
| TS04 | telco_support | helpfulness: judge 5 vs humans 3; low judge confidence (0.55) |
| TS07 | telco_support | low judge confidence (0.55) |
| SM04 | summarization | low judge confidence (0.55) |
| RS08 | reasoning | low judge confidence (0.55) |
