# Report: runs\run2_500ctx\repolicyqa\test100 (repolicyqa, n=100)

## Answer quality and retrieval

| n | EM | F1 | LLM judge | Yes/no acc | Doc recall@10 | Pool recall |
|---|---|---|---|---|---|---|
| 100 | 0.320 | 0.495 | 0.605 | 0.762 (n=42) | 0.950 | 1.000 |

## Cost

| Latency mean (s) | Latency p95 (s) | Total time (min) | LLM calls / q (pipeline count) | LLM requests / q (measured) | Prompt tokens / q | Completion tokens / q | Total tokens / q | Total tokens (run) |
|---|---|---|---|---|---|---|---|---|
| 11.7 | 16.7 | 19.6 | 2.9 | 29.5 | 7600 | 580 | 8180 | 817,995 |

## By question type

| Type | n | EM | F1 | Judge | Doc recall@10 |
|---|---|---|---|---|---|
| Descriptive | 20 | 0.000 | 0.345 | 0.450 | 0.950 |
| Factual | 20 | 0.200 | 0.455 | 0.650 | 0.950 |
| Legal-Obligation | 20 | 0.000 | 0.269 | 0.500 | 0.950 |
| Yes-No | 20 | 0.800 | 0.800 | 0.800 | 1.000 |
| Yes-No-cond | 20 | 0.600 | 0.606 | 0.625 | 0.900 |

## Where it fails (wrong answers only)

| Correct (F1≥0.5) | Answer form only* | **Failures** | Retrieval | Reranking | Generation: step not found | Generation: wrong value |
|---|---|---|---|---|---|---|
| 47 | 5 | **48** | 4 (8%) | 5 (10%) | 18 (38%) | 21 (44%) |

*F1 < 0.5 but judged correct (right fact, other wording); not counted as a failure.
Per-question rows: failures.csv
