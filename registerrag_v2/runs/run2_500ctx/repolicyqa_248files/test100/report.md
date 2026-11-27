# Report: runs\run2_500ctx\repolicyqa\test100 (repolicyqa, n=100)

## Answer quality and retrieval

| n | EM | F1 | LLM judge | Yes/no acc | Doc recall@10 | Pool recall |
|---|---|---|---|---|---|---|
| 100 | 0.320 | 0.499 | 0.630 | 0.762 (n=42) | 1.000 | 1.000 |

## Cost

| Latency mean (s) | Latency p95 (s) | Total time (min) | LLM calls / q (pipeline count) | LLM requests / q (measured) | Prompt tokens / q | Completion tokens / q | Total tokens / q | Total tokens (run) |
|---|---|---|---|---|---|---|---|---|
| 11.9 | 17.7 | 19.9 | 3.0 | 24.4 | 6854 | 528 | 7381 | 738,149 |

## By question type

| Type | n | EM | F1 | Judge | Doc recall@10 |
|---|---|---|---|---|---|
| Descriptive | 20 | 0.000 | 0.347 | 0.525 | 1.000 |
| Factual | 20 | 0.200 | 0.406 | 0.650 | 1.000 |
| Legal-Obligation | 20 | 0.000 | 0.333 | 0.550 | 1.000 |
| Yes-No | 20 | 0.750 | 0.750 | 0.750 | 1.000 |
| Yes-No-cond | 20 | 0.650 | 0.656 | 0.675 | 1.000 |

## Where it fails (wrong answers only)

| Correct (F1≥0.5) | Answer form only* | **Failures** | Retrieval | Reranking | Generation: step not found | Generation: wrong value |
|---|---|---|---|---|---|---|
| 50 | 3 | **47** | 7 (15%) | 3 (6%) | 16 (34%) | 21 (45%) |

*F1 < 0.5 but judged correct (right fact, other wording); not counted as a failure.
Per-question rows: failures.csv
