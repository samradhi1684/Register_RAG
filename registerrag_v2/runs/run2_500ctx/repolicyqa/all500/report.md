# Report: runs\run2_500ctx\repolicyqa\all500 (repolicyqa, n=500)

## Answer quality and retrieval

| n | EM | F1 | LLM judge | Yes/no acc | Doc recall@10 | Pool recall |
|---|---|---|---|---|---|---|
| 500 | 0.328 | 0.485 | 0.590 | 0.742 (n=213) | 0.976 | 0.994 |

## Cost

| Latency mean (s) | Latency p95 (s) | Total time (min) | LLM calls / q (pipeline count) | LLM requests / q (measured) | Prompt tokens / q | Completion tokens / q | Total tokens / q | Total tokens (run) |
|---|---|---|---|---|---|---|---|---|
| 11.8 | 17.2 | 98.7 | 3.0 | 29.3 | 7435 | 569 | 8003 | 4,001,655 |

## By question type

| Type | n | EM | F1 | Judge | Doc recall@10 |
|---|---|---|---|---|---|
| Descriptive | 100 | 0.020 | 0.315 | 0.435 | 0.980 |
| Factual | 100 | 0.240 | 0.461 | 0.635 | 0.990 |
| Legal-Obligation | 100 | 0.010 | 0.266 | 0.465 | 0.970 |
| Yes-No | 100 | 0.670 | 0.676 | 0.695 | 0.970 |
| Yes-No-cond | 100 | 0.700 | 0.711 | 0.720 | 0.970 |

## Where it fails (wrong answers only)

| Correct (F1≥0.5) | Answer form only* | **Failures** | Retrieval | Reranking | Generation: step not found | Generation: wrong value |
|---|---|---|---|---|---|---|
| 230 | 19 | **251** | 26 (10%) | 28 (11%) | 104 (41%) | 93 (37%) |

*F1 < 0.5 but judged correct (right fact, other wording); not counted as a failure.
Per-question rows: failures.csv
