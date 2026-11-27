# Report: runs/run1/repolicyqa/test100 (repolicyqa, n=100)

## Answer quality and retrieval

| n | EM | F1 | LLM judge | Yes/no acc | Doc recall@10 | Pool recall |
|---|---|---|---|---|---|---|
| 100 | 0.320 | 0.483 | 0.600 | 0.810 (n=42) | 0.990 | 0.990 |

## Cost

| Latency mean (s) | Latency p95 (s) | Total time (min) | LLM calls / q (pipeline count) | LLM requests / q (measured) | Prompt tokens / q | Completion tokens / q | Total tokens / q | Total tokens (run) |
|---|---|---|---|---|---|---|---|---|
| 12.3 | 17.7 | 20.5 | 3.0 | 23.5 | 6577 | 526 | 7103 | 710,268 |

## By question type

| Type | n | EM | F1 | Judge | Doc recall@10 |
|---|---|---|---|---|---|
| Descriptive | 20 | 0.000 | 0.324 | 0.475 | 1.000 |
| Factual | 20 | 0.100 | 0.369 | 0.525 | 0.950 |
| Legal-Obligation | 20 | 0.000 | 0.217 | 0.500 | 1.000 |
| Yes-No | 20 | 0.800 | 0.800 | 0.775 | 1.000 |
| Yes-No-cond | 20 | 0.700 | 0.706 | 0.725 | 1.000 |

## Where it fails (wrong answers only)

| Correct (F1≥0.5) | Answer form only* | **Failures** | Retrieval | Reranking | Generation: step not found | Generation: wrong value |
|---|---|---|---|---|---|---|
| 47 | 3 | **50** | 7 (14%) | 8 (16%) | 17 (34%) | 18 (36%) |

*F1 < 0.5 but judged correct (right fact, other wording); not counted as a failure.
Per-question rows: failures.csv
