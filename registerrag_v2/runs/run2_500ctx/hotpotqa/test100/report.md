# Report: runs\run2_500ctx\hotpotqa\test100 (hotpotqa, n=100)

## Answer quality and retrieval

| n | EM | F1 | LLM judge | Yes/no acc | Doc recall@10 | Pool recall |
|---|---|---|---|---|---|---|
| 100 | 0.540 | 0.666 | 0.680 | 0.692 (n=13) | 0.970 | 0.990 |

## Cost

| Latency mean (s) | Latency p95 (s) | Total time (min) | LLM calls / q (pipeline count) | LLM requests / q (measured) | Prompt tokens / q | Completion tokens / q | Total tokens / q | Total tokens (run) |
|---|---|---|---|---|---|---|---|---|
| 10.9 | 15.8 | 18.2 | 3.2 | 17.8 | 4914 | 293 | 5207 | 520,722 |

## By question type

| Type | n | EM | F1 | Judge | Doc recall@10 |
|---|---|---|---|---|---|
| bridge | 50 | 0.480 | 0.572 | 0.630 | 0.940 |
| comparison | 50 | 0.600 | 0.759 | 0.730 | 1.000 |

## Where it fails (wrong answers only)

| Correct (F1≥0.5) | Answer form only* | **Failures** | Retrieval | Reranking | Generation: step not found | Generation: wrong value |
|---|---|---|---|---|---|---|
| 69 | 3 | **28** | 2 (7%) | 4 (14%) | 8 (29%) | 14 (50%) |

*F1 < 0.5 but judged correct (right fact, other wording); not counted as a failure.
Per-question rows: failures.csv
