# Report: runs/run1/2wiki/test100 (2wiki, n=100)

## Answer quality and retrieval

| n | EM | F1 | LLM judge | Yes/no acc | Doc recall@10 | Pool recall |
|---|---|---|---|---|---|---|
| 100 | 0.490 | 0.592 | 0.575 | 0.714 (n=7) | 0.863 | 0.882 |

## Cost

| Latency mean (s) | Latency p95 (s) | Total time (min) | LLM calls / q (pipeline count) | LLM requests / q (measured) | Prompt tokens / q | Completion tokens / q | Total tokens / q | Total tokens (run) |
|---|---|---|---|---|---|---|---|---|
| 11.9 | 17.6 | 19.9 | 3.4 | 20.0 | 6076 | 265 | 6341 | 634,137 |

## By question type

| Type | n | EM | F1 | Judge | Doc recall@10 |
|---|---|---|---|---|---|
| bridge_comparison | 25 | 0.480 | 0.480 | 0.440 | 0.690 |
| comparison | 25 | 0.960 | 0.960 | 0.940 | 1.000 |
| compositional | 25 | 0.280 | 0.432 | 0.580 | 0.960 |
| inference | 25 | 0.240 | 0.498 | 0.340 | 0.800 |

## Where it fails (wrong answers only)

| Correct (F1≥0.5) | Answer form only* | **Failures** | Retrieval | Reranking | Generation: step not found | Generation: wrong value |
|---|---|---|---|---|---|---|
| 64 | 2 | **34** | 15 (44%) | 2 (6%) | 3 (9%) | 14 (41%) |

*F1 < 0.5 but judged correct (right fact, other wording); not counted as a failure.
Per-question rows: failures.csv
