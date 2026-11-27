# Report: runs\run2_500ctx\hotpotqa\all500 (hotpotqa, n=500)

## Answer quality and retrieval

| n | EM | F1 | LLM judge | Yes/no acc | Doc recall@10 | Pool recall |
|---|---|---|---|---|---|---|
| 500 | 0.530 | 0.624 | 0.660 | 0.789 (n=76) | 0.962 | 0.975 |

## Cost

| Latency mean (s) | Latency p95 (s) | Total time (min) | LLM calls / q (pipeline count) | LLM requests / q (measured) | Prompt tokens / q | Completion tokens / q | Total tokens / q | Total tokens (run) |
|---|---|---|---|---|---|---|---|---|
| 11.2 | 15.8 | 93.7 | 3.2 | 20.4 | 5614 | 322 | 5936 | 2,967,979 |

## By question type

| Type | n | EM | F1 | Judge | Doc recall@10 |
|---|---|---|---|---|---|
| bridge | 250 | 0.448 | 0.535 | 0.592 | 0.932 |
| comparison | 250 | 0.612 | 0.713 | 0.728 | 0.992 |

## Where it fails (wrong answers only)

| Correct (F1≥0.5) | Answer form only* | **Failures** | Retrieval | Reranking | Generation: step not found | Generation: wrong value |
|---|---|---|---|---|---|---|
| 321 | 16 | **163** | 16 (10%) | 9 (6%) | 46 (28%) | 92 (56%) |

*F1 < 0.5 but judged correct (right fact, other wording); not counted as a failure.
Per-question rows: failures.csv
