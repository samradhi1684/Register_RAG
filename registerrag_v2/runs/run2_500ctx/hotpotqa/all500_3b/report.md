# Report: runs\run2_500ctx\hotpotqa\all500_3b (hotpotqa, n=500)

## Answer quality and retrieval

| n | EM | F1 | LLM judge | Yes/no acc | Doc recall@10 | Pool recall |
|---|---|---|---|---|---|---|
| 500 | 0.392 | 0.466 | 0.500 | 0.500 (n=76) | 0.934 | 0.965 |

## Cost

| Latency mean (s) | Latency p95 (s) | Total time (min) | LLM calls / q (pipeline count) | LLM requests / q (measured) | Prompt tokens / q | Completion tokens / q | Total tokens / q | Total tokens (run) |
|---|---|---|---|---|---|---|---|---|
| 10.8 | 16.4 | 90.1 | 3.1 | 29.8 | 8540 | 357 | 8897 | 4,448,392 |

## By question type

| Type | n | EM | F1 | Judge | Doc recall@10 |
|---|---|---|---|---|---|
| bridge | 250 | 0.324 | 0.405 | 0.440 | 0.898 |
| comparison | 250 | 0.460 | 0.528 | 0.560 | 0.970 |

## Where it fails (wrong answers only)

| Correct (F1≥0.5) | Answer form only* | **Failures** | Retrieval | Reranking | Generation: step not found | Generation: wrong value |
|---|---|---|---|---|---|---|
| 242 | 10 | **248** | 25 (10%) | 18 (7%) | 52 (21%) | 153 (62%) |

*F1 < 0.5 but judged correct (right fact, other wording); not counted as a failure.
Per-question rows: failures.csv
