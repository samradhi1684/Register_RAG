# Report: runs\run2_500ctx\2wiki\all500 (2wiki, n=500)

## Answer quality and retrieval

| n | EM | F1 | LLM judge | Yes/no acc | Doc recall@10 | Pool recall |
|---|---|---|---|---|---|---|
| 500 | 0.546 | 0.634 | 0.606 | 0.615 (n=52) | 0.912 | 0.934 |

## Cost

| Latency mean (s) | Latency p95 (s) | Total time (min) | LLM calls / q (pipeline count) | LLM requests / q (measured) | Prompt tokens / q | Completion tokens / q | Total tokens / q | Total tokens (run) |
|---|---|---|---|---|---|---|---|---|
| 11.5 | 16.6 | 96.2 | 3.4 | 19.1 | 6024 | 270 | 6293 | 3,146,693 |

## By question type

| Type | n | EM | F1 | Judge | Doc recall@10 |
|---|---|---|---|---|---|
| bridge_comparison | 125 | 0.640 | 0.645 | 0.608 | 0.796 |
| comparison | 125 | 0.840 | 0.840 | 0.792 | 1.000 |
| compositional | 125 | 0.352 | 0.491 | 0.580 | 0.952 |
| inference | 125 | 0.352 | 0.558 | 0.444 | 0.900 |

## Where it fails (wrong answers only)

| Correct (F1≥0.5) | Answer form only* | **Failures** | Retrieval | Reranking | Generation: step not found | Generation: wrong value |
|---|---|---|---|---|---|---|
| 332 | 5 | **163** | 48 (29%) | 15 (9%) | 13 (8%) | 87 (53%) |

*F1 < 0.5 but judged correct (right fact, other wording); not counted as a failure.
Per-question rows: failures.csv
