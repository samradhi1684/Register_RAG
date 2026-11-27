# Report: runs\run2_500ctx\2wiki\all500_3b (2wiki, n=500)

## Answer quality and retrieval

| n | EM | F1 | LLM judge | Yes/no acc | Doc recall@10 | Pool recall |
|---|---|---|---|---|---|---|
| 500 | 0.394 | 0.475 | 0.443 | 0.327 (n=52) | 0.899 | 0.953 |

## Cost

| Latency mean (s) | Latency p95 (s) | Total time (min) | LLM calls / q (pipeline count) | LLM requests / q (measured) | Prompt tokens / q | Completion tokens / q | Total tokens / q | Total tokens (run) |
|---|---|---|---|---|---|---|---|---|
| 10.6 | 16.9 | 88.0 | 3.2 | 27.3 | 8394 | 312 | 8705 | 4,352,723 |

## By question type

| Type | n | EM | F1 | Judge | Doc recall@10 |
|---|---|---|---|---|---|
| bridge_comparison | 125 | 0.552 | 0.558 | 0.540 | 0.928 |
| comparison | 125 | 0.624 | 0.628 | 0.608 | 0.952 |
| compositional | 125 | 0.288 | 0.388 | 0.460 | 0.920 |
| inference | 125 | 0.112 | 0.324 | 0.164 | 0.796 |

## Where it fails (wrong answers only)

| Correct (F1≥0.5) | Answer form only* | **Failures** | Retrieval | Reranking | Generation: step not found | Generation: wrong value |
|---|---|---|---|---|---|---|
| 237 | 6 | **257** | 44 (17%) | 41 (16%) | 21 (8%) | 151 (59%) |

*F1 < 0.5 but judged correct (right fact, other wording); not counted as a failure.
Per-question rows: failures.csv
