# Report: runs/run1/musique/test100 (musique, n=100)

## Answer quality and retrieval

| n | EM | F1 | LLM judge | Yes/no acc | Doc recall@10 | Pool recall |
|---|---|---|---|---|---|---|
| 100 | 0.260 | 0.333 | 0.335 | – | 0.752 | 0.824 |

## Cost

| Latency mean (s) | Latency p95 (s) | Total time (min) | LLM calls / q (pipeline count) | LLM requests / q (measured) | Prompt tokens / q | Completion tokens / q | Total tokens / q | Total tokens (run) |
|---|---|---|---|---|---|---|---|---|
| 9.9 | 13.7 | 16.5 | 2.2 | 31.6 | 8433 | 334 | 8767 | 876,668 |

## By question type

| Type | n | EM | F1 | Judge | Doc recall@10 |
|---|---|---|---|---|---|
| 2hop | 34 | 0.382 | 0.404 | 0.426 | 0.853 |
| 3hop | 33 | 0.242 | 0.333 | 0.348 | 0.808 |
| 4hop | 33 | 0.152 | 0.260 | 0.227 | 0.591 |

## Where it fails (wrong answers only)

| Correct (F1≥0.5) | Answer form only* | **Failures** | Retrieval | Reranking | Generation: step not found | Generation: wrong value |
|---|---|---|---|---|---|---|
| 34 | 1 | **65** | 38 (58%) | 6 (9%) | 7 (11%) | 14 (22%) |

*F1 < 0.5 but judged correct (right fact, other wording); not counted as a failure.
Per-question rows: failures.csv
