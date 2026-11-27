# Report: runs\run2_500ctx\musique\test100 (musique, n=100)

## Answer quality and retrieval

| n | EM | F1 | LLM judge | Yes/no acc | Doc recall@10 | Pool recall |
|---|---|---|---|---|---|---|
| 100 | 0.280 | 0.372 | 0.370 | – | 0.776 | 0.853 |

## Cost

| Latency mean (s) | Latency p95 (s) | Total time (min) | LLM calls / q (pipeline count) | LLM requests / q (measured) | Prompt tokens / q | Completion tokens / q | Total tokens / q | Total tokens (run) |
|---|---|---|---|---|---|---|---|---|
| 9.8 | 14.1 | 16.4 | 2.3 | 34.6 | 9369 | 336 | 9704 | 970,414 |

## By question type

| Type | n | EM | F1 | Judge | Doc recall@10 |
|---|---|---|---|---|---|
| 2hop | 34 | 0.353 | 0.415 | 0.441 | 0.882 |
| 3hop | 33 | 0.212 | 0.316 | 0.318 | 0.798 |
| 4hop | 33 | 0.273 | 0.385 | 0.348 | 0.644 |

## Where it fails (wrong answers only)

| Correct (F1≥0.5) | Answer form only* | **Failures** | Retrieval | Reranking | Generation: step not found | Generation: wrong value |
|---|---|---|---|---|---|---|
| 40 | 1 | **59** | 28 (47%) | 7 (12%) | 9 (15%) | 15 (25%) |

*F1 < 0.5 but judged correct (right fact, other wording); not counted as a failure.
Per-question rows: failures.csv
