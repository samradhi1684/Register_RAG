# Report: runs\run2_500ctx\musique\all500 (musique, n=500)

## Answer quality and retrieval

| n | EM | F1 | LLM judge | Yes/no acc | Doc recall@10 | Pool recall |
|---|---|---|---|---|---|---|
| 500 | 0.252 | 0.349 | 0.354 | – | 0.755 | 0.819 |

## Cost

| Latency mean (s) | Latency p95 (s) | Total time (min) | LLM calls / q (pipeline count) | LLM requests / q (measured) | Prompt tokens / q | Completion tokens / q | Total tokens / q | Total tokens (run) |
|---|---|---|---|---|---|---|---|---|
| 9.9 | 14.4 | 82.3 | 2.3 | 35.5 | 9783 | 335 | 10117 | 5,058,655 |

## By question type

| Type | n | EM | F1 | Judge | Doc recall@10 |
|---|---|---|---|---|---|
| 2hop | 167 | 0.359 | 0.462 | 0.461 | 0.862 |
| 3hop | 167 | 0.210 | 0.322 | 0.350 | 0.792 |
| 4hop | 166 | 0.187 | 0.263 | 0.250 | 0.608 |

## Where it fails (wrong answers only)

| Correct (F1≥0.5) | Answer form only* | **Failures** | Retrieval | Reranking | Generation: step not found | Generation: wrong value |
|---|---|---|---|---|---|---|
| 189 | 7 | **304** | 166 (55%) | 37 (12%) | 42 (14%) | 59 (19%) |

*F1 < 0.5 but judged correct (right fact, other wording); not counted as a failure.
Per-question rows: failures.csv
