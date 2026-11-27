# Report: runs\run2_500ctx\2wiki\test100 (2wiki, n=100)

## Answer quality and retrieval

| n | EM | F1 | LLM judge | Yes/no acc | Doc recall@10 | Pool recall |
|---|---|---|---|---|---|---|
| 100 | 0.550 | 0.645 | 0.645 | 0.714 (n=7) | 0.895 | 0.917 |

## Cost

| Latency mean (s) | Latency p95 (s) | Total time (min) | LLM calls / q (pipeline count) | LLM requests / q (measured) | Prompt tokens / q | Completion tokens / q | Total tokens / q | Total tokens (run) |
|---|---|---|---|---|---|---|---|---|
| 11.5 | 16.7 | 19.2 | 3.4 | 22.2 | 6699 | 274 | 6973 | 697,305 |

## By question type

| Type | n | EM | F1 | Judge | Doc recall@10 |
|---|---|---|---|---|---|
| bridge_comparison | 25 | 0.680 | 0.680 | 0.640 | 0.720 |
| comparison | 25 | 0.960 | 0.960 | 0.940 | 1.000 |
| compositional | 25 | 0.280 | 0.437 | 0.620 | 0.980 |
| inference | 25 | 0.280 | 0.502 | 0.380 | 0.880 |

## Where it fails (wrong answers only)

| Correct (F1≥0.5) | Answer form only* | **Failures** | Retrieval | Reranking | Generation: step not found | Generation: wrong value |
|---|---|---|---|---|---|---|
| 68 | 2 | **30** | 8 (27%) | 3 (10%) | 4 (13%) | 15 (50%) |

*F1 < 0.5 but judged correct (right fact, other wording); not counted as a failure.
Per-question rows: failures.csv
