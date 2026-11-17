# ref - iterative refinement

Decompose the question into sub-questions (JSON mode, 384-token cap), retrieve sentences separately for each, concatenate everything,
answer. No register, no verification, no carried-over answers. Kept as a reference row; the published baselines B6 and B8 supersede it.

- **Code:** `iterative.py`. **Run:** `python scripts/run_baselines.py --methods iterative`.
- **Check decomposition quality:** `python scripts/compiler_stats.py` (a `1` in `#requirements` means no real decomposition).
- **Tests:** `tests/test_iterative.py`. **Results:** `results/`.
