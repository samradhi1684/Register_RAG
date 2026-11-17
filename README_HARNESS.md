# RegisterRAG harness (v0.4: all nine baselines implemented)

One shared package (`registerrag/`) measures every system the same way; every baseline lives in its own folder under
`baselines/` with its code, README, extra dependencies, tests and results (see `baselines/README.md`).

```
registerrag/      shared machinery: units, data, hybrid retrieval, rerank, generators, ledger, metrics, runner, registry, CLI
baselines/<id>/   baseline.py, <method>.py, README.md, requirements.txt, tools/, tests/, results/
scripts/          cross-baseline tools: run_baselines, summarize_runs, baseline_table, by_type, stability, compare helpers, data tools
tests/            tests of the shared package
datasets/         the frozen 100-question testbeds and gold-program drafts
```

## Everyday commands (PowerShell or bash, from the repo root, venv active)
```
python -m pytest -q                                   # all tests, offline (stub models); no PYTHONPATH needed
python -m registerrag.cli baselines                   # which baselines exist, status, number of finished runs
python scripts/run_baselines.py                       # floor, B1, B2, iterative reference  (3B, 2Wiki + MuSiQue)
python scripts/run_baselines.py --generator qwen2.5:7b --methods none,chunk,sentence
python scripts/run_baselines.py --methods pyrag       # B8, same-model and code-model planners
python scripts/run_baselines.py --methods chunk --rep 1    # stability repetition -> baselines/b1_vanilla_chunk/results/stability/rep1
python scripts/summarize_runs.py [--hops]             # every finished run; or pass a baseline folder / results dir
python scripts/baseline_table.py                      # the frozen table -> baselines/baseline_table.md
python scripts/by_type.py                             # EM per 2Wiki question type (shows the guessing floor)
python scripts/stability.py --baseline b1_vanilla_chunk
python -m registerrag.cli compare <runA>/results.jsonl <runB>/results.jsonl     # paired bootstrap + McNemar
python baselines/b8_pyrag/tools/pyrag_stats.py        # did PyRAG really run as PyRAG, or fall back to plain RAG?
```
Single run: `python -m registerrag.cli run --system pyrag --dataset 2wiki --data datasets/wiki2_data/2wiki_100.json --generator qwen2.5:3b --prepend-title --prompt v2`
(output goes to that baseline's `results/` unless `--out` is given; method settings via `--method-arg key=value`).

## Baselines B1-B9 (details and fidelity tables in each `baselines/<id>/README.md`)
`chunk` B1, `sentence` B2, `georag` B3, `hlg` B4, `surerag` B5, `evimem` B6, `lychee` B7, `pyrag` B8, `liveledger` B9, plus reference rows `none`, `iterative`.
`python scripts/run_baselines.py --methods <system|folder|b#|all> [--generator qwen2.5:7b] [--datasets 2wiki,musique] [--limit N --out runs_smoke]`.
B4/B6/B7 have an offline index; `scripts/build_index.py` builds it (cached, resumable) and `run_baselines.py` calls it automatically.

## Troubleshooting
- **`Windows fatal exception: access violation` in a native library (torch, numpy)** - intermittent on this machine and not caused by the
  harness. `run_baselines.py` sets single-threaded BLAS (fewer crashes), retries (`--attempts 3`) and resumes. In `pytest`, a lone failing
  test that passes on rerun is this; real logic failures reproduce every time. Optional for interactive shells:
  `$env:OPENBLAS_NUM_THREADS=1; $env:OMP_NUM_THREADS=1`.
- **Ollama:** `ollama list` must show the generator; `ollama pull qwen2.5:7b` etc. The runner pings the model before every run.
- **First question much slower than the rest:** an index was built inside the run - use `scripts/build_index.py` first (done automatically by `run_baselines.py`).

## Moving earlier results into the new layout
`python scripts/migrate_runs.py` shows what would move (dry run); add `--apply` to move `runs/` and `runs_stab/` into the
baseline folders. Run folders are named by a hash of their config, so moved runs still resume and still match new runs.

## Frozen settings (change only deliberately; the config hash records them)
| Setting | Value |
|---|---|
| title prepended to units | on (`--prepend-title`) |
| prompt | v2, forced answer (v1 may abstain and halved EM) |
| top-k | 5 (budget sweeps: `--top-k 20 --budget 512..4096`) |
| retrieval | BM25 + BGE-small hybrid (lambda 0.5), MiniLM cross-encoder rerank |
| generators | qwen2.5:3b and qwen2.5:7b via Ollama (every baseline at both sizes) |
| corpus | pooled over the loaded questions (state this in every table) |

## Known limits
- Run-to-run noise on a GPU is about 1-2 EM points at n=100 (Ollama is not bit-deterministic): compare paired, report mean +/- sd over repetitions, use 500 questions for gated comparisons.
- MuSiQue has paragraph-level gold only; sentence-level recall is reported for 2Wiki/HotpotQA.
- This Windows machine has crashed intermittently in native libraries; `run_baselines.py` retries and resumes, and `summary.json` records the environment.
- Legacy stand-alone scripts in `baselines/{vector-rag,sentence-rag,vector-rag-crossencoder,policy-rag}` are untouched.
