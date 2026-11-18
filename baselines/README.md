# baselines/

One folder per baseline. Each folder holds everything that belongs to that baseline: its code, its documentation, its
extra dependencies, its tests and its results. Shared machinery (retrieval, reranking, generators, metrics, run logging)
stays in `registerrag/` so every baseline is measured the same way.

| Folder | `--system` | Method | Status |
|---|---|---|---|
| `b1_vanilla_chunk` | `chunk` | vanilla RAG over 256-word chunks | implemented |
| `b2_vanilla_sentence` | `sentence` | vanilla RAG over single sentences | implemented |
| `b3_georag` | `georag` | GeoRAG, information-demand coverage (arXiv 2606.29328) | implemented (training-free, paper algorithms) |
| `b4_hlg_rag` | `hlg` | Hierarchical Lexical Graph RAG (arXiv 2506.08074) | implemented (in-memory graph, own prompts) |
| `b5_sure_rag` | `surerag` | SURE-RAG, sufficiency-aware verification (arXiv 2605.03534) | implemented (architecture; trained parts replaced, see README) |
| `b6_evimem` | `evimem` | EviMem, evidence-gap-driven retrieval (arXiv 2604.27695) | implemented (port of the released IRIS code) |
| `b7_lycheememory` | `lychee` | LycheeMemory V2 (arXiv 2608.12990) | implemented (structure-faithful, reconstructed prompts) |
| `b7b_lychee_acl` | `lycheeacl` | LycheeMemory ACL 2026 (arXiv 2602.08382), released 7B checkpoint | implemented as inference-only reference (no Gate released; own venv, see its README) |
| `b8_pyrag` | `pyrag` | PyRAG, executable multi-hop reasoning (arXiv 2605.12975) | implemented (port of the authors' code, training-free) |
| `b9_liveledger` | `liveledger` | LiveLedger, illusory completion (arXiv 2602.07549) | implemented (prompted updater, out-of-domain adaptation) |
| `ref_no_retrieval` | `none` | reference floor, no retrieval | implemented |
| `ref_iterative_refinement` | `iterative` | reference, decompose and concatenate | implemented |

`python -m registerrag.cli baselines` prints the live version of this table, including the number of finished runs.

## Folder contract
```
baselines/<id>/
  baseline.py        declares SYSTEM, LABEL, DEFAULT_UNIT, pipeline_class(), variants()   (makes it runnable)
  <method>.py        the pipeline code of this baseline (only if it is not the shared one-shot pipeline)
  README.md          paper, official code, what follows the paper, what is ours, how to run, how to read the results
  requirements.txt   dependencies on top of the root requirements.txt
  tools/             scripts that only make sense for this baseline (e.g. diagnostics)
  tests/             tests that run offline with stub models:  python -m pytest -q
  results/           one folder per run, named <system>_<dataset>_<config hash>   (results.jsonl, summary.json, config.json)
    stability/rep<N>/  repetitions of the same config for the run-to-run noise check
  third_party/       (git-ignored) cloned official code, if used for a labelled reference row
  data/              (git-ignored) baseline-specific indexes or caches
```
A folder is picked up automatically as soon as it contains a `baseline.py` (see `registerrag/registry.py`);
without one it is listed as `planned`.

## Reproduction order and commands (PowerShell, repo root, venv active)
```powershell
..\.venv\Scripts\Activate.ps1
python -m pytest -q                                            # 111 tests, offline. If ONE test dies with "Windows fatal exception", rerun (see README_HARNESS.md, Troubleshooting)
ollama pull qwen2.5:3b ; ollama pull qwen2.5:7b ; ollama pull qwen2.5-coder:3b ; ollama pull qwen2.5-coder:7b

# per baseline, in your agreed order (each README has the details, fidelity table and how to read the output):
python scripts/run_baselines.py --methods none,chunk,sentence             # floor, B1, B2            (add --generator qwen2.5:7b for 7B)
python scripts/run_baselines.py --methods pyrag                           # B8
python scripts/run_baselines.py --methods evimem                          # B6  (builds its tuple index first)
python scripts/run_baselines.py --methods georag                          # B3
python scripts/run_baselines.py --methods surerag                         # B5  (then tools/fit_aggregator.py, tools/selective.py)
python scripts/run_baselines.py --methods liveledger                      # B9
python scripts/run_baselines.py --methods hlg                             # B4  (graph build ~1-2 h, cached)
python scripts/run_baselines.py --methods lychee                          # B7  (memory build ~0.5-1 h, cached)
python scripts/run_baselines.py --methods lycheeacl                       # B7b (own venv: run tools/setup_env.ps1 once; stop Ollama models first)
python scripts/baseline_table.py                                          # -> baselines/baseline_table.md
```
Every command is resumable: finished runs are skipped, an interrupted run continues at the next question, an interrupted index
build continues at the next document. Index caches (`baselines/<id>/index/`) and results are meant to be committed.

## Adding a baseline
1. `cp -r baselines/_template baselines/b<N>_<name>` and edit `baseline.py` and `README.md`.
2. Write the pipeline as a subclass of `registerrag.pipelines.BasePipeline`, returning a `PipelineResult`. Use
   `self._retrieve_rank` for retrieval and `self._lm(kind, ...)` for every model call so tokens, calls and latency are
   counted exactly like the other systems. Method-specific settings go through `--method-arg key=value`.
3. Add offline tests under `tests/` (use `registerrag.generate.StubGenerator` or a scripted fake model).
4. Run it: `python scripts/run_baselines.py --methods b<N>`.

## Results
Runs land in `baselines/<id>/results/` automatically. Cross-baseline views:
`python scripts/summarize_runs.py` (all runs), `python scripts/baseline_table.py` (the frozen table, written to
`baselines/baseline_table.md`), `python scripts/by_type.py`, and `python -m registerrag.cli compare <A> <B>` for paired tests.
Per-baseline diagnostics live in that baseline's `tools/`.

## DSIRE (RePolicyQA) - how every baseline sees it (no paper uses this dataset)
- **Data:** `datasets/dsire_data/repolicyqa_100.json`, 100 questions; loader `registerrag/data.py::parse_dsire_records`. One record = (policy chunk, question, free-text answer); the **corpus is the unique policy chunks** of all questions (see `docs` in each run's `index_stats`), pooled across questions. Question types (20 each): Descriptive, Legal-Obligation, Yes-No-cond, Factual, Yes-No. MuSiQue-style pysbd sentence splitting is used.
- **Gold:** document-level only (the question's own chunk); there are **no gold sentences**, so sentence recall/precision are not defined. Because that chunk is in the pool and the pool is small, `doc_recall@10` and "all gold in prompt" are 0.9-1.0 for almost every method: DSIRE hardly separates retrievers, it mostly measures generation.
- **Metric: use token-F1.** Gold answers are free text (mean 21 words); 37 of 100 are plain yes/no. EM is about 0.92 on those 37 and about 0.06 on the other 63, so DSIRE "EM" is yes/no accuracy. `baseline_table.md` therefore shows **F1 as the primary metric for DSIRE** and EM for 2Wiki/MuSiQue.
- **Indexed methods (B4, B6, B7)** build their graph / tuple index / memory over the DSIRE chunks with the same frozen generator, one index per generator (separate cache files). `fallback_chunks` / `fallback_sentences` / `fallback_segments` in `index_stats` say how much of it fell back to verbatim text.
- **Every method is out of domain here** (the papers evaluate Wikipedia multi-hop QA, LoCoMo dialogue, web search or conversation memory); the loader sets `n_hops=1` for every DSIRE question (not annotated) and the answer sits in one policy chunk, so multi-hop machinery (PyRAG programs, LiveLedger constraints, EviMem iterations: 66-72 % of EviMem questions stop after iteration 1) has little to do.
- **B7b (LycheeMemory ACL)** has no DSIRE rows.

## Legacy scripts
`vector-rag/`, `sentence-rag/`, `vector-rag-crossencoder/` and `policy-rag/` are the earlier stand-alone scripts. They are not
part of the registry and are left untouched so the earlier numbers stay reproducible.
