# B8 - PyRAG: executable multi-hop reasoning (Sun et al., 2026)

- **Paper:** "Retrieval is Cheap, Show Me the Code: Executable Multi-Hop Reasoning for Retrieval-Augmented Generation", arXiv 2605.12975
- **Official code:** https://github.com/GasolSun36/PyRAG (read: `runner.py`, `plan_agent.py`, `decompose_agent.py`, `tools.py`, `code_executor.py`)
- **Status:** ported from the authors' code, **training-free** variant only (the RL-trained 7B `gasolsun/PyRAG-7b` is not used). Tests: `python -m pytest baselines/b8_pyrag -q`.
- **Code here:** `pyrag.py` (pipeline + sandbox), `baseline.py`, `tools/pyrag_stats.py` (did it really run as PyRAG?).

## Method as implemented (authors' behaviour, from their code)
1. **Decompose** agent -> JSON list of atomic sub-queries (3 attempts, fallback `[question]`).
2. **Plan** agent writes a Python program over `retrieve(query)` and `answer(query, docs=None)`; syntax errors go back to the agent (3 attempts).
3. The program is **executed**. A runtime error returns to the Plan agent (`FIX_PROMPT`: failing code + error) for up to **3 fix rounds** (4 executions).
4. **Adaptive retrieval:** if an `answer()` result contains an insufficiency marker (`unknown`, `not found`, `cannot answer`, ...), the `retrieve()` call just before it gets top-k **10** and the **whole program is re-executed once** (not only that step).
5. `answer()` with docs -> evidence prompt, `<thinking>/<answer>` tags, text inside `<answer>` is returned; without docs -> aggregation prompt (final step: `final_answer = answer("Given: ... Answer the question: <original question>")`).
All prompts (decompose, plan, fix, syntax feedback, both answer prompts, the one-shot code example) are the authors', verbatim (`pyrag.py`, top).

## What differs from the authors (state this in the paper)
| Item | Authors | Here |
|---|---|---|
| Retriever / corpus | E5-base over Wikipedia-2018 (Search-R1 setup), top-5 | our BM25+BGE-small hybrid + MiniLM cross-encoder over the benchmark's own paragraphs (256-word chunks), top-5 |
| Agent models | Qwen2.5-7B-Instruct (decompose, answer) + Qwen2.5-Coder-7B (plan), vLLM | frozen generator for every role; code model as planner is a separate, labelled variant |
| Sampling | temperature 0.7 | temperature 0 (harness-wide, reproducible) - so the adaptive re-execution is deterministic except for the boosted calls |
| Sandbox | bare `exec()` | AST validation, restricted builtins, line budget (best effort, research use only) |
| Tool cap | none | 20 tool calls / question (`max_tool_calls`) |
| Final answer | whatever `final_answer` holds | if the program fails or ends "insufficient": ONE forced answer from everything retrieved (same policy as all other baselines); logged as `fallback` |
| Answer length | up to 1024 tokens | 384 (`answer_tokens`); an unclosed `<answer>` is tolerated |

## Run it
```powershell
# one-time (venv lives one level above the repo folder)
..\.venv\Scripts\Activate.ps1
ollama pull qwen2.5:3b ; ollama pull qwen2.5:7b ; ollama pull qwen2.5-coder:3b ; ollama pull qwen2.5-coder:7b
python -m pytest baselines/b8_pyrag -q                        # offline, ~2 s

# quick real-model check (8 questions, throw-away output folder)
python scripts/run_baselines.py --methods pyrag --datasets 2wiki --limit 8 --out runs_smoke

# the real runs: 100 questions x {2wiki, musique} x {same-model planner, code-model planner}
python scripts/run_baselines.py --methods pyrag                               # qwen2.5:3b
python scripts/run_baselines.py --methods pyrag --generator qwen2.5:7b        # qwen2.5:7b
# a crash or interruption is retried and RESUMES; re-running the same command skips finished runs
```
Single run: `python -m registerrag.cli run --system pyrag --dataset 2wiki --data datasets/wiki2_data/2wiki_100.json --generator qwen2.5:3b --prepend-title --prompt v2 --method-arg plan_generator=qwen2.5-coder:3b`

Knobs (`--method-arg k=v`): `plan_generator`, `max_repairs` (3), `topk_retry` (10), `max_tool_calls` (20), `max_lines` (5000), `adaptive` (true), `answer_tokens` (384). `--unit sentence` gives a sentence-retrieval variant (not in the paper).

## Reading the result (always do this before quoting an EM)
```powershell
python baselines/b8_pyrag/tools/pyrag_stats.py --show 2
```
It reports how many questions ran without repair, needed repair, or **fell back to plain RAG**, adaptive-retrieval rate, and EM for each group. If most questions fall back the row measures plain RAG, not PyRAG. Per question the run stores `sub_queries`, `code`, `repairs`, `exec_log`, `retried_with_topk`, `fallback`, `untagged_answers` (answers where the model skipped the `<answer>` tag) in `results.jsonl` -> `extra`.

Expectation set by the paper (Qwen2.5-7B, Wikipedia-2018, EM %): 2Wiki 33.4, MuSiQue 11.8 vs vanilla RAG 18.9 / 4.7. Numbers are **not** comparable to ours (different corpus, 100 questions); use them only to sanity-check direction. At 3B the planner often cannot write valid programs - that is a finding, check `pyrag_stats`.

## Not done
Official code + 7B RL weights as a reference row (needs vLLM and an HTTP `/retrieve` server over our index).
