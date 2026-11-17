# B6 - EviMem: evidence-gap-driven iterative retrieval (Li et al., 2026)

- **Paper:** "EviMem: Evidence-Gap-Driven Iterative Retrieval for Long-Term Conversational Memory", arXiv 2604.27695 (v2 read in full)
- **Official code:** https://github.com/AIGeeksGroup/EviMem (read: `IRIS_retriever_v5.py`, `IRIS_runner_v5.py`, `retriever.py`, `models.py`)
- **Status:** training-free port of the released retrieval code (IRIS) + our re-creation of the LaceMem memory build. Tests: `python -m pytest baselines/b6_evimem -q`.
- **Code here:** `evimem.py`, `baseline.py`. Index cache: `index/tuples_<generator>_<version>.jsonl` (committed - it is an intermediate artifact).

## Mapping conversational memory -> Wikipedia QA
| LaceMem (paper) | Here |
|---|---|
| Raw layer: verbatim speaker turns | the corpus' verbatim **sentences** (`--unit sentence`) |
| Index layer: LLM-factorised (subject, predicate, object, time) tuples linked to their turn | same: one LLM extraction call per **document** (12 sentences per call); a sentence the model skips enters the index verbatim (`fallback_sentences`, never lost) |
| Edge layer: same-source edges + semantic-similarity edges | same-sentence edges (weight 1.0) + top-3 cosine neighbours with sim >= 0.7 (weight = sim) |
| one conversation = one memory | `--corpus-scope pooled` (default, all questions' documents) or `per_question` (only that question's documents, closest to LoCoMo) |

## IRIS loop (followed from the released CODE where it differs from the paper)
`k = 3` iterations. Each: **anchor** retrieval (original question, top `max(5, k_ret//2)` = 5, 6, 8) + **refined** retrieval (diagnosis-driven query, top `10+3(i-1)` = 10, 13, 16) + 1-hop edge expansion (edge weight >= 0.5) -> accumulated, de-duplicated (first occurrence wins; order = anchor, refined, expanded) -> **sufficiency judge** (sees the first 20 facts) returns EXACT / INFERRABLE / PARTIAL, CONFIDENCE, MISSING -> stop if EXACT and conf >= 0.7 (0.85 temporal) or INFERRABLE and conf >= 0.7, else **refine** the query from MISSING with the rule-based per-iteration strategy -> after the loop: optional multi-hop reasoning-chain call, then **answer** from the first 30 facts with a tier-adaptive instruction. Sufficiency / refinement / answer / reasoning-chain prompts are the authors', verbatim.

## Where this deviates from the paper or the authors' setup
| Item | Authors | Here |
|---|---|---|
| Paper vs code | Alg. 2: same budget for both paths, entity-count downgrade (delta=2, conf cap 0.6), entity hints in refinement | **code followed**: halved anchor budget; entity tracker is log-only in their code, so not implemented |
| Models | GPT-4o (answer), GPT-4o-mini (judge/refine), text-embedding-3-small | one frozen generator for every role; BGE-small embeddings (cosine threshold 0.5 kept - with BGE it rarely binds, top-k bounds the set) |
| Tuple extraction prompt | not published in the files we could read | ours (`EXTRACT_SYSTEM`) |
| Temperature | 0 / 0.2 / paper says 0.3 | 0 everywhere (harness) |
| "Not mentioned" abstention | returns the literal "Not mentioned" | **disabled** (forced-answer policy of all baselines); `would_abstain` is logged (`allow_abstain=true` enables it) |
| Answer format | sentence-style, LLM judge | + one line asking for the shortest answer (EM/F1 are the metrics here); `short_answer=false` removes it |
| Judge parser | substring `exact:` (misreads `EXACT_MATCH:`) | accepts both |
| Temporal detection | LoCoMo-tuned regexes | `when / how long / what date|year|time / which year / how many years` |

## Run it
```powershell
..\.venv\Scripts\Activate.ps1
python -m pytest baselines/b6_evimem -q

# 1) build the Index layer once (cached, RESUMABLE; ~1 s/document at 3B -> ~15 min for 2wiki-100, ~30 min for musique-100)
python scripts/build_index.py --system evimem --dataset 2wiki   --data datasets/wiki2_data/2wiki_100.json       --generator qwen2.5:3b
python scripts/build_index.py --system evimem --dataset musique --data datasets/musique_data/musique_100.json   --generator qwen2.5:3b

# 2) runs (scripts/run_baselines.py also does step 1 automatically; add --no-prebuild to skip it)
python scripts/run_baselines.py --methods evimem                              # 3B : variant A (answer from tuples) + variant B (grounded in raw sentences)
python scripts/run_baselines.py --methods evimem --generator qwen2.5:7b      # 7B (separate index: tuples depend on the extractor)
```
Single run: `python -m registerrag.cli run --system evimem --dataset 2wiki --data datasets/wiki2_data/2wiki_100.json --generator qwen2.5:3b --prepend-title --prompt v2 --method-arg answer_context=raw`

Knobs (`--method-arg`): `max_iterations` 3, `judge_facts` 20, `answer_facts` 30, `embedding_threshold` 0.5, `edge_k` 3, `edge_min_sim` 0.7, `edge_min_weight` 0.5, `answer_context` facts|raw, `reasoning_chain` true, `short_answer` true, `allow_abstain` false, `window` 12, `cache_dir`.

## Reading the result
Per question, `extra` holds every iteration (query, #anchor/#refined/#expanded, judge verdict + confidence + parse flag, stop reason), the facts shown to the answer prompt, `would_abstain`, `judge_parse_failures`, and `index_stats` (tuples vs verbatim-fallback sentences). Check `judge_parse_failures` and the share of runs stopping at iteration 1 vs 3 before drawing conclusions. Latency of the first question of a run includes index loading only; building is a separate step.
Caveats to state: the Index quality depends on the 3B extractor; the sufficiency judge sees only the first 20 facts and edge-expanded facts come last (authors' ordering), so expansion often does not reach the judge.
