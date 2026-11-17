# B7 - LycheeMemory V2: segment-level consolidation (Li et al., 2026)

- **Paper:** "LycheeMemory V2: Efficient Long-Term Memory for LLM Agents via Semantic Segment-Level Consolidation", arXiv 2608.12990 (v2). Sections 3.1-3.5, Algorithms 1-2 and Appendices A-D read verbatim; Appendix E prompts only as fragments.
- **Official code / checkpoint:** none found (`github.com/lycheemem/lycheemem` returned 404; no Hugging Face weights). It is **training-free**.
- **Open item for you:** the *ACL 2026 LycheeMemory* (Compressor + Gate + RL-trained Reasoner, RULER-HQA, 7B checkpoint) described in an earlier chat is a **different paper** and could not be found. If it is the one you meant for this baseline (it is conceptually closer to a bounded working memory), send its arXiv/ACL link and I will read it and build it as a separate `b7b` row. Until then B7 = V2, the paper you linked (arXiv 2608.12990).
- **Status:** structure-faithful re-implementation; prompts reconstructed from fragments. Tests: `python -m pytest baselines/b7_lycheememory -q`. It is the weakest fit of all nine (dialogue memory, unbounded append-only store, no Wikipedia results in the paper).
- **Code here:** `lychee.py`, `baseline.py`. Memory cache: `index/memory_<generator>_<version>.jsonl` (committed).

## Write path (Alg. 1) - implemented
Each document is a "session", its sentences the "exchanges". Segment boundary probability `p_t = σ(−1.10 + score)` from semantic surprise (`z=(s−0.20)/0.14`, clipped), cohesion drop, token-length pressure (−1.30 below 70% of 300 tokens, rising to 3.00 at 900) and turn-count pressure (−0.85, −0.15, 0.15, min(1, 0.30+0.15(n−4))); finalise at `p ≥ 0.50`, 900 tokens or 10 exchanges. **One LLM call per segment** -> records (`fact|event|preference|constraint|procedure|failure_pattern|tool_affordance`, self-contained text, entities, tags, temporal, evidence turns) + a disambiguation note carried to the next segment. Evidence nodes: entity, topic, entity-topic, temporal (year / month), event-frame. No write-time conflict resolution. Wikipedia paragraphs are short, so most documents stay **one segment**.

## Read path (Alg. 2) - implemented
1 planner LLM call -> 3-8 semantic queries + evidence routes. Per route four channels: record-vector search, evidence-node search (entity/topic keys found in the query -> linked records), temporal keys, raw-sentence vector search (-> records citing those sentences). Cross-encoder rerank (MiniLM instead of bge-reranker-v2-m3), **RRF over routes (κ=60)**, route-coverage then **MMR** (`0.75·rel − 0.25·sig`), `k = max(top_k, 15)` records, 1 answer LLM call (answer prompt fragments verbatim: "Avoid double-counting ...", "prioritize the most recent memory", "less than 5-6 words", "Output only the answer phrase").
Note: B7 hands the generator **15 records + their raw sentences** (its own k_plan), not 5 - state that when comparing prompt sizes (`peak_prompt_tokens`).

## Fidelity table
| Item | Authors | Here |
|---|---|---|
| Models | GPT-4.1-mini / 4o-mini, text-embedding-3-small, bge-reranker-v2-m3 | frozen qwen2.5 generator, BGE-small, MiniLM cross-encoder |
| Stores | LanceDB + SQLite FTS5 | in-memory numpy + python dicts |
| Weights inside `score` | not given (`σ(−1.10 + score)`) | ours: `0.5·z_s + 2.0·d + L + N` |
| Encoder / planner prompts | Appendix E (dialogue-specific rules: speakers, "I/me/my") | reconstructed from the published fragments, dialogue rules dropped (`ENCODE_SYSTEM`, `PLAN_SYSTEM`) |
| Channel score fusion, node scoring | not specified | max over channels; node channel scored by cosine to the linked records; temporal +0.05 bonus |
| Conversation memory | one memory per conversation | pooled corpus (or `--corpus-scope per_question`) |

## Run it
```powershell
..\.venv\Scripts\Activate.ps1
python -m pytest baselines/b7_lycheememory -q

# 1) consolidation (cached, RESUMABLE; ~2 s/document at 3B -> ~35 min for 2wiki-100, ~70 min for musique-100)
python scripts/build_index.py --system lychee --dataset 2wiki   --data datasets/wiki2_data/2wiki_100.json     --generator qwen2.5:3b
python scripts/build_index.py --system lychee --dataset musique --data datasets/musique_data/musique_100.json --generator qwen2.5:3b

# 2) runs (build is automatic if not done; --no-prebuild to skip)
python scripts/run_baselines.py --methods lychee                              # full method + "no planner" ablation row
python scripts/run_baselines.py --methods lychee --generator qwen2.5:7b
```
Knobs (`--method-arg`): `delta` 0.5, `seg_min_tokens` 300, `seg_max_tokens` 900, `seg_max_exchanges` 10, `k_plan` 15, `rrf_kappa` 60, `rerank` true, `mmr` true, `plan` true, `short_answer` true, `cache_dir`.

## Reading the result
Per question: `plan_ok`, semantic queries, #routes, #candidates, the records in the prompt, `index_stats` (documents, segments, records, `fallback_segments` = segments where the 3B encoder failed and sentences were stored verbatim). Do not compare its EM to the paper's LoCoMo 89% (different task, GPT-4.1-mini).
