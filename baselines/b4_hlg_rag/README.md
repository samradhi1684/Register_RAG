# B4 - HLG-RAG: Hierarchical Lexical Graph (Ghassel et al., KDD 2025)

- **Paper:** "Hierarchical Lexical Graph for Enhanced Multi-Hop Retrieval", arXiv 2506.08074. Read verbatim: abstract through Sec. 5.2 step 3 (graph model, StatementGraphRAG, TopicGraphRAG, diversity filter). **Not read verbatim:** experiments, results tables, appendices - hyper-parameter values below are from a summary and marked.
- **Official code:** AWS `graphrag-toolkit` lexical-graph (`pip install graphrag-lexical-graph`; needs Neptune/Neo4j/FalkorDB + OpenSearch/Postgres). Its retrievers are *analogues* (`TopicBeamSearch`, TF-IDF reranker by default), not the paper's `StatementGraphRAG`. **Not used** as the primary row.
- **Status:** in-memory re-implementation of the three-tier graph and both retrievers. Tests: `python -m pytest baselines/b4_hlg_rag -q`.
- **Code here:** `hlg.py`, `baseline.py`. Graph cache: `index/graph_<generator>_<version>.jsonl` (committed).

## Graph (built offline with the generator, 2 LLM calls per chunk)
- **Lineage:** Source (document) -> Chunk (sequentially linked, <= 256 words).
- **Summarisation:** *Statement* = atomic proposition (the retrieval unit; embedded), *Topic* (per document; embedded with its statements), *Fact* (subject, predicate, object) that SUPPORTS statements.
- **Entity-relationship:** *Entity* (name, classification) linked to the statements/facts that mention it.
- Extraction: call 1 = propositions (atomic, coreferences resolved); call 2 = topics / entities / facts that **refer to propositions by number** (never rewrites them). Chunks whose extraction fails keep their sentences as statements with capitalised-span entities (`fallback_chunks`).

## Retrieval
- **StatementGraphRAG** (default): keywords+synonyms (LLM) -> `Score_kw(s) = |keywords matching an entity of s|`, ties by cosine -> top `kw_k`; vector search top `vss_k`; union -> **graph beam search** over statements sharing an entity (path embedding = attention-weighted sum, weights softmax of cosine to the query; path score = cos(query, path); expand only while a child improves its parent; depth `max_depth`, beam `beam_width`) -> cross-encoder rerank -> **TF-IDF diversity filter** (drop a statement whose similarity to a higher-scored one exceeds 1−τ, τ = 0.5%) -> top-k statements into the prompt.
- **TopicGraphRAG** (`retriever=topic`): top-down topic search (`topic_k` topics by cosine to topic embeddings) + bottom-up entity search -> merged -> beam search scored by the reranker along the path -> rerank -> filter -> top-k.

## Fidelity table
| Item | Authors | Here |
|---|---|---|
| Graph store / vector store | Neptune Analytics + OpenSearch | in-memory dicts + numpy |
| Extraction LLM | Claude 3 Sonnet (summary) | frozen qwen2.5 generator; **prompts are ours** (the AWS prompts were not verifiable verbatim); JSON instead of pipe-delimited output |
| Embeddings / reranker | Cohere embed v3 / bge-reranker-v2-minicpm-layerwise (summary, unverified) | BGE-small / MiniLM cross-encoder (harness-wide) |
| Beam search | per-start expansion | one **global** beam of width B over all seeds; neighbours capped at `max_neighbors=50` per statement by cosine |
| Hyper-parameters | symbols only in the verbatim text; summary: 100 VSS seeds, beam 50, depth 3, 50 topics, τ=0.5% (**unverified**) | exactly those values (`vss_k=100, beam_width=50, max_depth=3, topic_k=50, diversity_tau=0.005`), `kw_k=50` ours |
| Context | 10 chunks (~3,000 tokens) | the harness' top-5 **statements** (frozen top-k, like B2); `--top-k 10` for more |
| Gold-unit metrics | n/a | each statement is mapped to its nearest sentence of the same document, **for the recall/precision metrics only** |

## Run it
```powershell
..\.venv\Scripts\Activate.ps1
python -m pytest baselines/b4_hlg_rag -q

# 1) graph build (cached, RESUMABLE; ~3 s per document at 3B -> about 1 h for 2wiki-100, about 2 h for musique-100; do it overnight)
python scripts/build_index.py --system hlg --dataset 2wiki   --data datasets/wiki2_data/2wiki_100.json     --generator qwen2.5:3b
python scripts/build_index.py --system hlg --dataset musique --data datasets/musique_data/musique_100.json --generator qwen2.5:3b

# 2) runs (run_baselines.py also does step 1 if it was not done; add --no-prebuild to skip)
python scripts/run_baselines.py --methods hlg                              # StatementGraphRAG + TopicGraphRAG rows
python scripts/run_baselines.py --methods hlg --generator qwen2.5:7b      # 7B builds its own graph (the extractor differs)
```
Knobs (`--method-arg`): `retriever` statement|topic, `vss_k`, `kw_k`, `beam_width`, `max_depth`, `max_neighbors`, `diversity_tau`, `topic_k`, `chunk_words`, `rerank`, `cache_dir`.

## Reading the result
Per question: keywords, #seeds / #after beam / #after diversity filter, the statements shown to the generator, `index_stats` (statements, topics, entities, facts, fallback chunks). If `fallback_chunks` is a large share, the 3B extractor failed and the graph degenerates to sentence retrieval with weak entities - say so. The paper's gain (23.1% relative retrieval recall over chunk RAG) was measured with a strong extractor and 5 other datasets.
