# B3 - GeoRAG: information-demand coverage optimisation (Zhang, Jia, Zhu, 2026)

- **Paper:** "Covering the Unseen: Information Demand Coverage Optimization for Retrieval-Augmented Generation", arXiv 2606.29328 (full HTML text read, all equations and both algorithms)
- **Official code:** none linked in the paper.
- **Not the other "GeoRAG"** (Wang et al. 2025, geographic QA with a BERT query classifier) - different paper.
- **Status:** training-free re-implementation of Algorithms 1 and 2. Tests: `python -m pytest baselines/b3_georag -q` (Sinkhorn cost checked against analytic values, greedy picks one chunk per demand peak, etc.).
- **Code here:** `georag.py`, `baseline.py`.

## Method (paper defaults, all implemented)
1. Candidate pool `C` = top **K=200** of the base retriever (here: BM25+BGE hybrid, **no reranker** - GeoRAG replaces top-k truncation).
2. **Demand proxy:** the LLM writes **n=20** sub-queries (temperature **0.7**); greedy max-min cosine keeps **m=10** (Eq. 6). *Reverse validation* (Eq. 7): retrieve top-20 per sub-query, quality = overlap with `C` x novelty vs the other sub-queries; drop quality < **0.05**; normalise (Eq. 8). Candidate weights `w_i ∝ r_i^α · L_i^β`, `r = softmax(retrieval score)`, `L_i = Σ_j q̃_j max(cos(c_i,q_j),0)`, α=β=1 (Eqs. 9-11).
3. **Greedy selection** (Alg. 1) of **k=5**: cold start = cosine top-1; then `argmax (1 − max_{s∈S} cos(c,s)) · (W_ε(P_Q,E_S) − W_ε(P_Q,E_{S∪c}))` with `M_ij = 1 − cos`, **ε=0.01**, **100** log-domain Sinkhorn iterations (Alg. 2). The gate is unclipped, as in Eq. 12.
4. The 5 chunks go to the generator in selection order.

## What is ours (the paper does not specify it)
| Item | Authors | Here |
|---|---|---|
| Sub-query prompt | not in the paper | ours (`SUBQ_SYSTEM`), one call asking for 20 sub-queries as JSON |
| Softmax temperature of Eq. 9 | not stated | 1.0 (`prior_temp`); note that with scores in [0,1] the retrieval prior is then nearly flat and `L_i` dominates |
| Start of the max-min filter | not stated | the sub-query closest to the question |
| Embedding model / retriever | Qwen3-Embedding-8B, dense, 100-word chunks, gold-injected 1M-chunk corpus | BGE-small + BM25 hybrid, 256-word chunks (same units as B1), benchmark's own paragraphs |
| LLM | Qwen3-4B for sub-queries and answers | frozen generator (sampling copy at T=0.7 for sub-queries; answers at T=0) |
| Numerics | - | demand marginal floored at 1e-30 before the log |
| Answer prompt | not reported | the harness' frozen forced-answer prompt v2 (same as B1/B2) |

**Caveat from the paper itself:** the 1-1/e guarantee is proven for the exact facility-location objective, not for the Sinkhorn surrogate that is deployed; and the headline tables use a corpus that contains the gold passages of every query. Our pooled benchmark corpus is similar in spirit (it contains the gold paragraphs plus distractors).

## Run it
```powershell
..\.venv\Scripts\Activate.ps1
python -m pytest baselines/b3_georag -q
python scripts/run_baselines.py --methods georag --datasets 2wiki --limit 8 --out runs_smoke   # quick real-model check

python scripts/run_baselines.py --methods georag                              # 3B: chunk variant + sentence variant, both datasets
python scripts/run_baselines.py --methods georag --generator qwen2.5:7b      # 7B
```
Expected cost: 1 retriever call + 10 reverse-validation retrievals + 1 sub-query LLM call + 1 answer call; ~3 s/question at 3B. No offline index.

Knobs (`--method-arg`): `pool_k` 200, `sub_n` 20, `sub_m` 10, `sub_temperature` 0.7, `tau_rv` 0.05, `alpha`, `beta`, `epsilon` 0.01, `sinkhorn_iters` 100, `prior_temp` 1.0, `rv_k` 20. Budget sweeps: `--top-k 20 --budget 1024` works (the top-k selected chunks are cut to the budget).

## Reading the result
`extra` per question: generated/kept sub-queries, their quality weights, how many survived the 0.05 filter (`n_alive`), `selected_ranks` (positions in the retriever's order), `demand_entropy`, and whether it fell back to a question-only demand (`fallback_demand`: sub-query JSON failed). In `ranking`, the 5 selected chunks come first, so `doc_recall@5/sent_recall@5` measure GeoRAG's selection directly. Compare against B1 (cosine/reranked top-5 of the same chunks) - the paper's claim is +6.5-7.5 EM over top-k truncation.
