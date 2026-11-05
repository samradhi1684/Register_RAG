# B5 - SURE-RAG: sufficiency- and uncertainty-aware evidence verification (Qiu, Han, Huang, 2026)

- **Paper:** "SURE-RAG: Sufficiency and Uncertainty-Aware Evidence Verification for Selective Retrieval-Augmented Generation", arXiv 2605.03534 (v2 HTML read in full)
- **Official code / data:** none released ("we plan to release ..."). The verifier checkpoint and HotpotQA-RAG v3 are not available.
- **Status:** the **architecture** is reproduced; the **trained components cannot be**, and are replaced as listed below. It is an answer-gating layer, not a retriever: it sits on top of the B1 chunk-RAG pipeline. Tests: `python -m pytest baselines/b5_sure_rag -q`.
- **Code here:** `sure_rag.py`, `tools/fit_aggregator.py`, `tools/selective.py`, `baseline.py`.

## Pipeline as implemented
`question -> B1 retrieval (hybrid + MiniLM rerank, top-5 chunks) -> answer a -> claim c (one LLM call: declarative sentence from (q, a)) -> pair verifier h(c, e_i) = [p_sup, p_ref, p_neu] for every passage -> features z (Table I blocks) -> aggregator g(z) = P(Safe) -> s = P(Safe) − β·u -> answer iff s ≥ τ, else abstain`.
One claim per answer (the paper's main experiments also use single-claim short answers).

## What is replaced (read before citing a B5 number)
| Component | Paper | Here |
|---|---|---|
| Pair verifier `h_θ(q,c,e)` | DeBERTa-v3-base **trained** on HotpotQA-RAG v3 claim-evidence pairs (20,130 pairs) | public NLI cross-encoder `cross-encoder/nli-deberta-v3-base` (same backbone family, **no task training**); hypothesis = declarative claim, premise = passage |
| Features `z` | Table I: coverage fractions, max-support / max-refute / mean-neutral, entropy / disagreement / conflict, (retrieval uncertainty, optional) | same blocks; **formulas are ours** (the paper gives none): `max_sup, max_ref, mean_neu, entropy` (mean normalised pair entropy), `disagreement` (std of p_sup), `conflict = min(max_sup,max_ref)`, `coverage_deficit = 1 − max_sup`, claim-level argmax one-hot as the three fractions. Retrieval-uncertainty omitted (optional in the paper) |
| Aggregator `g_φ` | logistic regression on the training split | logistic regression fitted by `tools/fit_aggregator.py` on **our own labelled runs** (label: base answer is exact-match correct). Binary Safe/Unsafe (the paper's Refuted vs Insufficient needs construction labels we do not have) |
| `τ`, `β` | fitted on the dev split | `τ` = max safe-F1 on out-of-fold scores, `β = 0`; without an aggregator file a heuristic `max_sup·(1−max_ref)` with `τ = 0.5` is used and logged as `heuristic` |
| Calibration | post-hoc (Platt) | none |
| Data | HotpotQA-RAG v3: curated sufficiency/refutation conditions, shortcut-controlled | ordinary multi-hop QA with the B1 pipeline's own answers; the verifier must judge answers that are only multi-hop-entailed by the *set* of passages |

**Known limitation (measured):** with per-passage pairs, a correct multi-hop answer is usually not entailed by any single passage, so `p_safe ≈ 0` for correct answers (30-question pilot: out-of-fold AUROC ≈ 0.3, i.e. uninformative). The paper's own "concat" comparator (all passages in one premise) gave AUROC ≈ 0.73 on a 60-question pilot, so it is registered as a second, clearly-labelled row (`premise=concat`). The pilot sizes are tiny; re-measure on the full 100 / 500 questions.

## Run it (three steps; steps 2-3 need no LLM)
```powershell
..\.venv\Scripts\Activate.ps1
python -m pytest baselines/b5_sure_rag -q
pip install scikit-learn                                      # only for tools/fit_aggregator.py (already present in the venv)

# 1) scoring pass: answers are NEVER withheld, every question stores its features + P(Safe)   (pairs row + concat row)
python scripts/run_baselines.py --methods surerag
python scripts/run_baselines.py --methods surerag --generator qwen2.5:7b

# 2) fit the aggregator on a finished run (cross-fitted; prints out-of-fold AUROC, safe-F1, coverage/accuracy), per row
python baselines/b5_sure_rag/tools/fit_aggregator.py baselines\b5_sure_rag\results\surerag_2wiki_<hash>\results.jsonl --out baselines\b5_sure_rag\aggregator\agg_2wiki_pairs.json

# 3) selective-answering view: coverage vs risk, using the out-of-fold scores (no leakage)
python baselines/b5_sure_rag/tools/selective.py baselines\b5_sure_rag\results\surerag_2wiki_<hash>\results.jsonl --oof baselines\b5_sure_rag\aggregator\agg_2wiki_pairs_oof_scores.json
```
Gated run (abstain when unsafe; withheld answers count as wrong in EM) with a fitted aggregator:
`python -m registerrag.cli run --system surerag --dataset 2wiki --data datasets/wiki2_data/2wiki_100.json --generator qwen2.5:3b --prepend-title --prompt v2 --method-arg aggregator=baselines/b5_sure_rag/aggregator/agg_2wiki_pairs.json`
(`<hash>` = the run folder printed by the run / listed by `python -m registerrag.cli baselines`.)
**Caution:** fitting and gating on the same 100 questions overstates the gain - report the out-of-fold numbers from step 3, or fit on `scripts/make_dev500.py` data and gate on the frozen 100.

Knobs (`--method-arg`): `gate` (true), `aggregator`, `verifier` (HF id; `stub` is test-only), `tau`, `beta`, `premise` (pairs|concat), `claim_tokens` (64).

## Reading the result
Per question: `claim`, `pair_probs` (k×3), `features`, `p_safe`, `u`, `s`, `decision`, `raw_answer`, which verifier and aggregator were used. EM of a scoring pass equals B1's (same answers); the interesting outputs are risk-coverage and AUROC of `p_safe` for "answer correct". The paper reports Risk@30 coverage 0.16 vs 0.26 (mean-pool) on its own benchmark; those numbers do not transfer to this data.
