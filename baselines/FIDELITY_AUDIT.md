# Baseline fidelity audit (2026-11-18)

Scope: B1-B9, B7b and the two reference rows, every variant that has a row in `baselines/baseline_table.md`
(2Wiki, MuSiQue, DSIRE; qwen2.5:3b and :7b; 100 questions each).

**Short answer: none of the baselines is a 100 % faithful reproduction of its paper, and none can be with this setup.**
Every row shares the same departures (section 1). On top of that each method has its own (section 3). The rows are
*controlled comparisons under one harness*, not reproductions of the published numbers. Section 4 defines the ground
each row can honestly claim, and section 5 lists what would have to change to get closer.

## 0. How this audit was done (what is verified, what is not)

| Check | Result |
|---|---|
| Authors' PyRAG repo cloned (`GasolSun36/PyRAG`) and every prompt string compared with `b8_pyrag/pyrag.py` | decompose prompt 100 % identical; the two answer prompts 97-98 % (removed: a few placeholder tokens and Chinese temporal keywords "谁先 / 何时") |
| Authors' EviMem repo cloned (`AIGeeksGroup/EviMem`), `IRIS_retriever_v5.py` prompt segments compared with `b6_evimem/evimem.py` | 18 of 24 prompt segments verbatim. Not verbatim: the entity-extraction prompt (entity tracker not implemented, log-only upstream) and four answer-tier instructions (temporal / inference / direct) |
| EviMem tuple-extraction prompt | The README said it was not published; **it is** (`src/memory_save_system/prompts/index_manager.py`, 10 KB). README now corrected. It is written for conversations (speaker, `record_time`, memory types, output via `memory_index_insert` tool calls), so it is **not a drop-in** for Wikipedia text; our `EXTRACT_SYSTEM` stays a labelled deviation. (My first version of this audit called it "avoidable"; that overstated it.) |
| LycheeMemory ACL repo cloned (`owoakuma/LycheeMemory`) | contains `example.py`, `test.json`, `src/` only: no Gate, no evaluation script, no prompts. Matches the B7b README. |
| Code defaults vs README for B3 (K, n, m, tau, eps, 100 Sinkhorn iters, T=0.7), B4 (100/50/3/50, tau 0.005), B7 (delta 0.5, bias -1.10, z=(s-0.20)/0.14, 300/900, kappa 60, 0.75/0.25), B9 (6 steps, lag, quote guard) | all match |
| Run-to-run noise (`scripts/stability.py`) | **B1 2Wiki, identical config, 3 repetitions: EM 0.38, 0.39, 0.37; 8 questions get a different answer. The table row for the same config says 0.35.** MuSiQue: 0.09 x3, but 11 questions differ. |
| NOT verified | the papers themselves were not re-read in this audit; fidelity to the paper text beyond the constants above rests on the READMEs. B3/B4/B5/B7/B9 have no authors' code to diff against. |

## 1. Departures shared by every row (the "harness gap")

| # | Item | Papers | Here | Effect |
|---|---|---|---|---|
| H1 | Generator | GPT-4o / GPT-4o-mini / Qwen3-4B / Qwen2.5-7B (RL) / gpt-oss-120b, depending on the paper | one frozen qwen2.5:3b or :7b (Q4 via Ollama) for **every** role (planner, judge, extractor, updater, answerer) | methods that need strong instruction following (B4, B6, B7, B8, B9) are degraded; the README for each says so |
| H2 | Corpus and retriever | Wikipedia-2018 + E5 / Qwen3-Embedding-8B / web search | the benchmark's own paragraphs (gold + distractors), BM25 + BGE-small hybrid, MiniLM reranker | much easier retrieval than open-domain; `doc_recall@10` is 0.6-0.9 |
| H3 | Context | method-specific (10 chunks, 30 facts, ...) | frozen top-5 units, forced-answer prompt v2, temperature 0, no abstention | a method that wins by showing more context or abstaining is neutralised |
| H4 | Sample | full test sets or 500-1000 questions | 100 questions per dataset | 95 % CI is about +/- 9 EM points; most between-method gaps are inside it |
| H5 | Nondeterminism | - | a temperature-0 config does not reproduce itself (see stability result above). The machine has a confirmed CPU fault (silent wrong results in the per-core test), and Ollama batching can also change outputs | treat gaps under about 4 points as noise |
| H6 | DSIRE | none of the papers uses it | indexes (B4 graph, B6 tuples, B7 memory) are built per dataset with the same 3B/7B model; now documented in `baselines/README.md` ("DSIRE" section). **Metric gap found and fixed:** DSIRE answers are free text (mean 21 words, 37/100 plain yes/no), so EM is yes/no accuracy (0.92 on those, 0.06 on the rest). The table now reports F1 as primary for DSIRE | DSIRE rows are out-of-domain for every method, and the EM numbers quoted for DSIRE earlier in this project measured yes/no only |

## 2. Fidelity grade per baseline

| Row | Grade | Why |
|---|---|---|
| B1 chunk, B2 sentence | **control** (no paper) | shared `OneShotRAG`; only the unit differs |
| ref none, ref iterative | control / reference | iterative is our own decompose-and-concatenate, not a published method |
| B8 PyRAG | **A-: port of the authors' code, training-free** | prompts essentially verbatim; loop (3 fix rounds, adaptive top-10 re-execution) from their code; no RL weights; own sandbox |
| B6 EviMem | **B+: port of the released retrieval code + our memory build** | IRIS loop and prompts mostly verbatim; Index build is ours (and the authors' prompt was available but not used) |
| B3 GeoRAG | **B: algorithms 1-2 from the paper, no authors' code** | all constants match; sub-query prompt, prior temperature and filter start are ours |
| B9 LiveLedger | **C+: method re-implemented, core component swapped** | fine-tuned 4B updater replaced by the frozen prompted model; agent, tool and domain all different |
| B4 HLG-RAG | **C+: re-implementation, hyper-parameters unverified** | graph model and both retrievers implemented; extraction prompts ours; 100/50/3/50/0.5 % come from a summary, not the verbatim paper text |
| B7 LycheeMemory V2 | **C: structure-faithful, prompts reconstructed** | no code or weights exist; prompt fragments only; dialogue memory applied to Wikipedia paragraphs |
| B5 SURE-RAG | **C-: architecture only** | the trained DeBERTa verifier and trained aggregator are replaced by an untrained public NLI model and a logistic fit on our own labels; known measured failure (AUROC about 0.3) |
| B7b LycheeMemory ACL | **D: not a RAG baseline; strict EM is meaningless for it** | released inference code without the Gate, different generator (own 7B), vendor patch. EM 0.00 is an **output-format effect** (the checkpoint answers in full sentences; all 200 questions were answered, no errors): lenient contains-acc is 0.65 (2Wiki) and 0.35 (MuSiQue), see `b7b_lychee_acl/README.md`. (First version of this audit wrongly blamed timeouts.) |

## 3. Per-baseline analysis of the differences, and why they exist

### B1 / B2 - vanilla RAG (control)
No deviation from a paper because there is none. Role: the floor every method must beat. Because the gap between methods is mostly a few points (H4, H5), B1 and B2 are the only rows whose noise is directly measured.

### B3 - GeoRAG (arXiv 2606.29328)
- Faithful: pool K=200, 20 sub-queries -> 10 by max-min cosine, reverse validation (tau 0.05), Sinkhorn eps 0.01 with 100 log-domain iterations, greedy k=5 with the unclipped novelty gate, no reranker.
- Ours (paper silent): sub-query prompt, softmax temperature of Eq. 9 (1.0; with scores in [0,1] the retrieval prior is nearly flat), start of the max-min filter, floor 1e-30.
- Different by necessity: Qwen3-Embedding-8B dense retrieval over 100-word chunks of a 1M-chunk corpus replaced by BGE+BM25 over 256-word chunks; Qwen3-4B replaced by the frozen generator.
- Evidence the deviation matters: `doc_recall@10` is as high as B1's (0.78 vs 0.83 on 2Wiki 3B) but "all gold in prompt" collapses to 0.05 (B1 0.53). The selector picks diverse chunks and drops the gold pair. Not a bug per se, but it means the paper's claim (+6.5-7.5 EM) is not reproduced here (B3 is 7 points *below* B1 on 2Wiki at 3B, 0.28 vs 0.35, and 14 points below at 7B, 0.23 vs 0.37).
- Note on the two table rows: "unit=sentence" is **not in the paper**; it is our extra variant.

### B4 - HLG-RAG (arXiv 2506.08074)
- Faithful: three-tier graph (lineage, summarisation, entity-relationship), StatementGraphRAG (keywords -> entity score, vector seeds, beam search with attention-weighted path embedding, rerank, TF-IDF diversity filter) and TopicGraphRAG.
- Ours: all extraction prompts and JSON output format (AWS prompts not verifiable verbatim); global instead of per-start beam; neighbour cap 50; `kw_k=50`.
- Unverified: the hyper-parameters (100 seeds, beam 50, depth 3, 50 topics, tau 0.5 %) were taken from a summary; the README says the experiment section and appendices were not read verbatim.
- Cause of the damage: the paper used Claude 3 Sonnet for extraction; a 3B model produces weak propositions/entities (`fallback_chunks`). Context is top-5 *statements*, not 10 chunks (about 3,000 tokens) as in the paper.
- Result pattern: 2Wiki 3B 0.36, but 2Wiki 7B 0.27 (lower than 3B): a 7B extractor builds a *different* graph, which shows the row measures the extractor as much as the method.

### B5 - SURE-RAG (arXiv 2605.03534)
- Faithful: the pipeline shape (answer -> claim -> pair verifier -> features -> aggregator -> score s = P(Safe) - beta*u -> gate).
- Replaced: the DeBERTa verifier **trained** on 20,130 HotpotQA-RAG v3 pairs by an off-the-shelf NLI model; the feature formulas are ours (paper gives none); the aggregator is fitted on our own labels; no calibration.
- **Important: the table rows are gate=False ("scores only").** SURE-RAG is an answer-gating layer; with the gate off the EM equals B1 by construction. The table rows therefore do **not** measure SURE-RAG at all. The documented finding (per-passage verifier AUROC about 0.3) says the component is uninformative here.
- Discrepancy to explain: README says scoring-pass EM equals B1; the table shows 0.37 vs 0.35 (2Wiki 3B) and 0.38 vs 0.37 (7B). Given the stability result this is noise, but the claim "identical answers" does not hold in practice.
- Honest claim: B5 rows = B1 + an unfitted verifier. Not comparable to the paper's Risk@30 results.

### B6 - EviMem (arXiv 2604.27695)
- Faithful (verified against the cloned code): IRIS loop with k=3, anchor + refined paths, 1-hop edge expansion (weight >= 0.5), sufficiency judge (first 20 facts), stop rules (EXACT >= 0.7/0.85, INFERRABLE >= 0.7), rule-based refinement, answer from first 30 facts. Prompts 18/24 verbatim.
- Code followed instead of the paper where they disagree (halved anchor budget, no entity downgrade): this is the right choice for a *code* port but it means the paper's Algorithm 2 is not what is measured.
- Ours: the LaceMem build (tuple extraction prompt, edge layer thresholds); no entity tracker; "Not mentioned" abstention off; shortest-answer line added.
- The authors' `index_manager.py` prompt exists upstream but is conversation- and tool-call-specific (not a drop-in), see section 0.
- Mapping risk: conversational memory -> Wikipedia paragraphs, pooled corpus. The paper's setting is per-conversation LoCoMo memory.
- Variants: `-` = answer from Index tuples (authors' code); `answer_context=raw` = answer from linked raw sentences (the paper's text). These are the authors' own two descriptions, so both are legitimate rows.

### B7 - LycheeMemory V2 (arXiv 2608.12990)
- Faithful: Algorithm 1 (boundary probability, segment encoding, evidence nodes) and Algorithm 2 (planner, four channels, rerank, RRF kappa 60, route coverage, MMR 0.75/0.25) as far as the constants are given.
- Reconstructed: encoder and planner prompts (paper gives fragments, dialogue-specific rules dropped); the weights inside `score`; channel fusion; node scoring.
- No code, no checkpoint exists, so the row cannot be validated. Wikipedia paragraphs are short, so most documents stay a single segment and the segmentation machinery is largely idle.
- B7 hands the generator 15 records + raw sentences (its own k_plan), not 5: its prompt is about 3x larger than B2's. Any win has a context-size confound.
- Stale text in the B7 README: it still asks you to send the ACL paper link; that paper is now B7b.

### B7b - LycheeMemory ACL 2026 (arXiv 2602.08382)
- It is a long-context model reference, not retrieval: it ignores `--generator`, reads all of the question's paragraphs, has no Gate, and needed a vendor patch for sdpa.
- **Table rows (EM 0.00) are a scoring artefact, not a failed run.** No errors; the model writes sentences (73 words mean on 2Wiki, 197 on MuSiQue), so strict EM/F1 are about 0 by construction. `tools/rescore.py` gives contains-acc 0.65 / 0.35 and a refusal ("memory does not contain it") rate of 0.19 / 0.59. The row is still filed under the "qwen2.5:3b" generator although that model is not used (a column artefact). No DSIRE B7b rows exist.

### B8 - PyRAG (arXiv 2605.12975)
- Faithful (verified): all prompts, decompose -> plan (code) -> execute with 3 fix rounds, adaptive top-10 re-execution, `answer()` with and without docs.
- Different: models (Qwen2.5-7B + Coder-7B with an RL variant available upstream vs qwen2.5 3B/7B), retriever/corpus, temperature 0.7 -> 0, restricted sandbox and 20-call cap, forced fallback answer, answer length 384.
- The RL-trained `gasolsun/PyRAG-7b` is not used. At 3B the planner often cannot write valid programs; the README requires `tools/pyrag_stats.py` before quoting any number. **That check has to be run and reported for the table rows** (fraction of questions that fell back to plain RAG).
- Variants: `-` = same model plans (paper's single-model setting); `plan_generator=qwen2.5-coder:Nb` = coder plans, as in the paper's Plan agent. The coder variant is the closer one. It is **missing at 7B on DSIRE**.

### B9 - LiveLedger (arXiv 2602.07549)
- Faithful: constraint extraction prompt, ledger update prompt (Appendix F verbatim), objective evidence with verbatim quote, ledger replacing the old table, asynchronous lag.
- Replaced: a fine-tuned Qwen3.5-4B updater (4,278 distilled samples) by the prompted generator; released search agents with 30 tool calls and web tools by our JSON ReAct loop with 6 steps and the local retriever; observation = top-5 sentences.
- The paper itself says chained multi-hop QA is future work, so 2Wiki/MuSiQue/DSIRE are all out of domain.
- Variants: `-` = agent + ledger; `ledger=False` = same agent without it (the paper's own control). The meaningful quantity is the difference between the two, and with n=100 it is within noise (2Wiki 3B 0.27 vs 0.29; DSIRE 0.28 vs 0.28).

## 4. The faithful ground: what each table row may claim

Rule: a row is "faithful" only for the property listed. Everything else must not be claimed.

| Row (as printed in the table) | Exact configuration | May claim | Must NOT claim |
|---|---|---|---|
| B1 vanilla RAG (chunk) | OneShotRAG, 256-word chunks, hybrid + MiniLM rerank, top-5 | control accuracy of vanilla RAG on this harness | any paper number |
| B2 vanilla RAG (sentence) | same, sentence units | effect of the finer unit | - |
| B3 GeoRAG `-` | chunk units, K=200, paper constants, no reranker | "GeoRAG algorithms 1-2 with paper defaults on our retriever and a 3B/7B LLM" | reproduction of +6.5-7.5 EM |
| B3 GeoRAG `unit=sentence` | same on sentences | our ablation | a paper variant |
| B4 HLG-RAG `-` | StatementGraphRAG, summary hyper-parameters | "HLG-RAG as described, 3B/7B-built graph" | the paper's 23.1 % recall gain |
| B4 HLG-RAG `retriever=topic` | TopicGraphRAG | the paper's second retriever | - |
| B5 SURE-RAG `gate=False` (and `premise=concat`) | B1 answers + scores, nothing withheld | "B1 plus an unfitted claim verifier" | any SURE-RAG accuracy or Risk@30 |
| B6 EviMem `-` | IRIS code, answer from tuples | the authors' code path | the paper's LoCoMo results |
| B6 EviMem `answer_context=raw` | answer from raw sentences | the paper's text description | - |
| B7 LycheeMemory V2 `-` | planner + 4 channels + RRF + MMR | "structure-faithful V2" | the 89 % LoCoMo figure; equal context to B1 |
| B7 LycheeMemory V2 `plan=False` | single query | planner ablation | - |
| B7b `timeout_s=3600` | released 7B checkpoint, long-context | strict EM is 0 by output format; lenient contains-acc 0.65 (2Wiki) / 0.35 (MuSiQue) via rescore.json, as a separate non-EM metric | any comparison with the other rows |
| B8 PyRAG `-` | same model plans | the authors' code, training-free, same-model | the RL-7B result |
| B8 PyRAG `plan_generator=qwen2.5-coder:N` | coder plans | closest to the paper's agent split | - |
| B9 LiveLedger `-` | agent + prompted ledger | "LiveLedger with a prompted updater" | the fine-tuned-updater gains |
| B9 LiveLedger `ledger=False` | same agent, no ledger | the paper's own control | - |
| ref none / ref iterative | floor / decompose-concatenate | references | published baselines |

### Protocol that makes the comparison valid (the common ground)
1. Same 100 questions, same corpus, same retriever stack and reranker for every row; same frozen prompt v2; temperature 0; top-5; no abstention. (Already enforced by `baseline_table.py`'s frozen filter.)
2. Always report n and the 95 % CI. Do not rank two methods whose CIs overlap. With n=100 that rules out most gaps in the table.
3. Quote the stability noise next to every table (B1 2Wiki: 0.35-0.39 for the same config).
4. For B8 report the PyRAG-vs-fallback split (`pyrag_stats.py`); for B4/B6/B7 report the extractor failure rate (`fallback_chunks`, `judge_parse_failures`, `fallback_segments`); for B9 `n_quote_guard_rejected` and `agent_parse_failures`. A row whose component mostly failed measures the fallback, not the method.
5. Use the paired test (`registerrag.cli compare`) for any claim that method X beats Y.

## 5. What would close the gap (ordered by value per effort)

1. ~~Fix the data issues first~~ **done 2026-11-18** (see section 7).
2. ~~Use the authors' EviMem extraction prompt~~ not a drop-in; only an adaptation is possible (section 7).
3. **Run the diagnostics already written** (`pyrag_stats.py`, B4/B7 fallback counts) and put the numbers next to the rows.
4. **Bigger samples:** the 500-question dev set (`scripts/make_dev500.py`) gives about +/- 4 points instead of +/- 9.
5. **Stronger generator for the roles that need it** (e.g. the already-installed qwen2.5:14b/32b, qwen3:14b, gpt-oss:20b) as an additional column, so the "model too weak" confound is separated from the "method" effect.
6. **Fix the hardware fault** (see crash RCA) before trusting any difference under 4 points.
7. **Methods that cannot be made faithful without outside artifacts:** B5 (needs the trained verifier, not released), B7 (no code), B9 updater (fine-tuned model not released). These should stay labelled "architecture only" in every table and paper.
8. For B3, B4, B9 contact the authors or check for later code releases; re-run the fidelity check when code appears (the clone-and-diff method used here takes minutes).

## 6. Measured effect of the deviations on the results (2026-11-18; paired bootstrap vs B1, n=100)

Counterfactual runs under the papers' own models/corpora are impossible here, so what is measured is (a) how often each
method's special component actually worked, (b) the difference to B1 with a paired 95 % interval. Differences marked * exclude 0.
Order of the four numbers: 2Wiki 3B, MuSiQue 3B, 2Wiki 7B, MuSiQue 7B (EM points).

| Baseline | Component health (measured) | vs B1 (EM) | DSIRE F1 vs B1 (3B / 7B) |
|---|---|---|---|
| B8 PyRAG | fell back to plain RAG on 33 % (2Wiki 3B), 59 % (MuSiQue 3B), 16 % / 52 % at 7B, 13-29 % on DSIRE | +9, +6, 0, +1 (none significant) | -12*, -10* |
| B9 LiveLedger | fallback 29-53 %; quote guard rejects 0.5-1.3 entries/q; ledger on-vs-off effect: -2, +5, -7, -1 (paper: +4 to +16) | -8, +6, +12*, +14* (ledger=False gives the same gains) | -9* / -5* |
| B3 GeoRAG | all-gold-in-prompt 0.05 vs B1 0.53; sub-query failure 11-19 % at 3B, 0-2 % at 7B | -7, -4, -14*, -8* (paper: +6.5 to +7.5) | -6 / -8* |
| B4 HLG-RAG | extraction fallbacks 1-2 % of chunks (graph mostly OK) | +1, +2, -10*, +2 | -8* / -3 |
| B6 EviMem | stops at iteration 1 on 11-37 % (2Wiki) and 66-72 % (DSIRE): the loop is mostly idle on DSIRE; 9 % verbatim-fallback tuples | -2, -2, -1, +7 | +6 / -12* |
| B7 Lychee V2 | planner always parsed; planner adds nothing (plan=False equal or better); 12-23 % verbatim segments | +2, +4, +2, +6 | -11* / -11* |
| B5 SURE-RAG | verifier AUROC 0.36-0.61 (chance 0.5); gate off | -1 to +2 (noise) | 0 / 0 |
| B7b | answers in sentences; strict EM 0 | n/a (contains-acc 0.65 / 0.35, lenient) | no rows |

Reading: (1) only about 15 of about 100 EM comparisons are significant, so most method differences are not resolvable at n=100;
(2) the ranking flips with generator size (2Wiki best: PyRAG at 3B, LiveLedger at 7B), so the model deviation H1 is first-order;
(3) the paper-claimed gains reproduce for no method (GeoRAG goes the wrong way, LiveLedger's ledger effect is about 0, HLG shows no
EM gain, PyRAG only at 3B and not significantly); (4) on DSIRE F1, B1 is the best or tied on nearly every comparison.

## 7. Status of the gaps (what was fixed, what cannot be)

| Gap | Status |
|---|---|
| B7b rows read as "invalid, timeouts" | **Corrected**: output-format effect; `tools/rescore.py` + README + table note added |
| DSIRE scored with EM (yes/no accuracy only) | **Fixed**: `baseline_table.py` shows F1 as primary for DSIRE, with a note; DSIRE section added to `baselines/README.md` |
| B6 README said the extraction prompt was unpublished | **Fixed** (README); using it needs an adaptation, not done |
| B5 README claimed identical answers to B1 | **Fixed** (README: 0-4 differences, gate off, AUROC measured) |
| B7 README stale "send me the ACL link" | **Fixed** |
| Missing DSIRE 7B PyRAG coder-planner row | **Run started**; regenerate the table when it finishes |
| Table note on noise / B5 gate / B7b | **Added** to `baseline_table.md` (generated by the script) |
| Component-health numbers next to rows | **Added** (section 6) |
| Rows with trained or unreleased parts (B5 verifier, B7 code, B9 updater, B7b Gate) | **Cannot be fixed** without artifacts that are not released; stay labelled "architecture only" |
| Generator and corpus differ from the papers (H1, H2) | **Cannot be removed**; a stronger-generator column is possible but is a multi-day run |
| n=100, noise, CPU fault | **Not fixed**: needs the 500-question set (long runs) and the hardware repair |
