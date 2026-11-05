# B9 - LiveLedger: explicit constraint-state tracking against illusory completion (2026)

- **Paper:** "When Is Enough Not Enough? Illusory Completion in Search Agents", arXiv 2602.07549 (v3 HTML read in full, incl. Appendix F prompts)
- **Official code:** none linked.
- **Correction to an earlier note:** the "11.6-point" gain is gpt-oss-120b accuracy in Appendix B (3-run mean). The paper's headline is +4.4 to +16.1 accuracy points / −6.2 to −16.7 unsubstantiated-answer-rate points (Table 2).
- **Status:** inference-time method re-implemented as a ReAct-style search agent + prompted ledger updater. Tests: `python -m pytest baselines/b9_liveledger -q`.
- **Code here:** `liveledger.py`, `baseline.py`.

## Method (from the paper)
- **Constraint extraction once** from the question (App. F prompt, verbatim) -> `C1..Cn`.
- After **every tool-using step** an updater reads (thinking, query, results) + the current ledger and records, per **(candidate, constraint)**, *objective* evidence: `obj = true | false | null` with a **verbatim quote** (App. F "LiveLedger update" prompt, verbatim). Evidence only - no agent belief column.
- The ledger is shown to the agent as a **table after the tool results, replacing the previous one**; it is the only addition to the agent's context; the agent never writes it.
- **Asynchronous lag:** the agent's call t+1 sees the ledger updated through call t−1 (`lag=true`).
- No stopping rule is defined by the paper; the agent decides when to answer.

## What is ours / differs
| Item | Paper | Here |
|---|---|---|
| Agent | released ReAct-style search agents (gpt-oss, DeepSeek-V4, Search-R1, ...), up to 30 tool calls | our small JSON ReAct loop (`AGENT_SYSTEM`), frozen generator, `max_steps=6` |
| Tool | Serper search + Jina browse (web) | our hybrid retriever + reranker over the benchmark corpus; observation = top-5 **sentences** (`--unit sentence`) |
| Updater | **fine-tuned** Qwen3.5-4B (4,278 distilled samples), tool-call output | the frozen generator, **prompted**, same arguments emitted as JSON |
| Quote rule | prompt only | + enforced: an entry whose quote is not found verbatim (normalised) in the observation is rejected and logged (`quote_guard`, `n_quote_guard_rejected`) |
| Domain | multi-constraint web questions; the paper explicitly leaves chained multi-hop QA to future work | 2Wiki / MuSiQue chains - **out of domain**; constraints extracted from a chain question are coarse |
| Budget exhausted / unparseable agent | n/a | one forced answer from everything retrieved (harness policy), logged as `fallback` |

## Run it
```powershell
..\.venv\Scripts\Activate.ps1
python -m pytest baselines/b9_liveledger -q
python scripts/run_baselines.py --methods liveledger --datasets 2wiki --limit 8 --out runs_smoke   # real-model check

# both rows: agent + LiveLedger, and the same agent WITHOUT the ledger (the paper's own with/without comparison)
python scripts/run_baselines.py --methods liveledger
python scripts/run_baselines.py --methods liveledger --generator qwen2.5:7b
```
Knobs (`--method-arg`): `ledger` (true), `max_steps` (6), `lag` (true), `search_k`, `obs_chars` (1500), `quote_guard` (true).
The meaningful comparison is **ledger=true vs ledger=false** (same agent, same seed): the effect of the ledger itself.

## Reading the result
Per question: `steps` (thought, query, retrieved unit ids, updates applied), `constraints`, `final_ledger` (evidence per candidate x constraint), `final_candidate`, `final_candidate_states`, `fully_verified` (all constraints of the answer's candidate are *Satisfied* - the complement of the paper's unsubstantiated-answer rate, defined per candidate), `n_quote_guard_rejected`, `agent_parse_failures`, `fallback`. The expected 3B failure mode is the updater quoting loosely (high rejection count) or the agent not following the JSON action format (check `agent_parse_failures`).
