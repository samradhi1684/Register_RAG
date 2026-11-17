"""LiveLedger (paper "When Is Enough Not Enough? Illusory Completion in Search Agents", arXiv:2602.07549) on the
RegisterRAG harness: a ReAct-style search agent whose context is extended by an explicit, evidence-only constraint ledger.

From the paper (Sec. 4, App. C.2, App. F):
  * constraints are extracted ONCE from the question (App. F "Prompt for Constraint Extraction", verbatim below);
  * after every tool-using step an updater reads the latest (thinking, query, results) and the current ledger and records,
    per (candidate, constraint), objective evidence: obj = true | false | null with a VERBATIM quote (App. F "Prompt for
    LiveLedger Update from Search Results", verbatim below, JSON output instead of a tool call);
  * the ledger is the only addition to the agent context; it is shown as a table AFTER the tool results and REPLACES the
    previous one; the agent itself never writes the ledger;
  * updates are asynchronous: the agent's call t+1 sees the ledger updated through call t-1 (lag=true, as in the paper).
Not in the paper (ours): the search-agent prompt (the paper reuses released agents), the table layout, the verbatim-quote
guard, the search tool (our hybrid retriever + reranker over the benchmark corpus instead of Serper/Jina), the updater model
(the frozen generator, prompted - the paper fine-tunes a 4B updater on distilled traces) and max_steps (default 6; paper: 30).
The paper explicitly leaves chained multi-hop questions to future work: this is an out-of-domain adaptation.
--method-arg: ledger (true), max_steps (6), lag (true), search_k (= --top-k), obs_chars (1500), quote_guard (true).
"""
from __future__ import annotations

import json
import re

from registerrag.pipelines import BasePipeline, PipelineResult

PROMPT_VERSION = "liveledger-paper-prompts+agent-v1"

# --------------------------------------------------------------------------------------- App. F prompts (verbatim)
CONSTRAINT_PROMPT = """### Task Instruction:
Please extract explicit, externally verifiable constraints from a question that an answer must satisfy.
Your task is to read a question and produce a constraint list in JSON format. Each constraint must describe a condition that the answer must satisfy and must be objectively verifiable using an external source (e.g., factual records or publicly available information).
### Instructions:
1. Decompose the question into a list of atomic constraints that cannot be further decomposed into smaller conditions.
2. For each constraint:
 - It must be **explicitly mentioned** in the question.
 - It must be objectively verifiable using an external source (e.g., factual records or publicly available information).
 - **Exclude** implicit, assumed, or inferred constraints, as well as constraints about the **answer format** (e.g., length, structure, naming style).
3. Output **only** valid JSON in the specified format.
### Output Format (JSON only):
{{ "constraint_1": {{ "constraint": "<externally verifiable condition the answer must satisfy>" }}, "constraint_2": {{ "constraint": "<externally verifiable condition the answer must satisfy>" }} }}
### Examples:
**Example 1**
**Question:** Name a publicly traded technology company that was founded before 1990, is headquartered in Japan, and is listed on the Tokyo Stock Exchange.
**Output:**
{{ "constraint_1": {{ "constraint": "The company must be a publicly traded technology company" }}, "constraint_2": {{ "constraint": "The company must have been founded before 1990" }}, "constraint_3": {{ "constraint": "The company must be headquartered in Japan" }}, "constraint_4": {{ "constraint": "The company must be listed on the Tokyo Stock Exchange" }} }}
**Example 2**
**Question:** Name a film that won the Academy Award for Best Picture, was directed by a woman, and was released in the 2010s.
**Output:**
{{ "constraint_1": {{ "constraint": "The film must have won the Academy Award for Best Picture" }}, "constraint_2": {{ "constraint": "The film must have been directed by a woman" }}, "constraint_3": {{ "constraint": "The film must have been released in the 2010s" }} }}
**Example 3**
**Question:** Name an Olympic athlete who has won gold medals in at least three different Summer Olympic Games and competed for the United States.
**Output:**
{{ "constraint_1": {{ "constraint": "The person must be an Olympic athlete" }}, "constraint_2": {{ "constraint": "The person must have won gold medals in at least three different Summer Olympic Games" }}, "constraint_3": {{ "constraint": "The person must have competed representing the United States" }} }}
### Inputs:
- **Question:** {question}
### Output:
Only the JSON object."""

UPDATE_SYSTEM = "You are a reasoning assistant that updates an epistemic ledger based on search results."
UPDATE_PROMPT = """### Task:
Analyze the search results and update the ledger with any new objective evidence found.
### Input Fields:
- **Question:** the multi-constraint question being answered
- **Constraints:** labeled conditions (C1, C2, …) that candidates must satisfy
- **Current Ledger:** evidence collected so far
- **Latest Step:** the most recent search iteration, containing:
 - **Thinking:** reasoning that motivated the search
 - **Search Query:** the query executed
 - **Search Results:** documents returned by the search
### Instructions:
1. Carefully review the Search Results.
2. Identify any candidates not present in the current ledger.
3. Identify any evidence that verifies or refutes constraints for candidates.
4. Call `update_ledger` with entries for each piece of new evidence found.
### Status Guidelines:
- `obj = true`: Search Results contain clear, direct evidence that proves the constraint is satisfied.
- `obj = false`: Search Results contain clear, direct evidence that disproves the constraint.
- `obj = null`: Evidence is ambiguous, indirect, or not found.
**Important Rules:**
- Include an entry for each candidate not in the current ledger with `obj = null`.
- Set `true` or `false` ONLY when evidence is direct and unambiguous.
- Always include a verbatim quote in `obj_evidence` when `obj` is non-null.
- If no relevant evidence is found, call `update_ledger(entries=[])`.
### Output Requirement:
Always call `update_ledger` based on the Search Results, including:
- New candidates with `obj = null`.
- Any (candidate, constraint) pair with newly discovered evidence.
### Tool Call Format:
update_ledger(entries=[{{ "candidate": "<Candidate Name>", "constraint": "<constraint_id>", "obj": true||false||null, "obj_evidence": "<verbatim quote or null>" }}])
### Inputs:
- **Question:** {question}
- **Constraints:** {constraints}
- **Current Ledger:** {ledger}
- **Thinking:** {thinking}
- **Search Query:** {search_query}
- **Search Results:** {retrieval_results}
### Output:
Call `update_ledger` with your findings."""
# adaptation: the paper's updater emits a tool call; a prompted general model emits the same arguments as JSON
UPDATE_JSON_SUFFIX = ('\n\nIMPORTANT: respond with JSON only, the arguments of the call, exactly: '
                      '{"entries": [{"candidate": "...", "constraint": "C1", "obj": true, "obj_evidence": "..."}]}. '
                      'Use {"entries": []} when there is nothing to record.')

# ------------------------------------------------------------------------------------------------- OURS: agent prompt
AGENT_SYSTEM = (
    "You answer a question by searching a document collection, step by step. At each step reply with JSON only:\n"
    '{"thought": "<what you know and what you need next>", "action": "search" or "answer", '
    '"query": "<search query, when action is search>", "answer": "<final answer, when action is answer>"}\n'
    "A search returns the most relevant passages. Search for ONE missing fact at a time. When you can answer, use action "
    '"answer" with the shortest possible answer (an entity, date, number, or yes/no) and no explanation.')
LEDGER_NOTE = ("\n\nConstraint ledger (what the retrieved evidence has objectively established so far; Satisfied = proved, "
               "Disproved = contradicted, Unknown = not yet established). Do not answer while a constraint of your "
               "candidate is still Unknown unless no further search can help:\n{table}")


# ------------------------------------------------------------------------------------------------------- ledger
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


class Ledger9:
    """Evidence-only constraint ledger: candidate -> constraint id -> {obj: True|False|None, evidence: str|None}."""

    def __init__(self, constraints: dict):
        self.constraints = constraints            # {"C1": "text", ...}
        self.rows: dict = {}
        self.rejected: list = []                  # entries refused by the quote guard (kept for audit)

    def apply(self, entries, observation: str, quote_guard=True):
        obs_n, applied = norm(observation), 0
        for e in entries if isinstance(entries, list) else []:
            if not isinstance(e, dict):
                continue
            cand = str(e.get("candidate") or "").strip()
            cid = str(e.get("constraint") or "").strip().upper()
            if not cand:
                continue
            row = self.rows.setdefault(cand, {c: {"obj": None, "evidence": None} for c in self.constraints})
            if cid not in self.constraints:
                continue
            obj, ev = e.get("obj"), e.get("obj_evidence")
            if obj not in (True, False):
                continue
            if quote_guard and not (isinstance(ev, str) and ev.strip() and norm(ev) and norm(ev) in obs_n):
                self.rejected.append({"candidate": cand, "constraint": cid, "obj": obj, "evidence": ev})
                continue
            row[cid] = {"obj": obj, "evidence": ev}
            applied += 1
        return applied

    def table(self) -> str:
        head = "Constraints: " + " | ".join(f"{c}: {t}" for c, t in self.constraints.items())
        if not self.rows:
            return head + "\n(no candidate yet)"
        lines = [head]
        for cand, cells in self.rows.items():
            parts = []
            for c, v in cells.items():
                st = "Satisfied" if v["obj"] is True else "Disproved" if v["obj"] is False else "Unknown"
                q = f' "{v["evidence"][:90]}"' if v["evidence"] else ""
                parts.append(f"{c}={st}{q}")
            lines.append(f"- {cand}: " + "; ".join(parts))
        return "\n".join(lines)

    def snapshot(self) -> dict:
        return {k: {c: dict(v) for c, v in cells.items()} for k, cells in self.rows.items()}

    def for_prompt(self) -> str:
        return json.dumps({k: {c: v["obj"] for c, v in cells.items()} for k, cells in self.rows.items()})


def parse_constraints(text: str, question: str):
    try:
        m = re.search(r"\{.*\}", text, re.S)
        d = json.loads(m.group(0))
        items = [str(v["constraint"]).strip() for k, v in d.items() if isinstance(v, dict) and v.get("constraint")]
        if items:
            return {f"C{i + 1}": t for i, t in enumerate(items)}, True
    except Exception:  # noqa: BLE001
        pass
    return {"C1": question}, False


def parse_json_obj(text: str):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
        return v if isinstance(v, dict) else None
    except ValueError:
        return None


# ------------------------------------------------------------------------------------------------------ pipeline
class LiveLedgerAgent(BasePipeline):
    name = "liveledger"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        m = self.margs = dict(self.cfg.method_args or {})
        self.use_ledger = bool(m.get("ledger", True))
        self.max_steps = int(m.get("max_steps", 6))
        self.lag = bool(m.get("lag", True))
        self.search_k = int(m.get("search_k", self.cfg.top_k))
        self.obs_chars = int(m.get("obs_chars", 1500))
        self.guard = bool(m.get("quote_guard", True))

    def _observation(self, units):
        txt, used = [], 0
        for i, u in enumerate(units, 1):
            t = f"[{i}] {self.retriever.index_text(u)}"
            if used + len(t) > self.obs_chars and txt:
                break
            txt.append(t); used += len(t)
        return "\n".join(txt) if txt else "(no results)"

    def run(self, ex, ledger):
        q = ex.question
        led9, constraints_ok = None, None
        if self.use_ledger:
            try:
                r = self._lm("compiler", "You extract constraints from questions.", CONSTRAINT_PROMPT.format(question=q),
                             ledger, json_mode=True, max_new_tokens=400)
                cons, constraints_ok = parse_constraints(r.text, q)
            except RuntimeError:
                cons, constraints_ok = {"C1": q}, False
            led9 = Ledger9(cons)
        history, steps, snaps, answer, parse_fail = [], [], [], None, 0
        all_units, rankings, stage1, shown = {}, [], [], {}
        for step in range(1, self.max_steps + 1):
            ledger_block = ""
            if led9 is not None:
                vis = snaps[-2] if (self.lag and len(snaps) >= 2) else (snaps[-1] if (not self.lag and snaps) else None)
                if vis is not None:
                    tmp = Ledger9(led9.constraints); tmp.rows = vis
                    ledger_block = LEDGER_NOTE.format(table=tmp.table())
                else:
                    ledger_block = LEDGER_NOTE.format(table=Ledger9(led9.constraints).table())
            user = f"Question: {q}\n\n" + ("\n\n".join(history) + "\n\n" if history else "") + ledger_block.strip() + "\n\nNext step (JSON):"
            act = None
            for _ in range(2):
                try:
                    r = self._lm("generator", AGENT_SYSTEM, user.strip(), ledger, json_mode=True, max_new_tokens=300)
                except RuntimeError:
                    r = None
                act = parse_json_obj(r.text) if r else None
                if act and act.get("action") in ("search", "answer"):
                    break
                act, parse_fail = None, parse_fail + 1
            if act is None:
                steps.append({"step": step, "error": "unparseable action"}); break
            thought = str(act.get("thought") or "")
            if act["action"] == "answer" and str(act.get("answer") or "").strip():
                answer = str(act["answer"]).strip()
                steps.append({"step": step, "thought": thought, "action": "answer", "answer": answer})
                break
            query = str(act.get("query") or "").strip() or q
            s1, rk = self._retrieve_rank(query, ex, ledger)
            stage1 += s1; rankings.append(rk)
            res = rk[: self.search_k]
            for u in res:
                all_units.setdefault(u.unit_id, u)
            obs = self._observation(res)
            history.append(f"Step {step}\nThought: {thought}\nSearch: {query}\nResults:\n{obs}")
            rec = {"step": step, "thought": thought, "action": "search", "query": query,
                   "results": [u.unit_id for u in res]}
            if led9 is not None:
                uu = UPDATE_PROMPT.format(question=q, constraints=json.dumps(led9.constraints), ledger=led9.for_prompt(),
                                          thinking=thought, search_query=query, retrieval_results=obs) + UPDATE_JSON_SUFFIX
                try:
                    r = self._lm("verifier", UPDATE_SYSTEM, uu, ledger, json_mode=True, max_new_tokens=500)
                    d = parse_json_obj(r.text)
                    rec["updates_applied"] = led9.apply((d or {}).get("entries", []), obs, self.guard)
                    rec["update_parse_ok"] = d is not None
                except RuntimeError as e:
                    rec["update_error"] = str(e)[:200]
                snaps.append(led9.snapshot())
            steps.append(rec)
        fallback = answer is None
        if fallback:     # step budget exhausted / agent failed: same forced-answer policy as the other baselines
            if not all_units:
                s1, rk = self._retrieve_rank(q, ex, ledger); stage1 += s1; rankings.append(rk)
                for u in rk[: self.search_k]:
                    all_units.setdefault(u.unit_id, u)
            pool = [u for rk in rankings for u in rk if u.unit_id in all_units]
            pool = list({u.unit_id: u for u in pool}.values())
            pred, kept, trunc = self._answer(ex, pool, ledger)
        else:
            pred, kept, trunc = answer, list(all_units.values()), False
        ranking, seen = [], set()
        for i in range(max((len(x) for x in rankings), default=0)):
            for rk in rankings:
                if i < len(rk) and rk[i].unit_id not in seen:
                    seen.add(rk[i].unit_id); ranking.append(rk[i])
        extra = {"steps": steps, "n_searches": sum(1 for s in steps if s.get("action") == "search"),
                 "fallback": fallback, "agent_parse_failures": parse_fail, "ledger_enabled": self.use_ledger,
                 "prompt_version": PROMPT_VERSION}
        if led9 is not None:
            final = norm(pred)
            match = [c for c in led9.rows if norm(c) and (norm(c) in final or final in norm(c)) and final]
            states = led9.rows[match[0]] if match else None
            extra.update(constraints=led9.constraints, constraints_parse_ok=constraints_ok, final_ledger=led9.snapshot(),
                         quote_guard_rejected=led9.rejected[:20], n_quote_guard_rejected=len(led9.rejected),
                         final_candidate=match[0] if match else None,
                         final_candidate_states=None if states is None else {c: v["obj"] for c, v in states.items()},
                         fully_verified=bool(states and all(v["obj"] is True for v in states.values())))
        return PipelineResult(pred, ranking, stage1, list(all_units.values()), kept, trunc, extra)
