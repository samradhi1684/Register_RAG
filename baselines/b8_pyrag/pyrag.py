"""PyRAG (Sun et al., 2026, arXiv:2605.12975) - training-free re-implementation inside the RegisterRAG harness.

Ported from the authors' public code (github.com/GasolSun36/PyRAG: runner.py, plan_agent.py, decompose_agent.py,
tools.py, code_executor.py). The prompts below are the authors' prompts, copied verbatim (the only change: the code
spells the reasoning tag <redacted_thinking>, an artefact of the published file; the paper's Figs. 9-10 use <thinking>).

Pipeline (RAGProgramRunner.run):
  1. Decompose agent  -> JSON list of atomic sub-queries (3 attempts, fallback [question]).
  2. Plan agent       -> Python program over retrieve(query) and answer(query, docs=None); syntax errors are fed back
                         (3 attempts).
  3. Execute; a runtime error goes back to the Plan agent with the failing code (MAX_FIX_ROUNDS = 3 -> up to 4 runs).
  4. Execution-driven adaptive retrieval: if an answer() result signals insufficient information, the retrieve() call
     that preceded it gets top-k 10 and the WHOLE program is re-executed ONCE (not just that step).

What is OURS and not the authors' (report this in the paper): retriever/corpus (our hybrid index over the benchmark's
own paragraphs, not E5 over Wikipedia-2018), frozen small generators run at temperature 0 (the authors sample at 0.7),
the sandbox below (the authors exec() with no restrictions), the 20-call tool cap, and the final forced-answer
fallback (the authors keep whatever final_answer holds).
Tunable via --method-arg: plan_generator, max_repairs (3), topk_retry (10), max_tool_calls (20), max_lines (5000),
adaptive (true), answer_tokens (384).
"""
from __future__ import annotations

import ast
import builtins
import json
import re
import sys

from registerrag.pipelines import BasePipeline, PipelineResult

PROMPT_VERSION = "pyrag-paper-v1"

# ----------------------------------------------------------------------------------------- authors' prompts (verbatim)
DECOMPOSE_SYSTEM = (
    "You are a question decomposition agent for multi-hop QA. Break a complex question into a minimal list of atomic "
    "single-hop sub-queries. Each sub-query should be independently answerable by a search engine. "
    "Return a JSON list of strings ONLY. No explanation, no markdown, no extra text.")

DECOMPOSE_USER = """Original question:
{query}

Example:
Original question:
In what year was the director of Inception born?

JSON output:
["Who directed Inception?", "When was Christopher Nolan born?"]

Now return a JSON list of atomic sub-queries for the original question above.
Return ONLY the JSON list of strings."""

PLAN_SYSTEM = '''You are a planning agent that writes Python code to answer questions via RAG.

You have exactly two functions available:

  retrieve(query: str) -> List[str]
      Searches a retrieval system and returns a list of relevant document strings.
      Do NOT pass a topk argument; the retrieval count is controlled externally.

  answer(query: str, docs: List[str]) -> str
      Calls an LLM to answer a question given a list of documents.
      Returns ONLY the short text inside the model's <answer>...</answer> block (the runtime extracts it). Use that return value directly in f-strings — it is a short phrase, not the full model reply.

  answer(query: str) -> str          (no docs — aggregation / synthesis mode)
      Same as above but with NO documents. The model answers using ONLY the information already present in the query string. Use this for the FINAL aggregation step.

Rules:
- Use retrieve() + answer() step by step to answer each sub-query.
- If you call retrieve() for a sub-question, pass that return value into the matching answer() for that sub-question. Do not pass [] for that step unless the answer is intentionally retrieval-free.
- Build on intermediate answers by interpolating them into later queries using f-strings.
- Variable names MUST be valid Python identifiers: only letters, digits, and underscores, NO spaces.
- Every variable you reference MUST be assigned BEFORE it is used. Never use a name that was not explicitly assigned in a previous line.
- Do NOT parse answer() return values in generated code (no .split(), regex, or indexing). The runtime already returns the short <answer> span.
- If you use a for/if/while block that might conditionally set final_answer, initialise final_answer = "" BEFORE the block so it is always defined.
- The last line of your code MUST be: final_answer = answer(<synthesis question>)
  Do NOT pass docs to this final answer() call — it is aggregation-only.
- NEVER use an f-string or string concatenation as final_answer. final_answer MUST ALWAYS be the return value of an answer() call so the result goes through the model and has a proper <answer> tag.
- CRITICAL: The <synthesis question> must use this TWO-PART format:
    "Given: <fact1>, <fact2>, ... Answer the question: <ORIGINAL QUESTION>"
  where the first part lists intermediate results as background facts, and the second part REPEATS the ORIGINAL question VERBATIM after "Answer the question:".
  Do NOT rephrase the original question or embed intermediate answers INTO the question itself.
  BAD:  f"Which American director, {name}, who is {nationality}, hosted ...?"  (answer leaks into the question → LLM says "yes")
  GOOD: f"Given: {name} hosted the event, {name} is {nationality}. Answer the question: Which American director hosted ...?"
- Valid Python 3 only: use ASCII double or single quotes for strings; never put backticks around variable names or expressions.
- Return ONLY the Python code, no explanation, no imports, no markdown prose.'''

CODE_EXAMPLE = '''# Example for: "Who was born earlier, the director of Inception or Jurassic Park?"
docs1 = retrieve("Who directed Inception?")
director1 = answer("Who directed Inception?", docs1)

docs2 = retrieve("Who directed Jurassic Park?")
director2 = answer("Who directed Jurassic Park?", docs2)

docs3 = retrieve(f"When was {director1} born?")
birth1 = answer(f"When was {director1} born?", docs3)

docs4 = retrieve(f"When was {director2} born?")
birth2 = answer(f"When was {director2} born?", docs4)

# ALWAYS end with answer() — list facts first, then repeat the ORIGINAL question verbatim
final_answer = answer(
    f"Given: {director1} directed Inception and was born {birth1}, "
    f"{director2} directed Jurassic Park and was born {birth2}. "
    f"Answer the question: Who was born earlier, the director of Inception or Jurassic Park?"
)'''

PLAN_USER = """Original question:
{original_query}

Sub-queries to resolve:
{sub_queries}

Reference example (do NOT copy, write code for the actual question above):
{code_example}

Now write the Python code for the original question.
End with: final_answer = answer(f"Given: <facts>. Answer the question: {original_query}")"""

FIX_PROMPT = """The following Python code was generated to answer the question but raised a runtime error.
Fix the code so it runs correctly. Return ONLY the corrected Python code.

=== Original question ===
{original_query}

=== Failing code ===
{failed_code}

=== Runtime error ===
{error_msg}

=== Fix instructions ===
- Do not parse answer() return values in generated code; the runtime returns short <answer> text.
- Every variable you reference must be explicitly assigned earlier.
- If retrieve() was called, pass its docs into answer(); do not use answer(..., []) for that step.
- Initialise final_answer = "" before any loop/conditional that might set it.
- The last line MUST be: final_answer = answer(<synthesis question>) — call answer() with NO docs
  to aggregate all intermediate results. NEVER use an f-string or string as final_answer directly."""

SYNTAX_FEEDBACK = """

---
Your previous output is NOT valid Python (syntax error).
Details: {error_detail}

Fix it. Use only ASCII ' or " for strings; never wrap variable names in backticks; output code only.

Previous attempt:
```python
{failed_code}
```
Return ONLY corrected Python code."""

ANSWER_WITH_DOCS = """You are given a question and retrieved documents.
You MUST answer using ONLY information from the retrieved documents.
Even for yes/no questions, decide yes or no by reasoning from facts in the documents.

Output format (STRICT):
<thinking> ... </thinking>
<answer> ... </answer>

Evidence citation rule:
- Whenever you use evidence from the documents in your reasoning, you MUST cite it inline as Doc [i] (matching the document indices shown in the retrieved block, e.g. [Doc 1] → Doc [1]).
- Only cite documents that are actually relevant.
- Keep <thinking> concise (1–3 sentences).

Answer rules:
- The <answer> should be a short phrase, preferably taken directly from the documents when possible.
- Match the answer TYPE to the QUESTION: WHO / which person / born first / earlier → a person's NAME in <answer>, not only a date; WHEN → date or time; yes/no → exactly yes / no / unknown when the documents do not support a definite answer.
- Do NOT output anything outside <thinking> and <answer>.

Example (do NOT copy the content, only follow the style):
<thinking>Doc [1] states that Future Ted serves as the narrator, and Doc [4] confirms the voice actor.</thinking>
<answer> Ted Mosby </answer>"""

ANSWER_NO_DOCS = """There are NO retrieved documents. The question text itself contains background facts (after 'Given:') and the actual question to answer (after 'Answer the question:').
You MUST use the provided facts to answer the ACTUAL QUESTION.

CRITICAL: Your job is to ANSWER the question, NOT to confirm whether the facts are true.
- If the question asks WHO / WHICH person → reply with a person's NAME.
- If the question asks WHEN → reply with a date or time.
- If the question asks WHERE → reply with a place.
- ONLY answer yes/no when the question is explicitly a yes/no question (e.g. 'Are both ...?', 'Is it true ...?').
- NEVER answer 'yes' or 'no' to a WHO/WHICH/WHEN/WHERE question.

Output format (STRICT):
<thinking> ... </thinking>
<answer> ... </answer>

- In <thinking>, identify the actual question type and combine the given facts to produce the answer (1–2 sentences).
- The <answer> must directly answer the question — a name, date, place, etc. — NOT 'yes' or 'no' unless the question is truly yes/no.
- Do NOT output anything outside <thinking> and <answer>."""

ANSWER_USER = """=== QUESTION ===
{query}
=== END QUESTION ===

=== RETRIEVED DOCUMENTS ===
{docs}
=== END DOCUMENTS ==="""

# runner._answer_indicates_insufficient_info markers (case-insensitive substring match, as in the authors' code)
INSUFFICIENT_MARKERS = ("not enough information", "insufficient information", "no enough information", "cannot answer",
                        "unable to answer", "信息不足", "无法根据", "无法回答", "unknown", "not found")
MAX_FIX_ROUNDS = 3
GEN_ATTEMPTS = 3

FORCED_SYSTEM_NOTE = "fallback uses the harness forced-answer prompt (QA_SYSTEMS), not a PyRAG prompt"


# ----------------------------------------------------------------------------- sandbox (best effort, research use)
class PlanRejected(Exception):
    """The generated program uses something outside the allowed surface."""


FORBIDDEN_NAMES = {"exec", "eval", "open", "compile", "__import__", "globals", "locals", "getattr", "setattr",
                   "delattr", "vars", "input", "breakpoint", "exit", "quit", "help", "dir", "type", "object", "super",
                   "classmethod", "staticmethod", "property", "memoryview", "bytearray"}
_ALLOWED_BUILTINS = ["len", "str", "int", "float", "bool", "list", "dict", "set", "tuple", "min", "max", "sum",
                     "sorted", "range", "enumerate", "zip", "abs", "round", "any", "all", "isinstance", "reversed",
                     "map", "filter", "repr"]
SAFE_BUILTINS = {n: getattr(builtins, n) for n in _ALLOWED_BUILTINS}


def validate(code: str) -> None:
    """Static checks before exec(). Not a security boundary against a determined attacker - the models here are
    small local ones and the machine is the researcher's own - but it stops imports, file/network access, dunder
    tricks and str.format attribute probing. SyntaxError propagates (it is a repairable error)."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise PlanRejected("imports are not allowed")
        if isinstance(node, ast.ClassDef):
            raise PlanRejected("class definitions are not allowed")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.Await, ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith,
                             ast.Yield, ast.YieldFrom)):
            raise PlanRejected(f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in ("format", "format_map")):
            raise PlanRejected(f"attribute '{node.attr}' is not allowed")
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            raise PlanRejected(f"name '{node.id}' is not allowed")


def run_limited(code: str, namespace: dict, max_lines: int = 5000) -> None:
    """exec() with a line-event budget so a runaway loop raises TimeoutError instead of hanging (works on Windows)."""
    count = [0]

    def local_tracer(frame, event, arg):
        if event == "line":
            count[0] += 1
            if count[0] > max_lines:
                raise TimeoutError(f"program exceeded {max_lines} executed lines (infinite loop?)")
        return local_tracer

    def global_tracer(frame, event, arg):
        return local_tracer if frame.f_code.co_filename == "<plan>" else None

    compiled = compile(code, "<plan>", "exec")
    old = sys.gettrace()
    sys.settrace(global_tracer)
    try:
        exec(compiled, namespace)  # noqa: S102 - validated, restricted builtins, line budget
    finally:
        sys.settrace(old)


# ----------------------------------------------------------------------------- helpers (utils.py of the authors)
def extract_code(text: str) -> str:
    t = (text or "").strip()
    t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    t = re.sub("[​‌‍﻿]", "", t)
    m = re.search(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", t, re.S)      # tolerates an unclosed fence
    return (m.group(1) if m else t).strip("\n")


def extract_json_list(text: str):
    t = (text or "").strip()
    for cand in (t, *(m.group(1) for m in re.finditer(r"```(?:json)?\s*(.*?)```", t, re.S)),
                 *(re.findall(r"\[.*\]", t, re.S))):
        try:
            v = json.loads(cand)
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                return [x.strip() for x in v if x.strip()]
        except ValueError:
            continue
    return None


def extract_answer_tag(text: str) -> str:
    """tools.extract_answer_tag: text inside <answer>...</answer>, else the whole stripped output. We additionally
    accept an unclosed <answer> (generation cut off by the token cap)."""
    t = (text or "").strip()
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", t, re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"<answer>\s*(.*)\Z", t, re.S | re.I)
    return (m.group(1).strip() if m else t)


def is_insufficient(text: str) -> bool:
    low = (text or "").lower()
    return not low.strip() or any(m in low for m in INSUFFICIENT_MARKERS)


def format_docs(docs) -> str:
    return "\n\n".join(f"[Doc {i + 1}]\n{d}" for i, d in enumerate(docs)) if docs else "No retrieved documents."


class InfraError(Exception):
    """A model/server failure inside a tool call (not a bug in the generated program, so no repair)."""


class ToolLimit(Exception):
    pass


class _Session:
    """Per-question state behind the two tools. Memoises identical calls (deterministic at temperature 0) so the
    adaptive re-execution only pays for what changed. `log` is per execution pass; the collectors span all passes."""

    def __init__(self, pipe, ex, ledger):
        self.p, self.ex, self.led = pipe, ex, ledger
        self.cache_r, self.cache_a = {}, {}
        self.stage1, self.rankings = [], []
        self.text2unit, self.retrieved, self.selected = {}, {}, {}
        self.truncated, self.untagged = False, 0
        self.max_calls = int(pipe.margs.get("max_tool_calls", 20))
        self.boost_k = int(pipe.margs.get("topk_retry", 10))
        self.answer_tokens = int(pipe.margs.get("answer_tokens", 384))
        self.trace: list = []                       # human-readable, all passes
        self.begin_pass({})

    def begin_pass(self, boost):
        self.boost, self.log, self.n_retrieve, self.n_calls = dict(boost), [], 0, 0

    def _tick(self):
        self.n_calls += 1
        if self.n_calls > self.max_calls:
            raise ToolLimit(f"more than {self.max_calls} tool calls")

    def print(self, *a, **k):
        self.trace.append("print: " + " ".join(str(x) for x in a)[:200])

    def retrieve(self, query, topk=None):
        q = str(query)
        idx = self.n_retrieve
        self.n_retrieve += 1
        kk = max(int(topk) if topk else self.p.cfg.top_k, self.boost.get(idx, 0))
        self._tick()
        if (q, kk) not in self.cache_r:
            try:
                stage1, ranking = self.p._retrieve_rank(q, self.ex, self.led)
            except Exception as e:  # noqa: BLE001
                raise InfraError(str(e)) from e
            self.stage1 += stage1
            self.rankings.append(ranking)
            texts = []
            for u in ranking[:kk]:
                t = self.p.retriever.index_text(u)
                self.text2unit.setdefault(t, u)
                self.retrieved.setdefault(u.unit_id, u)
                texts.append(t)
            self.cache_r[(q, kk)] = texts
        self.log.append({"type": "retrieve", "idx": idx, "topk": kk, "query": q})
        self.trace.append(f"retrieve#{idx}({q[:80]!r}, k={kk})")
        return list(self.cache_r[(q, kk)])

    def _fit(self, q, texts):
        """Respect --budget: drop trailing docs until the answer prompt fits."""
        budget = self.p.cfg.budget_tokens
        if budget is None:
            return texts, False
        c = self.p.counter
        used = c.count(ANSWER_WITH_DOCS) + c.count(ANSWER_USER.format(query=q, docs=""))
        kept = []
        for t in texts:
            n = c.count(t) + 8
            if used + n > budget:
                return kept, True
            kept.append(t)
            used += n
        return kept, False

    def answer(self, query, docs=None):
        q = str(query)
        texts = [str(d) for d in docs] if docs else []
        key = (q, tuple(texts))
        self._tick()
        if key not in self.cache_a:
            kept, trunc = self._fit(q, texts)
            self.truncated |= trunc
            for t in kept:
                u = self.text2unit.get(t)
                if u is not None:
                    self.selected.setdefault(u.unit_id, u)
            system = ANSWER_WITH_DOCS if texts else ANSWER_NO_DOCS
            user = ANSWER_USER.format(query=q, docs=format_docs(kept))
            try:
                r = self.p._lm("generator", system, user, self.led, max_new_tokens=self.answer_tokens)
            except RuntimeError as e:
                raise InfraError(str(e)) from e
            if not re.search(r"<answer>", r.text, re.I):
                self.untagged += 1
            self.cache_a[key] = extract_answer_tag(r.text)
        out = self.cache_a[key]
        self.log.append({"type": "answer", "query": q, "out": out})
        self.trace.append(f"answer({q[:80]!r}, docs={len(texts)}) -> {out[:80]!r}")
        return out


# ----------------------------------------------------------------------------- the pipeline
class PyRAG(BasePipeline):
    name = "pyrag"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.margs = dict(self.cfg.method_args or {})
        self.plan_gen = self.aux.get("plan")            # None -> the main generator plays the Plan role too

    # -- agents ---------------------------------------------------------------
    def _decompose(self, ex, ledger):
        err = None
        for _ in range(GEN_ATTEMPTS):
            try:
                r = self._lm("compiler", DECOMPOSE_SYSTEM, DECOMPOSE_USER.format(query=ex.question), ledger,
                             max_new_tokens=256)
                subs = extract_json_list(r.text)
                if subs:
                    return subs, None
                err = "no JSON list of strings in the reply"
            except Exception as e:  # noqa: BLE001
                err = str(e)
        return [ex.question], err

    def _plan_user(self, ex, subs):
        return PLAN_USER.format(original_query=ex.question, sub_queries=repr(subs), code_example=CODE_EXAMPLE)

    def _generate_code(self, ex, subs, ledger):
        user, last_err = self._plan_user(ex, subs), None
        for _ in range(GEN_ATTEMPTS):
            try:
                r = self._lm("compiler", PLAN_SYSTEM, user, ledger, gen=self.plan_gen, max_new_tokens=600)
                code = extract_code(r.text)
                compile(code, "<plan>", "exec")
                return code
            except SyntaxError as e:
                user = self._plan_user(ex, subs) + SYNTAX_FEEDBACK.format(error_detail=str(e), failed_code=code)
                last_err = f"SyntaxError: {e}"
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
        raise RuntimeError(last_err or "plan generation failed")

    def _fix_code(self, ex, code, err, ledger):
        user, last = FIX_PROMPT.format(original_query=ex.question, failed_code=code, error_msg=err), None
        for _ in range(GEN_ATTEMPTS):
            try:
                r = self._lm("compiler", PLAN_SYSTEM, user, ledger, gen=self.plan_gen, max_new_tokens=600)
                fixed = extract_code(r.text)
                compile(fixed, "<plan>", "exec")
                return fixed
            except Exception as e:  # noqa: BLE001
                last = e
        raise RuntimeError(f"fix failed: {last}")

    # -- execution ------------------------------------------------------------
    def _execute_with_fixes(self, ex, code, S, ledger, repairs):
        """Returns (final_answer or None, code, program_error). InfraError propagates."""
        max_fix = int(self.margs.get("max_repairs", MAX_FIX_ROUNDS))
        err = None
        for attempt in range(max_fix + 1):
            S.begin_pass(S.boost)
            try:
                validate(code)
                ns = {"__builtins__": {**SAFE_BUILTINS, "print": S.print}, "retrieve": S.retrieve, "answer": S.answer}
                run_limited(code, ns, int(self.margs.get("max_lines", 5000)))
                fa = ns.get("final_answer")
                if fa is None or callable(fa):
                    return None, code, "final_answer was never assigned"
                return str(fa).strip(), code, None
            except InfraError:
                raise
            except Exception as e:  # noqa: BLE001 - any program error is repairable feedback
                err = f"{type(e).__name__}: {e}"
                repairs.append(err)
                if attempt < max_fix:
                    code = self._fix_code(ex, code, err, ledger)
        return None, code, err

    @staticmethod
    def _boost_map(log, final, boost_k):
        boost, last = {}, None
        for ev in log:
            if ev["type"] == "retrieve":
                last = ev
            elif is_insufficient(ev["out"]) and last is not None and last["topk"] < boost_k:
                boost[last["idx"]] = boost_k
        if not boost and final is not None and is_insufficient(final):
            ret = [e for e in log if e["type"] == "retrieve"]
            if ret and ret[-1]["topk"] < boost_k:
                boost[ret[-1]["idx"]] = boost_k
        return boost

    def run(self, ex, ledger):
        S = _Session(self, ex, ledger)
        subs, decompose_err = self._decompose(ex, ledger)
        code, plan_err, repairs, final, prog_err, infra_err, retried = "", None, [], None, None, None, False
        try:
            code = self._generate_code(ex, subs, ledger)
        except Exception as e:  # noqa: BLE001
            plan_err = str(e)
        if code:
            try:
                final, code, prog_err = self._execute_with_fixes(ex, code, S, ledger, repairs)
                if self.margs.get("adaptive", True) and prog_err is None:
                    boost = self._boost_map(S.log, final, S.boost_k)
                    if boost:
                        retried = True
                        S.trace.append(f"adaptive retrieval: re-executing whole program, boost={boost}")
                        S.boost = boost
                        final2, code, prog_err = self._execute_with_fixes(ex, code, S, ledger, repairs)
                        if final2 is not None:
                            final = final2
            except InfraError as e:
                infra_err = str(e)
        fallback = final is None or is_insufficient(final)
        if fallback:     # keep baselines forced-answer (same policy as the other systems); logged so it cannot hide
            if not S.retrieved:
                try:
                    S.begin_pass({})
                    S.retrieve(ex.question)
                except Exception as e:  # noqa: BLE001
                    infra_err = infra_err or str(e)
            pred, kept, trunc = self._answer(ex, list(S.retrieved.values()), ledger)
            S.truncated |= trunc
            for u in kept:
                S.selected.setdefault(u.unit_id, u)
        else:
            pred = final
        ranking, seen = [], set()
        for i in range(max((len(x) for x in S.rankings), default=0)):          # round-robin over retrieve() calls
            for rk in S.rankings:
                if i < len(rk) and rk[i].unit_id not in seen:
                    seen.add(rk[i].unit_id)
                    ranking.append(rk[i])
        return PipelineResult(
            pred, ranking, S.stage1, list(S.retrieved.values()), list(S.selected.values()), S.truncated,
            {"sub_queries": subs, "code": code, "repairs": repairs, "retried_with_topk": retried,
             "n_tool_calls": S.n_calls, "fallback": fallback, "program_error": prog_err, "decompose_error": decompose_err,
             "plan_error": plan_err, "infra_error": infra_err, "exec_log": S.trace[:80], "prompt_version": PROMPT_VERSION,
             "untagged_answers": S.untagged, "raw_final_answer": final,
             "plan_model": getattr(self.plan_gen or self.generator, "name", "?")})
