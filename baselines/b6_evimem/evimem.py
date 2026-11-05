"""EviMem (Li et al., 2026, arXiv:2604.27695) - training-free re-implementation inside the RegisterRAG harness.

Ported from the authors' code (github.com/AIGeeksGroup/EviMem, src/memory_retrieval_system/IRIS_retriever_v5.py); where the
paper's pseudocode and the released code disagree, the CODE is followed and the difference is listed in README.md.

  LaceMem (memory)   Raw layer   = the verbatim sentence units of the corpus (a "turn" in LoCoMo = a sentence here)
                     Index layer = atomic (subject, predicate, object, time) tuples, one LLM extraction per document
                     Edge layer  = same-source edges (weight 1.0) + semantic-similarity edges between tuples
  IRIS (retrieval)   up to 3 iterations of: dual-path retrieval (anchor = original question, refined = diagnosis-driven
                     query) -> 1-hop edge expansion -> LLM sufficiency judgement (EXACT / INFERRABLE / PARTIAL, confidence,
                     MISSING) -> stop, or refine the query from the MISSING text with a rule-based strategy
  Answer             one generation over the accumulated facts, with tier-adaptive instructions

Prompts for sufficiency / refinement / answering / reasoning-chain are the authors' (verbatim). The tuple-extraction
prompt is OURS (the authors' LaceMem ingest prompts were not published in the files we could read).
Tunable via --method-arg: max_iterations (3), judge_facts (20), answer_facts (30), embedding_threshold (0.5),
edge_k (3), edge_min_sim (0.7), edge_min_weight (0.5), answer_context (facts|raw), reasoning_chain (true),
short_answer (true), allow_abstain (false), window (12 sentences per extraction call).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np

from registerrag.pipelines import BasePipeline, PipelineResult

PROMPT_VERSION = "evimem-code-v5+tuples-v1"
HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------------- authors' prompts (verbatim)
SUFFICIENCY_SYSTEM = "You are a helpful assistant that evaluates information sufficiency."
SUFFICIENCY_USER = """Question: {question}

Retrieved Facts:
{facts_text}
{extra_instruction}

Evaluate if these facts can answer the question:

1. EXACT_MATCH: Can answer precisely with specific details? (yes/no)
2. INFERRABLE: Can reasonably infer the answer from related information? (yes/no)
3. PARTIAL_MATCH: Have related information but not enough to answer? (yes/no)
4. CONFIDENCE: Your confidence level (0.0-1.0)
5. MISSING: What specific information is missing? (or "none" if exact/inferrable)

Respond in EXACTLY this format:
EXACT: yes/no
INFERRABLE: yes/no
PARTIAL: yes/no
CONFIDENCE: 0.0-1.0
MISSING: <missing information or "none">"""

TEMPORAL_EXTRA = """
IMPORTANT: This is a TEMPORAL question asking for specific dates/times.
- EXACT_MATCH: Has precise date/time (e.g., "January 19, 2023", "2023-01-19")
- INFERRABLE: Has temporal clues that allow reasonable inference
  * Example: "as of February 2023" → event likely in February 2023
  * Example: "after opening in January" → subsequent events after January
- PARTIAL_MATCH: Has related but insufficient temporal information
- Vague terms like "recently", "not long ago" are PARTIAL, not EXACT
"""

REFINE_SYSTEM = "You are a helpful assistant that refines search queries."
REFINE_USER = """Original question: {original_question}
Current search query: {current_query}
Missing information: {missing_info}
Iteration: {iteration}/{max_iterations}

Strategy: {strategy}

Generate an improved search query to find the missing information.
Keep it concise and focused. Return ONLY the query (no explanation)."""
STRATEGIES = {
    ("temporal", 1): "Focus on DATE, TIME, and temporal keywords (when, started, launched, opened).",
    ("temporal", 2): "Search for specific DATE FORMATS and temporal relations (as of, after, before, during).",
    ("temporal", 3): "Try broader temporal context: related events, milestones, timeframes.",
    ("general", 1): "Focus on specific keywords, entity names, and key concepts.",
    ("general", 2): "Try different angle: related events, attributes, or contextual information.",
    ("general", 3): "Use synonyms, broader concepts, or implied relationships.",
}

ANSWER_SYSTEM = "You are a helpful assistant that answers questions based on provided facts."
ANSWER_USER = """{instruction}
{reasoning_context}
Question: {question}

Relevant Facts:
{facts_text}

Answer (be concise and direct):"""
INSTR_TEMPORAL_EXACT = ('This is a TEMPORAL question. Answer with the PRECISE date/time from the facts.\n'
                        'Use standard formats: "January 19, 2023", "2023-01-19", "May 2023".\n'
                        'Do NOT use vague terms like "around", "approximately".')
INSTR_TEMPORAL_INFER = ('The facts contain temporal clues that allow REASONABLE INFERENCE.\nMake a careful inference:\n'
                        '- If "as of February 2023", the event likely happened in/around February 2023\n'
                        '- If "after opening in January", the event is after January\n'
                        '- Be clear but concise. Use "in" or "around" if inferring timeframe\n'
                        '- Do NOT be overly cautious with phrases like "it is likely that" or "based on the facts".')
INSTR_TEMPORAL_ELSE = "The facts don't contain sufficient temporal information. State what's missing briefly."
INSTR_CONFIDENT = ('Answer the question directly and confidently using the facts.\n'
                   'Do NOT use hedging phrases like "likely", "it seems", "probably", "based on the facts".\n'
                   'If you have the information, state it clearly.')
INSTR_INFER = ('Answer based on reasonable inference from the facts.\nUse clear language. If inferring, use:\n'
               '- "Based on the facts, [answer]" OR\n- Simply state the answer with brief reasoning\n'
               'Do NOT overuse uncertainty markers ("likely", "it seems", "probably").')
INSTR_ELSE = "Answer based on available facts. If key information is missing, state it concisely."
SHORT_ANSWER_NOTE = ("\nOutput format: give the shortest possible answer (an entity, date, number, or yes/no) with no "
                     "explanation. Always give your best answer, even if the facts seem incomplete.")

CHAIN_SYSTEM = "You are a helpful assistant that analyzes reasoning requirements."
CHAIN_USER = """Analyze if this question requires multi-hop reasoning.

Question: {question}
Entities: {entities_text}

Facts available:
{facts_text}

If this requires connecting multiple facts (multi-hop reasoning):
1. List the reasoning steps needed
2. Identify which facts support each step

If it's a simple direct question, respond "DIRECT".

Response format:
TYPE: MULTI-HOP / DIRECT
STEPS:
- Step 1: [description]
- Step 2: [description]
...
FACTS_USED: [comma-separated fact indices]"""

# ------------------------------------------------------------------------------------- OURS: tuple extraction prompt
EXTRACT_SYSTEM = (
    "You factorize text into atomic semantic tuples for a memory index. For every numbered sentence, write each fact it "
    "states as one tuple (subject, predicate, object, time). Rules: one fact per tuple; the subject and object must be "
    "full, self-contained entity names (replace pronouns such as 'he', 'she', 'it', 'the film' using the page title and "
    "the other sentences); keep the wording of the text; 'time' is the date/year the fact refers to, or an empty string. "
    "Skip sentences with no factual content. Reply with JSON only: "
    '{"facts":[{"sent":1,"subject":"...","predicate":"...","object":"...","time":""}]}')
EXTRACT_USER = "Page title: {title}\n\nSentences:\n{numbered}"

TEMPORAL_RE = re.compile(r"\bwhen\b|\bhow\s+long\b|\bwhat\s+(?:date|year|time)\b|\bwhich\s+year\b|\bhow\s+many\s+years\b",
                         re.I)


# ------------------------------------------------------------------------------------------------ small helpers
def is_temporal(q: str) -> bool:
    return bool(TEMPORAL_RE.search(q or ""))


def fact_string(f: dict) -> str:
    if f.get("fallback"):
        return f["text"]
    s = f"{f['subject']} {f['predicate']} {f['object']}".strip()
    return f"{s} (time: {f['time']})" if f.get("time") else s


def parse_sufficiency(text: str):
    """-> dict(exact, inferrable, partial, confidence, missing, parse_ok). Accepts EXACT: and EXACT_MATCH: (the authors'
    parser only matches the former, so a model that echoes the long label would silently read as 'no')."""
    t = text or ""
    low = t.lower()

    def yn(key):
        m = re.search(rf"{key}(?:_match)?\s*:\s*[*\s]*(yes|no)", low)
        return (m.group(1) == "yes") if m else None

    exact, inf, part = yn("exact"), yn("inferrable"), yn("partial")
    m = re.search(r"confidence\s*:\s*[*\s]*([0-9]*\.?[0-9]+)", low)
    conf = float(m.group(1)) if m else None
    if conf is not None and conf > 1.0:
        conf = conf / 100.0 if conf <= 100 else 1.0
    m = re.search(r"missing\s*:\s*(.*)", t, re.I | re.S)
    missing = (m.group(1).strip().splitlines() or [""])[0].strip().strip('"') if m else ""
    ok = exact is not None and conf is not None
    return {"exact": bool(exact), "inferrable": bool(inf), "partial": bool(part),
            "confidence": conf if conf is not None else 0.0,
            "missing": "" if missing.lower() in ("", "none", "n/a") else missing, "parse_ok": ok}


def calibrate(v: dict, temporal: bool) -> dict:
    """Paper Alg. 2 calibration (temporal questions only): exact -> c>=0.85, inferrable -> c<=0.75, partial -> c<=0.50."""
    if not temporal:
        return v
    c = v["confidence"]
    if v["exact"]:
        c = max(c, 0.85)
    elif v["inferrable"]:
        c = min(c, 0.75)
    elif v["partial"]:
        c = min(c, 0.50)
    return {**v, "confidence": c}


def should_stop(v: dict, temporal: bool, allow_inference: bool = True) -> bool:
    """retrieve_with_irr: exact & c >= thr (0.85 temporal / 0.7 otherwise), or inferrable & c >= 0.7."""
    thr = 0.85 if temporal else 0.7
    if v["exact"] and v["confidence"] >= thr:
        return True
    return bool(v["inferrable"] and v["confidence"] >= 0.7 and allow_inference)


def answer_instruction(v: dict, temporal: bool) -> str:
    if temporal:
        if v["exact"]:
            return INSTR_TEMPORAL_EXACT
        if v["inferrable"]:
            return INSTR_TEMPORAL_INFER
        return INSTR_TEMPORAL_ELSE
    if v["exact"] or (v["inferrable"] and v["confidence"] >= 0.75):
        return INSTR_CONFIDENT
    if v["inferrable"] or (v["partial"] and v["confidence"] >= 0.5):
        return INSTR_INFER
    return INSTR_ELSE


def would_say_not_mentioned(v: dict, iterations_done: int, n_facts: int, max_iter: int) -> bool:
    """_detect_not_mentioned (logged only; the harness keeps every baseline on the forced-answer policy)."""
    if not (v["exact"] or v["partial"] or v["inferrable"]) and iterations_done >= 2:
        return True
    return v["confidence"] < 0.3 and n_facts < 3 and iterations_done >= max_iter


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


# --------------------------------------------------------------------------------------------------- the memory
class LaceMem:
    """Index tuples (+ cache), embeddings, edges. Built lazily from the harness' UnitStore."""

    def __init__(self, store, embedder, gen_name, margs, cache_dir=None, log=print):
        self.store, self.embedder, self.log = store, embedder, log
        self.window = int(margs.get("window", 12))
        self.edge_k = int(margs.get("edge_k", 3))
        self.edge_min_sim = float(margs.get("edge_min_sim", 0.7))
        self.cache_dir = cache_dir if cache_dir is not None else os.path.join(HERE, "index")
        self.cache_path = None
        self.facts: list = []
        self.doc_done: dict = {}
        self.stats = {"docs": 0, "tuples": 0, "fallback_sentences": 0, "failed_calls": 0}
        self._gen_name = gen_name
        self._built = False

    # -- extraction ---------------------------------------------------------
    def _load_cache(self):
        if self.cache_path or not self.cache_dir:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, f"tuples_{_safe(self._gen_name)}_{PROMPT_VERSION}.jsonl")
        if os.path.exists(self.cache_path):
            for l in open(self.cache_path, encoding="utf-8"):
                try:
                    r = json.loads(l)
                    self.doc_done[r["doc_id"]] = r["facts"]
                except (ValueError, KeyError):
                    continue

    def _extract_doc(self, doc, generator):
        sents = [(i, s) for i, s in enumerate(doc.sentences) if any(ch.isalnum() for ch in s)]
        facts, failed = [], 0
        for a in range(0, len(sents), self.window):
            part = sents[a:a + self.window]
            numbered = "\n".join(f"{n + 1}. {s}" for n, (_, s) in enumerate(part))
            got, covered = None, set()
            for _ in range(2):
                try:
                    r = generator.generate(EXTRACT_SYSTEM, EXTRACT_USER.format(title=doc.title, numbered=numbered),
                                           json_mode=True, max_new_tokens=900)
                    m = re.search(r"\{.*\}", r.text, re.S)
                    got = json.loads(m.group(0))["facts"]
                    break
                except Exception:  # noqa: BLE001
                    got = None
            if got is None:
                failed += 1
                got = []
            for f in got:
                try:
                    n = int(f["sent"]) - 1
                    subj, pred, obj = (str(f.get(k, "")).strip() for k in ("subject", "predicate", "object"))
                    if 0 <= n < len(part) and subj and pred and obj:
                        facts.append({"sent_idx": part[n][0], "subject": subj, "predicate": pred, "object": obj,
                                      "time": str(f.get("time") or "").strip(), "fallback": False})
                        covered.add(n)
                except (KeyError, ValueError, TypeError):
                    continue
            for n, (i, s) in enumerate(part):       # never lose a sentence: uncovered ones enter the index verbatim
                if n not in covered:
                    facts.append({"sent_idx": i, "text": f"{doc.title}: {s}", "fallback": True})
        return facts, failed

    def build(self, generator, doc_ids=None):
        """Ensure tuples exist for doc_ids (default: every document), then (re)build embeddings and edges."""
        self._load_cache()
        docs = self.store.docs
        todo = [d for d in (doc_ids if doc_ids is not None else docs) if d not in self.doc_done]
        t0 = time.perf_counter()
        for n, d in enumerate(todo, 1):
            facts, failed = self._extract_doc(docs[d], generator)
            self.doc_done[d] = facts
            self.stats["failed_calls"] += failed
            if self.cache_path:
                with open(self.cache_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"doc_id": d, "facts": facts}) + "\n")
            if n % 25 == 0 or n == len(todo):
                self.log(f"  [LaceMem index] {n}/{len(todo)} docs extracted ({time.perf_counter() - t0:.0f}s)")
        self._assemble()

    def _assemble(self):
        self.facts = []
        for d, fl in self.doc_done.items():
            if d not in self.store.docs:
                continue
            for f in fl:
                self.facts.append({**f, "doc_id": d, "unit_id": f"{d}#s{f['sent_idx']}"})
        texts = [fact_string(f) for f in self.facts]
        self.F = self.embedder.encode_docs(texts) if texts else np.zeros((0, 1), np.float32)
        self.doc_of = np.array([f["doc_id"] for f in self.facts], dtype=object)
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi([re.findall(r"\w+", t.lower()) or ["_"] for t in texts]) if texts else None
        self._edges()
        self.stats.update(docs=len(self.doc_done), tuples=sum(not f["fallback"] for f in self.facts),
                          fallback_sentences=sum(f["fallback"] for f in self.facts))
        self._built = True

    def _edges(self):
        n = len(self.facts)
        adj = [dict() for _ in range(n)]
        by_unit: dict = {}
        for i, f in enumerate(self.facts):
            by_unit.setdefault(f["unit_id"], []).append(i)
        for ids in by_unit.values():                 # same_raw edges, weight 1.0
            for a in ids:
                for b in ids:
                    if a != b:
                        adj[a][b] = 1.0
        if n > 1 and self.edge_k > 0:                # semantic_similarity edges: top edge_k neighbours with sim >= edge_min_sim
            for a0 in range(0, n, 2048):
                S = self.F[a0:a0 + 2048] @ self.F.T
                for r in range(S.shape[0]):
                    S[r, a0 + r] = -1.0
                kk = min(self.edge_k, n - 1)
                top = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
                for r in range(S.shape[0]):
                    for b in top[r]:
                        w = float(S[r, b])
                        if w >= self.edge_min_sim:
                            a = a0 + r
                            adj[a][int(b)] = max(adj[a].get(int(b), 0), w)
                            adj[int(b)][a] = max(adj[int(b)].get(a, 0), w)
        self.adj = adj

    # -- search ---------------------------------------------------------------
    def search(self, query, k, thr, mask=None):
        """search_method='auto': embedding (cosine >= thr), else BM25, else keyword overlap."""
        if not self.facts:
            return []
        cand = np.arange(len(self.facts)) if mask is None else np.flatnonzero(mask)
        if cand.size == 0:
            return []
        s = self.F[cand] @ self.embedder.encode_query(query)
        order = np.argsort(-s, kind="stable")
        hits = [int(cand[o]) for o in order[:k] if s[o] >= thr]
        if hits:
            return hits
        b = self.bm25.get_scores(re.findall(r"\w+", query.lower()))[cand]
        order = np.argsort(-b, kind="stable")
        hits = [int(cand[o]) for o in order[:k] if b[o] > 0]
        if hits:
            return hits
        q = set(re.findall(r"\w+", query.lower()))
        sc = [len(q & set(re.findall(r"\w+", fact_string(self.facts[int(c)]).lower()))) for c in cand]
        order = np.argsort(-np.asarray(sc), kind="stable")
        return [int(cand[o]) for o in order[:k] if sc[o] > 0]

    def expand(self, ids, min_weight, mask=None):
        out = []
        for i in ids:
            for j, w in self.adj[i].items():
                if w >= min_weight and (mask is None or mask[j]):
                    out.append(j)
        return out


# ------------------------------------------------------------------------------------------------- the pipeline
class EviMem(BasePipeline):
    name = "evimem"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        m = self.margs = dict(self.cfg.method_args or {})
        self.K = int(m.get("max_iterations", 3))
        self.judge_facts = int(m.get("judge_facts", 20))
        self.answer_facts = int(m.get("answer_facts", 30))
        self.thr = float(m.get("embedding_threshold", 0.5))
        self.edge_min_weight = float(m.get("edge_min_weight", 0.5))
        self.answer_context = str(m.get("answer_context", "facts"))
        if self.answer_context not in ("facts", "raw"):
            raise ValueError("answer_context must be facts or raw")
        self.use_chain = bool(m.get("reasoning_chain", True))
        self.short = bool(m.get("short_answer", True))
        self.allow_abstain = bool(m.get("allow_abstain", False))
        self.mem = LaceMem(self.store, self.retriever.embedder, getattr(self.generator, "name", "gen"), m,
                           cache_dir=m.get("cache_dir"))
        self.lock_scope = self.cfg.corpus_scope

    def build_index(self, doc_ids=None):
        self.mem.build(self.generator, doc_ids)
        return self.mem.stats

    def _mask(self, ex):
        if self.cfg.corpus_scope != "per_question":
            return None
        allowed = set(ex.doc_ids)
        return np.array([d in allowed for d in self.mem.doc_of], dtype=bool)

    # -- LLM roles --------------------------------------------------------------
    def _judge(self, question, facts, temporal, ledger):
        facts_text = "\n".join(f"- {fact_string(f)}" for f in facts[: self.judge_facts])
        user = SUFFICIENCY_USER.format(question=question, facts_text=facts_text,
                                       extra_instruction=TEMPORAL_EXTRA if temporal else "")
        r = self._lm("verifier", SUFFICIENCY_SYSTEM, user, ledger, max_new_tokens=200)
        v = parse_sufficiency(r.text)
        return calibrate(v, temporal), r.text

    def _refine(self, question, cur, missing, it, temporal, ledger):
        strat = STRATEGIES[("temporal" if temporal else "general", min(it, 3))]
        user = REFINE_USER.format(original_question=question, current_query=cur, missing_info=missing, iteration=it,
                                  max_iterations=self.K, strategy=strat)
        try:
            r = self._lm("compiler", REFINE_SYSTEM, user, ledger, max_new_tokens=100)
        except RuntimeError:
            return cur
        q = r.text.strip().strip('"\'').splitlines()[0].strip() if r.text.strip() else ""
        return q or cur

    def _entities(self, q):
        spans = re.findall(r"(?:[A-Z][\w'’.-]*)(?:\s+(?:of|the|de|and|in|on|for|[A-Z][\w'’.-]*))*", q)
        spans = [s.strip() for s in spans if len(s) > 2 and s.strip() not in ("Who", "What", "Which", "When", "Where",
                                                                           "Are", "Is", "Was", "Do", "Does", "Did", "How")]
        return ", ".join(dict.fromkeys(spans)) or "none"

    # -- main loop --------------------------------------------------------------
    def run(self, ex, ledger):
        if not self.mem._built:
            if self.cfg.corpus_scope == "per_question":
                self.mem.build(self.generator, list(self.store.docs))
            else:
                self.mem.build(self.generator)
        mask = self._mask(ex)
        mem, q = self.mem, ex.question
        temporal = is_temporal(q)
        orig_all, ref_all, exp_all, trace, cur, verdict, raw_txt = [], [], [], [], q, None, ""
        done = 0
        for it in range(1, self.K + 1):
            k_ret = 10 + 3 * (it - 1)
            k_anchor = max(5, k_ret // 2)                          # the CODE halves the anchor budget (paper: same k_ret)
            ledger.call("retriever", 2)
            A = mem.search(q, k_anchor, self.thr, mask)
            R = mem.search(cur, k_ret, self.thr, mask)
            E = mem.expand(list(dict.fromkeys(A + R)), self.edge_min_weight, mask)
            orig_all += A; ref_all += R; exp_all += E
            combined = list(dict.fromkeys(orig_all + ref_all + exp_all))
            verdict, raw_txt = self._judge(q, [mem.facts[i] for i in combined], temporal, ledger)
            done = it
            stop = should_stop(verdict, temporal)
            trace.append({"iter": it, "query": cur, "n_anchor": len(A), "n_refined": len(R), "n_expanded": len(E),
                          "n_total": len(combined), "judge": {k: verdict[k] for k in
                                                              ("exact", "inferrable", "partial", "confidence", "missing", "parse_ok")},
                          "stop": stop})
            if stop or it == self.K:
                break
            if verdict["missing"]:
                cur = self._refine(q, cur, verdict["missing"], it, temporal, ledger)
        combined = list(dict.fromkeys(orig_all + ref_all + exp_all))
        abstain = would_say_not_mentioned(verdict, done, len(combined), self.K)
        pred, ans_units, ans_facts, trunc, chain = self._answer(ex, q, combined, verdict, temporal, abstain, ledger)
        units = [self.store.by_id[mem.facts[i]["unit_id"]] for i in combined if mem.facts[i]["unit_id"] in self.store.by_id]
        rank, seen = [], set()
        for u in units:
            if u.unit_id not in seen:
                seen.add(u.unit_id); rank.append(u)
        return PipelineResult(
            pred, rank, rank, rank, ans_units, trunc,
            {"iterations": trace, "stopped_at": done, "would_abstain": abstain, "answer_context": self.answer_context,
             "answer_facts": [fact_string(mem.facts[i]) for i in ans_facts][:self.answer_facts],
             "reasoning_chain": chain, "judge_raw_last": raw_txt[:400], "temporal": temporal,
             "judge_parse_failures": sum(not t["judge"]["parse_ok"] for t in trace),
             "index_stats": dict(mem.stats), "prompt_version": PROMPT_VERSION})

    def _answer(self, ex, q, combined, v, temporal, abstain, ledger):
        mem = self.mem
        facts_idx = list(dict.fromkeys(combined))[: self.answer_facts]
        if self.answer_context == "raw":                            # variant: ground in the linked Raw layer (sentences)
            seen, lines, units = set(), [], []
            for i in facts_idx:
                uid = mem.facts[i]["unit_id"]
                u = self.store.by_id.get(uid)
                if u is not None and uid not in seen:
                    seen.add(uid); units.append(u); lines.append(self.retriever.index_text(u))
            lines = lines[: int(self.margs.get("raw_k", 10))]; units = units[: len(lines)]
        else:
            lines = [fact_string(mem.facts[i]) for i in facts_idx]
            units, seen = [], set()
            for i in facts_idx:
                uid = mem.facts[i]["unit_id"]
                if uid not in seen and uid in self.store.by_id:
                    seen.add(uid); units.append(self.store.by_id[uid])
        trunc = False
        if self.cfg.budget_tokens is not None:                       # honour --budget: drop trailing facts until it fits
            fixed = self.counter.count(ANSWER_SYSTEM) + self.counter.count(ANSWER_USER.format(
                instruction=INSTR_THIS_MAX, reasoning_context="", question=q, facts_text=""))
            used, keep = fixed, []
            for l in lines:
                t = self.counter.count(l) + 4
                if used + t > self.cfg.budget_tokens:
                    trunc = True
                    break
                keep.append(l); used += t
            if trunc:
                lines = keep; units = units[: len(keep)] if self.answer_context == "raw" else units
        chain = None
        if self.use_chain and (v["inferrable"] or v["partial"]):
            ctext = "\n".join(f"[{n}] {l}" for n, l in enumerate(lines[:15]))
            try:
                r = self._lm("compiler", CHAIN_SYSTEM, CHAIN_USER.format(question=q, entities_text=self._entities(q),
                                                                        facts_text=ctext), ledger, max_new_tokens=400)
                chain = r.text.strip()
            except RuntimeError:
                chain = None
        reasoning = ""
        if chain and re.search(r"TYPE:\s*MULTI", chain, re.I):
            m = re.search(r"STEPS:\s*(.*?)(?:FACTS_USED:|\Z)", chain, re.S | re.I)
            if m and m.group(1).strip():
                reasoning = f"\n\nReasoning steps required:\n{m.group(1).strip()}\n"
        instr = answer_instruction(v, temporal) + (SHORT_ANSWER_NOTE if self.short else "")
        user = ANSWER_USER.format(instruction=instr, reasoning_context=reasoning, question=q,
                                  facts_text="\n".join(f"- {l}" for l in lines) or "(no facts)")
        r = self._lm("generator", ANSWER_SYSTEM, user, ledger)
        pred = r.text
        if abstain and self.allow_abstain:
            pred = "Not mentioned"
        return pred, units, facts_idx, trunc, bool(reasoning)


INSTR_THIS_MAX = INSTR_TEMPORAL_INFER + SHORT_ANSWER_NOTE      # longest instruction, for the budget estimate
