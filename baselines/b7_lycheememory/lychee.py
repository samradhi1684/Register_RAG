"""LycheeMemory V2 (Li et al. 2026, "Efficient Long-Term Memory for LLM Agents via Semantic Segment-Level Consolidation",
arXiv:2608.12990) re-implemented, training-free, on the RegisterRAG harness. There is no released code or checkpoint.

Write path (Sec. 3.2, Alg. 1)  exchanges (here: the sentences of one document, a "session") are grouped into semantic segments
by a no-LLM boundary score (semantic surprise, cohesion drop, token-length pressure, turn-count pressure); every finalised
segment is consolidated by ONE LLM call into self-contained records (type, text, entities, tags, temporal, evidence turns);
records get entity / topic / entity-topic / temporal / event-frame evidence nodes. Conflicts are not resolved at write time.
Read path (Sec. 3.5, Alg. 2)   one planner LLM call -> semantic queries + evidence routes; per route four recall channels
(record vector, evidence-node, temporal, raw-turn vector); merge, optional cross-encoder rerank; reciprocal-rank fusion over
routes (kappa=60); route-coverage + MMR diversity selection; serialise records; one answer LLM call.

Numbers taken from the paper (Appendix B): delta=0.50, bias -1.10, surprise z=(s-0.20)/0.14, chunk tokens min/max 300/900,
max 10 exchanges, length pressure -1.30 below 70% of min rising to 3.00 at max, turn signal -0.85/-0.15/0.15/min(1,0.30+0.15(n-4)),
k_plan=15, lambda_node=lambda_record=max(3k,30), lambda_turn=max(k,20), RRF kappa=60, s_select = 0.75 s_rel - 0.25 s_sig.
NOT in the paper text we could read (ours): the weights inside "score", the exact encoder / planner prompts (we reconstruct them
from the published fragments), the node-channel scoring, the fusion of channel scores, and a MiniLM cross-encoder in place of
bge-reranker-v2-m3. Wikipedia pages are not dialogue: speaker / "I-me-my" rules of the original prompts are dropped.
--method-arg: delta (0.5), seg_min_tokens (300), seg_max_tokens (900), seg_max_exchanges (10), k_plan (15), rrf_kappa (60),
rerank (true), mmr (true), plan (true), cache_dir.
"""
from __future__ import annotations

import json
import math
import os
import re
import time

import numpy as np

from registerrag.pipelines import BasePipeline, PipelineResult

PROMPT_VERSION = "lychee-v2-reconstructed-v1"
HERE = os.path.dirname(os.path.abspath(__file__))
MEMORY_TYPES = ("fact", "preference", "event", "constraint", "procedure", "failure_pattern", "tool_affordance")

ENCODE_SYSTEM = (
    "You are a memory encoder. Given a conversation chunk, extract atomic memory records that are self-contained and "
    "retrievable without the original dialogue.\n"
    "Rules:\n1. One fact per record.\n2. Convert relative dates to YYYY-MM-DD when the date can be determined.\n"
    "3. Resolve pronouns and abbreviations to the full entity names, using the page title.\n"
    "4. Skip content with no durable information.\n5. disambiguation_context is a short note that helps resolve references in "
    "the next chunk.\n6. If nothing durable, return {\"records\": [], \"disambiguation_context\": \"\"}.\n"
    "Output raw JSON only, no markdown fences:\n"
    '{"records":[{"memory_type":"fact|event|preference|constraint|procedure|failure_pattern|tool_affordance",'
    '"semantic_text":"...","entities":["..."],"temporal":{"t_ref":"","t_valid_from":"","t_valid_to":""},'
    '"tags":["..."],"evidence_turns":[0]}],"disambiguation_context":"..."}')
ENCODE_USER = ("<SESSION_DATE>{session_date}</SESSION_DATE>\n<REFERENCE_CONTEXT>\n{reference_context}\n</REFERENCE_CONTEXT>\n"
               "<CURRENT_TURNS>\n{current_turns}\n</CURRENT_TURNS>")

PLAN_SYSTEM = (
    "Write a JSON plan for answering a user question from past conversation memory.\n"
    "Look only at the user-visible wording in <USER_QUERY> and <RECENT_CONTEXT>. Do not decide whether the answer is "
    "available or unavailable.\nProduce 3 to 8 semantic queries and one evidence route by default.\n"
    "Return raw JSON only:\n"
    '{"reason":"...","question_type":"...","semantic_queries":["..."],"temporal_filter":"",'
    '"evidence_target":"...","evidence_constraints":[],"constraints":[],'
    '"evidence_routes":[{"goal":"...","queries":["..."],"constraints":[],"temporal_filter":""}]}')
PLAN_USER = "<USER_QUERY>{question}</USER_QUERY>\n<RECENT_CONTEXT>\n(none)\n</RECENT_CONTEXT>"

ANSWER_SYSTEM = "You are an intelligent memory assistant tasked with answering questions using conversation memories."
ANSWER_USER = """# Episodic/Semantic Memories
{records}

# Raw Memories
{raw}

Instructions:
- Avoid double-counting redundant evidence.
- If memories contain contradictory information, prioritize the most recent memory.
- The final answer should usually be less than 5-6 words.
- Output only the answer phrase.{short}

Question: {question}
Answer:"""
SHORT_NOTE = " Give the shortest possible answer (an entity, date, number, or yes/no); always give your best answer."


# ------------------------------------------------------------------------------------------------- segmentation
def _cos(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def length_pressure(tokens, tmin=300, tmax=900):
    lo = 0.7 * tmin
    if tokens < lo:
        return -1.30
    if tokens >= tmax:
        return 3.00
    return -1.30 + (tokens - lo) / (tmax - lo) * (3.00 + 1.30)


def turn_signal(n):
    return {1: -0.85, 2: -0.15, 3: 0.15}.get(n, min(1.0, 0.30 + 0.15 * (n - 4)))


def boundary_probability(embs, tokens, new_emb, new_tokens, w_s=0.5, w_c=2.0, bias=-1.10, tmin=300, tmax=900):
    """p_t for appending exchange t to the active segment (embs: its exchange embeddings, tokens: its token count)."""
    c = np.mean(embs, axis=0)
    s = 1.0 - max(_cos(new_emb, c), _cos(new_emb, embs[-1]))
    z = float(np.clip((s - 0.20) / 0.14, -2.0, 4.0))
    coh = lambda V: float(np.mean([_cos(v, np.mean(V, axis=0)) for v in V]))
    d = max(0.0, coh(embs) - coh(list(embs) + [new_emb]))
    score = w_s * z + w_c * d + length_pressure(tokens + new_tokens, tmin, tmax) + turn_signal(len(embs) + 1)
    return 1.0 / (1.0 + math.exp(-(bias + score)))


def segment(embs, tok_counts, delta=0.5, tmin=300, tmax=900, max_ex=10):
    """-> list of (start, end) index ranges. Finalise on semantic boundary (p >= delta) or capacity."""
    segs, start = [], 0
    for t in range(1, len(embs)):
        cur, ctok = embs[start:t], sum(tok_counts[start:t])
        cap = (ctok + tok_counts[t] > tmax) or (t - start + 1 > max_ex)
        if cap or boundary_probability(cur, ctok, embs[t], tok_counts[t], tmin=tmin, tmax=tmax) >= delta:
            segs.append((start, t)); start = t
    if embs:
        segs.append((start, len(embs)))
    return segs


def rrf(lists, kappa=60.0):
    sc = {}
    for lst in lists:
        for r, rid in enumerate(lst, 1):
            sc[rid] = sc.get(rid, 0.0) + 1.0 / (kappa + r)
    return sc


def mmr_select(cands, rel, emb, k, lam_rel=0.75, lam_sig=0.25):
    """s_select = 0.75 s_rel - 0.25 s_sig, s_sig = max cosine to the already selected records."""
    sel, pool = [], list(cands)
    while pool and len(sel) < k:
        best = max(pool, key=lambda i: lam_rel * rel[i] - lam_sig * (max((float(emb[i] @ emb[j]) for j in sel), default=0.0)))
        sel.append(best); pool.remove(best)
    return sel


def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except ValueError:
        return None


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


# ------------------------------------------------------------------------------------------------------- memory
class Memory:
    def __init__(self, store, retriever, gen_name, margs, log=print):
        self.store, self.ret, self.log = store, retriever, log
        self.m = margs
        self.cache_dir = margs.get("cache_dir") or os.path.join(HERE, "index")
        self.gen_name, self.cache_path, self.done = gen_name, None, {}
        self.records, self.nodes, self.built = [], {}, False
        self.stats = {"docs": 0, "segments": 0, "records": 0, "fallback_segments": 0}
        self.pos = {u.unit_id: j for j, u in enumerate(store.units)}
        self.by_sent = {}

    def _cache(self):
        if self.cache_path:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, f"memory_{re.sub(r'[^A-Za-z0-9_.-]+', '_', self.gen_name)}_{PROMPT_VERSION}.jsonl")
        if os.path.exists(self.cache_path):
            for l in open(self.cache_path, encoding="utf-8"):
                try:
                    r = json.loads(l); self.done[r["doc_id"]] = r["segments"]
                except (ValueError, KeyError):
                    continue

    def _consolidate(self, doc, gen):
        sent_units = [self.store.by_id.get(f"{doc.doc_id}#s{i}") for i in range(len(doc.sentences))]
        keep = [(i, u) for i, u in enumerate(sent_units) if u is not None]
        if not keep:
            return []
        pos = self.pos
        embs = [self.ret.emb[pos[u.unit_id]] for _, u in keep]
        toks = [max(1, int(len(doc.sentences[i].split()) * 1.3)) for i, _ in keep]
        segs = segment(embs, toks, float(self.m.get("delta", 0.5)), int(self.m.get("seg_min_tokens", 300)),
                       int(self.m.get("seg_max_tokens", 900)), int(self.m.get("seg_max_exchanges", 10)))
        out, ref = [], ""
        for a, b in segs:
            turns = "\n".join(f"[{n}] {doc.sentences[keep[a + n][0]]}" for n in range(b - a))
            recs, disamb, fb = None, "", False
            for _ in range(2):
                try:
                    r = gen.generate(ENCODE_SYSTEM, ENCODE_USER.format(session_date=doc.title, reference_context=ref[:2400],
                                                                      current_turns=turns), json_mode=True, max_new_tokens=900)
                    d = parse_json(r.text)
                    recs = [x for x in d["records"] if isinstance(x, dict) and str(x.get("semantic_text", "")).strip()]
                    disamb = str(d.get("disambiguation_context", ""))[:1200]
                    break
                except Exception:  # noqa: BLE001
                    recs = None
            if recs is None:
                fb = True
                recs = [{"memory_type": "fact", "semantic_text": f"{doc.title}: {doc.sentences[keep[a + n][0]]}", "entities": [],
                         "temporal": {}, "tags": [], "evidence_turns": [n]} for n in range(b - a)]
            ref = (disamb + "\n" + "\n".join(r["semantic_text"] for r in recs[-12:]))[:2400]
            clean = []
            for r in recs:
                ev = [keep[a + int(t)][0] for t in r.get("evidence_turns", []) if str(t).lstrip("-").isdigit() and 0 <= int(t) < b - a]
                clean.append({"type": r.get("memory_type") if r.get("memory_type") in MEMORY_TYPES else "fact",
                              "text": str(r["semantic_text"]).strip(), "entities": [str(e) for e in r.get("entities", [])][:8],
                              "tags": [str(t) for t in r.get("tags", [])][:6], "temporal": r.get("temporal") or {},
                              "sent_idx": ev or [keep[a][0]]})
            out.append({"range": [a, b], "records": clean, "fallback": fb})
        return out

    def build(self, gen):
        self._cache()
        t0, todo = time.perf_counter(), [d for d in self.store.docs if d not in self.done]
        for n, d in enumerate(todo, 1):
            segs = self._consolidate(self.store.docs[d], gen)
            self.done[d] = segs
            with open(self.cache_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"doc_id": d, "segments": segs}) + "\n")
            if n % 20 == 0 or n == len(todo):
                self.log(f"  [Lychee memory] {n}/{len(todo)} docs ({time.perf_counter() - t0:.0f}s)")
        self.records, self.nodes, self.by_sent = [], {}, {}
        nseg = nfb = 0
        for d, segs in self.done.items():
            if d not in self.store.docs:
                continue
            for si, sg in enumerate(segs):
                nseg += 1; nfb += bool(sg["fallback"])
                for r in sg["records"]:
                    rid = len(self.records)
                    self.records.append({**r, "rid": rid, "doc_id": d, "seg": si, "title": self.store.docs[d].title})
                    for sj in r["sent_idx"]:
                        self.by_sent.setdefault((d, sj), []).append(rid)
                    keys = [f"frame:{d}:{si}"] + [f"ent:{norm(e)}" for e in r["entities"]] + [f"topic:{norm(t)}" for t in r["tags"]]
                    keys += [f"et:{norm(e)}|{norm(t)}" for e in r["entities"] for t in r["tags"]]
                    for ts in (r["temporal"] or {}).values():
                        m = re.match(r"(\d{4})(?:-(\d{2}))?", str(ts or ""))
                        if m:
                            keys.append(f"year:{m.group(1)}")
                            if m.group(2):
                                keys.append(f"month:{m.group(1)}-{m.group(2)}")
                    for k in keys:
                        self.nodes.setdefault(k, set()).add(rid)
        texts = [r["text"] for r in self.records]
        self.R = self.ret.embedder.encode_docs(texts) if texts else np.zeros((0, 1), np.float32)
        self.stats.update(docs=len(self.done), segments=nseg, records=len(self.records), fallback_segments=nfb)
        self.built = True


# ----------------------------------------------------------------------------------------------------- pipeline
class LycheeMemory(BasePipeline):
    name = "lychee"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        m = self.margs = dict(self.cfg.method_args or {})
        self.k_plan = int(m.get("k_plan", 15)); self.kappa = float(m.get("rrf_kappa", 60.0))
        self.use_rerank = bool(m.get("rerank", True)); self.use_mmr = bool(m.get("mmr", True))
        self.use_plan = bool(m.get("plan", True)); self.short = bool(m.get("short_answer", True))
        self.mem = Memory(self.store, self.retriever, getattr(self.generator, "name", "gen"), m)

    def build_index(self):
        self.mem.build(self.generator)
        return self.mem.stats

    def _plan(self, q, ledger):
        try:
            r = self._lm("compiler", PLAN_SYSTEM, PLAN_USER.format(question=q), ledger, json_mode=True, max_new_tokens=500)
            d = parse_json(r.text) or {}
            qs = [str(x).strip() for x in d.get("semantic_queries", []) if str(x).strip()][:8]
            routes = [x for x in d.get("evidence_routes", []) if isinstance(x, dict)]
            if not qs:
                return [q], [{"queries": [q], "temporal_filter": ""}], d, False
            rts = [{"queries": [str(z) for z in (x.get("queries") or qs)][:8], "temporal_filter": str(x.get("temporal_filter") or "")}
                   for x in routes] or [{"queries": qs, "temporal_filter": str(d.get("temporal_filter") or "")}]
            return qs, rts, d, True
        except Exception:  # noqa: BLE001
            return [q], [{"queries": [q], "temporal_filter": ""}], {}, False

    def _route(self, ex, route, k, mask, qemb):
        mem = self.mem
        lam_rec, lam_turn = max(3 * k, 30), max(k, 20)
        cand = np.flatnonzero(mask) if mask is not None else np.arange(len(mem.records))
        cand_set = set(cand.tolist())
        score = {}
        for q in route["queries"]:
            qv = qemb(q)
            sims = mem.R[cand] @ qv                                          # channel 1: record vector
            for i in np.argsort(-sims, kind="stable")[:lam_rec]:
                rid = int(cand[i]); score[rid] = max(score.get(rid, -1), float(sims[i]))
            qn = norm(q)                                                     # channel 2: evidence nodes -> linked records
            linked = set()
            for key, rids in mem.nodes.items():
                kind, _, name = key.partition(":")
                if kind in ("ent", "topic") and len(name) > 2 and name in qn:
                    linked |= rids
            linked &= cand_set
            for rid in sorted(linked, key=lambda r: -float(mem.R[r] @ qv))[:lam_rec]:
                score[rid] = max(score.get(rid, -1), float(mem.R[rid] @ qv))
            tf = route.get("temporal_filter", "")                            # channel 3: temporal keys
            for y in re.findall(r"\b(1\d{3}|20\d{2})\b", tf + " " + q):
                for rid in (mem.nodes.get(f"year:{y}", set()) & cand_set):
                    score[rid] = max(score.get(rid, -1), float(mem.R[rid] @ qv) + 0.05)
            # channel 4: raw turns (sentence units) -> the records that cite them
            pos = self.retriever.retrieve(q, lam_turn, self.store.allowed_for(ex.doc_ids) if mask is not None else None)
            for h in pos:
                u = self.store.units[h.idx]
                for rid in mem.by_sent.get((u.doc_id, u.sent_idx), []):
                    if rid not in cand_set:
                        continue
                    score[rid] = max(score.get(rid, -1), float(h.score))
        order = sorted(score, key=lambda r: -score[r])[: max(100, 4 * k)]
        if self.reranker is not None and self.use_rerank and order:
            ranked = self.reranker.rerank(ex.question, [mem.records[r] for r in order], lambda r: f"{r['title']}. {r['text']}")
            order = [r["rid"] for r, _ in ranked]
        return order

    def run(self, ex, ledger):
        mem = self.mem
        if not mem.built:
            mem.build(self.generator)
        q = ex.question
        mask = None
        if self.cfg.corpus_scope == "per_question":
            allowed = set(ex.doc_ids); mask = np.array([r["doc_id"] in allowed for r in mem.records], dtype=bool)
        cache = {}
        def qemb(s):
            if s not in cache:
                cache[s] = self.retriever.embedder.encode_query(s)
            return cache[s]
        qs, routes, plan, plan_ok = self._plan(q, ledger) if self.use_plan else ([q], [{"queries": [q], "temporal_filter": ""}], {}, True)
        k = max(self.cfg.top_k, self.k_plan, 1)
        ledger.call("retriever", len(routes))
        lists = [self._route(ex, r, k, mask, qemb) for r in routes]
        fused = rrf(lists, self.kappa)
        cand = sorted(fused, key=lambda r: -fused[r])
        cover = [l[0] for l in lists if l]                                   # route coverage: each route's best record first
        cand = list(dict.fromkeys(cover + cand))
        if self.use_mmr and cand:
            rel = {r: fused[r] / max(fused.values()) for r in cand}
            sel = mmr_select(cand[: max(3 * k, 30)], rel, mem.R, k)
        else:
            sel = cand[:k]
        recs = [mem.records[r] for r in sel]
        line = lambda r: f"[{r['type']}] {r['title']}: {r['text']}"
        raw_lines = []
        for r in recs[:k]:
            doc = self.store.docs[r["doc_id"]]
            for si in r["sent_idx"][:1]:
                raw_lines.append(f"{doc.title}: {doc.sentences[si]}")
        raw_lines = list(dict.fromkeys(raw_lines))
        if self.cfg.budget_tokens is not None:                                # honour --budget
            fixed = self.counter.count(ANSWER_SYSTEM) + self.counter.count(ANSWER_USER.format(records="", raw="", question=q, short=SHORT_NOTE))
            used, keep_r = fixed, []
            for r in recs:
                t = self.counter.count(line(r)) + 4
                if used + t > self.cfg.budget_tokens:
                    break
                keep_r.append(r); used += t
            trunc = len(keep_r) < len(recs); recs = keep_r; raw_lines = raw_lines[: max(0, len(recs) // 3)]
        else:
            trunc = False
        user = ANSWER_USER.format(records="\n".join(f"- {line(r)}" for r in recs) or "(none)",
                                  raw="\n".join(f"- {l}" for l in raw_lines) or "(none)", question=q,
                                  short=SHORT_NOTE if self.short else "")
        r = self._lm("generator", ANSWER_SYSTEM, user, ledger)
        sent_units = lambda rs: list({u.unit_id: u for u in (self.store.by_id.get(f"{x['doc_id']}#s{si}") for x in rs for si in x["sent_idx"]) if u}.values())
        ranking = sent_units([mem.records[i] for i in cand])
        return PipelineResult(r.text, ranking, ranking, ranking, sent_units(recs), trunc,
                              {"plan_ok": plan_ok, "semantic_queries": qs, "n_routes": len(routes), "k": k,
                               "records_in_prompt": [line(x) for x in recs], "n_candidates": len(cand),
                               "index_stats": dict(mem.stats), "prompt_version": PROMPT_VERSION})
