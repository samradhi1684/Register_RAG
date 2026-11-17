"""HLG-RAG (Ghassel et al., KDD 2025, "Hierarchical Lexical Graph for Enhanced Multi-Hop Retrieval", arXiv:2506.08074) -
local, in-memory re-implementation on the RegisterRAG harness (no Neptune / OpenSearch / Bedrock).

Three-tier graph built offline with the generator LLM (2 calls per chunk):
  lineage       Source (document) -> Chunk (sequentially linked)
  summarisation Statement (atomic proposition, the retrieval unit) <- Topic (thematic group, per document)
                Fact (subject, predicate, object) SUPPORTS Statement
  entity        Entity (name + classification), linked to the statements/facts that mention it
Retrieval (paper Sec. 4):
  StatementGraphRAG  keyword->entity scoring + vector search -> union -> graph beam search over statements that share an
                     entity (attention-weighted path embedding) -> cross-encoder rerank -> TF-IDF diversity filter
  TopicGraphRAG      top-down topic search + bottom-up entity search -> beam search guided by the reranker -> rerank -> filter
Hyper-parameters named in the paper only symbolically (B, D_max, k) are set to the values reported by the paper's
experiment section as far as we could verify them (vss_k=100, beam_width=50, max_depth=3, diversity tau=0.5%); see README.

Ours (not in the paper): both extraction prompts (the AWS toolkit prompts were not verifiable verbatim), JSON output instead of
pipe-delimited text, the keyword prompt, a global (not per-start) beam, a neighbour cap per entity, nearest-sentence mapping
of statements to gold units for the harness' recall metrics.
--method-arg: retriever (statement|topic), vss_k (100), kw_k (50), beam_width (50), max_depth (3), max_neighbors (50),
diversity_tau (0.005), topic_k (50), chunk_words (256), cache_dir, rerank (true).
"""
from __future__ import annotations

import json
import math
import os
import re
import time

import numpy as np

from registerrag.generate import build_prompt
from registerrag.pipelines import BasePipeline, PipelineResult
from registerrag.prompts import QA_SYSTEMS

PROMPT_VERSION = "hlg-v1"
HERE = os.path.dirname(os.path.abspath(__file__))

PROPOSITION_SYSTEM = (
    "You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph. "
    "Break the text into simple, atomic statements (propositions). Preserve the original phrasing, resolve coreferences, "
    "replace pronouns and acronyms with the entities they refer to, and make every statement understandable on its own. "
    "Do not use knowledge that is not in the text. Reply with JSON only: {\"propositions\": [\"...\", \"...\"]}")
PROPOSITION_USER = "Source: {title}\n\nText:\n{text}"

TOPIC_SYSTEM = (
    "You organise numbered propositions into a knowledge graph. (1) Group the propositions into topics (short thematic "
    "names). (2) List the entities (people, places, works, organisations, dates...) with a one-word classification. "
    "(3) List facts as (subject, predicate, object) triples, predicate in UPPER_SNAKE_CASE, subject and object being "
    "entities. Refer to propositions by their number; do NOT rewrite them. Reply with JSON only: "
    '{"topics":[{"name":"...","statement_ids":[1,2]}],'
    '"entities":[{"name":"...","classification":"...","statement_ids":[1]}],'
    '"facts":[{"subject":"...","predicate":"...","object":"...","statement_ids":[1]}]}')
TOPIC_USER = "Source: {title}\n\nPropositions:\n{numbered}"

KEYWORD_SYSTEM = (
    "Extract the keywords of the search query: the named entities and the key concepts, each with at most two close "
    'synonyms or alternate spellings. Reply with JSON only: {"keywords": ["...", "..."]}')


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


def cap_spans(text: str) -> list:
    return [m.strip() for m in re.findall(r"(?:[A-Z][\w'’.-]*)(?:\s+(?:of|the|de|and|[A-Z][\w'’.-]*))*", text or "")
            if len(m.strip()) > 2]


def parse_json(text: str):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except ValueError:
        return None


def tfidf_vectors(texts):
    toks = [re.findall(r"\w+", norm(t)) for t in texts]
    df: dict = {}
    for t in toks:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    vocab = {w: i for i, w in enumerate(df)}
    n = len(texts)
    X = np.zeros((n, len(vocab)))
    for i, t in enumerate(toks):
        for w in t:
            X[i, vocab[w]] += 1
    idf = np.array([math.log((1 + n) / (1 + df[w])) + 1 for w in vocab]) if vocab else np.zeros(0)
    X = X * idf
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def diversity_filter(items, scores, texts, tau):
    """Paper Sec. 4.3: normalise, TF-IDF cosine; of two statements more similar than (1 - tau) keep the higher scored."""
    if len(items) < 2 or tau <= 0:
        return list(range(len(items)))
    X = tfidf_vectors(texts)
    order = sorted(range(len(items)), key=lambda i: -scores[i])
    keep, kept = [], np.zeros((len(items), X.shape[1]))              # one matrix-vector product per candidate
    for i in order:
        if not keep or float((kept[: len(keep)] @ X[i]).max()) <= 1.0 - tau:
            kept[len(keep)] = X[i]
            keep.append(i)
    return keep


# ------------------------------------------------------------------------------------------------------ the graph
class LexicalGraph:
    def __init__(self, store, embedder, gen_name, margs, prepend_title, cache_dir=None, log=print):
        self.store, self.embedder, self.log, self.prepend = store, embedder, log, prepend_title
        self.chunk_words = int(margs.get("chunk_words", 256))
        self.cache_dir = cache_dir if cache_dir is not None else os.path.join(HERE, "index")
        self.gen_name, self.cache_path, self.done = gen_name, None, {}
        self.statements, self.topics, self.entities = [], [], {}
        self.stats = {"chunks": 0, "statements": 0, "topics": 0, "entities": 0, "facts": 0, "fallback_chunks": 0}
        self.built = False

    def _chunks(self, doc):
        out, cur, n = [], [], 0
        for s in doc.sentences:
            w = len(s.split())
            if cur and n + w > self.chunk_words:
                out.append(" ".join(cur)); cur, n = [], 0
            cur.append(s); n += w
        if cur:
            out.append(" ".join(cur))
        return out

    def _cache(self):
        if self.cache_path or not self.cache_dir:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.gen_name)
        self.cache_path = os.path.join(self.cache_dir, f"graph_{safe}_{PROMPT_VERSION}.jsonl")
        if os.path.exists(self.cache_path):
            for l in open(self.cache_path, encoding="utf-8"):
                try:
                    r = json.loads(l); self.done[r["doc_id"]] = r["chunks"]
                except (ValueError, KeyError):
                    continue

    def _extract_chunk(self, title, text, gen):
        """-> dict(statements, topics, entities, facts, fallback). Never returns an empty statement list."""
        props, fallback = None, False
        for _ in range(2):
            try:
                r = gen.generate(PROPOSITION_SYSTEM, PROPOSITION_USER.format(title=title, text=text), json_mode=True,
                                 max_new_tokens=900)
                d = parse_json(r.text)
                props = [str(p).strip() for p in d["propositions"] if str(p).strip()]
                if props:
                    break
            except Exception:  # noqa: BLE001
                props = None
        if not props:
            props = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            fallback = True
        topics, ents, facts = [], [], []
        try:
            numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(props))
            r = gen.generate(TOPIC_SYSTEM, TOPIC_USER.format(title=title, numbered=numbered), json_mode=True,
                             max_new_tokens=1200)
            d = parse_json(r.text) or {}
            ok = lambda ids: [int(i) - 1 for i in (ids or []) if str(i).lstrip("-").isdigit() and 0 < int(i) <= len(props)]
            topics = [{"name": str(t.get("name", "")).strip(), "ids": ok(t.get("statement_ids"))} for t in d.get("topics", [])
                      if isinstance(t, dict) and t.get("name")]
            ents = [{"name": str(e.get("name", "")).strip(), "cls": str(e.get("classification", "")).strip(),
                     "ids": ok(e.get("statement_ids"))} for e in d.get("entities", []) if isinstance(e, dict) and e.get("name")]
            facts = [{"s": str(f.get("subject", "")).strip(), "p": str(f.get("predicate", "")).strip(),
                      "o": str(f.get("object", "")).strip(), "ids": ok(f.get("statement_ids"))}
                     for f in d.get("facts", []) if isinstance(f, dict) and f.get("subject") and f.get("object")]
        except Exception:  # noqa: BLE001
            fallback = True
        if not topics:
            topics = [{"name": title, "ids": list(range(len(props)))}]
        for i, p in enumerate(props):                     # entity safety net: capitalised spans of statements without entity
            if not any(i in e["ids"] for e in ents):
                for sp in cap_spans(p)[:6]:
                    ents.append({"name": sp, "cls": "", "ids": [i]})
        return {"statements": props, "topics": topics, "entities": ents, "facts": facts, "fallback": fallback}

    def build(self, gen):
        self._cache()
        t0, todo = time.perf_counter(), [d for d in self.store.docs if d not in self.done]
        for n, d in enumerate(todo, 1):
            doc = self.store.docs[d]
            chunks = [self._extract_chunk(doc.title, c, gen) for c in self._chunks(doc)]
            self.done[d] = chunks
            if self.cache_path:
                with open(self.cache_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"doc_id": d, "chunks": chunks}) + "\n")
            if n % 20 == 0 or n == len(todo):
                self.log(f"  [HLG index] {n}/{len(todo)} docs ({time.perf_counter() - t0:.0f}s)")
        self._assemble()

    def _assemble(self):
        self.statements, self.topics, self.entities = [], [], {}
        nfact = nfb = 0
        for d, chunks in self.done.items():
            if d not in self.store.docs:
                continue
            doc = self.store.docs[d]
            sent_ids = [self.store.by_id.get(f"{d}#s{i}") for i in range(len(doc.sentences))]
            for ci, ch in enumerate(chunks):
                base = len(self.statements)
                nfb += bool(ch["fallback"])
                for i, text in enumerate(ch["statements"]):
                    self.statements.append({"sid": base + i, "doc_id": d, "title": doc.title, "text": text, "chunk": ci,
                                            "ents": set(), "topics": []})
                for t in ch["topics"]:
                    self.topics.append({"tid": len(self.topics), "doc_id": d, "name": t["name"],
                                        "sids": [base + i for i in t["ids"]]})
                    for i in t["ids"]:
                        self.statements[base + i]["topics"].append(len(self.topics) - 1)
                for e in ch["entities"]:
                    key = norm(e["name"])
                    if not key:
                        continue
                    for i in e["ids"]:
                        self.statements[base + i]["ents"].add(key)
                    self.entities.setdefault(key, {"name": e["name"], "cls": e["cls"], "sids": set()})["sids"].update(base + i for i in e["ids"])
                for f in ch["facts"]:
                    nfact += 1
                    for i in f["ids"]:
                        for part in (f["s"], f["o"]):
                            k = norm(part)
                            if k:
                                self.statements[base + i]["ents"].add(k)
                                self.entities.setdefault(k, {"name": part, "cls": "", "sids": set()})["sids"].add(base + i)
        texts = [(f"{s['title']}. {s['text']}" if self.prepend else s["text"]) for s in self.statements]
        self.S = self.embedder.encode_docs(texts) if texts else np.zeros((0, 1), np.float32)
        ttexts = [f"{t['name']}. " + " ".join(self.statements[i]["text"] for i in t["sids"][:12]) for t in self.topics]
        self.T = self.embedder.encode_docs(ttexts) if ttexts else np.zeros((0, 1), np.float32)
        self.ent_index = {}
        for k, v in self.entities.items():
            for sid in v["sids"]:
                self.ent_index.setdefault(k, []).append(sid)
        self.stats.update(chunks=sum(len(c) for c in self.done.values()), statements=len(self.statements),
                          topics=len(self.topics), entities=len(self.entities), facts=nfact, fallback_chunks=nfb)
        self.built = True

    def map_to_sentence(self, st, retriever):
        """Nearest source sentence of the statement's document (only for the harness' gold-recall metrics)."""
        cand = self.store.doc2idx.get(st["doc_id"])
        if cand is None or len(cand) == 0:
            return None
        sims = retriever.emb[cand] @ self.S[st["sid"]] if retriever.emb.shape[1] == self.S.shape[1] else np.zeros(len(cand))
        return self.store.units[int(cand[int(np.argmax(sims))])]


# --------------------------------------------------------------------------------------------------- the pipeline
class HLGRAG(BasePipeline):
    name = "hlg"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        m = self.margs = dict(self.cfg.method_args or {})
        self.mode = str(m.get("retriever", "statement"))
        if self.mode not in ("statement", "topic"):
            raise ValueError("retriever must be statement or topic")
        self.vss_k = int(m.get("vss_k", 100)); self.kw_k = int(m.get("kw_k", 50))
        self.B = int(m.get("beam_width", 50)); self.D = int(m.get("max_depth", 3))
        self.max_nbr = int(m.get("max_neighbors", 50)); self.tau = float(m.get("diversity_tau", 0.005))
        self.topic_k = int(m.get("topic_k", 50)); self.use_rerank = bool(m.get("rerank", True))
        self.g = LexicalGraph(self.store, self.retriever.embedder, getattr(self.generator, "name", "gen"), m,
                              self.cfg.prepend_title, cache_dir=m.get("cache_dir"))

    def build_index(self):
        self.g.build(self.generator)
        return self.g.stats

    # -- retrieval steps ------------------------------------------------------
    def _keywords(self, q, ledger):
        try:
            r = self._lm("compiler", KEYWORD_SYSTEM, f"Query: {q}", ledger, json_mode=True, max_new_tokens=200)
            kws = [norm(k) for k in (parse_json(r.text) or {}).get("keywords", []) if norm(str(k))]
            if kws:
                return list(dict.fromkeys(kws))
        except Exception:  # noqa: BLE001
            pass
        return list(dict.fromkeys(norm(s) for s in cap_spans(q))) or [norm(q)]

    def _keyword_hits(self, kws, qv, mask):
        g, score = self.g, {}
        for k in kws:
            for ek, sids in g.ent_index.items():
                if ek == k or (len(k) > 3 and (k in ek or ek in k) and len(ek) > 3):
                    for sid in set(sids):
                        score.setdefault(sid, set()).add(k)
        cand = [(len(v), float(g.S[sid] @ qv), sid) for sid, v in score.items() if mask is None or mask[sid]]
        cand.sort(key=lambda t: (-t[0], -t[1]))
        return [sid for _, _, sid in cand[: self.kw_k]]

    def _neighbours(self, sid, qv, mask):
        g, out = self.g, set()
        for e in g.statements[sid]["ents"]:
            out.update(g.ent_index.get(e, []))
        out.discard(sid)
        out = [n for n in out if mask is None or mask[n]]
        if len(out) > self.max_nbr:
            out.sort(key=lambda n: -float(g.S[n] @ qv))
            out = out[: self.max_nbr]
        return out

    def _beam(self, starts, qv, mask, rerank_score=None):
        """Global beam search over statements sharing an entity. Statement mode: path score = cos(e_Q, attention-weighted
        path embedding), weights softmax(sim(e_Q, e_s)). Topic mode: mean reranker score along the path."""
        g = self.g
        sim = lambda sid: float(g.S[sid] @ qv)

        def path_score(path):
            if rerank_score is not None:
                return float(np.mean([rerank_score(s) for s in path]))
            w = np.exp(np.array([sim(s) for s in path])); w /= w.sum()
            e = (w[:, None] * g.S[path]).sum(axis=0)
            return float(e @ qv / max(np.linalg.norm(e), 1e-12))

        beam = [([s], path_score([s])) for s in starts]
        beam = sorted(beam, key=lambda t: -t[1])[: self.B]
        found = set()
        for _ in range(self.D):
            children = []
            for path, sc in beam:
                for n in self._neighbours(path[-1], qv, mask):
                    if n in path:
                        continue
                    np_ = path + [n]
                    ns = path_score(np_)
                    if ns > sc:                                  # expand only while the child improves its parent
                        children.append((np_, ns))
            if not children:
                break
            beam = sorted(children, key=lambda t: -t[1])[: self.B]
            for p, _ in beam:
                found.update(p)
        return found

    def _rerank(self, q, sids):
        if not sids:
            return []
        txt = lambda sid: f"{self.g.statements[sid]['title']}. {self.g.statements[sid]['text']}"
        if self.reranker is None or not self.use_rerank:
            qv = self.retriever.embedder.encode_query(q)
            return sorted(((sid, float(self.g.S[sid] @ qv)) for sid in sids), key=lambda t: -t[1])
        items = [{"sid": s} for s in sids]
        ranked = self.reranker.rerank(q, items, lambda it: txt(it["sid"]))
        return [(it["sid"], sc) for it, sc in ranked]

    def run(self, ex, ledger):
        g = self.g
        if not g.built:
            g.build(self.generator)
        q = ex.question
        qv = self.retriever.embedder.encode_query(q).astype(np.float32)
        mask = None
        if self.cfg.corpus_scope == "per_question":
            allowed = set(ex.doc_ids)
            mask = np.array([s["doc_id"] in allowed for s in g.statements], dtype=bool)
        cand = np.arange(len(g.statements)) if mask is None else np.flatnonzero(mask)
        kws = self._keywords(q, ledger)
        ledger.call("retriever")
        info = {"keywords": kws, "mode": self.mode}
        if self.mode == "statement":
            vs = g.S[cand] @ qv
            vss = [int(cand[i]) for i in np.argsort(-vs, kind="stable")[: self.vss_k]]
            kwh = self._keyword_hits(kws, qv, mask)
            init = list(dict.fromkeys(kwh + vss))
            beam = self._beam(init, qv, mask)
            final = list(dict.fromkeys(init + sorted(beam)))
            info.update(n_vss=len(vss), n_kw=len(kwh), n_init=len(init), n_beam=len(beam), n_final=len(final))
        else:
            tmask = np.ones(len(g.topics), bool) if mask is None else np.array([mask[t["sids"][0]] if t["sids"] else False for t in g.topics])
            tc = np.flatnonzero(tmask)
            ts = g.T[tc] @ qv
            top_t = [int(tc[i]) for i in np.argsort(-ts, kind="stable")[: self.topic_k]]
            topic_sids = [s for t in top_t for s in g.topics[t]["sids"]]
            kwh = self._keyword_hits(kws, qv, mask)
            init = list(dict.fromkeys(topic_sids + kwh))
            cache = {}
            ledger.call("reranker")

            def rs(sid):
                if sid not in cache:
                    cache[sid] = dict(self._rerank(q, [sid]))[sid] if self.reranker is not None else float(g.S[sid] @ qv)
                return cache[sid]
            if self.reranker is not None:                       # score the seed set in one batch (fast path)
                cache.update(dict(self._rerank(q, init)))
            beam = self._beam(init[: self.B * 4], qv, mask, rerank_score=rs)
            final = list(dict.fromkeys(init + sorted(beam)))
            info.update(n_topics=len(top_t), n_kw=len(kwh), n_init=len(init), n_beam=len(beam), n_final=len(final))
        if self.reranker is not None and self.use_rerank:
            ledger.call("reranker")
            ledger.side_tokens(sum(self.counter.count(q) + self.counter.count(g.statements[s]["text"]) for s in final))
        ranked = self._rerank(q, final)
        sids = [s for s, _ in ranked]
        keep = diversity_filter(sids, [sc for _, sc in ranked], [g.statements[s]["text"] for s in sids], self.tau)
        sids = [sids[i] for i in keep]
        top = [g.statements[s] for s in sids[: self.cfg.top_k]]
        system = QA_SYSTEMS[self.cfg.prompt_version]
        idx_text = lambda st: (f"{st['title']}. {st['text']}" if self.cfg.prepend_title else st["text"])
        user, kept, trunc = build_prompt(self.counter, system, q, top, idx_text, self.cfg.budget_tokens)
        r = self._lm("generator", system, user, ledger)
        mapped = lambda sts: list({u.unit_id: u for u in (g.map_to_sentence(st, self.retriever) for st in sts) if u is not None}.values())
        ranking = mapped([g.statements[s] for s in sids])
        stage1 = mapped([g.statements[s] for s in final])
        info.update(statements_in_prompt=[st["text"] for st in kept], n_after_diversity=len(sids),
                    unit_mapping="statement -> nearest source sentence (metrics only)", prompt_version=PROMPT_VERSION,
                    index_stats=dict(g.stats))
        return PipelineResult(r.text, ranking, stage1, ranking, mapped(kept), trunc, info)
