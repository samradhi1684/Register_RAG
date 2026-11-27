"""
Hybrid retriever + 1-hop graph expansion.

Improvements over Stage 1 baseline:
  - Multi-query rank fusion: build up to 3 query variants per requirement,
    run hybrid search for each, fuse via Reciprocal Rank Fusion (RRF k=60).
    Activated by config.multi_query_enabled (default True).
  - Title-boost: units from an article whose title matches a resolved
    dep-binding get +title_boost_weight added to their score.
    Critical for 2-hop bridging in a global index.
    Activated by config.title_boost_enabled (default True).
  - Falls back to original single-query path when multi_query_enabled=False.
"""
from __future__ import annotations
import logging
import re

import numpy as np

from registerrag.config import Config
from registerrag.corpus.graph import EvidenceGraph
from registerrag.corpus.indexer import HybridIndex, embed_query
from registerrag.models import EvidenceUnit, RegisterEntry, Requirement, RetrievedCandidate

logger = logging.getLogger(__name__)

# RRF constant — k=60 is the standard default from the original paper
_RRF_K = 60


class Retriever:
    def __init__(
        self,
        index: HybridIndex,
        graph: EvidenceGraph,
        config: Config,
    ) -> None:
        self.index = index
        self.graph = graph
        self.config = config
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(
            config.embedding_model, device=config.embedding_device
        )

    def retrieve(
        self,
        requirement: Requirement,
        register: dict[str, RegisterEntry],
    ) -> list[RetrievedCandidate]:
        """
        Retrieve candidate evidence units for one requirement.
        Uses multi-query RRF + title-boost when config.multi_query_enabled.
        """
        bindings = {
            rid: e.binding
            for rid, e in register.items()
            if e.is_supported and e.binding
        }

        if self.config.multi_query_enabled:
            cands = self._retrieve_multi(requirement, bindings)
        else:
            cands = self._retrieve_single(requirement, bindings)

        # Entity anchoring: a dependent step is about its dependency's entity, so
        # that entity's own article is the right place to look ("Where was
        # [R1] born?" with R1=Bob Seger → the "Bob Seger" article). Inject the
        # article's units so the reranker can see them even if lexical /
        # dense search missed them.
        have = {c.unit.unit_id for c in cands}
        floor = min((c.hybrid_score for c in cands), default=0.0)
        anchored = False
        if self.config.entity_anchor_enabled and requirement.depends_on:
            for rid in requirement.depends_on:
                anchored |= self._inject(cands, have, floor, self.anchor_units(bindings.get(rid, "")),
                                         requirement, bindings)

        # Root-step title anchoring: a root step names its entity in the text
        # ("Who directed the film Caminito De Gloria?"); inject the article(s)
        # whose title is mentioned there, matched loosely (case, quotes,
        # "kinda"/"kind of", "(1999 Film)" disambiguators).
        if getattr(self.config, "root_title_anchor_enabled", True) and not requirement.depends_on:
            titles = self.mentioned_titles(requirement.information_need)
            if titles:
                logger.debug("Root title anchor: %s -> %s", requirement.id, titles)
            for t in titles:
                anchored |= self._inject(cands, have, floor, self._docs_by_title()[t],
                                         requirement, bindings)

        # Question-scope fallback: a step that found no anchor of its own is
        # searched inside the scope(s) the question names (fuzzy match, see
        # pipeline/scope.py) — e.g. a policy question's program.
        if not anchored and self._question_scopes:
            for name in self._question_scopes:
                units = [u for d in self._scope_matcher().scopes.get(name, [])
                         for u in self._units_by_doc().get(d, [])]
                self._inject(cands, have, floor, units, requirement, bindings)
        return cands

    # ─────────────────────────────────────────────────────────────────────────
    # Anchoring helpers
    # ─────────────────────────────────────────────────────────────────────────

    _question_scopes: list = []

    def set_question(self, question: str) -> list[str]:
        """Called once per question: fix the scope(s) the question names."""
        self._question_scopes = []
        if getattr(self.config, "scope_anchor_enabled", False):
            self._question_scopes = self._scope_matcher().match(
                question, min_coverage=self.config.scope_min_coverage)
            if self._question_scopes:
                logger.info("Question scope: %s", self._question_scopes)
        return self._question_scopes

    def _scope_matcher(self):
        if not hasattr(self, "_scopes"):
            from registerrag.pipeline.scope import ScopeMatcher
            self._scopes = ScopeMatcher.for_index(self.config.index_dir, self.index.units)
        return self._scopes

    def _units_by_doc(self) -> dict[str, list[EvidenceUnit]]:
        if not hasattr(self, "_doc_units"):
            d: dict[str, list[EvidenceUnit]] = {}
            for u in self.index.units:
                d.setdefault(u.document_id, []).append(u)
            self._doc_units = d
        return self._doc_units

    def _inject(self, cands, have, floor, units, requirement, bindings) -> bool:
        """
        Add anchored units as candidates.  A small document (≤
        anchor_inject_all_max units — every 2Wiki article) is injected whole;
        from a large one (a policy program with hundreds of units) only the
        top anchor_scope_top_n units of a hybrid search inside it.
        Returns True if the anchor resolved to any unit.
        """
        if not units:
            return False
        if len(units) > getattr(self.config, "anchor_inject_all_max", 10 ** 9):
            q = self._build_base_query(requirement, bindings)
            emb = self._encoder.encode([q], normalize_embeddings=False)[0].astype(np.float32)
            hits = self.index.search_within(q, emb, [u.unit_id for u in units],
                                            top_k=self.config.anchor_scope_top_n)
            units = [self.index.get_unit(uid) for uid, _ in hits]
        for unit in units:
            if unit is not None and unit.unit_id not in have:
                have.add(unit.unit_id)
                cands.append(RetrievedCandidate(unit=unit, hybrid_score=floor, rank=len(cands) + 1))
        return True

    _NORM_SUBS = ((re.compile(r"[’‘`]"), "'"), (re.compile(r"'s\b"), ""),(re.compile(r"\bkinda\b"), "kind of"),
                  (re.compile(r"\bgonna\b"), "going to"), (re.compile(r"[^\w'()]+"), " "),
                  (re.compile(r"(?<!\w)'|'(?!\w)"), " "))

    @classmethod
    def _norm_title_text(cls, s: str) -> str:
        s = s.lower()
        for pat, rep in cls._NORM_SUBS:
            s = pat.sub(rep, s)
        return " " + " ".join(s.split()) + " "

    def mentioned_titles(self, text: str, max_docs: int = 2) -> list[str]:
        """
        Titles of articles named in `text`. A full title (with its "(… film)"
        disambiguator) beats a bare base-title match; when only the base title
        matches several articles (eight "Charley's Aunt" films) none is taken.
        Single-word titles need ≥6 chars and a capitalised mention.
        """
        if not hasattr(self, "_title_norms"):
            self._title_norms = [
                (t, self._norm_title_text(t), self._norm_title_text(self._PAREN.sub("", t)))
                for t in self._docs_by_title() if t
            ]
        nt = self._norm_title_text(text)
        full, base = [], {}
        for t, tn, bn in self._title_norms:
            words = bn.split()
            if not words or len(bn.strip()) < 4:
                continue
            if len(words) == 1:
                w = self._PAREN.sub("", t).strip()
                if len(w) < 6 or not re.search(rf"\b{re.escape(w)}\b", text):
                    continue
            if tn != bn and tn in nt:
                full.append(t)
            elif bn in nt:
                base.setdefault(bn, []).append(t)
        full_bases = {self._norm_title_text(self._PAREN.sub("", t)) for t in full}
        kind = re.search(r"\b(film|movie|song|album|novel|band)\b", text, re.I)
        for bn, ts in base.items():
            if bn in full_bases:
                continue                     # the disambiguated title already matched
            if len(ts) > 1 and kind:         # "the film The Yellow Birds" → "The Yellow Birds (film)"
                k = kind.group(1).lower().replace("movie", "film")
                ts = [t for t in ts if k in t.lower()[len(self._PAREN.sub("", t)):]]
            if len(ts) == 1:
                full.append(ts[0])
        hits = full
        # drop titles strictly contained in a longer hit ("Sight" inside "Love at Second Sight")
        bases = {t: self._norm_title_text(self._PAREN.sub("", t)) for t in hits}
        hits = [t for t in hits
                if not any(bases[t] != bases[o] and bases[t] in bases[o] for o in hits)]
        hits.sort(key=len, reverse=True)
        return hits[:max_docs]

    # ─────────────────────────────────────────────────────────────────────────
    # Entity anchoring
    # ─────────────────────────────────────────────────────────────────────────

    _PAREN = re.compile(r"\s*\([^)]*\)\s*$")

    def _docs_by_title(self) -> dict[str, list[EvidenceUnit]]:
        if not hasattr(self, "_title_docs"):
            docs: dict[str, list[EvidenceUnit]] = {}
            for u in self.index.units:
                docs.setdefault(u.document_title or "", []).append(u)
            self._title_docs = docs
        return self._title_docs

    @classmethod
    def title_matches(cls, binding: str, title: str) -> bool:
        """
        True if `title` names the entity `binding`: equal after dropping a
        trailing "(…)" disambiguator, or one starts with the other at a word
        boundary (" " or ","), e.g. "John Chaworth" ↔ "John Chaworth, 2nd
        Viscount Chaworth". Single-word bindings must match exactly.
        """
        b = " ".join(binding.lower().split())
        t = cls._PAREN.sub("", title.lower()).strip()
        if len(b) < 4 or not t:
            return False
        if b == t:
            return True
        if len(b.split()) < 2 or len(t.split()) < 2:
            return False
        return (t.startswith(b) and t[len(b):len(b) + 1] in (" ", ",")) or \
               (b.startswith(t) and b[len(t):len(t) + 1] in (" ", ","))

    def anchor_units(self, binding: str, max_docs: int = 2) -> list[EvidenceUnit]:
        """All sentences of the (at most `max_docs`) articles whose title names `binding`."""
        binding = (binding or "").strip()
        if not binding:
            return []
        docs = self._docs_by_title()
        exact = [t for t in docs if self._PAREN.sub("", t.lower()).strip() == binding.lower()]
        titles = exact or [t for t in docs if self.title_matches(binding, t)]
        out: list[EvidenceUnit] = []
        for t in titles[:max_docs]:
            out.extend(docs[t])
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-query path
    # ─────────────────────────────────────────────────────────────────────────

    def _retrieve_multi(
        self,
        requirement: Requirement,
        bindings: dict[str, str],
    ) -> list[RetrievedCandidate]:
        """Build several query variants, run hybrid search for each, fuse with RRF."""
        variants = self._build_query_variants(requirement, bindings)

        # Deeper per-list pool when fusing several ranked lists. max() keeps the
        # Stage-2 gap-fill boost (which temporarily raises top_k_initial) effective.
        pool_k = max(self.config.top_k_multi_query, self.config.top_k_initial)

        rrf_scores: dict[str, float] = {}   # unit_id → cumulative RRF score
        unit_by_id: dict[str, EvidenceUnit] = {}

        for variant in variants:
            q_emb = self._encoder.encode(
                [variant], normalize_embeddings=False
            )[0].astype(np.float32)
            ranked = self.index.search(variant, q_emb, top_k=pool_k)
            for rank, (uid, _score) in enumerate(ranked):
                rrf_scores[uid] = rrf_scores.get(uid, 0.0) + 1.0 / (_RRF_K + rank + 1)
                if uid not in unit_by_id:
                    unit = self.index.get_unit(uid)
                    if unit:
                        unit_by_id[uid] = unit

        # Graph expansion from top RRF seeds
        seed_ids = sorted(rrf_scores, key=lambda u: rrf_scores[u], reverse=True)[
            : self.config.graph_expansion_top_n
        ]
        expanded_ids = self.graph.expand(seed_ids, hops=self.config.graph_expansion_hops)

        # Score expanded units with the primary query (dense only, discounted)
        primary_query = variants[0]
        primary_emb = self._encoder.encode(
            [primary_query], normalize_embeddings=False
        )[0].astype(np.float32)
        # Expanded units must live on the RRF scale (≈0.016 per list hit), not the
        # raw [0,1] cosine scale — otherwise every graph neighbour outranks every
        # directly-retrieved unit. Treat each as a single-list hit ranked by dense
        # similarity, placed after the retrieved pool.
        if expanded_ids:
            extra = self._score_dense(
                [u for u in expanded_ids if u not in rrf_scores], primary_emb
            )
            extra.sort(key=lambda x: x[1], reverse=True)
            for i, (uid, _dense) in enumerate(extra):
                rrf_scores[uid] = 1.0 / (_RRF_K + pool_k + i + 1)
                unit = self.index.get_unit(uid)
                if unit:
                    unit_by_id[uid] = unit

        # Title boost on the same RRF scale. The default weight 0.2 maps to +1/61,
        # i.e. the equivalent of one extra rank-1 list hit (was a flat +0.2,
        # ~10x the largest possible RRF score, which buried everything else).
        title_boost = (self.config.title_boost_weight / 0.2) / (_RRF_K + 1)

        # Title-boost: units whose article title matches a resolved dep-binding
        title_boost_names = {
            v.lower() for v in bindings.values() if v and len(v.split()) <= 5
        }

        candidates = []
        for uid, score in rrf_scores.items():
            unit = unit_by_id.get(uid)
            if unit is None:
                continue

            if self.config.title_boost_enabled and title_boost_names:
                doc_title = unit.document_title.lower() if unit.document_title else ""
                if doc_title and any(
                    name in doc_title or doc_title in name
                    for name in title_boost_names
                    if len(name) > 3
                ):
                    score += title_boost

            candidates.append(RetrievedCandidate(unit=unit, hybrid_score=score, rank=0))

        candidates.sort(key=lambda c: c.hybrid_score, reverse=True)
        candidates = candidates[:pool_k]
        for i, c in enumerate(candidates):
            c.rank = i + 1

        return candidates

    # ─────────────────────────────────────────────────────────────────────────
    # Single-query path (original Stage 1 fallback)
    # ─────────────────────────────────────────────────────────────────────────

    def _retrieve_single(
        self,
        requirement: Requirement,
        bindings: dict[str, str],
    ) -> list[RetrievedCandidate]:
        """Original Stage 1 single-query hybrid + graph expansion."""
        query = self._build_base_query(requirement, bindings)
        q_emb = self._encoder.encode(
            [query], normalize_embeddings=False
        )[0].astype(np.float32)

        scored_ids = self.index.search(query, q_emb, top_k=self.config.top_k_initial)

        seed_ids = [uid for uid, _ in scored_ids[: self.config.graph_expansion_top_n]]
        expanded_ids = self.graph.expand(seed_ids, hops=self.config.graph_expansion_hops)

        if expanded_ids:
            extra_scores = self._score_dense(list(expanded_ids), q_emb)
            existing_ids = {uid for uid, _ in scored_ids}
            for uid, score in extra_scores:
                if uid not in existing_ids:
                    scored_ids.append((uid, score * 0.9))

        scored_ids.sort(key=lambda x: x[1], reverse=True)
        scored_ids = scored_ids[: self.config.top_k_initial]

        candidates = []
        for rank, (uid, score) in enumerate(scored_ids):
            unit = self.index.get_unit(uid)
            if unit is None:
                continue
            candidates.append(RetrievedCandidate(unit=unit, hybrid_score=score, rank=rank + 1))

        return candidates

    # ─────────────────────────────────────────────────────────────────────────
    # Query variant builder
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_placeholders(text: str, bindings: dict[str, str]) -> str:
        """
        Replace references to earlier requirements with their bindings, e.g.
          "[R1]", "[R1's father]", "[answer from R1]"         → "<R1 binding>"
          "the director identified in R1", "the person found in R1",
          "the answer from R1", "result of R1", bare "R1"     → "<R1 binding>"
        References whose requirement has no binding are left untouched.
        """
        out = text
        for rid, val in bindings.items():
            val = (val or "").strip()
            if not val or not re.search(r"\b" + re.escape(rid) + r"\b", out):
                continue
            r = re.escape(rid)
            # Bracketed forms: [R1], [R1's X], [answer from R1], [result of R1]
            out = re.sub(r"\[[^\]]*\b" + r + r"\b[^\]]*\]", val, out)
            # "the <noun> identified/found/named/... in/by/from R1"
            out = re.sub(
                r"\b(?:the\s+)?(?:[\w\-]+\s+)?"
                r"(?:identified|found|named|given|obtained|mentioned|determined|returned|retrieved)"
                r"\s+(?:in|by|from|at)\s+(?:step\s+)?" + r + r"\b",
                val, out, flags=re.IGNORECASE,
            )
            # "the answer/result/output from/of/to R1"
            out = re.sub(
                r"\b(?:the\s+)?(?:answer|result|output|entity|person|value)\s+(?:from|of|to|in)\s+(?:step\s+)?"
                + r + r"\b",
                val, out, flags=re.IGNORECASE,
            )
            # Bare reference: "R1"
            out = re.sub(r"\b" + r + r"\b", val, out)
        return " ".join(out.split())

    def _build_query_variants(
        self,
        req: Requirement,
        bindings: dict[str, str],
    ) -> list[str]:
        """
        Build up to multi_query_max_variants (default 3) distinct query strings,
        in priority order:

        1. Concrete-need: dep binding substituted into the info-need text
           (only when the need contains a [Rk] placeholder that resolves).
        2. Dep+need: dep binding values followed by the info need.
        3. Entity-title: the dep binding alone (pulls the entity's own article).

        The former reversed / relation-term / birthplace / other-bindings variants
        were dropped: with 7 lists fused by RRF they diluted the strong signals
        and pushed gold units out of the pool (recall 0.84 → 0.525).
        """
        dep_values = [
            bindings[rid].strip()
            for rid in req.depends_on
            if rid in bindings and bindings[rid].strip()
        ]

        raw_need = req.information_need.strip()
        # Substitute [R1] / "identified in R1" / … with the actual dep bindings
        target = self._resolve_placeholders(raw_need, bindings)
        dep_text = " ".join(dep_values)

        variants: list[str] = []

        # 1. Concrete-need (placeholders resolved to dep bindings)
        if target != raw_need:
            variants.append(target)

        # 2. Dep+need — skip prepending dep values already present in the need
        missing_deps = " ".join(v for v in dep_values if v.lower() not in target.lower())
        variants.append(" ".join(p for p in (missing_deps, target) if p))

        # 3. Entity-title
        for dep_val in dep_values:
            if len(dep_val.split()) <= 5:
                variants.append(dep_val)

        # De-duplicate preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for q in variants:
            q = " ".join(q.split())
            key = q.lower()
            if q and key not in seen:
                seen.add(key)
                unique.append(q)

        return unique[: self.config.multi_query_max_variants]

    def _build_base_query(
        self,
        requirement: Requirement,
        bindings: dict[str, str],
    ) -> str:
        """Single-query fallback (original Stage 1 logic)."""
        base = self._resolve_placeholders(requirement.information_need, bindings)
        known_parts = [
            bindings[dep_id]
            for dep_id in requirement.depends_on
            if dep_id in bindings and bindings[dep_id]
            and bindings[dep_id].lower() not in base.lower()
        ]
        if known_parts:
            return f"{base} {' '.join(known_parts)}"
        return base

    # ─────────────────────────────────────────────────────────────────────────
    # Dense scoring helper
    # ─────────────────────────────────────────────────────────────────────────

    def _score_dense(
        self,
        unit_ids: list[str],
        q_emb: np.ndarray,
    ) -> list[tuple[str, float]]:
        """Score unit_ids against a query embedding (cosine similarity)."""
        import faiss
        embs = self.index.get_embeddings()
        unit_id_list = [u.unit_id for u in self.index.units]
        uid_to_idx = {uid: i for i, uid in enumerate(unit_id_list)}

        q = q_emb.reshape(1, -1).copy()
        faiss.normalize_L2(q)

        results = []
        for uid in unit_ids:
            idx = uid_to_idx.get(uid)
            if idx is None:
                continue
            e = embs[idx].copy().reshape(1, -1)
            faiss.normalize_L2(e)
            score = float(np.dot(q[0], e[0]))
            results.append((uid, (score + 1.0) / 2.0))

        return results
