"""
Stage 2 orchestrator — RegisterRAG with gap filling, eviction, and answer verification.

Pipeline per question
─────────────────────
1.  compile_requirements                       (LLM call #1)
2.  topological_sort requirements
3.  First pass — for each requirement in topo order:
      a. retrieve candidates (hybrid + 1-hop graph)
      b. rerank with cross-encoder
      c. write to register  (eviction enforced if over budget)
4.  Gap detection — find requirements with missing / low-confidence bindings
5.  Gap-fill loop (up to gap_fill_max_iter iterations):
      for each gap requirement:
        a. build bridge query  (substitute resolved dep bindings)
        b. retrieve with boosted top_k
        c. rerank
        d. update register (was_gap_fill=True)
      break early if no gaps remain
6.  build proof certificate (greedy set cover)
7.  order register entries by utility for generator
8.  generate answer  (LLM call #2 … #2+retries)
9.  answer verification + retry (cross-encoder, up to answer_verify_max_retries)
10. log MD trace + metrics

LLM calls: 2 baseline + up to answer_verify_max_retries extra (default = 3 max).
"""
from __future__ import annotations
import logging
import time

from registerrag.config import Config
from registerrag.corpus.graph import EvidenceGraph
from registerrag.corpus.indexer import HybridIndex
from registerrag.models import (
    EvidenceUnit, ProofCertificate, QuestionResult,
    RegisterEntry, Requirement, RetrievedCandidate,
)
from registerrag.pipeline.compiler import compile_requirements, topological_order
from registerrag.pipeline.generator import generate_answer_stage2
from registerrag.pipeline.proof import build_proof_certificate
from registerrag.pipeline.register import RegisterManager
from registerrag.pipeline.retriever import Retriever
from registerrag.pipeline.verifier import CrossEncoderVerifier

logger = logging.getLogger(__name__)


class Stage2Pipeline:
    """
    Stage 2: adds gap filling, register eviction, ordered context, and
    answer verification on top of the Stage 1 single-pass pipeline.

    Parameters
    ----------
    index    : pre-built HybridIndex
    graph    : pre-built EvidenceGraph
    verifier : pre-loaded CrossEncoderVerifier
    nlp      : spaCy model (for RegisterManager binding extraction)
    config   : Config
    runs_dir : where to write per-question MD traces
    """

    def __init__(
        self,
        index: HybridIndex,
        graph: EvidenceGraph,
        verifier: CrossEncoderVerifier,
        nlp,
        config: Config,
        runs_dir: str,
        llm_client=None,
    ) -> None:
        self.index = index
        self.graph = graph
        self.verifier = verifier
        self.config = config
        self.runs_dir = runs_dir
        self.llm_client = llm_client  # for LLM binding extraction (optional)
        self.retriever = Retriever(index, graph, config)
        self.register_mgr = RegisterManager(config, nlp)

    # ─────────────────────────────────────────────────────────────────────────
    # Public
    # ─────────────────────────────────────────────────────────────────────────

    def run_question(
        self,
        question_id: str,
        question: str,
        gold_answer: str,
        supporting_facts: list,
    ) -> QuestionResult:
        t0 = time.time()
        from registerrag.llm_cache import usage_snapshot
        usage0 = usage_snapshot()
        total_llm_calls = 0
        self.register_mgr.reset()
        self.register_mgr.set_question(question)
        self.retriever.set_question(question)

        # ── Step 1–2: compile + sort ──────────────────────────────────────────
        requirements, calls = compile_requirements(question, self.config)
        total_llm_calls += calls
        ordered_reqs = topological_order(requirements)

        # ── Step 3: first pass (priority-ordered scheduling) ─────────────────
        # Always process requirements whose deps are resolved first.
        # Among ready requirements, pick the one with the most urgent downstream
        # deps (most other reqs waiting on it → process it first).
        all_candidates: dict[str, list[RetrievedCandidate]] = {}
        unit_pool: dict[str, EvidenceUnit] = {}

        resolved_ids: set[str] = set()
        unsupported = list(ordered_reqs)
        processing_order: list[Requirement] = []

        while unsupported:
            ready = [r for r in unsupported if all(d in resolved_ids for d in r.depends_on)]
            if not ready:
                ready = unsupported[:1]  # circular / unresolvable — force progress

            def _urgency(req: Requirement) -> int:
                return sum(1 for r in unsupported if req.id in r.depends_on)

            ready.sort(key=_urgency, reverse=True)
            chosen = ready[0]
            processing_order.append(chosen)
            unsupported.remove(chosen)
            resolved_ids.add(chosen.id)

        for req in processing_order:
            candidates = self.retriever.retrieve(req, self.register_mgr.entries)
            ranked = self.verifier.rerank(
                req, candidates, top_k=self.config.top_k_final,
                register=self.register_mgr.entries,
            )
            self.register_mgr.add(
                req, ranked, ordered_reqs,
                was_gap_fill=False,
                llm_client=self.llm_client,
            )
            all_candidates[req.id] = ranked
            for c in ranked:
                unit_pool[c.unit.unit_id] = c.unit

        # ── Step 4–5: gap detection + iterative gap filling ───────────────────
        gap_fill_stats = {"iterations": 0, "gaps_filled": 0}

        if self.config.gap_fill_enabled:
            for iteration in range(self.config.gap_fill_max_iter):
                gaps = self.register_mgr.gaps(ordered_reqs)
                if not gaps:
                    break  # all requirements satisfied

                gap_fill_stats["iterations"] += 1
                logger.debug(
                    "[%s] Gap-fill iteration %d: %d gaps — %s",
                    question_id, iteration + 1, len(gaps),
                    [g.id for g in gaps],
                )

                for gap_req in gaps:
                    bridge_query = self._build_bridge_query(gap_req, ordered_reqs)
                    # Boost top_k for gap retrieval
                    orig_top_k = self.config.top_k_initial
                    self.config.top_k_initial = orig_top_k + self.config.gap_fill_top_k_boost
                    gap_candidates = self.retriever.retrieve(gap_req, self.register_mgr.entries)
                    self.config.top_k_initial = orig_top_k  # restore

                    gap_ranked = self.verifier.rerank(
                        gap_req, gap_candidates, top_k=self.config.top_k_final,
                        register=self.register_mgr.entries,
                    )
                    old_entry = self.register_mgr.get(gap_req.id)
                    new_entry = self.register_mgr.add(
                        gap_req, gap_ranked, ordered_reqs,
                        was_gap_fill=True,
                        llm_client=self.llm_client,
                    )

                    # Update unit pool + candidate record
                    all_candidates[gap_req.id] = gap_ranked
                    for c in gap_ranked:
                        unit_pool[c.unit.unit_id] = c.unit

                    # Count as filled if we improved on the old entry
                    old_conf = old_entry.confidence if old_entry else 0.0
                    if new_entry.is_supported and new_entry.confidence > old_conf:
                        gap_fill_stats["gaps_filled"] += 1
                        logger.debug(
                            "[%s] Gap filled: %s  binding=%r  conf=%.2f → %.2f",
                            question_id, gap_req.id,
                            new_entry.binding, old_conf, new_entry.confidence,
                        )

        if getattr(self.config, "backtrack_enabled", False):
            self._backtrack(ordered_reqs, all_candidates, unit_pool)

        if getattr(self.config, "replan_enabled", False):
            total_llm_calls += self._replan_missing(question, ordered_reqs, all_candidates, unit_pool)

        if gap_fill_stats["iterations"] > 0:
            logger.debug(
                "[%s] Gap fill: %d iterations, %d gaps filled",
                question_id, gap_fill_stats["iterations"], gap_fill_stats["gaps_filled"],
            )

        # ── Step 6: proof certificate ─────────────────────────────────────────
        proof = build_proof_certificate(
            self.register_mgr.entries, unit_pool, all_candidates=all_candidates
        )

        # ── Step 7: order register entries for generator ──────────────────────
        ordered_pairs = self.register_mgr.ordered_entries(ordered_reqs, ordered_reqs)

        # ── Step 8–9: generate + verify ───────────────────────────────────────
        # Evidence for yes/no and description answers, which answer from all
        # findings together: each step's binding passages, then the top
        # reranked SUPPORT passages of every step, interleaved by rank.
        extra_units = [
            unit_pool[uid]
            for e in self.register_mgr.entries.values()
            for uid in e.supporting_unit_ids
            if uid in unit_pool
        ]
        per_req = [
            [c.unit for c in all_candidates.get(r.id, []) if c.verification == "SUPPORT"][:4]
            for r in ordered_reqs
        ]
        for i in range(4):
            extra_units.extend(lst[i] for lst in per_req if i < len(lst))
        missing_units = []
        if getattr(self.config, "missing_step_evidence_enabled", False):
            k = self.config.missing_step_evidence_k
            for r in ordered_reqs:
                e = self.register_mgr.get(r.id)
                if e is None or not e.is_supported:
                    missing_units.extend(
                        [c.unit for c in all_candidates.get(r.id, []) if c.verification == "SUPPORT"][:k])
        answer, calls = generate_answer_stage2(
            question,
            ordered_reqs,
            self.register_mgr.entries,
            proof,
            self.config,
            ordered_pairs=ordered_pairs,
            verifier=self.verifier,
            extra_units=extra_units,
            missing_units=missing_units,
        )
        total_llm_calls += calls
        from registerrag.pipeline.answer_kind import BOOLEAN, DESCRIPTION, classify
        from registerrag.pipeline.generator import answer_evidence, kind_router_on
        if kind_router_on(self.config) and classify(question) in (BOOLEAN, DESCRIPTION):
            answer_unit_ids = [u.unit_id for u in answer_evidence(proof, extra_units)]
        elif missing_units:
            answer_unit_ids = [u.unit_id for u in answer_evidence(proof, missing_units)]
        else:
            answer_unit_ids = [u.unit_id for u in proof.units]

        latency = time.time() - t0

        # ── Step 10: evaluate ─────────────────────────────────────────────────
        from registerrag.evaluation.metrics import exact_match, token_f1, doc_recall_score
        em = exact_match(answer, gold_answer)
        f1 = token_f1(answer, gold_answer)
        recall = doc_recall_score(unit_pool, supporting_facts, self.index)

        from registerrag.evaluation.failure import classify_failure
        failure_type = classify_failure(
            em, recall, gold_answer, [u.text for u in proof.units]
        )

        # ── Step 10b: write MD trace ──────────────────────────────────────────
        from registerrag.tracing.trace import write_trace
        trace_path = write_trace(
            runs_dir=self.runs_dir,
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
            predicted_answer=answer,
            requirements=ordered_reqs,
            all_candidates=all_candidates,
            register=self.register_mgr.entries,
            proof=proof,
            em=em,
            f1=f1,
            doc_recall=recall,
            failure_type=failure_type,
            latency_s=latency,
            llm_calls=total_llm_calls,
        )

        supported_reqs = sum(
            1 for e in self.register_mgr.entries.values() if e.is_supported
        )

        result = QuestionResult(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
            predicted_answer=answer,
            em=em,
            f1=f1,
            doc_recall=recall,
            failure_type=failure_type,
            latency_s=latency,
            llm_calls=total_llm_calls,
            requirements_count=len(requirements),
            supported_requirements=supported_reqs,
            proof_units_count=len(proof.units),
            trace_path=trace_path,
            proof_unit_ids=[u.unit_id for u in proof.units],
            answer_unit_ids=answer_unit_ids,
        )
        usage1 = usage_snapshot()
        d = {k: usage1.get(k, 0) - usage0.get(k, 0) for k in ("requests", "prompt_tokens", "completion_tokens")}
        result.llm_requests = d["requests"]
        result.prompt_tokens = d["prompt_tokens"]
        result.completion_tokens = d["completion_tokens"]
        result.total_tokens = d["prompt_tokens"] + d["completion_tokens"]

        logger.info(
            "[%s] EM=%.0f F1=%.2f recall=%.2f latency=%.1fs llm=%d %s"
            "  gap_iters=%d filled=%d",
            question_id, em, f1, recall, latency, total_llm_calls, failure_type,
            gap_fill_stats["iterations"], gap_fill_stats["gaps_filled"],
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Private
    # ─────────────────────────────────────────────────────────────────────────

    def _run_step(self, req, reqs, all_candidates, unit_pool, gap=False):
        candidates = self.retriever.retrieve(req, self.register_mgr.entries)
        ranked = self.verifier.rerank(req, candidates, top_k=self.config.top_k_final,
                                      register=self.register_mgr.entries)
        entry = self.register_mgr.add(req, ranked, reqs, was_gap_fill=gap, llm_client=self.llm_client)
        all_candidates[req.id] = ranked
        for c in ranked:
            unit_pool[c.unit.unit_id] = c.unit
        return entry, ranked

    def _backtrack(self, ordered_reqs, all_candidates, unit_pool) -> None:
        """
        Binding beam by backtracking: a dependent step that is still NOT FOUND
        re-runs once per alternative value of the step it depends on (list items,
        values from that step's next passages). The first alternative under which
        it is supported is kept (and downstream missing steps re-run); otherwise
        the register is restored.
        """
        width = getattr(self.config, "backtrack_width", 3)
        for req in ordered_reqs:
            e = self.register_mgr.get(req.id)
            if (e is not None and e.is_supported) or not req.depends_on:
                continue
            for d in req.depends_on:
                de = self.register_mgr.get(d)
                if not (de and de.is_supported and de.binding):
                    continue
                dreq = next((r for r in ordered_reqs if r.id == d), None)
                if dreq is None:
                    continue
                alts = self.register_mgr.alternative_bindings(
                    dreq, all_candidates.get(d, []), llm_client=self.llm_client,
                    max_calls=getattr(self.config, "backtrack_extract_calls", 3))[:width]
                done = False
                for alt, uid in alts:
                    snap = self.register_mgr.snapshot()
                    saved = all_candidates.get(req.id)
                    self.register_mgr.rebind(d, alt, uid)
                    entry, _ = self._run_step(req, ordered_reqs, all_candidates, unit_pool, gap=True)
                    if entry.is_supported:
                        logger.info("Backtrack: %s %r -> %r made %s found %r",
                                    d, de.binding, alt, req.id, entry.binding)
                        done = True
                        break
                    logger.info("Backtrack: %s -> %r did not help %s", d, alt, req.id)
                    self.register_mgr.restore(snap)
                    if saved is not None:
                        all_candidates[req.id] = saved
                if done:
                    for later in ordered_reqs:
                        le = self.register_mgr.get(later.id)
                        if req.id in later.depends_on and (le is None or not le.is_supported):
                            self._run_step(later, ordered_reqs, all_candidates, unit_pool, gap=True)
                    break

    def _replan_missing(self, question, ordered_reqs, all_candidates, unit_pool) -> int:
        """
        Re-plan on failure: each still-unsupported step whose inputs are known is
        rewritten into 1-2 sub-steps (pipeline/replan.py) and retried; a found
        answer is adopted by the step, and its missing dependants are re-run.
        Returns the LLM calls used (the binding calls are counted elsewhere).
        """
        from registerrag.pipeline.replan import replan_step
        calls, budget = 0, getattr(self.config, "replan_max_per_question", 2)
        for req in ordered_reqs:
            if budget <= 0:
                break
            e = self.register_mgr.get(req.id)
            if e is not None and e.is_supported:
                continue
            if not all((self.register_mgr.get(d) and self.register_mgr.get(d).is_supported)
                       for d in req.depends_on):
                continue
            budget -= 1
            titles = [c.unit.document_title for c in all_candidates.get(req.id, [])][:8]
            subs, n = replan_step(question, req, self.register_mgr.entries, titles, self.config)
            calls += n
            if not subs:
                continue
            reqs = list(ordered_reqs) + subs
            last = None
            for s in subs:
                last, ranked = self._run_step(s, reqs, all_candidates, unit_pool, gap=True)
                if not last.is_supported:
                    break
            found = last is not None and last.is_supported
            if found:
                self.register_mgr.adopt(subs[-1].id, req.id)
                all_candidates[req.id] = all_candidates[subs[-1].id]
                logger.info("Replan %s found %r", req.id, last.binding)
            for s in subs:
                self.register_mgr.drop(s.id)
                all_candidates.pop(s.id, None)
            if found:
                # dependants that were missing get one more pass with the new binding
                for dep in ordered_reqs:
                    de = self.register_mgr.get(dep.id)
                    if req.id in dep.depends_on and (de is None or not de.is_supported):
                        self._run_step(dep, ordered_reqs, all_candidates, unit_pool, gap=True)
        return calls

    def _build_bridge_query(
        self,
        gap_req: Requirement,
        all_requirements: list[Requirement],
    ) -> str:
        """
        Build a targeted bridge query for a gap requirement.

        Strategy: substitute resolved dep bindings directly into the
        information need as noun phrases, then append the binding as an
        anchor term. This gives BM25 a concrete entity to match against.

        Example:
          information_need = "birthplace of the director of Inception"
          R1 binding       = "Christopher Nolan"
          bridge query     = "Christopher Nolan birthplace born nationality country"
        """
        base = gap_req.information_need
        resolved_bindings = []

        for dep_id in gap_req.depends_on:
            dep_entry = self.register_mgr.get(dep_id)
            if dep_entry and dep_entry.is_supported and dep_entry.binding:
                resolved_bindings.append(dep_entry.binding)

        if resolved_bindings:
            # Put bindings first so BM25 anchors on the entity
            binding_str = " ".join(resolved_bindings)
            return f"{binding_str} {base}"

        return base
