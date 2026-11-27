"""
Stage 1 orchestrator — single-pass RegisterRAG.

Pipeline per question
─────────────────────
1. compile_requirements  (LLM call #1)
2. topological_sort requirements
3. For each requirement in order:
   a. retrieve candidates (hybrid + 1-hop graph)
   b. rerank with cross-encoder (not an LLM call)
   c. write to register (spaCy binding extraction)
4. build proof certificate (greedy set cover)
5. generate answer          (LLM call #2)
6. log MD trace + metrics

Total LLM calls: exactly 2 per question.
No gap generation (Stage 2 feature).
No eviction (Stage 5 feature) — budget is tracked and warned only.
"""
from __future__ import annotations
import logging
import time
from pathlib import Path

from registerrag.config import Config
from registerrag.corpus.graph import EvidenceGraph
from registerrag.corpus.indexer import HybridIndex
from registerrag.models import (
    EvidenceUnit, ProofCertificate, QuestionResult,
    RegisterEntry, Requirement, RetrievedCandidate,
)
from registerrag.pipeline.compiler import compile_requirements, topological_order
from registerrag.pipeline.generator import generate_answer
from registerrag.pipeline.proof import build_proof_certificate
from registerrag.pipeline.register import RegisterManager
from registerrag.pipeline.retriever import Retriever
from registerrag.pipeline.verifier import CrossEncoderVerifier

logger = logging.getLogger(__name__)


class Stage1Pipeline:
    """
    Instantiate once and call run_question() for each question.

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
    ) -> None:
        self.index = index
        self.graph = graph
        self.verifier = verifier
        self.config = config
        self.runs_dir = runs_dir
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
        supporting_facts: list,          # HotpotQA format: [["title", sent_idx], …]
    ) -> QuestionResult:
        """
        Run the full Stage 1 pipeline for one question.
        Returns a QuestionResult with all metrics.
        """
        t0 = time.time()
        total_llm_calls = 0
        self.register_mgr.reset()
        self.register_mgr.set_question(question)

        # ── Step 1: compile requirements ──────────────────────────────────
        requirements, calls = compile_requirements(question, self.config)
        total_llm_calls += calls
        ordered_reqs = topological_order(requirements)

        # ── Step 2–3: per-requirement retrieve → verify → register ────────
        all_candidates: dict[str, list[RetrievedCandidate]] = {}
        unit_pool: dict[str, EvidenceUnit] = {}

        for req in ordered_reqs:
            candidates = self.retriever.retrieve(req, self.register_mgr.entries)
            ranked = self.verifier.rerank(
                req, candidates, top_k=self.config.top_k_final,
                register=self.register_mgr.entries,
            )
            self.register_mgr.add(req, ranked, ordered_reqs)

            all_candidates[req.id] = ranked
            for c in ranked:
                unit_pool[c.unit.unit_id] = c.unit

        # ── Step 4: proof certificate ─────────────────────────────────────
        proof = build_proof_certificate(
            self.register_mgr.entries, unit_pool, all_candidates=all_candidates
        )

        # ── Step 5: generate answer ───────────────────────────────────────
        answer, calls = generate_answer(
            question, ordered_reqs, self.register_mgr.entries, proof, self.config
        )
        total_llm_calls += calls

        latency = time.time() - t0

        # ── Step 6: evaluate ──────────────────────────────────────────────
        from registerrag.evaluation.metrics import exact_match, token_f1, doc_recall_score
        em = exact_match(answer, gold_answer)
        f1 = token_f1(answer, gold_answer)
        recall = doc_recall_score(unit_pool, supporting_facts, self.index)

        from registerrag.evaluation.failure import classify_failure
        failure_type = classify_failure(
            em, recall, gold_answer, [u.text for u in proof.units]
        )

        # ── Step 7: write MD trace ────────────────────────────────────────
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
        )

        logger.info(
            "[%s] EM=%.0f F1=%.2f recall=%.2f latency=%.1fs llm=%d %s",
            question_id, em, f1, recall, latency, total_llm_calls, failure_type,
        )
        return result
