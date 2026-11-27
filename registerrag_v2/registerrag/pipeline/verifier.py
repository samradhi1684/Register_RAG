"""
Cross-encoder verification layer.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to rerank candidates.
NOT an LLM call — fast, cheap, batched.

The verifier:
  1. Takes (requirement_text, evidence_text) pairs
  2. Returns a rerank score per pair
  3. Top-k by score → labelled SUPPORT; rest → NEUTRAL

This is NOT a 3-class classifier (SUPPORT/NEUTRAL/REFUTE).
REFUTE detection is for Stage 4 conflict resolution.
Stage 1: SUPPORT vs NEUTRAL only.
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np

from registerrag.config import Config
from registerrag.models import RegisterEntry, RetrievedCandidate, Requirement
from registerrag.pipeline.proof import binding_in_text

logger = logging.getLogger(__name__)

# Added to the ms-marco logit of candidates containing the requirement's binding
BINDING_BONUS = 0.30


def unit_context(unit) -> str:
    """
    Where a unit sits: "Title > Section > Subsection" (title omitted when the
    text already names it). Wikipedia sentences have no sections, so they get
    just the title, as before.
    """
    title = (getattr(unit, "document_title", "") or "").strip()
    parts = [] if (not title or title.lower() in unit.text.lower()) else [title]
    sec = [s for s in (getattr(unit, "section_path", None) or []) if s and s.lower() != title.lower()]
    return " > ".join(parts + sec[-2:])          # the two innermost headings are enough


def _with_title(unit) -> str:
    """Passage text as the cross-encoder sees it: '<context>: text' (see unit_context)."""
    ctx = unit_context(unit)
    return f"{ctx}: {unit.text}" if ctx else unit.text


class CrossEncoderVerifier:
    """
    Wraps a HuggingFace cross-encoder for relevance scoring.

    Parameters
    ----------
    model_name : e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device : "cpu" | "cuda"
    batch_size : how many pairs to score in one forward pass
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("pip install sentence-transformers")
        logger.info("Loading cross-encoder: %s on %s", model_name, device)
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size

    def rerank(
        self,
        requirement: Requirement,
        candidates: list[RetrievedCandidate],
        top_k: int = 10,
        register: Optional[dict[str, RegisterEntry]] = None,
    ) -> list[RetrievedCandidate]:
        """
        Rerank `candidates` for `requirement`.

        Returns the top_k candidates sorted by rerank_score descending.
        Candidates in the top_k are labelled SUPPORT; rest NEUTRAL.

        Binding-presence bonus: if `register` already holds a binding for this
        requirement (e.g. on a Stage-2 gap-fill re-rank), candidates whose text
        contains that binding (case-insensitive substring) get +BINDING_BONUS.
        Root / first-pass requirements have no binding yet, so no bonus.
        """
        if not candidates:
            return []

        # Resolve [R1]-style references so the cross-encoder knows which entity
        # the step is about ("Where was [R1] born?" → "Where was Bob Seger born?"),
        # and prefix each passage with its article title so subject-less
        # sentences ("He was born in London.") carry their subject.
        query = requirement.information_need
        if register:
            from registerrag.pipeline.retriever import Retriever
            query = Retriever._resolve_placeholders(
                query, {rid: e.binding for rid, e in register.items() if e.binding}
            )
        pairs = [(query, _with_title(c.unit)) for c in candidates]

        # Batch inference
        scores = self._score_pairs(pairs)

        for cand, score in zip(candidates, scores):
            cand.rerank_score = float(score)

        entry = register.get(requirement.id) if register else None
        binding = (entry.binding or "").strip().lower() if entry else ""
        if binding:
            for cand in candidates:
                if binding_in_text(binding, cand.unit.text):
                    cand.rerank_score += BINDING_BONUS
                    logger.info(
                        "Binding bonus +%.2f fired: req=%s binding='%s' unit_snippet='%s'",
                        BINDING_BONUS, requirement.id, entry.binding, cand.unit.text[:60],
                    )

        # Sort by rerank score
        sorted_cands = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)

        # Label top_k as SUPPORT
        for i, cand in enumerate(sorted_cands):
            cand.verification = "SUPPORT" if i < top_k else "NEUTRAL"
            cand.rank = i + 1

        return sorted_cands

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Run cross-encoder inference in batches."""
        all_scores: list[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            batch_scores = self.model.predict(batch, show_progress_bar=False)
            if hasattr(batch_scores, "tolist"):
                batch_scores = batch_scores.tolist()
            all_scores.extend(batch_scores)
        return all_scores
