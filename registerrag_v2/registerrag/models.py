"""
Core data models for RegisterRAG.
All dataclasses — immutable-friendly, easy to serialise, no hidden logic.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Corpus layer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvidenceUnit:
    """
    Atomic piece of evidence.  One sentence from one document.
    Preserves full provenance so the trace can cite it exactly.
    """
    unit_id: str                    # e.g. "doc7_sent3"
    unit_type: str                  # "source_sentence"
    text: str                       # exact sentence text
    document_id: str
    document_title: str
    section_path: list[str]         # ["History", "Early life"] — empty if flat
    sentence_idx: int               # position in doc (0-based)
    char_start: int
    char_end: int
    entity_mentions: list[str] = field(default_factory=list)
    embedding: Optional[list[float]] = None  # not stored in JSON traces

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "embedding"}

    @property
    def text_truncated(self) -> str:
        return self.text[:120] + "…" if len(self.text) > 120 else self.text


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline layer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Requirement:
    """
    One information need in the decomposition.
    Uses a flat list with depends_on annotations (DAG, not tree)
    to handle diamond dependencies cleanly.
    """
    id: str                         # "R1", "R2", …
    information_need: str           # natural-language sub-question
    depends_on: list[str]           # e.g. ["R1"] — empty for root requirements
    expected_answer_type: str       # "person" | "place" | "date" | "number" | "fact" | …
    criticality: str = "required"   # "required" | "optional"

    @property
    def is_root(self) -> bool:
        return len(self.depends_on) == 0


@dataclass
class RetrievedCandidate:
    """One retrieved candidate before / after verification."""
    unit: EvidenceUnit
    hybrid_score: float
    rerank_score: float = 0.0
    verification: str = "NEUTRAL"   # "SUPPORT" | "NEUTRAL"
    rank: int = 0


@dataclass
class RegisterEntry:
    """
    One slot in the evidence register.
    Binding = compact answer (a few words, not full sentence).
    """
    requirement_id: str
    binding: str                    # e.g. "Jonas Salk"
    status: str                     # "supported" | "missing" | "conflicting"
    confidence: float
    supporting_unit_ids: list[str]
    conflicting_unit_ids: list[str] = field(default_factory=list)
    token_cost: int = 0
    was_gap_fill: bool = False       # for utility scoring in later stages

    @property
    def is_supported(self) -> bool:
        return self.status == "supported"

    @property
    def is_missing(self) -> bool:
        return self.status == "missing"


@dataclass
class ProofCertificate:
    """
    Minimum sufficient set of evidence units (greedy set cover).
    The LLM final-answer call sees ONLY this, not all retrieved evidence.
    """
    units: list[EvidenceUnit]
    covered_requirements: list[str]  # requirement IDs fully covered
    dropped_unit_ids: list[str]      # redundant units removed by minimiser

    @property
    def total_tokens(self) -> int:
        return sum(len(u.text.split()) for u in self.units)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation / logging
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    """Full record for one question — written to results.jsonl."""
    question_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    em: float                   # 0 or 1
    f1: float                   # token-level F1
    doc_recall: float           # |retrieved ∩ gold| / |gold|
    failure_type: str           # "none" | "retrieval" | "evidence_selection" | "partial_retrieval" | "generation"
    latency_s: float
    llm_calls: int
    requirements_count: int
    supported_requirements: int
    proof_units_count: int
    trace_path: str             # path to per-question .md trace
    proof_unit_ids: list = field(default_factory=list)   # for post-hoc evidence scoring
    answer_unit_ids: list = field(default_factory=list)  # evidence shown to the answer generator
    # measured at the LLM client (all requests of this question, incl. binding extraction)
    llm_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class RunSummary:
    """Aggregate statistics for a full evaluation run."""
    total_questions: int
    em: float
    f1: float
    doc_recall: float
    retrieval_failures: int
    generation_failures: int
    partial_retrieval: int
    mean_latency_s: float
    p95_latency_s: float
    mean_llm_calls: float
    mean_requirements: float
    mean_proof_units: float
