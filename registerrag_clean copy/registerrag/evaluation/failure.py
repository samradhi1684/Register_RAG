"""
Failure classification.

Mutually exclusive categories (checked in this order):

  retrieval        : doc_recall == 0.0
                     The system retrieved none of the gold supporting facts.
                     Fix: better retrieval (BM25/dense/graph tuning).

  evidence_selection : doc_recall > 0.5, EM == 0, and no proof-certificate unit
                     contains > 50% of the gold answer's tokens.
                     Evidence was retrieved but the proof handed to the generator
                     lacks the answer. Fix: binding extraction / proof selection.
                     (Skipped for yes/no gold answers, which never appear in text.)

  partial_retrieval : 0 < doc_recall < 1.0 and EM == 0
                     Some evidence found but not all; answer wrong.
                     Fix: better coverage — more hops, better query expansion.

  generation       : doc_recall == 1.0 and EM == 0
                     All evidence was retrieved but answer was wrong.
                     Fix: better binding extraction, proof minimisation, or generation.

  none             : EM == 1.0 (correct answer)
"""
from __future__ import annotations

from typing import Iterable, Optional

from registerrag.evaluation.metrics import normalize_answer

_YES_NO = {"yes", "no"}


def proof_contains_gold(gold_answer: str, proof_texts: Iterable[str]) -> bool:
    """True iff some proof unit text contains > 50% of the gold answer's tokens."""
    gold_tokens = normalize_answer(gold_answer).split()
    if not gold_tokens:
        return True
    for text in proof_texts:
        unit_tokens = set(normalize_answer(text).split())
        hits = sum(1 for t in gold_tokens if t in unit_tokens)
        if hits / len(gold_tokens) > 0.5:
            return True
    return False


def classify_failure(
    em: float,
    doc_recall: float,
    gold_answer: Optional[str] = None,
    proof_texts: Optional[Iterable[str]] = None,
) -> str:
    """
    Parameters
    ----------
    em          : exact match (0 or 1)
    doc_recall  : fraction of gold evidence retrieved [0, 1]
    gold_answer : gold answer string (enables the evidence_selection check)
    proof_texts : texts of the proof-certificate units given to the generator

    Returns
    -------
    "none" | "retrieval" | "evidence_selection" | "partial_retrieval" | "generation"
    """
    if em == 1.0:
        return "none"

    if doc_recall == 0.0:
        return "retrieval"

    if (
        doc_recall > 0.5
        and gold_answer is not None
        and proof_texts is not None
        and normalize_answer(gold_answer) not in _YES_NO
        and not proof_contains_gold(gold_answer, proof_texts)
    ):
        return "evidence_selection"

    if doc_recall < 1.0:
        return "partial_retrieval"

    # doc_recall == 1.0, EM == 0
    return "generation"


def failure_analysis(results: list[dict]) -> dict:
    """
    Detailed breakdown to understand what to fix first.

    Parameters
    ----------
    results : list of QuestionResult.to_dict()

    Returns
    -------
    dict with counts, fractions, and example question IDs per category
    """
    n = len(results)
    if n == 0:
        return {}

    categories = {
        "none": [],
        "retrieval": [],
        "evidence_selection": [],
        "partial_retrieval": [],
        "generation": [],
    }
    for r in results:
        ft = r.get("failure_type", "none")
        categories.setdefault(ft, []).append(r["question_id"])

    analysis: dict = {}
    for cat, qids in categories.items():
        analysis[cat] = {
            "count": len(qids),
            "fraction": round(len(qids) / n, 4),
            "example_ids": qids[:5],
        }

    # Retrieval failure drill-down: how many had partial evidence?
    partial_recalls = [
        r["doc_recall"]
        for r in results
        if r.get("failure_type") == "partial_retrieval"
    ]
    if partial_recalls:
        analysis["partial_retrieval"]["mean_recall"] = round(
            sum(partial_recalls) / len(partial_recalls), 4
        )

    return analysis
