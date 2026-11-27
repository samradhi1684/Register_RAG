"""
Evaluation metrics.

EM  : exact match after normalisation (lowercase, strip articles + punct)
F1  : token-level F1 (SQuAD-style)
Doc recall : |retrieved_docs ∩ gold_docs| / |gold_docs|

All metrics are computed over ALL questions including:
  - questions where generation failed
  - questions where retrieval failed
  - no abstain → every question gets a score (zeros count)
"""
from __future__ import annotations
import re
import string
from collections import Counter
from typing import Optional

from registerrag.corpus.indexer import HybridIndex
from registerrag.models import EvidenceUnit


# ─────────────────────────────────────────────────────────────────────────────
# Text normalisation (SQuAD-style)
# ─────────────────────────────────────────────────────────────────────────────

ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, strip articles, collapse whitespace."""
    text = text.lower()
    text = text.translate(PUNCT_TABLE)
    text = ARTICLES_RE.sub(" ", text)
    text = " ".join(text.split())
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Per-question metrics
# ─────────────────────────────────────────────────────────────────────────────

def exact_match(predicted: str, gold: str) -> float:
    """Return 1.0 if predictions match after normalisation, else 0.0."""
    return 1.0 if normalize_answer(predicted) == normalize_answer(gold) else 0.0


def token_f1(predicted: str, gold: str) -> float:
    """
    Token-level F1 (SQuAD 1.1 formula).
    Operates on normalised token bags.
    """
    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall    = num_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def doc_recall_score(
    unit_pool: dict[str, EvidenceUnit],
    supporting_facts: list,
    index: HybridIndex,
) -> float:
    """
    HotpotQA doc recall.

    supporting_facts : [["title", sent_idx], …]
    unit_pool        : all evidence units retrieved during this question
    index            : to look up unit_id by (title, sent_idx)

    Returns |retrieved ∩ gold| / |gold|
    If supporting_facts is empty, returns 1.0 (vacuously satisfied).
    """
    if not supporting_facts:
        return 1.0

    # Document-level gold: [{"doc_ids": [...]}, …] — each entry is one gold
    # source that may live in any of several documents (e.g. a program's meta
    # file, its content files, or a law shared with other programs).
    if isinstance(supporting_facts[0], dict):
        got = {u.document_id for u in unit_pool.values()}
        hits = sum(1 for sf in supporting_facts if got & set(sf.get("doc_ids", [])))
        return round(hits / len(supporting_facts), 4)

    # Build a lookup: (doc_title.lower(), sent_idx) → unit_id
    title_sent_to_uid: dict[tuple, str] = {}
    for uid, unit in {u.unit_id: u for u in index.units}.items():
        key = (unit.document_title.lower(), unit.sentence_idx)
        title_sent_to_uid[key] = uid

    gold_unit_ids: set[str] = set()
    for title, sent_idx in supporting_facts:
        key = (title.lower(), sent_idx)
        uid = title_sent_to_uid.get(key)
        if uid:
            gold_unit_ids.add(uid)

    if not gold_unit_ids:
        return 0.0   # gold evidence not found in corpus at all → retrieval failure

    retrieved_ids = set(unit_pool.keys())
    intersection = gold_unit_ids & retrieved_ids
    return round(len(intersection) / len(gold_unit_ids), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate over a run
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(results: list[dict]) -> dict:
    """
    Compute aggregate stats from a list of QuestionResult.to_dict() records.
    Includes ALL questions — no abstain exclusion.
    """
    if not results:
        return {}

    n = len(results)
    em_scores = [r["em"] for r in results]
    f1_scores = [r["f1"] for r in results]
    recall_scores = [r["doc_recall"] for r in results]
    latencies = sorted(r["latency_s"] for r in results)
    llm_calls = [r["llm_calls"] for r in results]
    req_counts = [r["requirements_count"] for r in results]
    proof_counts = [r["proof_units_count"] for r in results]

    failure_counts = Counter(r["failure_type"] for r in results)

    p95_idx = int(0.95 * n)

    return {
        "total_questions": n,
        "em": round(sum(em_scores) / n, 4),
        "f1": round(sum(f1_scores) / n, 4),
        "doc_recall": round(sum(recall_scores) / n, 4),
        "correct": int(sum(em_scores)),
        "retrieval_failures": failure_counts.get("retrieval", 0),
        "generation_failures": failure_counts.get("generation", 0),
        "partial_retrieval": failure_counts.get("partial_retrieval", 0),
        "evidence_selection_failures": failure_counts.get("evidence_selection", 0),
        "mean_latency_s": round(sum(latencies) / n, 2),
        "p95_latency_s": round(latencies[p95_idx], 2),
        "mean_llm_calls": round(sum(llm_calls) / n, 2),
        "mean_requirements": round(sum(req_counts) / n, 2),
        "mean_proof_units": round(sum(proof_counts) / n, 2),
    }
