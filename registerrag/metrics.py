from __future__ import annotations

import re
import string
from collections import Counter


# ------------------------------------------------------------------ answer quality
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def _f1(p: str, g: str) -> float:
    pt, gt = normalize_answer(p).split(), normalize_answer(g).split()
    common = Counter(pt) & Counter(gt)
    n = sum(common.values())
    if not pt or not gt or n == 0:
        return float(pt == gt)
    pr, rc = n / len(pt), n / len(gt)
    return 2 * pr * rc / (pr + rc)


def f1_score(pred: str, golds) -> float:
    return max(_f1(pred, g) for g in golds)


# ------------------------------------------------------------------ retrieval
def covered(units) -> set:
    out = set()
    for u in units:
        out |= u.covered_ids()
    return out


def requirement_recall(ranked_units, req, k: int = 10):
    """Recall@k of one requirement's gold evidence. Sentence-level if available, else document-level."""
    top = ranked_units[:k]
    if req.gold_unit_ids:
        g = set(req.gold_unit_ids)
        return len(g & covered(top)) / len(g), "sentence"
    if req.gold_doc_ids:
        g = set(req.gold_doc_ids)
        return len(g & {u.doc_id for u in top}) / len(g), "document"
    return None, None


def gold_complete(units, ex) -> bool:
    if ex.gold_unit_ids:
        return set(ex.gold_unit_ids) <= covered(units)
    return set(ex.gold_doc_ids) <= {u.doc_id for u in units}


def retrieval_report(ranking, selected, ex, ks=(5, 10, 20)) -> dict:
    gd = set(ex.gold_doc_ids)
    gu = set(ex.gold_unit_ids) if ex.gold_unit_ids else None
    out = {}
    for k in ks:
        top = ranking[:k]
        out[f"doc_recall@{k}"] = len(gd & {u.doc_id for u in top}) / len(gd) if gd else None
        if gu:
            out[f"sent_recall@{k}"] = len(gu & covered(top)) / len(gu)
            out[f"unit_precision@{k}"] = sum(1 for u in top if u.covered_ids() & gu) / max(len(top), 1)
    out["ctx_doc_recall"] = len(gd & {u.doc_id for u in selected}) / len(gd) if gd else None
    if gu:
        out["ctx_sent_recall"] = len(gu & covered(selected)) / len(gu)
        out["ctx_unit_precision"] = sum(1 for u in selected if u.covered_ids() & gu) / max(len(selected), 1)
    out["ctx_all_gold"] = gold_complete(selected, ex)
    hops = []
    for i, r in enumerate(ex.gold_requirements, 1):
        rec, level = requirement_recall(ranking, r, 10)
        hops.append({"rid": r.rid, "hop": i, "recall@10": rec, "level": level})
    out["per_hop"] = hops
    return out


def classify_failure(em: float, error, ex, selected, pool, stage1, truncated, reranker_used) -> str:
    """Automatic first-pass label for baselines. Taxonomy of the roadmap plus two baseline-only labels:
    CONTEXT_OVERFLOW (gold retrieved but dropped by the token budget) and SYSTEM_ERROR. Manual review
    should confirm; the label is flagged as auto in the log."""
    if error:
        return "SYSTEM_ERROR"
    if em >= 1.0:
        return ""
    if gold_complete(selected, ex):
        return "GENERATION_FAILURE"
    if truncated and gold_complete(pool, ex):
        return "CONTEXT_OVERFLOW"
    if gold_complete(stage1, ex) and reranker_used:
        return "RERANKING_FAILURE"
    if gold_complete(stage1, ex):
        return "RETRIEVAL_FAILURE"
    return "RETRIEVAL_FAILURE"
