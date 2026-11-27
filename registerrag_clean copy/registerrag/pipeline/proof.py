"""
Proof certificate minimiser — greedy set cover.

Given all SUPPORT evidence units collected across all requirements,
find the MINIMUM SUFFICIENT SUBSET that covers all supported requirements.

Algorithm
---------
1. For each supported requirement, record which units cover it
   (i.e. the unit is in its RegisterEntry.supporting_unit_ids)
2. Greedy: add the unit that covers the most uncovered requirements
3. Repeat until all requirements are covered (or no more units)
4. Record dropped unit IDs for the trace

The LLM final-answer call receives ONLY the proof certificate units,
not all retrieved evidence.
"""
from __future__ import annotations
import logging
import re
from collections import defaultdict
from typing import Optional

from registerrag.models import EvidenceUnit, ProofCertificate, RegisterEntry, RetrievedCandidate

logger = logging.getLogger(__name__)

_NON_ALNUM = re.compile(r"[^a-z0-9]")


def _tok(s: str) -> set[str]:
    return set(_NON_ALNUM.sub(" ", s.lower()).split())


def binding_in_text(binding: str, text: str) -> bool:
    """
    Fuzzy check that `binding` is present in `text`: case-insensitive substring,
    or >= 80% of the binding's alphanumeric tokens appear in the text. Handles
    LLM paraphrases ("Louis de Luxembourg" vs "Louis de Luxembourg, Count of
    Saint-Pol"). Bindings shorter than 3 characters never match.
    """
    if not binding or len(binding) < 3:
        return False
    if binding.lower() in text.lower():
        return True
    b, t = _tok(binding), _tok(text)
    if not b:
        return False
    return len(b & t) / len(b) >= 0.80


def _binding_swap(
    final_ids: list[str],
    supported_entries: dict[str, RegisterEntry],
    unit_to_reqs: dict[str, set[str]],
    unit_pool: dict[str, EvidenceUnit],
    all_candidates: dict[str, list[RetrievedCandidate]],
) -> list[str]:
    """
    For each requirement whose selected proof unit(s) don't contain its binding
    (case-insensitive substring), bring in that requirement's highest-scored
    candidate that does. If the old unit covered only this requirement it is
    replaced; if it also covers other requirements it is kept and the new unit
    is added alongside it. Requirements with no binding-containing candidate
    are left unchanged.

    Mutates `unit_to_reqs` / `unit_pool` so coverage bookkeeping stays correct.
    """
    result = list(final_ids)
    for rid, entry in supported_entries.items():
        binding = (entry.binding or "").strip().lower()
        if not binding:
            continue
        covering = [uid for uid in result if rid in unit_to_reqs.get(uid, set())]
        if any(binding_in_text(binding, unit_pool[uid].text) for uid in covering if uid in unit_pool):
            logger.info("Proof swap skipped: req=%s proof already contains binding", rid)
            continue

        cands = sorted(
            all_candidates.get(rid, []), key=lambda c: c.rerank_score, reverse=True
        )
        best = next((c for c in cands if binding_in_text(binding, c.unit.text)), None)
        old_text = " | ".join(unit_pool[u].text for u in covering if u in unit_pool) or "(none)"
        logger.info(
            "Proof swap: req=%s binding='%s' swap_found=%s old_unit='%s' new_unit='%s'",
            rid, entry.binding, best is not None, old_text[:50],
            best.unit.text[:50] if best is not None else "N/A",
        )
        if best is None:
            continue

        new_uid = best.unit.unit_id
        unit_pool.setdefault(new_uid, best.unit)
        unit_to_reqs[new_uid].add(rid)

        for uid in covering:
            unit_to_reqs[uid].discard(rid)
            if not unit_to_reqs[uid] and uid in result:
                result.remove(uid)   # covered only this requirement → replaced
        if new_uid not in result:
            result.append(new_uid)
    return result


def build_proof_certificate(
    register: dict[str, RegisterEntry],
    unit_pool: dict[str, EvidenceUnit],
    all_candidates: Optional[dict[str, list[RetrievedCandidate]]] = None,
) -> ProofCertificate:
    """
    Parameters
    ----------
    register       : {req_id → RegisterEntry}  current register state
    unit_pool      : {unit_id → EvidenceUnit}  all units seen during retrieval
    all_candidates : {req_id → reranked candidates}  enables the binding swap pass

    Returns
    -------
    ProofCertificate with minimal sufficient evidence set
    """
    # Only consider supported requirements
    supported_entries = {
        rid: e for rid, e in register.items() if e.is_supported
    }

    if not supported_entries:
        logger.debug("No supported requirements — empty proof certificate")
        return ProofCertificate(units=[], covered_requirements=[], dropped_unit_ids=[])

    # Map: unit_id → set of requirement IDs it supports
    unit_to_reqs: dict[str, set[str]] = defaultdict(set)
    for rid, entry in supported_entries.items():
        for uid in entry.supporting_unit_ids:
            if uid in unit_pool:
                unit_to_reqs[uid].add(rid)

    if not unit_to_reqs:
        logger.debug("No units in pool for supported requirements")
        return ProofCertificate(units=[], covered_requirements=[], dropped_unit_ids=[])

    # Greedy set cover
    all_req_ids = set(supported_entries.keys())
    uncovered = set(all_req_ids)
    selected_ids: list[str] = []

    candidate_pool = set(unit_to_reqs.keys())

    while uncovered and candidate_pool:
        # Pick unit that covers the most uncovered requirements
        best_uid = max(
            candidate_pool,
            key=lambda uid: len(unit_to_reqs[uid] & uncovered),
        )
        covered_by_best = unit_to_reqs[best_uid] & uncovered
        if not covered_by_best:
            break  # remaining uncovered reqs have no evidence
        selected_ids.append(best_uid)
        uncovered -= covered_by_best
        candidate_pool.discard(best_uid)

    # Try backward removal: remove any selected unit whose removal still keeps full coverage
    final_ids = _backward_prune(selected_ids, unit_to_reqs, all_req_ids)

    # Swap pass: make sure each requirement's proof contains its binding
    if all_candidates:
        final_ids = _binding_swap(
            final_ids, supported_entries, unit_to_reqs, unit_pool, all_candidates
        )

    # Build output
    all_candidate_ids = set(unit_to_reqs.keys())
    dropped_ids = list(all_candidate_ids - set(final_ids))

    proof_units = [unit_pool[uid] for uid in final_ids if uid in unit_pool]
    covered_reqs = list(
        {rid for uid in final_ids for rid in unit_to_reqs.get(uid, set())}
    )

    logger.debug(
        "Proof certificate: %d units cover %d/%d requirements (dropped %d)",
        len(proof_units), len(covered_reqs), len(all_req_ids), len(dropped_ids),
    )
    return ProofCertificate(
        units=proof_units,
        covered_requirements=covered_reqs,
        dropped_unit_ids=dropped_ids,
    )


def _backward_prune(
    selected_ids: list[str],
    unit_to_reqs: dict[str, set[str]],
    all_req_ids: set[str],
) -> list[str]:
    """Remove any unit that is redundant given the others."""
    result = list(selected_ids)
    for uid in list(result):
        without = [x for x in result if x != uid]
        covered_without = {rid for x in without for rid in unit_to_reqs.get(x, set())}
        if all_req_ids.issubset(covered_without):
            result = without   # uid is redundant — drop it
    return result
