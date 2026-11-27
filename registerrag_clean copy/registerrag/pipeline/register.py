"""
Evidence register manager.

The register is the bounded working memory:
  requirement_id → RegisterEntry

Stage 2 additions vs Stage 1:
  - Eviction: when budget overflows, lowest-utility non-pinned entry is removed
  - ordered_entries(): returns entries sorted by utility desc for generator
  - gaps(): returns requirements that still need filling
  - Dependency pinning fully enforced during eviction

Improvements over Stage 1 baseline:
  - is_valid_binding(): rejects pronouns, stubs, death-context, long phrases
  - add() iterates candidates until a valid binding is found
  - LLM binding extraction: optional (+1 LLM call per req, --llm-binding)
"""
from __future__ import annotations
import json
import os
import logging
import re
from typing import Optional

from registerrag.config import Config
from registerrag.models import EvidenceUnit, RegisterEntry, Requirement, RetrievedCandidate
from registerrag.pipeline.proof import binding_in_text
from registerrag.pipeline.answer_kind import binding_style, classify

# Relation step on a named subject ("Who is the father of X?")
_RELATION_NEED = re.compile(
    r"^\s*who\s+(?:is|was)\s+the\s+(?:father|mother|spouse|wife|husband|sibling|brother|"
    r"sister|child|son|daughter|parent)\s+of\b", re.I)

# Sentence whose grammatical subject is a pronoun (refers to its article's subject)
_PRONOUN_START = re.compile(r"^\W*(he|she|his|her|they|their|it|its|him)\b", re.I)

# JSON-example placeholders the model sometimes copies verbatim as the binding
_JUNK = {"name", "answer", "string", "example", "value", "entity", "text",
         "result", "unknown", "[name]", "[answer]", "your answer here"}


def _validate_binding(binding: str, candidates: list) -> bool:
    """Returns True if binding is found in any retrieved candidate."""
    for c in candidates:
        if binding_in_text(binding, c.unit.text):
            return True
    return False

logger = logging.getLogger(__name__)

# Text between two entity spans that makes them one name:
#   ", "  |  ", Lady Of "  |  ", Count of "
_TITLE_JOIN_RE = re.compile(r"^\s*,\s*(?:[A-Z]\w*\s+(?:[Oo]f|de|the|von)\s+)?$")


def estimate_tokens(text: str) -> int:
    """Fast approximate token count (1 word ≈ 1.3 tokens)."""
    return max(1, int(len(text.split()) * 1.3))


class RegisterManager:
    """
    Manages the evidence register: writes, reads, utility scoring, eviction.

    Parameters
    ----------
    config : Config
    nlp    : loaded spaCy model (for binding extraction)
    """

    _type_check = False   # class-level: is_valid_binding is a classmethod

    def __init__(self, config: Config, nlp) -> None:
        RegisterManager._type_check = bool(getattr(config, "binding_type_check", False))
        self.config = config
        self.nlp = nlp
        self._entries: dict[str, RegisterEntry] = {}
        self._subject: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Public
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def entries(self) -> dict[str, RegisterEntry]:
        return dict(self._entries)

    @property
    def total_tokens(self) -> int:
        return sum(e.token_cost for e in self._entries.values())

    def add(
        self,
        requirement: Requirement,
        candidates: list[RetrievedCandidate],
        all_requirements: list[Requirement],
        was_gap_fill: bool = False,
        llm_client=None,
    ) -> RegisterEntry:
        """
        Create or update the register entry for `requirement`.

        Improvement: iterate support candidates until a valid binding is found
        (is_valid_binding rejects pronouns, stubs, death-context phrases etc.).

        Parameters
        ----------
        llm_client : optional OpenAI-compat client for LLM binding extraction
                     (used only when config.llm_binding_extraction is True)
        """
        support_cands = [c for c in candidates if c.verification == "SUPPORT"]
        style = self._binding_style(requirement)
        max_words = 45 if style == "clause" else 12

        # Bindings already held by this requirement's dependencies: a dependent
        # step must resolve to a *new* entity, not echo the one it builds on.
        dep_bindings = {
            rid: self._entries[rid].binding
            for rid in requirement.depends_on
            if rid in self._entries and self._entries[rid].binding
        }
        # Subject check only applies to dependent steps — for "Who is X?" the
        # subject itself is the correct binding.
        # ...and for relation steps ("Who is the father of X?"), where X is the
        # subject and can't be its own father.
        subject = (
            self._subject
            if requirement.depends_on or _RELATION_NEED.match(requirement.information_need)
            else ""
        )

        # Dep entities named in the LLM binding prompt (and already rejected as
        # bindings by is_valid_binding via current_bindings=dep_bindings)
        dep_entities = [v.strip() for v in dep_bindings.values() if v and len(v.strip()) >= 3]

        # Source-aware ordering: for dependent steps, try candidates from the
        # dep entity's own article first ("Albert II" → article "Albert II of
        # Germany"), so hub sentences from unrelated articles ("…born in
        # London", "…grandfather Gaston IV") can't win just by naming an entity.
        # Stable within each group: rerank-score order is kept.
        if requirement.depends_on:
            dep_names = [
                v.strip().lower() for v in dep_bindings.values()
                if v and len(v.strip()) >= 3
            ]
            if dep_names and self.config.hub_filter_enabled:
                # Hub filter: "He was born in London." from some other person's
                # article can't answer a question about the dep entity — its
                # subject is a pronoun that refers to that article's subject.
                from registerrag.pipeline.retriever import Retriever
                def _foreign_pronoun(c) -> bool:
                    title = c.unit.document_title or ""
                    own = any(Retriever.title_matches(n, title) or n in title.lower()
                              for n in dep_names)
                    return not own and bool(_PRONOUN_START.match(c.unit.text))
                kept = [c for c in support_cands if not _foreign_pronoun(c)]
                if len(kept) < len(support_cands):
                    logger.debug("Hub filter: req=%s dropped %d pronoun sentences",
                                 requirement.id, len(support_cands) - len(kept))
                support_cands = kept
            if dep_names:
                support_cands.sort(
                    key=lambda c: (
                        0 if any(n in (c.unit.document_title or "").lower() for n in dep_names)
                        else 1,
                        -c.rerank_score,
                    )
                )

        binding = ""
        top_cand: Optional[RetrievedCandidate] = None

        # Joint-first: one call over the top SUPPORT passages before trying
        # passages one at a time
        if getattr(self.config, "joint_binding_first", False):
            binding, top_cand = self._try_joint(requirement, support_cands, llm_client, dep_entities,
                                                dep_bindings, subject, max_words)

        # Try support candidates in order until a valid binding is found
        nli_on = getattr(self.config, "nli_verify_enabled", False) and style != "clause"
        nli_fallback = None     # first value the NLI check found neutral (look-ahead mode)
        nli_look = 0
        for cand in ([] if binding else support_cands):
            if nli_fallback is not None:
                if nli_look >= self.config.nli_lookahead:
                    break
                nli_look += 1
            if self.config.llm_binding_extraction and llm_client is not None:
                raw = self._extract_binding_llm(
                    cand.unit, requirement, llm_client, dep_entities=dep_entities,
                    known_bindings={
                        rid: e.binding for rid, e in self._entries.items() if e.binding
                    },
                    candidates=candidates,
                )
            else:
                raw = self._extract_binding(cand.unit, requirement.expected_answer_type)

            if self.is_valid_binding(
                raw, requirement.expected_answer_type,
                current_bindings=dep_bindings, subject=subject, max_words=max_words,
            ):
                if nli_on:
                    label = self._nli_label(cand.unit, requirement, raw)
                    if label == "contradiction":
                        continue
                    if label == "neutral" and self.config.nli_lookahead > 0:
                        if nli_fallback is None:
                            nli_fallback = (raw, cand)
                        continue
                binding = raw
                top_cand = cand
                break
        if not binding and nli_fallback is not None:
            binding, top_cand = nli_fallback

        # Anchor fallback: a dependent step whose role-aware (v2) extraction
        # refused everywhere gets one more pass with the plain v1 prompt, but
        # ONLY over sentences from the dependency entity's own article — the
        # place where v1's answer is trustworthy (no foreign "He was born in…").
        if (not binding and requirement.depends_on and dep_entities
                and self.config.llm_binding_extraction and llm_client is not None
                and self.config.anchor_v1_fallback_enabled):
            from registerrag.pipeline.retriever import Retriever
            names = [n.lower() for n in dep_entities]
            own = [c for c in sorted(candidates, key=lambda c: -c.rerank_score)
                   if any(Retriever.title_matches(n, c.unit.document_title or "")
                          or n in (c.unit.document_title or "").lower() for n in names)]
            for cand in own[:6]:
                raw = self._llm_extract("v1", cand.unit, requirement, llm_client,
                                        dep_entities, None)[1]
                if self.is_valid_binding(raw, requirement.expected_answer_type,
                                         current_bindings=dep_bindings, subject=subject,
                                         max_words=max_words):
                    logger.info("Anchor v1 fallback: req=%s binding='%s' unit=%s",
                                requirement.id, raw, cand.unit.unit_id)
                    binding, top_cand = raw, cand
                    if cand not in support_cands:
                        support_cands.insert(0, cand)
                    break

        # Joint fallback: per-passage extraction found nothing. Read the top
        # SUPPORT passages together in one call — the answer may follow from
        # passages combined or be implied ("A and his brother B …"). Kept only
        # if it is grounded: the phrase must occur in one of those passages.
        if (not binding and getattr(self.config, "joint_binding_fallback_enabled", False)
                and not getattr(self.config, "joint_binding_first", False)):
            binding, top_cand = self._try_joint(requirement, support_cands, llm_client, dep_entities,
                                                dep_bindings, subject, max_words)

        # If no valid binding found, still use top support candidate (mark missing)
        if not binding and support_cands:
            top_cand = support_cands[0]

        if top_cand:
            status = "supported" if binding else "missing"
            # Normalise ms-marco logits → [0.4, 0.98]
            confidence = max(0.40, min(0.98, (top_cand.rerank_score + 10) / 20.0))
            if not binding:
                confidence *= 0.5  # halve confidence when binding failed validation
            unit_ids = [c.unit.unit_id for c in support_cands[:3]]
        else:
            binding = ""
            status = "missing"
            confidence = 0.0
            unit_ids = []

        token_cost = estimate_tokens(binding) + estimate_tokens(requirement.information_need)

        entry = RegisterEntry(
            requirement_id=requirement.id,
            binding=binding,
            status=status,
            confidence=confidence,
            supporting_unit_ids=unit_ids,
            token_cost=token_cost,
            was_gap_fill=was_gap_fill,
        )
        self._entries[requirement.id] = entry

        # ── Budget enforcement ────────────────────────────────────────────────
        if self.total_tokens > self.config.register_token_budget:
            if self.config.register_eviction_enabled:
                self._evict(all_requirements)
            else:
                logger.warning(
                    "Register over budget: %d / %d tokens (eviction disabled)",
                    self.total_tokens,
                    self.config.register_token_budget,
                )

        return entry

    def gaps(
        self,
        all_requirements: list[Requirement],
    ) -> list[Requirement]:
        """
        Return requirements that have no confident binding — i.e. are gaps.

        A requirement is a gap if:
          - its entry is missing/absent, OR
          - its confidence < gap_fill_confidence_threshold
        """
        threshold = self.config.gap_fill_confidence_threshold
        result = []
        for req in all_requirements:
            entry = self._entries.get(req.id)
            if entry is None or entry.is_missing or entry.confidence < threshold:
                result.append(req)
        return result

    def ordered_entries(
        self,
        all_requirements: list[Requirement],
        topo_order: list[Requirement],
    ) -> list[tuple[Requirement, RegisterEntry]]:
        """
        Return (requirement, entry) pairs ordered for the generator context.

        Ordering rule:
        1. Follow topological order (roots first so chain reads naturally)
        2. Within each "tier" (same topo depth), sort by utility descending
        This ensures the generator reads: R1 → R2 → R3 and always sees the
        most reliable entries first within each tier.
        """
        # Build topo-index map for stable ordering
        topo_idx = {req.id: i for i, req in enumerate(topo_order)}

        pairs = []
        for req in topo_order:
            entry = self._entries.get(req.id)
            if entry is not None:
                pairs.append((req, entry))

        # Sort: primarily topo order, secondarily utility desc within same deps
        def sort_key(pair):
            req, entry = pair
            # group by max dependency depth (topo tier)
            tier = topo_idx.get(req.id, 0)
            u = self.utility(entry, all_requirements)
            u_sort = -u if u != float("inf") else float("-inf")  # inf → first
            return (tier, u_sort)

        pairs.sort(key=sort_key)
        return pairs

    def get(self, requirement_id: str) -> Optional[RegisterEntry]:
        return self._entries.get(requirement_id)

    def adopt(self, src_id: str, dst_id: str) -> RegisterEntry:
        """Re-plan: the sub-step `src_id` answered step `dst_id` — move its entry there."""
        src = self._entries[src_id]
        entry = RegisterEntry(
            requirement_id=dst_id, binding=src.binding, status=src.status,
            confidence=src.confidence, supporting_unit_ids=list(src.supporting_unit_ids),
            conflicting_unit_ids=list(src.conflicting_unit_ids), token_cost=src.token_cost,
            was_gap_fill=True,
        )
        self._entries[dst_id] = entry
        return entry

    def drop(self, requirement_id: str) -> None:
        self._entries.pop(requirement_id, None)

    # ── Binding beam / backtracking support ──────────────────────────────────
    def snapshot(self) -> dict:
        return dict(self._entries)

    def restore(self, snap: dict) -> None:
        self._entries = dict(snap)

    def rebind(self, rid: str, binding: str, unit_id: str) -> None:
        """Give step `rid` an alternative binding (its source passage first)."""
        e = self._entries[rid]
        units = [unit_id] + [u for u in e.supporting_unit_ids if u != unit_id]
        self._entries[rid] = RegisterEntry(
            requirement_id=rid, binding=binding, status="supported", confidence=e.confidence,
            supporting_unit_ids=units[:3], conflicting_unit_ids=list(e.conflicting_unit_ids),
            token_cost=estimate_tokens(binding) + max(0, e.token_cost - estimate_tokens(e.binding)),
            was_gap_fill=True,
        )

    _LIST_SPLIT = re.compile(r"\s*(?:,\s*(?:and|or)\s+|,\s*|\s+(?:and|or)\s+|;\s*|/\s*)")

    def alternative_bindings(
        self,
        requirement: Requirement,
        candidates: list[RetrievedCandidate],
        llm_client=None,
        max_calls: int = 3,
    ) -> list[tuple[str, str]]:
        """
        Other plausible values for an already-bound step -> [(binding, unit_id)]:
          1. the items of a list-valued binding ("ABC, CBS, and NBC")
          2. values extracted from the next SUPPORT passages that do not
             contain the current binding (at most `max_calls` LLM calls)
        """
        e = self._entries.get(requirement.id)
        if e is None or not e.binding:
            return []
        cur = e.binding.strip()
        src = e.supporting_unit_ids[0] if e.supporting_unit_ids else ""
        dep_bindings = {rid: self._entries[rid].binding for rid in requirement.depends_on
                        if rid in self._entries and self._entries[rid].binding}
        out: list[tuple[str, str]] = []
        seen = {cur.lower()}

        def _add(b, uid):
            b = b.strip().strip(".")
            if b and b.lower() not in seen and self.is_valid_binding(
                    b, requirement.expected_answer_type, current_bindings=dep_bindings, max_words=12):
                seen.add(b.lower())
                out.append((b, uid))

        parts = [p for p in self._LIST_SPLIT.split(cur) if p and p.strip()]
        if len(parts) >= 2 and all(len(p.split()) <= 5 for p in parts):
            for p in parts:
                _add(p, src)
        if self._binding_style(requirement) == "clause" or not (
                self.config.llm_binding_extraction and llm_client is not None):
            return out
        calls = 0
        known = {rid: x.binding for rid, x in self._entries.items() if x.binding}
        dep_entities = [v.strip() for v in dep_bindings.values() if v and len(v.strip()) >= 3]
        for c in [c for c in candidates if c.verification == "SUPPORT"]:
            if calls >= max_calls:
                break
            if c.unit.unit_id == src or cur.lower() in c.unit.text.lower():
                continue
            calls += 1
            raw = self._extract_binding_llm(c.unit, requirement, llm_client,
                                            dep_entities=dep_entities, known_bindings=known,
                                            candidates=candidates)
            _add(raw, c.unit.unit_id)
        return out

    def is_pinned(self, entry: RegisterEntry, all_requirements: list[Requirement]) -> bool:
        """
        An entry is pinned (U=∞) if it is a known_input for at least one
        ACTIVE UNRESOLVED downstream requirement.
        """
        for req in all_requirements:
            if entry.requirement_id in req.depends_on:
                downstream = self._entries.get(req.id)
                if downstream is None or downstream.is_missing:
                    return True
        return False

    def utility(self, entry: RegisterEntry, all_requirements: list[Requirement]) -> float:
        """
        U(e) = 0.30·C + 0.30·D + 0.20·G + 0.15·V − 0.05·L

        C  confidence
        D  downstream dependency weight (fraction of unresolved downstream reqs)
        G  gap-fill bonus (1 if was_gap_fill, 0 otherwise)
        V  criticality value (1.0=required, 0.4=optional)
        L  length penalty (token_cost / 200)
        """
        if self.is_pinned(entry, all_requirements):
            return float("inf")

        req = next((r for r in all_requirements if r.id == entry.requirement_id), None)

        C = entry.confidence
        dependent_unresolved = [
            r for r in all_requirements
            if entry.requirement_id in r.depends_on
            and (self._entries.get(r.id) is None or self._entries[r.id].is_missing)
        ]
        D = len(dependent_unresolved) / max(len(all_requirements), 1)
        G = 1.0 if entry.was_gap_fill else 0.0
        V = 1.0 if (req and req.criticality == "required") else 0.4
        L = entry.token_cost / 200.0

        return 0.30 * C + 0.30 * D + 0.20 * G + 0.15 * V - 0.05 * L

    def reset(self) -> None:
        self._entries.clear()
        self._subject = ""

    # ─────────────────────────────────────────────────────────────────────────
    # Public — binding validation
    # ─────────────────────────────────────────────────────────────────────────

    def set_question(self, question: str) -> None:
        """Record the question so dependent bindings can be checked against its subject."""
        self._subject = self.extract_subject(question, self.nlp)

    @staticmethod
    def extract_subject(question: str, nlp=None) -> str:
        """
        Return the question's single named subject, or "" if there isn't exactly one.

        Comparison questions ("Who is older, A or B?") name two entities and the
        answer is one of them, so no subject is returned for those.
        """
        spans: list[str] = []
        if nlp is not None:
            try:
                ents = [
                    e for e in nlp(question).ents
                    if e.label_ not in ("DATE", "CARDINAL", "ORDINAL", "QUANTITY",
                                        "MONEY", "PERCENT", "TIME")
                ]
                # Merge entities joined only by ", " — "Catherine of Austria,
                # Lady of Coucy" is one name, not two.
                merged: list[tuple[int, int]] = []
                for e in ents:
                    gap = question[merged[-1][1]:e.start_char] if merged else ""
                    if merged and _TITLE_JOIN_RE.match(gap):
                        merged[-1] = (merged[-1][0], e.end_char)
                    else:
                        merged.append((e.start_char, e.end_char))
                spans = [question[a:b] for a, b in merged]
            except Exception:
                spans = []
        if not spans:
            # Fallback: runs of capitalised words (allowing of/de/the/von joiners
            # and "Name, Title Of Place" forms), skipping the first question word.
            body = re.sub(r"^\s*\w+\s+", " ", question)
            spans = re.findall(
                r"[A-Z][\w'\-\.]*(?:,?\s+(?:(?:of|de|the|von|van|da|di|del|la|le)\s+)?[A-Z][\w'\-\.]*)*",
                body,
            )
            spans = [x for x in spans if len(x) > 3]
        spans = [re.sub(r"'s$", "", x).strip(" ,?") for x in spans]
        uniq = list(dict.fromkeys(x for x in spans if x))
        return uniq[0] if len(uniq) == 1 else ""

    @staticmethod
    def _norm_entity(text: str) -> str:
        return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())

    @classmethod
    def _same_entity(cls, a: str, b: str) -> bool:
        """Equal after normalisation, or one contains the other (≥ 2 words)."""
        na, nb = cls._norm_entity(a), cls._norm_entity(b)
        if not na or not nb:
            return False
        if na == nb:
            return True
        short, long_ = sorted((na, nb), key=len)
        return len(short.split()) >= 2 and f" {short} " in f" {long_} "

    @classmethod
    def is_valid_binding(
        cls,
        binding: str,
        expected_type: str = "",
        current_bindings: Optional[dict[str, str]] = None,
        subject: str = "",
        max_words: int = 12,
    ) -> bool:
        """
        Return True iff `binding` is a plausible, non-garbage answer phrase.

        Rejects:
        - Empty / whitespace-only strings
        - Pure pronouns ("he", "she", "they", "it", "his", …)
        - Very short single characters or punctuation
        - Generic stubs ("unknown", "n/a", "none", "yes", "no")
        - Fragments longer than 12 words (too verbose for a binding)
        - Death-context sentences masquerading as birthplace bindings
        - The question subject (`subject`) — callers pass it for dependent steps
        - Any binding already held by a dependency (`current_bindings`)
        """
        if not binding or not binding.strip():
            return False

        b = binding.strip().lower()

        _PRONOUNS = {
            "he", "she", "they", "it", "him", "her", "them",
            "his", "hers", "their", "its", "i", "we", "us",
            "this", "that", "these", "those",
            "who", "which", "what",
        }
        if b in _PRONOUNS:
            return False
        # Possessive-pronoun phrases ("his father", "her husband")
        if re.match(r"^(his|her|their|its)\s", b):
            return False

        _STUBS = {"unknown", "n/a", "na", "none", "yes", "no", "not known", "unclear"}
        if b in _STUBS:
            return False

        # Reject single common stop-words / determiners
        _STOP_WORDS = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "of"}
        if b in _STOP_WORDS:
            return False

        # Truncated garbage: "1965 …"
        if re.fullmatch(r"\d{4}\s*…", b):
            return False

        # Too short
        if len(b) < 2:
            return False

        # Too long — likely a full sentence, not a binding (clause-style
        # bindings for yes/no and description steps may be longer)
        if len(binding.split()) > max_words:
            return False

        # Death-context sentence used as birthplace
        if expected_type.lower() in ("place_of_birth", "birthplace", "country_of_birth"):
            _DEATH_SIGNALS = ("died", "death", "passed away", "suicide", "killed", "murdered")
            if any(sig in b for sig in _DEATH_SIGNALS):
                return False

        # Type sanity (config.binding_type_check, set on the class by RegisterManager):
        # a bare number/year can't name a person/place/org; a date/number step needs a digit
        if cls._type_check:
            et = (expected_type or "").lower()
            bare_num = re.fullmatch(r"[\d\s.,/\-–]+(?:s)?", b) is not None
            if et in ("person", "place", "org", "university") and bare_num:
                return False
            if et in ("date", "year", "number", "amount") and not re.search(
                    r"\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                    r"hundred|thousand|million|billion|first|second|third|dozen|several|few|many)\b", b):
                return False

        # Echo of the question subject (e.g. grandfather-of-X → "X")
        if subject and cls._same_entity(binding, subject):
            return False

        # Echo of a dependency's binding
        if current_bindings:
            for v in current_bindings.values():
                if v and cls._same_entity(binding, v):
                    return False

        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Private — LLM binding extraction (optional, requires llm_client)
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_binding_llm(
        self,
        unit: EvidenceUnit,
        requirement: Requirement,
        llm_client,
        dep_entities: Optional[list[str]] = None,
        known_bindings: Optional[dict[str, str]] = None,
        candidates: Optional[list[RetrievedCandidate]] = None,
    ) -> str:
        """
        Extract the binding for `requirement` from one evidence unit via the LLM.
        Returns the extracted phrase, or "" on failure.

        dep_entities  : bindings this step builds on (first one is named in the
                        v2 prompt as the known entity).
        known_bindings: {req_id -> binding} used to resolve [R1]-style
                        placeholders in the v2 step text.
        candidates    : all retrieved candidates for this step -- the evidence
                        used to validate bindings in the cascade.

        Mode (env REGISTERRAG_BINDING_PROMPT, default "hybrid" -- best measured:
        hybrid_sc EM 0.45 vs cascade_sc 0.42):
          v1      -- original binding_fix_llm prompt everywhere
          v2      -- role-aware prompt everywhere
          hybrid  -- v1 for root steps, v2 for dependent steps
          cascade -- root: v1.  dependent: v2 if evidence-validated, else v1 if
                     evidence-validated, else whichever had a candidate (v2 first).
        """
        mode = os.environ.get("REGISTERRAG_BINDING_PROMPT", "hybrid").lower()
        is_root = not requirement.depends_on
        args = (unit, requirement, llm_client, dep_entities, known_bindings)

        if mode == "v1" or (mode in ("hybrid", "cascade") and is_root):
            return self._llm_extract("v1", *args)[1]
        if mode in ("v2", "hybrid"):
            return self._llm_extract("v2", *args)[1]

        # cascade, dependent step
        cands = candidates or []
        has_v1, b_v1 = False, ""
        has_v2, b_v2 = self._llm_extract("v2", *args)
        if has_v2 and _validate_binding(b_v2, cands):
            v2s, v1s, chosen, out = "ok", "skip", "v2", b_v2
        else:
            v2s = "fail"
            has_v1, b_v1 = self._llm_extract("v1", *args)
            if has_v1 and _validate_binding(b_v1, cands):
                v1s, chosen, out = "ok", "v1", b_v1
            else:
                v1s = "fail"
                out = b_v2 if has_v2 else (b_v1 if has_v1 else "")
                chosen = "unvalidated" if out else "none"
        # src_only: binding appears only in the unit it was extracted from
        src_only = bool(out) and not _validate_binding(
            out, [c for c in cands if c.unit.unit_id != unit.unit_id]
        )
        logger.info(
            "Binding cascade: req=%s v2=%s v1=%s chosen=%s binding='%s' src_only=%s",
            requirement.id, v2s, v1s, chosen, out, src_only,
        )
        return out

    def _nli_label(self, unit: EvidenceUnit, requirement: Requirement, value: str) -> str:
        """NLI check of one extracted value against its passage (pipeline/nli.py)."""
        from registerrag.pipeline import nli
        from registerrag.pipeline.retriever import Retriever
        need = Retriever._resolve_placeholders(
            requirement.information_need,
            {rid: e.binding for rid, e in self._entries.items() if e.binding})
        label, pe, pc = nli.judge(unit.text, need, value, self.config.nli_model)
        if label == "contradiction" and pc < self.config.nli_contradiction_min:
            label = "neutral"
        logger.info("NLI %s: req=%s value=%r p_ent=%.2f p_con=%.2f unit=%s",
                    label, requirement.id, value, pe, pc, unit.unit_id)
        return label

    def _binding_style(self, requirement: Requirement) -> str:
        """span (short phrase) or clause (verbatim evidence clause) — set by the answer-kind router."""
        if not getattr(self.config, "answer_kind_router_enabled", True):
            return "span"
        return binding_style(classify(requirement.information_need))

    def _llm_extract_clause(
        self,
        unit: EvidenceUnit,
        requirement: Requirement,
        llm_client,
        known_bindings: Optional[dict[str, str]] = None,
    ) -> tuple[bool, str]:
        """
        Clause-style extraction for yes/no and description steps: copy the
        shortest verbatim clause of the evidence that confirms, rules out or
        states what the need asks about. The yes/no verdict itself is made
        later by the generator, which sees all steps together.
        """
        from registerrag.pipeline.retriever import Retriever
        need = Retriever._resolve_placeholders(requirement.information_need, known_bindings or {})
        title = (unit.document_title or "").strip()
        section = " > ".join(getattr(unit, "section_path", []) or [])
        where = f' (from "{title}"' + (f', section "{section}"' if section else "") + ")" if title else ""
        system_msg = (
            "You copy the part of an evidence passage that answers an information need. "
            "Output ONLY a JSON object: {\"has_candidate\": bool, \"extracted_binding\": str}."
        )
        user_msg = (
            f"Information need: {need}\n"
            f"Evidence{where}: {unit.text[:900]}\n\n"
            "If the evidence answers the need — it confirms it, rules it out, or states the "
            "amount, condition, requirement or process asked about — set has_candidate=true and "
            "copy the shortest clause that does so, verbatim, at most 40 words. "
            "Otherwise set has_candidate=false."
        )
        try:
            resp = llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "system", "content": system_msg},
                          {"role": "user", "content": user_msg}],
                temperature=0.0,
                max_tokens=120,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                return False, ""
            data = json.loads(m.group())
            binding = " ".join(str(data.get("extracted_binding", "") or "").split())
            has = bool(data.get("has_candidate")) and bool(binding)
            if binding.lower() in _JUNK:
                return False, ""
            return has, binding if has else ""
        except Exception as exc:          # malformed JSON etc. → no candidate
            logger.debug("Clause extraction failed: %s", exc)
            return False, ""

    def _try_joint(self, requirement, support_cands, llm_client, dep_entities,
                   dep_bindings, subject, max_words):
        """Joint multi-passage extraction -> (binding, holding candidate) or ("", None)."""
        if not (support_cands and self._binding_style(requirement) == "span"
                and self.config.llm_binding_extraction and llm_client is not None):
            return "", None
        top = support_cands[: self.config.joint_binding_k]
        raw = self._llm_extract_joint(top, requirement, llm_client, dep_entities, {
            rid: e.binding for rid, e in self._entries.items() if e.binding})
        holder = next((c for c in top if binding_in_text(raw, c.unit.text)), None) if raw else None
        if holder and self.is_valid_binding(
                raw, requirement.expected_answer_type, current_bindings=dep_bindings,
                subject=subject, max_words=max_words):
            logger.info("Joint binding: req=%s binding='%s' unit=%s",
                        requirement.id, raw, holder.unit.unit_id)
            support_cands.remove(holder)
            support_cands.insert(0, holder)
            return raw, holder
        return "", None

    def _llm_extract_joint(
        self,
        cands: list[RetrievedCandidate],
        requirement: Requirement,
        llm_client,
        dep_entities: Optional[list[str]] = None,
        known_bindings: Optional[dict[str, str]] = None,
    ) -> str:
        """One LLM call over several passages -> short answer phrase, or ""."""
        from registerrag.pipeline.retriever import Retriever
        from registerrag.pipeline.verifier import unit_context
        need = Retriever._resolve_placeholders(requirement.information_need, known_bindings or {})
        passages = "\n".join(
            f"[{i + 1}] ({unit_context(c.unit)}) {c.unit.text[:700]}" for i, c in enumerate(cands))
        lines = [f"Question: {need}"]
        if dep_entities:
            lines.append(f"Known entity: {dep_entities[0]}")
        lines.append(f"Expected answer type: {requirement.expected_answer_type or 'short phrase'}")
        user_msg = (
            "\n".join(lines) + "\n\nPassages:\n" + passages + "\n\n"
            "Answer the question from the passages. The answer may be stated in one "
            "passage, follow from several passages together, or be clearly implied by "
            "what they say. Give the shortest answer phrase (1–8 words), copied from the "
            "passages. If the passages do not support an answer, give an empty string.\n"
            'Respond with JSON only: {"answer": "..."}'
        )
        try:
            resp = llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "system", "content":
                           "You answer one sub-question from retrieved passages. Output only JSON."},
                          {"role": "user", "content": user_msg}],
                temperature=0.0,
                max_tokens=60,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                return ""
            ans = " ".join(str(json.loads(m.group()).get("answer", "") or "").split())
            return "" if ans.lower() in _JUNK else ans
        except Exception as exc:
            logger.debug("Joint binding extraction failed: %s", exc)
            return ""

    def _llm_extract(
        self,
        version: str,
        unit: EvidenceUnit,
        requirement: Requirement,
        llm_client,
        dep_entities: Optional[list[str]] = None,
        known_bindings: Optional[dict[str, str]] = None,
    ) -> tuple[bool, str]:
        """One LLM extraction call with the v1 or v2 prompt -> (has_candidate, binding)."""
        if self._binding_style(requirement) == "clause":
            return self._llm_extract_clause(unit, requirement, llm_client, known_bindings)
        system_msg = (
            "You extract a concise answer phrase (1–8 words) from an evidence sentence. "
            "Output ONLY a JSON object: {\"has_candidate\": bool, \"extracted_binding\": str}. "
            "If the sentence does not answer the need, set has_candidate=false."
        )
        # Answer-first format: with {"has_candidate": bool, ...} first, the 7B
        # model commits to false before reading for the answer (it refused
        # "Which label did X acquire?" over "X ... the acquisition of Y").
        answer_first = getattr(self.config, "binding_answer_first", False)
        answer_check = answer_first and getattr(self.config, "binding_answer_check", False)
        if answer_first:
            fmt = ('{"answer": str, "answers_need": bool}' if answer_check else '{"answer": str}')
            system_msg = (
                "You extract a concise answer phrase (1–8 words) from an evidence sentence. "
                f"Output ONLY a JSON object: {fmt}. Copy the phrase from the evidence "
                "that answers the need; the evidence may word the fact differently from the "
                "question (e.g. 'passed away' for 'died', 'established' for 'founded'). "
                "If the evidence does not give the answer, use an empty string."
                + (" Then set answers_need: true only if that phrase is exactly the kind of "
                   "thing the need asks for (a person for 'who', a date for 'when', ...)."
                   if answer_check else "")
            )
        if version == "v1":
            title = (unit.document_title or "").strip()
            section = " > ".join((getattr(unit, "section_path", None) or [])[-2:])
            evidence_label = (
                f'Evidence (from the article "{title}"'
                + (f', section "{section}"' if section else "")
                + "; pronouns refer to its subject)"
                if title and self.config.v1_article_title else "Evidence"
            )
            user_msg = (
                f"Information need: {requirement.information_need}\n"
                f"Expected type: {requirement.expected_answer_type}\n"
                f"{evidence_label}: {unit.text}\n\n"
                "Extract the answer phrase."
            )
        else:
            from registerrag.pipeline.retriever import Retriever
            dep_entity = dep_entities[0] if dep_entities else ""
            article_title = unit.document_title or getattr(unit, "doc_id", "") or ""
            resolved_query = Retriever._resolve_placeholders(
                requirement.information_need, known_bindings or {}
            )
            ent = dep_entity or "the known entity"
            header = [f"Question: {resolved_query}"]
            if article_title:
                header.append(f"Article: {article_title}")
            section = " > ".join((getattr(unit, "section_path", None) or [])[-2:])
            if section:
                header.append(f"Section: {section}")
            if dep_entity:
                header.append(f"Known entity: {dep_entity}")
            header.append(
                f"Expected answer type: {requirement.expected_answer_type or 'name or short phrase'}"
            )
            user_msg = (
                "Extract one specific fact from the text below.\n\n"
                + "\n".join(header) + "\n\n"
                "Rules:\n"
                "- Return the direct answer to the question (a name, place, year, or yes/no).\n"
                f"- If the question asks for an ATTRIBUTE of {ent} (birthplace, "
                "nationality, occupation, death date), return that attribute value.\n"
                f"- If the question asks for a RELATIVE of {ent} (father, mother, "
                f"spouse, sibling, grandfather), return the relative's name, NOT {ent} itself.\n"
                + ("- If the text does not give the answer, return an empty string.\n\n"
                   if answer_first else
                   "- If the text does not contain enough information to answer, "
                   "return has_candidate: false.\n\n")
                + f"Text: {unit.text[:600]}\n\n"
                "Respond with JSON only:\n"
                + (('{"answer": "...", "answers_need": true/false}' if answer_check else '{"answer": "..."}')
                   if answer_first else
                   '{"has_candidate": true/false, "extracted_binding": "answer"}')
            )
        try:
            resp = llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=60,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r"\{.*?\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                if answer_first:
                    binding = str(data.get("answer", "") or "").strip()
                    has = bool(binding) and (not answer_check or data.get("answers_need") in (True, "true", "True"))
                else:
                    binding = str(data.get("extracted_binding", "") or "").strip()
                    has = bool(data.get("has_candidate")) and bool(binding)
                if binding.lower() in _JUNK:      # copied the JSON example
                    has, binding = False, ""
                return has, (binding if has else "")
        except Exception as exc:
            logger.debug("LLM binding extraction failed: %s", exc)
        return False, ""

    # ─────────────────────────────────────────────────────────────────────────
    # Private — eviction
    # ─────────────────────────────────────────────────────────────────────────

    def _evict(self, all_requirements: list[Requirement]) -> None:
        """
        Evict the single lowest-utility non-pinned entry until the register
        fits within budget. Called after every add() that overflows.
        """
        budget = self.config.register_token_budget
        # Keep evicting until under budget or nothing left to evict
        while self.total_tokens > budget:
            # Score every entry — skip pinned
            evictable = [
                (req_id, entry)
                for req_id, entry in self._entries.items()
                if not self.is_pinned(entry, all_requirements)
            ]
            if not evictable:
                logger.warning(
                    "Register over budget (%d/%d) but all entries are pinned — cannot evict.",
                    self.total_tokens, budget,
                )
                break

            # Evict the entry with the lowest utility
            worst_id = min(
                evictable,
                key=lambda pair: self.utility(pair[1], all_requirements),
            )[0]
            evicted = self._entries.pop(worst_id)
            logger.debug(
                "Evicted register entry %s (tokens=%d, utility=%.3f) — budget %d/%d",
                worst_id,
                evicted.token_cost,
                self.utility(evicted, all_requirements),  # will re-compute but entry is gone
                self.total_tokens,
                budget,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Private — binding extraction (spaCy NER heuristic)
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_binding(self, unit: EvidenceUnit, expected_type: str) -> str:
        """
        Extract a compact answer (a name, date, number) from the evidence unit
        using spaCy NER — no LLM call.

        Falls back to the first noun phrase, then the first sentence fragment.
        """
        ner_map = self.config.answer_type_to_ner
        target_labels = ner_map.get(expected_type.lower(), [])

        doc = self.nlp(unit.text)

        # Try targeted NER type first
        for ent in doc.ents:
            if ent.label_ in target_labels:
                binding = ent.text.strip()
                if binding:
                    return binding

        # Fallback 1: any named entity
        for ent in doc.ents:
            binding = ent.text.strip()
            if binding:
                return binding

        # Fallback 2: first noun chunk (requires dependency parser; skip if unavailable)
        try:
            for chunk in doc.noun_chunks:
                binding = chunk.text.strip()
                if len(binding.split()) <= 6 and binding:
                    return binding
        except ValueError:
            pass

        # Fallback 3: truncate sentence
        words = unit.text.split()
        return " ".join(words[:8]) + ("…" if len(words) > 8 else "")
