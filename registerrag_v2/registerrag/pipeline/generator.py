"""
Answer generator — LLM call per question.

Stage 1: simple evidence + bindings table, single call.
Stage 2: explicit multi-hop chain rendering, verification-driven retry.
"""
from __future__ import annotations
import logging

import dataclasses
import re
from collections import Counter

from registerrag.config import Config
from registerrag.llm_client import call_llm
from registerrag.models import EvidenceUnit, ProofCertificate, RegisterEntry, Requirement
from registerrag.pipeline.answer_kind import (
    ANSWER_FORMAT, BOOLEAN, DESCRIPTION, FACT, classify, votes as kind_votes,
)

logger = logging.getLogger(__name__)

# Prepended to the answer instructions of both generator prompts
_COMPARISON_RULE = (
    "If the question asks 'which [film/movie/person/book/song/album/show] "
    "[came first / is older / was released earlier / was born earlier]', "
    "your answer MUST be a name (film title, person name, etc.) — never "
    "a bare year, date, or number."
)

_YESNO_RULE = "If the answer is yes or no, reply with just 'yes' or 'no'."

_YESNO = re.compile(
    r"^\s*(do|does|did|are|is|was|were|have|has|had|can|could|would|should)\b",
    re.I,
)


_WHICH_FILM = re.compile(r"\bwhich\s+(?:film|movie)\b", re.I)
_WHICH_PERSON = re.compile(
    r"\bwhich\s+person\b|\bwho\s+(?:\w+\s+){0,2}?(?:lived|born|died)\b", re.I
)
_FILM_GUARD = "Important: the answer must be a film title, not a person's name."
_PERSON_GUARD = "Important: the answer must be a person's name, not a year or date."


def _system_prompt(base: str, question: str) -> str:
    """
    Swap the entity-name rule for the yes/no rule on yes/no questions, and
    append the answer-type guard after the CoT format for "which film/movie"
    and "which person / who lived|born|died" questions.
    """
    if _YESNO.match(question):
        return base.replace(_COMPARISON_RULE, _YESNO_RULE)
    guards = []
    if _WHICH_FILM.search(question):
        guards.append(_FILM_GUARD)
    elif _WHICH_PERSON.search(question):   # film guard wins ("which film has the director who died…")
        guards.append(_PERSON_GUARD)
    if guards:
        return base.replace(_COT_FORMAT, _COT_FORMAT + "\n" + "\n".join(guards))
    return base


_COT_FORMAT = (
    "Reason step by step from the evidence. Format your response as:\n"
    "Reasoning: [one sentence per evidence piece]\n"
    "Answer: [final answer only — a name, yes/no, or short phrase]"
)

# Room for the reasoning lines before "Answer:"
_COT_MAX_TOKENS = 320
_DESC_MAX_TOKENS = 480     # reasoning + a 1–3 sentence answer


def kind_router_on(config: Config) -> bool:
    return getattr(config, "answer_kind_router_enabled", True)


# Clause kinds (yes/no, description) answer the question itself from all the
# findings, instead of "the last step's binding is the answer".
_CLAUSE_SYSTEM = """You answer a question from an ordered list of findings and the evidence passages they came from.

Each step shows what was found for one part of the question ([NOT FOUND] if nothing was).
Read every step and the evidence, then answer the question itself.

Reason step by step from the evidence. Format your response as:
Reasoning: [one sentence per relevant evidence piece]
{answer_format}

Rules:
- Use only the evidence. Do not use outside knowledge.
- Keep the Answer on a single line.
"""


def _clause_system_prompt(kind: str, config: Config | None = None) -> str:
    fmt = ANSWER_FORMAT[kind]
    if kind == DESCRIPTION and config is not None and getattr(config, "description_complete_enabled", False):
        fmt = ANSWER_FORMAT["description_complete"]
    return _CLAUSE_SYSTEM.format(answer_format=fmt)


_COMPARISON = re.compile(
    r"\b(which film|which movie|who (?:lived|was born|died)|"
    r"which (?:director|actor|person|book|song|album))\b", re.I)


def _answer_call(messages: list[dict], question: str, config: Config,
                 kind: str = FACT) -> tuple[str, int]:
    """
    Main answer call -> (extracted answer, llm_calls). With self-consistency on
    and a comparison question (or, with the router, any closed answer kind:
    yes/no, choice), sample k answers at self_consistency_temperature and return
    the majority; no majority -> first sample. Yes/no samples vote on their
    leading Yes/No.
    """
    max_tokens = _DESC_MAX_TOKENS if kind == DESCRIPTION else _COT_MAX_TOKENS
    vote = config.self_consistency_enabled and (
        _COMPARISON.search(question) or (kind_router_on(config) and kind_votes(kind)))
    if not vote:
        text, calls = call_llm(messages, config, json_mode=False, max_tokens=max_tokens)
        return _extract_answer(text, kind), calls

    sc_config = dataclasses.replace(config, llm_temperature=config.self_consistency_temperature)
    answers, total = [], 0
    for _ in range(max(1, config.self_consistency_k)):
        text, calls = call_llm(messages, sc_config, json_mode=False, max_tokens=max_tokens)
        total += calls
        answers.append(_extract_answer(text, kind))
    # Choice questions: only samples that name one of the options may win the
    # vote ("No" or a bare year can't answer "Which film …, A or B?")
    m = _OPTIONS_Q.match(question)
    if m and kind_router_on(config):
        opts = [_toks(m.group(1)), _toks(m.group(2))]
        named = [a for a in answers if any(_toks(a) & o for o in opts)]
        if named and len(named) < len(answers):
            logger.info("Choice guard: dropped %s", [a for a in answers if a not in named])
            answers = named
    counts = Counter(a.lower().strip() for a in answers)
    key, n = counts.most_common(1)[0]
    chosen = next(a for a in answers if a.lower().strip() == key) if n > 1 else answers[0]
    logger.info("Self-consistency: samples=%s chosen=%r", answers, chosen)
    return chosen, total

_YEAR_RETRY_SYSTEM = (
    "You answer comparison questions with a single name. "
    "Output only the name — no year, no date, no explanation."
)

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 prompt (kept for reference / backwards compat)
# ─────────────────────────────────────────────────────────────────────────────
_STAGE1_SYSTEM = """You are a precise answer generator for a multi-hop QA system.

You are given:
1. A question
2. Decomposed requirements with their bindings (short extracted answers)
3. Supporting evidence sentences

Your task: generate a concise, accurate answer to the question.

""" + _COMPARISON_RULE + """

""" + _COT_FORMAT + """

Rules:
- Answer ONLY from the provided evidence. Do not use outside knowledge.
- The Answer line must be concise: 1–15 words for factual questions.
- If the bindings directly answer the question, use them.
- If evidence is insufficient, give your best guess from what is given.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 prompt — explicit multi-hop chain reasoning
# ─────────────────────────────────────────────────────────────────────────────
_STAGE2_SYSTEM = """You are a precise multi-hop answer generator.

The question requires chaining several intermediate findings together.
You are given an ordered chain of sub-questions, each with an extracted binding,
followed by supporting evidence sentences.

""" + _COMPARISON_RULE + """

""" + _COT_FORMAT + """

HOW TO ANSWER:
1. Read the chain top-to-bottom: each step builds on the previous one.
2. The final answer is what the LAST step's binding resolves to.
3. If a binding says [NOT FOUND], infer from the evidence below — do NOT invent.
4. The Answer line holds ONLY the final answer: 1–10 words, no punctuation
   beyond what is needed. Do NOT repeat the question.

Example chain:
  R1: "Who directed Inception?"  →  Christopher Nolan
  R2: "What nationality is Christopher Nolan?"  →  British
  Reasoning: Inception was directed by Christopher Nolan. Christopher Nolan is British.
  Answer: British
"""

_MINIMAL_SPAN_RULE = """
ANSWER SPAN: give the minimal span that answers the question, copied as written
in the evidence. Leave out descriptive titles or roles in front of a name
("the British Landgrave Edmund Bellinger" -> "Edmund Bellinger"), added
locations ("La Goulette, Tunisia" -> "La Goulette") and unit words the question
does not ask for ("ten years old" -> "ten").
"""

_STAGE2_RETRY_SYSTEM = """You are a precise factual answer extractor.

The previous answer was rejected because it may not be supported by the evidence.
Look ONLY at the supporting evidence sentences below.
Extract the shortest factual answer that directly answers the question.
1–8 words maximum. No explanation.
"""


def generate_answer(
    question: str,
    requirements: list[Requirement],
    register: dict[str, RegisterEntry],
    proof: ProofCertificate,
    config: Config,
    ordered_pairs: list[tuple[Requirement, RegisterEntry]] | None = None,
) -> tuple[str, int]:
    """
    Generate the final answer (Stage 1 path — simple call, no retry).

    Parameters
    ----------
    ordered_pairs : optional pre-sorted (req, entry) list from register.ordered_entries()
                    If None, falls back to requirements order.
    Returns
    -------
    (answer_text, llm_calls)
    """
    user_content = _build_user_message(question, requirements, register, proof, ordered_pairs)
    messages = [
        {"role": "system", "content": _system_prompt(_STAGE1_SYSTEM, question)},
        {"role": "user", "content": user_content},
    ]
    answer, calls = _answer_call(messages, question, config)
    answer, guard_calls = _year_guard(answer, question, proof, config)
    return answer, calls + guard_calls


def generate_answer_stage2(
    question: str,
    requirements: list[Requirement],
    register: dict[str, RegisterEntry],
    proof: ProofCertificate,
    config: Config,
    ordered_pairs: list[tuple[Requirement, RegisterEntry]] | None = None,
    verifier=None,
    extra_units: list[EvidenceUnit] | None = None,
    missing_units: list[EvidenceUnit] | None = None,
) -> tuple[str, int]:
    """
    Stage 2 generation: chain prompt + optional cross-encoder verification retry.

    With the answer-kind router, yes/no and description questions use the
    clause prompt (answer the question from all findings + their evidence,
    `extra_units` = the passages each step's binding came from); fact and
    choice questions keep the short-answer chain prompt.

    Returns
    -------
    (answer_text, llm_calls)
    """
    total_calls = 0
    kind = classify(question) if kind_router_on(config) else FACT
    clause_kind = kind in (BOOLEAN, DESCRIPTION)

    if clause_kind:
        units = answer_evidence(proof, extra_units)
        user_content = _build_chain_message(
            question, requirements, register, proof, ordered_pairs,
            evidence_units=units, final_line="Answer the question.")
        system = _clause_system_prompt(kind, config)
    elif missing_units:
        # Some step found no binding: also show that step's top passages, so
        # the answer can still be read off the evidence
        user_content = _build_chain_message(
            question, requirements, register, proof, ordered_pairs,
            evidence_units=answer_evidence(proof, missing_units))
        system = _system_prompt(_STAGE2_SYSTEM, question)
    else:
        user_content = _build_chain_message(question, requirements, register, proof, ordered_pairs)
        system = _system_prompt(_STAGE2_SYSTEM, question)
    if not clause_kind and getattr(config, "answer_minimal_span_enabled", False):
        system = system + _MINIMAL_SPAN_RULE
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    answer, calls = _answer_call(messages, question, config, kind)
    total_calls += calls
    logger.debug("Answer kind=%s answer=%r", kind, answer)

    # ── Verification + retry (short answers only: its retry asks for 1–8 words) ─
    shown = answer_evidence(proof, missing_units) if (missing_units and not clause_kind) else proof.units
    if (config.answer_verify_enabled and verifier is not None and shown
            and not clause_kind):
        verify_score = _verify_answer(answer, shown, verifier)
        logger.debug("Answer verify score=%.3f threshold=%.3f", verify_score, config.answer_verify_threshold)

        if verify_score < config.answer_verify_threshold and total_calls <= config.answer_verify_max_retries:
            logger.debug("Answer failed verification (score=%.3f) — retrying", verify_score)
            evidence_block = "\n".join(
                f'  "{u.text}"' for u in shown
            )
            retry_user = (
                f"Question: {question}\n\n"
                f"Evidence:\n{evidence_block}\n\n"
                f"Answer (1–8 words):"
            )
            retry_messages = [
                {"role": "system", "content": _STAGE2_RETRY_SYSTEM},
                {"role": "user", "content": retry_user},
            ]
            retry_text, retry_calls = call_llm(
                retry_messages, config, json_mode=False, max_tokens=32
            )
            total_calls += retry_calls
            answer = _clean(retry_text)
            logger.debug("Retry answer: %r", answer)

    if getattr(config, "answer_type_check_enabled", False):
        answer, guard_calls = _choice_type_check(answer, question, shown or proof.units, config)
    else:
        answer, guard_calls = _year_guard(answer, question, proof, config)
    total_calls += guard_calls
    if getattr(config, "compare_operator_enabled", False):
        from registerrag.pipeline.operators import resolve_comparison
        decided, op_calls = resolve_comparison(question, requirements, register, config)
        total_calls += op_calls
        if decided and decided.strip().lower() != answer.strip().lower():
            logger.info("Compare operator: %r -> %r", answer, decided)
            answer = decided
    if getattr(config, "choice_snap_enabled", False):
        from registerrag.pipeline.operators import snap_to_option
        snapped = snap_to_option(answer, question)
        if snapped and snapped != answer:
            logger.info("Choice snap: %r -> %r", answer, snapped)
            answer = snapped
    if config.date_compare_enabled:
        decided = date_compare_answer(question, requirements, register)
        if decided and decided.lower() != answer.strip().lower():
            logger.info("Date compare: %r -> %r", answer, decided)
            answer = decided
    if config.refusal_fallback_enabled and kind != BOOLEAN:
        fb = refusal_fallback(answer, question, requirements, register)
        if fb:
            logger.info("Refusal fallback: %r -> %r", answer, fb)
            answer = fb
    if config.yesno_compare_enabled:
        decided = same_value_answer(question, requirements, register)
        if decided and decided != answer.strip().lower():
            logger.info("Yes/no compare: %r -> %r", answer, decided)
            answer = decided
    if config.answer_normalize_enabled and kind != DESCRIPTION:
        answer = normalize_final_answer(answer, question)

    return answer, total_calls


_CLAUSE_MAX_UNITS = 12
# The local Ollama server keeps ~3k prompt tokens intact and silently cuts
# longer prompts (the start, i.e. the instructions, is lost) — so the evidence
# block gets a budget well inside that.
_EVIDENCE_TOKEN_BUDGET = 1600


def _dedupe_units(units: list[EvidenceUnit]) -> list[EvidenceUnit]:
    seen, out = set(), []
    for u in units:
        if u.unit_id not in seen:
            seen.add(u.unit_id)
            out.append(u)
    return out


def answer_evidence(proof: ProofCertificate, extra_units: list[EvidenceUnit] | None) -> list[EvidenceUnit]:
    """Evidence shown for clause-kind answers: proof units first, then extras, within the token budget."""
    out, used = [], 0
    for u in _dedupe_units(list(proof.units) + list(extra_units or [])):
        cost = int(len(u.text.split()) * 1.4) + 12
        if out and (used + cost > _EVIDENCE_TOKEN_BUDGET or len(out) >= _CLAUSE_MAX_UNITS):
            break
        out.append(u)
        used += cost
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic date comparison ("Which film came out first, A or B?")
# ─────────────────────────────────────────────────────────────────────────────

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}
_MIN_Q = re.compile(r"\b(earlier|earliest|first|older|oldest|sooner)\b", re.I)
_MAX_Q = re.compile(r"\b(later|latest|more recent(?:ly)?|most recent(?:ly)?|younger|youngest|last)\b", re.I)
_SKIP_Q = re.compile(r"\b(longer|longest|more|most|less|fewer)\b(?!\s+recent)", re.I)


def _parse_date(s: str) -> tuple[int, int, int] | None:
    """'24 December 1978' / 'October 5, 1894' / '1944' → (y, m, d); None if no year."""
    t = (s or "").lower()
    ym = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", t)
    if not ym:
        return None
    month = next((n for name, n in _MONTHS.items() if re.search(r"\b" + name + r"\b", t)), 0)
    dm = re.search(r"\b([0-3]?\d)(?:st|nd|rd|th)?\b(?!\d)", t.replace(ym.group(1), " "))
    day = int(dm.group(1)) if (dm and month and 1 <= int(dm.group(1)) <= 31) else 0
    return (int(ym.group(1)), month, day)


def _lifespan_answer(
    opts: list[str],
    requirements: list[Requirement],
    register: dict[str, RegisterEntry],
) -> str:
    """Pick the option with the longer (death − birth) span from born/died bindings."""
    spans = []
    for o in opts:
        ot = _toks(o)
        born = died = None
        for r in requirements:
            if not ot <= _toks(r.information_need):
                continue
            e = register.get(r.id)
            d = _parse_date(e.binding) if (e and e.is_supported and e.binding) else None
            if not d:
                continue
            if re.search(r"\bborn\b", r.information_need, re.I):
                born = d
            elif re.search(r"\b(die|died|death)\b", r.information_need, re.I):
                died = d
        if not (born and died):
            return ""
        spans.append((died[0] - born[0]) * 372 + (died[1] - born[1]) * 31 + (died[2] - born[2])
                     if born[1] and died[1] else (died[0] - born[0]) * 372)
    if spans[0] == spans[1]:
        return ""
    return opts[0] if spans[0] > spans[1] else opts[1]


def _ancestors_text(req: Requirement, by_id: dict[str, Requirement]) -> str:
    out, stack, seen = [], [req], set()
    while stack:
        r = stack.pop()
        if r.id in seen:
            continue
        seen.add(r.id)
        out.append(r.information_need)
        stack.extend(by_id[d] for d in r.depends_on if d in by_id)
    return " ".join(out)


def date_compare_answer(
    question: str,
    requirements: list[Requirement],
    register: dict[str, RegisterEntry],
) -> str:
    """
    "Which/Who …, A or B?" with an earlier/later-type comparison: when the two
    leaf bindings are dates, pick the option with the min (earlier/first/older)
    or max (later/more recently/younger) date. "" when not applicable/unsure.
    """
    m = _OPTIONS_Q.match(question)
    if not m:
        return ""
    head = question.split(",")[0]
    if re.search(r"\blive[sd]?\s+longer\b", head, re.I):
        return _lifespan_answer([m.group(1).strip(), m.group(2).strip()], requirements, register)
    if _SKIP_Q.search(head):
        return ""
    want_min, want_max = bool(_MIN_Q.search(head)), bool(_MAX_Q.search(head))
    if want_min == want_max:
        return ""
    opts = [m.group(1).strip(), m.group(2).strip()]
    by_id = {r.id: r for r in requirements}
    used = {d for r in requirements for d in r.depends_on}
    leaves = [r for r in requirements if r.id not in used]
    dated = []
    for r in leaves:
        e = register.get(r.id)
        d = _parse_date(e.binding) if (e and e.is_supported and e.binding) else None
        if d:
            dated.append((r, d))
    if len(dated) != 2:
        return ""
    # map each dated leaf to an option by token overlap with its requirement chain
    def overlap(r, o):
        ot = _toks(o)
        return len(_toks(_ancestors_text(r, by_id)) & ot) / max(1, len(ot))
    s00 = overlap(dated[0][0], opts[0]) + overlap(dated[1][0], opts[1])
    s01 = overlap(dated[0][0], opts[1]) + overlap(dated[1][0], opts[0])
    if s00 == s01:
        return ""
    pairs = [(opts[0], dated[0][1]), (opts[1], dated[1][1])] if s00 > s01 else \
            [(opts[1], dated[0][1]), (opts[0], dated[1][1])]
    (oa, da), (ob, db) = pairs
    # compare at the precision both dates share (year; then month/day if both known)
    ka, kb = [da[0]], [db[0]]
    if da[1] and db[1]:
        ka.append(da[1]); kb.append(db[1])
        if da[2] and db[2]:
            ka.append(da[2]); kb.append(db[2])
    if ka == kb:
        return ""
    earlier = oa if ka < kb else ob
    later = ob if ka < kb else oa
    return earlier if want_min else later


# ─────────────────────────────────────────────────────────────────────────────
# Refusal fallback
# ─────────────────────────────────────────────────────────────────────────────

_REFUSAL = re.compile(
    r"\b(not\s+(?:provided|specified|found|mentioned|available|given|stated|known)|"
    r"insufficient|cannot be determined|can't be determined|no information|unknown|"
    r"not enough|unclear)\b|\[not found\]", re.I)


def refusal_fallback(
    answer: str,
    question: str,
    requirements: list[Requirement],
    register: dict[str, RegisterEntry],
) -> str:
    """
    The generator refused ("Not provided", "Evidence insufficient") or answered
    yes/no to a non-yes/no question: fall back to the binding of the single
    final requirement, if it has one. "" → keep the answer.
    """
    a = (answer or "").strip()
    refused = bool(_REFUSAL.search(a)) or not a
    bad_yesno = a.lower() in ("yes", "no") and not _YESNO.match(question)
    if not (refused or bad_yesno):
        return ""
    used = {d for r in requirements for d in r.depends_on}
    leaves = [r for r in requirements if r.id not in used]
    if len(leaves) != 1:
        return ""
    e = register.get(leaves[0].id)
    if e and e.is_supported and e.binding:
        return e.binding
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic yes/no for "same country / nationality" comparisons
# ─────────────────────────────────────────────────────────────────────────────

_SAME_Q = re.compile(r"\bsame\s+(country|nationality|place|city)\b|\bboth\b.*\bfrom\b.*\bsame\b", re.I)

_DEMONYM = {
    "american": "united states", "usa": "united states", "us": "united states",
    "u.s.": "united states", "united states of america": "united states", "america": "united states",
    "british": "united kingdom", "english": "united kingdom", "scottish": "united kingdom",
    "welsh": "united kingdom", "uk": "united kingdom", "england": "united kingdom",
    "scotland": "united kingdom", "wales": "united kingdom", "great britain": "united kingdom",
    "italian": "italy", "french": "france", "german": "germany", "west german": "germany",
    "east german": "germany", "west germany": "germany", "east germany": "germany",
    "canadian": "canada", "indian": "india", "japanese": "japan", "chinese": "china",
    "russian": "russia", "soviet": "russia", "soviet union": "russia", "ussr": "russia",
    "spanish": "spain", "mexican": "mexico", "australian": "australia", "swedish": "sweden",
    "danish": "denmark", "norwegian": "norway", "finnish": "finland", "dutch": "netherlands",
    "the netherlands": "netherlands", "belgian": "belgium", "austrian": "austria",
    "swiss": "switzerland", "polish": "poland", "hungarian": "hungary", "czech": "czech republic",
    "greek": "greece", "turkish": "turkey", "iranian": "iran", "egyptian": "egypt",
    "brazilian": "brazil", "argentine": "argentina", "argentinian": "argentina",
    "filipino": "philippines", "korean": "south korea", "south korean": "south korea",
    "irish": "ireland", "portuguese": "portugal", "indonesian": "indonesia",
    "pakistani": "pakistan", "nigerian": "nigeria", "romanian": "romania",
    "bulgarian": "bulgaria", "serbian": "serbia", "croatian": "croatia",
    "ukrainian": "ukraine", "israeli": "israel", "thai": "thailand",
    "vietnamese": "vietnam", "new zealander": "new zealand", "libyan": "libya",
    "georgian": "georgia", "armenian": "armenia", "senegalese": "senegal",
}
_COUNTRIES = set(_DEMONYM.values())


def _to_country(binding: str) -> str:
    """Map a nationality / country / 'City, Country' binding to a canonical country, or ''."""
    b = re.sub(r"[^\w\s.\-]", " ", (binding or "").lower()).strip()
    if not b:
        return ""
    if b in _DEMONYM:
        return _DEMONYM[b]
    if b in _COUNTRIES:
        return b
    # "City, Region, Country": the last comma part is the most specific country
    # signal ("Mackenzie, British Columbia, Canada" is Canada, not British)
    if "," in binding:
        last = _to_country(binding.rsplit(",", 1)[1])
        if last:
            return last
    # any known country / demonym phrase inside, longest first ("Rome, Italy", "Italian-American")
    for key in sorted(list(_DEMONYM) + list(_COUNTRIES), key=len, reverse=True):
        if len(key) > 3 and re.search(r"\b" + re.escape(key) + r"\b", b):
            return _DEMONYM.get(key, key)
    return ""


def same_value_answer(
    question: str,
    requirements: list[Requirement],
    register: dict[str, RegisterEntry],
) -> str:
    """
    For "Are X and Y from the same country?"-style questions, compare the two
    leaf bindings (requirements no other requirement depends on) and return
    "yes"/"no". Returns "" when not applicable or either value is unknown.
    """
    if not (_YESNO.match(question) and _SAME_Q.search(question)):
        return ""
    used = {d for r in requirements for d in r.depends_on}
    leaves = [r for r in requirements if r.id not in used]
    if len(leaves) != 2:
        return ""
    vals = []
    for r in leaves:
        e = register.get(r.id)
        if not (e and e.is_supported and e.binding):
            return ""
        vals.append(e.binding)
    c1, c2 = _to_country(vals[0]), _to_country(vals[1])
    if c1 and c2:
        return "yes" if c1 == c2 else "no"
    n1, n2 = (" ".join(_TOK.findall(v.lower())) for v in vals)
    if n1 and n1 == n2:
        return "yes"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Final-answer normalisation (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

# Leading count before a name ("two Nandi Awards" → "Nandi Awards") — but not
# the day of a date ("10 June 1819") or a number with its unit.
_LEAD_COUNT = re.compile(
    r"^(?:two|three|four|five|six|several|multiple|\d+)\s+"
    r"(?!(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b)(?=[A-Z])", re.I)
_LEAD_HONORIFIC = re.compile(
    r"^(?:Lady|Lord|Sir|Dame|Count|Countess|Dr\.?|Mr\.?|Mrs\.?|Ms\.?|former|the late)\s+(?=\S+\s+\S)"
)
_PLACE_Q = re.compile(r"\b(where|place of (?:birth|death|burial)|birthplace|born|buried)\b", re.I)
_PLACE_IN = re.compile(r"^(.+?)\s+in\s+[A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)*$")
_OPTIONS_Q = re.compile(r"^\s*(?:which|who)\b[^,]*,\s*(.+)\s+or\s+(.+?)\s*\?*\s*$", re.I)
_TOK = re.compile(r"[a-z0-9]+")

# Kinship questions ("Who is X's father-in-law?", "Who is the wife of X?")
_KIN_Q = re.compile(
    r"^\s*who\s+(?:is|was)\b.*\b(?:father|mother|parent|grand\w+|\w+-in-law|step\w+|spouse|wife|"
    r"husband|child|son|daughter|sibling|brother|sister|uncle|aunt)\b", re.I)
# Regnal style before a name: "King Ngwane V" → "Ngwane V"
_REGNAL = re.compile(r"^(?:King|Queen|Emperor|Empress)\s+(?=[A-Z])")
# Title appositive after a regnal number or full name:
# "Louis VI, Elector Palatine" → "Louis VI",
# "George Chaworth, 1st Viscount Chaworth of Armagh" → "George Chaworth"
_TITLE_APPOSITIVE = re.compile(
    r"^(.+?\b[IVX]+|[A-Z][\w'-]+(?: [A-Z][\w'-]+)+?),\s+(?:\d+(?:st|nd|rd|th)\s+)?"
    r"(?:Elector|Prince|Princess|Duke|Duchess|Count|Countess|Viscount|Earl|Baron|"
    r"Marquess|Margrave|Landgrave|King|Queen)\b.*$")


def short_person_label(name: str) -> str:
    """
    Reduce a noble/royal name to its short label form (the form 2Wiki gold
    answers use): drop a leading King/Queen and a trailing ", <title> of …".
    """
    a = _REGNAL.sub("", name.strip())
    return _TITLE_APPOSITIVE.sub(r"\1", a).strip()


def _toks(s: str) -> set[str]:
    return set(_TOK.findall(s.lower()))


def normalize_final_answer(answer: str, question: str) -> str:
    """
    Deterministic clean-up of the final answer:
      - "which/who …, A or B?" → snap the answer to the closest option's exact text
      - drop leading counts ("two Nandi Awards") and honorifics ("Lady Rachel Billington")
      - place questions: keep the first comma part and drop a trailing " in <Place>"
        ("London, England, United Kingdom" → "London", "Komitas Pantheon in Yerevan" → "Komitas Pantheon")
    """
    a = (answer or "").strip()
    if not a or a.lower() in ("yes", "no"):
        return a

    m = _OPTIONS_Q.match(question)
    if m:
        opts = [m.group(1).strip(), m.group(2).strip()]
        at = _toks(a)
        scores = [len(at & _toks(o)) / max(1, len(_toks(o))) for o in opts]
        best = max(range(2), key=lambda i: scores[i])
        if scores[best] > 0 and scores[best] != scores[1 - best]:
            if opts[best] != a:
                logger.debug("Answer snapped to option: %r -> %r", a, opts[best])
            return opts[best]

    a = _LEAD_COUNT.sub("", a)
    a = _LEAD_HONORIFIC.sub("", a)
    if _KIN_Q.match(question):
        a = short_person_label(a)
    if _PLACE_Q.search(question):
        a = a.split(",")[0].strip()
        pm = _PLACE_IN.match(a)
        if pm:
            a = pm.group(1)
    return a.strip()


def _choice_type_check(
    answer: str,
    question: str,
    units: list[EvidenceUnit],
    config: Config,
) -> tuple[str, int]:
    """
    General answer-type check for choice questions ("…, A or B?"): the answer
    must name one of the options. If it names neither (a bare year, "No", a
    third entity), re-ask once constrained to the options.
    """
    from registerrag.pipeline.operators import question_options, snap_to_option
    opts = question_options(question)
    if not opts or classify(question) != "choice" or snap_to_option(answer, question):
        return answer, 0
    evidence = "\n".join(f'  "{u.text}"' for u in units) or "(none)"
    messages = [
        {"role": "system", "content": "Answer the question with exactly one of the given options, "
                                      "copied as written. Output only that option."},
        {"role": "user", "content": f"Question: {question}\nOptions: {opts[0]} | {opts[1]}\n"
                                    f"Evidence:\n{evidence}\nAnswer:"},
    ]
    retry, calls = call_llm(messages, config, json_mode=False, max_tokens=40)
    snapped = snap_to_option(_clean(retry), question)
    if snapped:
        logger.info("Choice type check: %r -> %r", answer, snapped)
        return snapped, calls
    return answer, calls


def _year_guard(
    answer: str,
    question: str,
    proof: ProofCertificate,
    config: Config,
) -> tuple[str, int]:
    """
    A bare 4-digit year answering a "which …" question is almost always the
    date binding leaking through instead of the entity it belongs to
    ("Which film came first, A or B?" → "1942"). Re-ask once, constrained to a
    name; keep the original answer if the retry is empty or still a year.
    """
    if not (re.fullmatch(r"\d{4}", answer) and re.search(r"\bwhich\b", question, re.I)):
        return answer, 0

    proof_text = " ".join(u.text for u in proof.units) if proof.units else "(none)"
    retry_prompt = (
        f"Question: {question}\n"
        f"Evidence: {proof_text}\n"
        f"The answer is a film or person name, not a year. "
        f"Answer with just the name."
    )
    messages = [
        {"role": "system", "content": _YEAR_RETRY_SYSTEM},
        {"role": "user", "content": retry_prompt},
    ]
    retry, calls = call_llm(messages, config, json_mode=False, max_tokens=20)
    cleaned = _clean(retry)
    if cleaned and not re.fullmatch(r"\d{4}", cleaned):
        logger.info("Year guard: %r → %r  (q=%r)", answer, cleaned, question[:80])
        return cleaned, calls
    logger.info("Year guard: retry still not a name (%r), keeping %r", cleaned, answer)
    return answer, calls


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_user_message(
    question: str,
    requirements: list[Requirement],
    register: dict[str, RegisterEntry],
    proof: ProofCertificate,
    ordered_pairs: list[tuple[Requirement, RegisterEntry]] | None,
) -> str:
    """Stage 1 flat bindings table + evidence block."""
    if ordered_pairs:
        req_order = [req for req, _ in ordered_pairs]
    else:
        req_order = requirements

    bindings_lines = []
    for req in req_order:
        entry = register.get(req.id)
        if entry and entry.is_supported:
            bindings_lines.append(
                f"  [{req.id}] {req.information_need}  →  **{entry.binding}**"
                f"  (confidence: {entry.confidence:.2f})"
            )
        else:
            bindings_lines.append(
                f"  [{req.id}] {req.information_need}  →  [NOT FOUND]"
            )

    bindings_block = "\n".join(bindings_lines) if bindings_lines else "(none)"

    evidence_lines = [
        f'  [{u.unit_id}] ({u.document_title})  "{u.text}"'
        for u in proof.units
    ]
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "(no evidence)"

    return (
        f"Question: {question}\n\n"
        f"Sub-question bindings:\n{bindings_block}\n\n"
        f"Supporting evidence:\n{evidence_block}\n\n"
        f"Answer (concise, factual):"
    )


def _build_chain_message(
    question: str,
    requirements: list[Requirement],
    register: dict[str, RegisterEntry],
    proof: ProofCertificate,
    ordered_pairs: list[tuple[Requirement, RegisterEntry]] | None,
    evidence_units: list[EvidenceUnit] | None = None,
    final_line: str = "Final answer (1–10 words):",
) -> str:
    """
    Stage 2 message: render requirements as an explicit chain showing
    dependency arrows, then supporting evidence (`evidence_units`, default
    the proof units; a unit's section heading is shown when it has one).
    """
    if ordered_pairs:
        req_order = [req for req, _ in ordered_pairs]
    else:
        req_order = requirements

    # Build chain lines showing dependency arrows explicitly
    chain_lines = []
    for i, req in enumerate(req_order):
        entry = register.get(req.id)
        binding_str = (
            f"**{entry.binding}**" if (entry and entry.is_supported and entry.binding)
            else "[NOT FOUND]"
        )
        conf_str = f"  conf={entry.confidence:.2f}" if (entry and entry.is_supported) else ""
        gap_flag = "  [gap-filled]" if (entry and entry.was_gap_fill) else ""

        # Show what this step depends on
        if req.depends_on:
            dep_bindings = []
            for dep_id in req.depends_on:
                dep_entry = register.get(dep_id)
                if dep_entry and dep_entry.binding:
                    dep_bindings.append(dep_entry.binding)
            if dep_bindings:
                chain_lines.append(
                    f"  Step {i+1} [{req.id}]: {req.information_need}"
                    f"\n           uses: {', '.join(dep_bindings)}"
                    f"\n           → {binding_str}{conf_str}{gap_flag}"
                )
            else:
                chain_lines.append(
                    f"  Step {i+1} [{req.id}]: {req.information_need}"
                    f"\n           → {binding_str}{conf_str}{gap_flag}"
                )
        else:
            chain_lines.append(
                f"  Step {i+1} [{req.id}]: {req.information_need}"
                f"\n           → {binding_str}{conf_str}{gap_flag}"
            )

    chain_block = "\n".join(chain_lines)

    def _src(u: EvidenceUnit) -> str:
        sec = " > ".join(getattr(u, "section_path", []) or [])
        return f"{u.document_title} — {sec}" if evidence_units is not None and sec else u.document_title

    units = proof.units if evidence_units is None else evidence_units
    evidence_lines = [f'  [{u.unit_id}] ({_src(u)})  "{u.text}"' for u in units]
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "(no evidence)"

    return (
        f"Question: {question}\n\n"
        f"Reasoning chain:\n{chain_block}\n\n"
        f"Supporting evidence:\n{evidence_block}\n\n"
        f"{final_line}"
    )


def _verify_answer(
    answer: str,
    proof_units: list[EvidenceUnit],
    verifier,
) -> float:
    """
    Score the answer against each proof unit using the cross-encoder.
    Returns the max score (ms-marco logit scale).
    A score above the threshold indicates the answer is grounded in the evidence.
    """
    if not answer or not proof_units:
        return -100.0
    pairs = [(answer, unit.text) for unit in proof_units]
    scores = verifier._score_pairs(pairs)
    return float(max(scores)) if scores else -100.0


# Verb heads that start a trailing predicate on an entity answer:
#   "Digby Tatham-Warter is older", "X came out earlier", "X was released first"
_TRAILING_VERBS = (
    "is", "was", "are", "were", "came", "comes", "come", "released", "premiered",
    "died", "lived", "has", "had", "did", "does", "appeared", "started", "began",
)
_TRAILING_PRED_RE = re.compile(
    r"^(?P<ent>.+?)\s+(?:" + "|".join(_TRAILING_VERBS) + r")\b(?P<rest>(?:\s+[\w\-']+){0,5})$"
)
# Clause after a comma that starts with a lowercase word is commentary, not part
# of the name ("X, who was born in 1917"). "Charles I, Count of Ligny" is kept.
_TRAILING_CLAUSE_RE = re.compile(r"^(?P<ent>.+?),\s+(?P<rest>[a-z].*)$")
_TRAILING_PUNCT = ".,;:!?"


_PRONOUNS = {"he", "she", "it", "they", "this", "that", "both", "neither", "none"}

_ANSWER_MARKER_RE = re.compile(r"answer\s*:", re.IGNORECASE)


_LEAD_YN = re.compile(r"^\W*(yes|no)\b", re.I)


def _extract_answer(text: str, kind: str = FACT) -> str:
    """
    Pull the final answer out of a "Reasoning: … / Answer: …" response.

    Takes the text after the LAST "Answer:" (first line only, in case the model
    keeps talking), then applies the usual cleanup. With no "Answer:" marker the
    whole response goes through the existing cleanup unchanged.

    kind=boolean     → "Yes" / "No" (the reason is dropped; no leading yes/no →
                       the cleaned text is kept)
    kind=description → the whole answer text, unshortened
    """
    if kind in (BOOLEAN, DESCRIPTION):
        matches = list(_ANSWER_MARKER_RE.finditer(text))
        tail = text[matches[-1].end():] if matches else text
        tail = " ".join(tail.split()).strip().strip('"').strip()
        if kind == BOOLEAN:
            m = _LEAD_YN.match(tail)
            return m.group(1).capitalize() if m else _clean(tail)
        return tail
    matches = list(_ANSWER_MARKER_RE.finditer(text))
    if not matches:
        return _clean(text)
    tail = text[matches[-1].end():].strip()
    first_line = tail.splitlines()[0] if tail else ""
    return _clean(first_line.strip().rstrip(_TRAILING_PUNCT).strip())


def _starts_proper_noun(text: str) -> bool:
    first = text.split()[0] if text.split() else ""
    return bool(first) and first[0].isupper() and first.lower() not in _PRONOUNS


def _postprocess_answer(answer: str) -> str:
    """
    Reduce a sentence-style answer to the bare entity.

      "Digby Tatham-Warter is older."                   → "Digby Tatham-Warter"
      "Diary For My Mother And Father came out earlier." → "Diary For My Mother And Father"
      "Charles I, Count of Ligny."                      → "Charles I, Count of Ligny"
      "No."                                             → "No"

    Only fires when the part before the verb/comma starts with a proper noun and
    the predicate after it is short; otherwise just trailing punctuation goes.
    """
    a = answer.strip().rstrip(_TRAILING_PUNCT).strip()

    # "Yes, both are Italian." / "No. They differ." → bare yes/no
    m = re.match(r"^(yes|no)\b[\s,.;:!]", a + " ", flags=re.IGNORECASE)
    if m and len(a.split()) > 1:
        return m.group(1)

    m = _TRAILING_CLAUSE_RE.match(a)
    if m and _starts_proper_noun(m.group("ent")):
        a = m.group("ent").strip()

    m = _TRAILING_PRED_RE.match(a)
    if m and _starts_proper_noun(m.group("ent")) and len(m.group("ent").split()) <= 10:
        a = m.group("ent").strip()

    return a.rstrip(_TRAILING_PUNCT).strip()


def _clean(text: str) -> str:
    """Strip common LLM artefacts from the generated answer."""
    answer = text.strip().strip('"').strip("'")
    if answer.lower().startswith("answer:"):
        answer = answer[7:].strip()
    if answer.lower().startswith("final answer:"):
        answer = answer[13:].strip()
    return _postprocess_answer(answer)
