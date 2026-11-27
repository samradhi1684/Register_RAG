"""
General comparison operator (dataset-agnostic).

Trigger is structural, not lexical: the plan has >= 2 leaf requirements (steps
no other step depends on) with supported bindings.  One small LLM call reads the
question plus the plan's findings and states whether the question compares the
SAME attribute of two or more entities, which operation it asks for, and which
step holds each entity's value.  The comparison itself is done in code:

  earliest / latest   dates (any precision; compared at the shared precision)
  largest / smallest  numbers ("1,234", "5.6 million", "two")
  same                the LLM's judgement whether the values denote the same thing

Returns "" whenever anything is missing or unparsable, so the LLM answer stands.
"""
from __future__ import annotations
import logging
import re

from registerrag.config import Config
from registerrag.llm_client import call_llm, parse_json_response
from registerrag.models import RegisterEntry, Requirement
from registerrag.pipeline.answer_kind import BOOLEAN, CHOICE, classify

logger = logging.getLogger(__name__)

OPS = ("earliest", "latest", "largest", "smallest", "same")

_SYSTEM = """You inspect a question and the findings a QA system already extracted for it.

Decide whether the question is answered by COMPARING THE SAME ATTRIBUTE of two or more
entities (e.g. which is older, which happened first, which is bigger, whether two things
are the same). If it is not such a comparison, set "comparison" to false.

If it is:
- "operation": one of earliest | latest | largest | smallest | same
    earliest = first / earlier / older (born or founded first) / sooner
    latest   = later / more recent / younger / last
    largest  = more / bigger / higher / longer / greater
    smallest = fewer / smaller / lower / shorter / less
    same     = whether the values are the same (same country, same nationality, same year)
  A question whether BOTH entities have some property ("Are A and B both singers?") is NOT
  a comparison: set "comparison" to false.
- "word": the comparison word or phrase copied from the question (e.g. "younger", "came out first")
- "options": the compared entities IN THE ORDER THEY APPEAR IN THE QUESTION, each with
  the step whose finding is that entity's value for the compared attribute
- "same_value": only for "same": true if the findings denote the same thing
  (e.g. "American" and "United States" are the same country), else false

Examples (invented):
Q: Which bridge opened earlier, Kestrel Bridge or Arden Bridge?
  R1 When did Kestrel Bridge open? = 1931 ; R2 When did Arden Bridge open? = 12 May 1928
  -> {"comparison": true, "operation": "earliest", "word": "earlier", "options": [{"name": "Kestrel Bridge", "step": "R1"}, {"name": "Arden Bridge", "step": "R2"}]}
Q: Are Lake Vira and Lake Osso in the same country?
  R1 Which country is Lake Vira in? = Chile ; R2 Which country is Lake Osso in? = Argentina
  -> {"comparison": true, "operation": "same", "word": "same country", "same_value": false, "options": [{"name": "Lake Vira", "step": "R1"}, {"name": "Lake Osso", "step": "R2"}]}
Q: What rebates does the Sunfield program offer for heat pumps and solar panels?
  -> {"comparison": false}

Return ONLY JSON."""


_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}
_MON_ABBR = {m[:3]: n for m, n in _MONTHS.items()}
_NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_TOK = re.compile(r"[a-z0-9]+")


def parse_date(s: str):
    """'24 December 1978' / 'Oct 5, 1894' / '1944' / '1990s' -> (y, m, d) with 0 for unknown; None."""
    t = (s or "").lower()
    ym = re.search(r"\b(\d{3,4})(?:s\b|\b)", t)
    if not ym:
        return None
    year = int(ym.group(1))
    if re.search(r"\b(bc|bce)\b", t):
        year = -year
    month = 0
    for w in re.findall(r"[a-z]+", t):
        if w in _MONTHS:
            month = _MONTHS[w]
            break
        if w[:3] in _MON_ABBR and len(w) <= 4:
            month = _MON_ABBR[w[:3]]
            break
    rest = t.replace(ym.group(1), " ")
    dm = re.search(r"\b([0-3]?\d)(?:st|nd|rd|th)?\b", rest)
    day = int(dm.group(1)) if (dm and month and 1 <= int(dm.group(1)) <= 31) else 0
    return (year, month, day)


# English comparison words -> operation. Grammar, not data: the LLM copies the
# word from the question and this table fixes the direction (a 7B model
# mislabels "younger" as earliest). Unknown words keep the LLM's operation.
_WORD_OP = [
    (r"\b(?:younger|youngest|later|latest|newer|newest|more recent(?:ly)?|most recent(?:ly)?|last|after)\b", "latest"),
    (r"\b(?:older|oldest|elder|eldest|earlier|earliest|first|sooner|before|prior)\b", "earliest"),
    (r"\b(?:more|most|larger|largest|bigger|biggest|higher|highest|taller|tallest|longer|longest|greater|greatest|heavier|wider|deeper)\b", "largest"),
    (r"\b(?:fewer|fewest|less|least|smaller|smallest|lower|lowest|shorter|shortest|lighter|narrower)\b", "smallest"),
]


def word_operation(word: str) -> str:
    w = (word or "").lower()
    for pat, op in _WORD_OP:
        if re.search(pat, w):
            return op
    return ""


def _date_like(s: str) -> bool:
    """A value that is a date, not a quantity ('3 July 1882', '1733')."""
    t = (s or "").lower()
    if any(re.search(r"\b" + m + r"\b", t) for m in list(_MONTHS) + list(_MON_ABBR)):
        return True
    nums = re.findall(r"\d+(?:[.,]\d+)?", t)
    return len(nums) == 1 and re.fullmatch(r"1\d{3}|20\d{2}", nums[0]) is not None


def parse_number(s: str):
    if _date_like(s):
        return None
    t = (s or "").lower().replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    if m:
        v = float(m.group(0))
        after = t[m.end():].split()
        if after and after[0] in _SCALE:
            v *= _SCALE[after[0]]
        return v
    for w in re.findall(r"[a-z]+", t):
        if w in _NUM_WORDS:
            return float(_NUM_WORDS[w])
    return None


def _leaves(requirements):
    used = {d for r in requirements for d in r.depends_on}
    return [r for r in requirements if r.id not in used]


def _findings_block(requirements, register):
    lines = []
    for r in requirements:
        e = register.get(r.id)
        val = e.binding if (e and e.is_supported and e.binding) else "[NOT FOUND]"
        dep = f"  (uses {', '.join(r.depends_on)})" if r.depends_on else ""
        lines.append(f"  {r.id} {r.information_need} = {val}{dep}")
    return "\n".join(lines)


def _toks(s):
    return set(_TOK.findall((s or "").lower()))


def _winner(op, vals):
    """vals: list of (name, parsed). Returns the winning name or '' on a tie."""
    if op in ("earliest", "latest"):
        # compare at the precision every date shares
        prec = 3
        for _, d in vals:
            prec = min(prec, 1 + (1 if d[1] else 0) + (1 if d[1] and d[2] else 0))
        keyed = [(n, d[:prec]) for n, d in vals]
    else:
        keyed = list(vals)
    best = min if op in ("earliest", "smallest") else max
    target = best(k for _, k in keyed)
    hits = [n for n, k in keyed if k == target]
    return hits[0] if len(hits) == 1 else ""


def resolve_comparison(
    question: str,
    requirements: list[Requirement],
    register: dict[str, RegisterEntry],
    config: Config,
) -> tuple[str, int]:
    """-> (decided answer or '', llm_calls)."""
    leaves = [r for r in _leaves(requirements)
              if register.get(r.id) and register[r.id].is_supported and register[r.id].binding]
    if len(leaves) < 2:
        return "", 0
    user = f"Question: {question}\n\nFindings:\n{_findings_block(requirements, register)}"
    text, calls = call_llm([{"role": "system", "content": _SYSTEM},
                            {"role": "user", "content": user}],
                           config, json_mode=True, max_tokens=200)
    spec = parse_json_response(text) or {}
    if not spec.get("comparison"):
        return "", calls
    op = str(spec.get("operation", "")).lower()
    if op != "same":
        op = word_operation(str(spec.get("word", ""))) or op
    opts = [o for o in spec.get("options", []) if isinstance(o, dict) and o.get("name")]
    if op not in OPS or len(opts) < 2:
        return "", calls
    kind = classify(question)

    if op == "same":
        if kind != BOOLEAN or "same_value" not in spec:
            return "", calls
        ans = "Yes" if spec["same_value"] in (True, "true", "True", "yes") else "No"
        logger.info("Compare op same -> %s", ans)
        return ans, calls

    parse = parse_date if op in ("earliest", "latest") else parse_number
    vals = []
    for o in opts:
        e = register.get(str(o.get("step", "")))
        v = parse(e.binding) if (e and e.is_supported and e.binding) else None
        if v is None:
            logger.info("Compare op %s: unparsable value for %r", op, o.get("name"))
            return "", calls
        vals.append((str(o["name"]).strip(), v))
    win = _winner(op, vals)
    if not win:
        return "", calls
    if kind == BOOLEAN:
        ans = "Yes" if win == vals[0][0] else "No"
    else:
        ans = win
    logger.info("Compare op %s over %s -> %r", op, vals, ans)
    return ans, calls


# ── Choice constraint ────────────────────────────────────────────────────────
_OPTIONS_Q = re.compile(r"^\s*(?:which|who)\b[^,]*,\s*(.+)\s+or\s+(.+?)\s*\?*\s*$", re.I)


def question_options(question: str) -> list[str]:
    m = _OPTIONS_Q.match(question or "")
    return [m.group(1).strip(), m.group(2).strip()] if m else []


def snap_to_option(answer: str, question: str) -> str:
    """'A or B?' questions: return the option the answer names (its exact text), else ''."""
    opts = question_options(question)
    if not opts or classify(question) != CHOICE:
        return ""
    at = _toks(answer)
    scores = [len(at & _toks(o)) / max(1, len(_toks(o))) for o in opts]
    best = max(range(len(opts)), key=lambda i: scores[i])
    if scores[best] > 0 and scores.count(scores[best]) == 1:
        return opts[best]
    return ""
