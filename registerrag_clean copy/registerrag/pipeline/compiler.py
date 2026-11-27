"""
Query compiler: LLM call #1 per question.

Decomposes a natural-language question into an ordered flat list of
Requirement objects (DAG via depends_on annotations).

Guarantees
----------
- 2–6 requirements
- depends_on only references IDs earlier in the list
- JSON-parsed with a fallback single-requirement if parsing fails
"""
from __future__ import annotations
import json
import re
import logging

from registerrag.config import Config
from registerrag.llm_client import call_llm, parse_json_response
from registerrag.models import Requirement

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a question decomposer for a multi-hop QA system.
Break the question into an ordered list of atomic information requirements.

Rules:
- 2 to 6 requirements maximum
- Each requirement must be answerable as a short phrase (a name, date, number, fact)
- Use `depends_on` to reference earlier requirement IDs (creates a DAG, not a tree)
- Root requirements (no dependencies) come first
- Keep `information_need` concise — it will be used as a search query
- `expected_answer_type` must be one of:
  person | place | university | org | date | year | number | amount | fact | other
- When a question involves grandfather, grandmother, father-in-law, mother-in-law,
  or any two-generation relationship, use ONE requirement PER GENERATION:
  grandparent = parent of parent (paternal → father's parent, maternal → mother's
  parent); parent-in-law = parent of spouse; sibling-in-law = spouse of sibling
  or sibling of spouse. Do NOT add a final step that merely restates an earlier
  answer — the last requirement must itself find the answer.

Examples (shown compactly; your output must use the JSON format below):

Q: "Who was the paternal grandfather of Marie Curie?"
  R1: Who was the father of Marie Curie?        depends_on: []      type: person
  R2: Who was the father of [R1]?               depends_on: [R1]    type: person

Q: "What was the birthplace of Michael Jordan's father-in-law?"
  R1: Who is the spouse of Michael Jordan?      depends_on: []      type: person
  R2: Who is the father of [R1]?                depends_on: [R1]    type: person
  R3: Where was [R2] born?                      depends_on: [R2]    type: place

Return ONLY valid JSON in this exact format:
{
  "requirements": [
    {
      "id": "R1",
      "information_need": "...",
      "depends_on": [],
      "expected_answer_type": "person",
      "criticality": "required"
    },
    {
      "id": "R2",
      "information_need": "... (use the answer to R1 to narrow this search)",
      "depends_on": ["R1"],
      "expected_answer_type": "place",
      "criticality": "required"
    }
  ]
}"""


# Extra rules + examples for nested descriptions ("the X of the Y where Z"),
# enabled by config.planner_nested_rules_enabled.  Examples are invented, not
# taken from any evaluation set.
NESTED_RULES = """- Unfold nested descriptions from the INSIDE OUT. Every description of an
  entity that is not given by name ("the city where X was born", "the team that
  signed Y", "the country that controlled Z") must become its OWN requirement,
  and the next requirement refers to its answer with [Rk].
- Each requirement asks exactly ONE relation about ONE entity (a name from the
  question or an earlier [Rk]). Never put two relations in one requirement.
- Never repeat the whole question as a requirement.
"""

NESTED_EXAMPLES = """
Q: "What is the population of the city where the author of Dune was born?"
  R1: Who wrote Dune?                           depends_on: []      type: person
  R2: In which city was [R1] born?              depends_on: [R1]    type: place
  R3: What is the population of [R2]?           depends_on: [R2]    type: number

Q: "Which river flows through the capital of the country where the inventor of the telephone was born?"
  R1: Who invented the telephone?               depends_on: []      type: person
  R2: In which country was [R1] born?           depends_on: [R1]    type: place
  R3: What is the capital of [R2]?              depends_on: [R2]    type: place
  R4: Which river flows through [R3]?           depends_on: [R3]    type: place

Q: "When did the club that signed the 2003 Ballon d'Or winner last win the league?"
  R1: Who won the 2003 Ballon d'Or?             depends_on: []      type: person
  R2: Which club signed [R1]?                   depends_on: [R1]    type: org
  R3: When did [R2] last win the league?        depends_on: [R2]    type: date
"""


def _system_prompt(config: Config) -> str:
    if not getattr(config, "planner_nested_rules_enabled", False):
        return SYSTEM_PROMPT
    marker = "\nExamples (shown compactly"
    head, tail = SYSTEM_PROMPT.split(marker, 1)
    tail = tail.replace("JSON format below):\n", "JSON format below):\n" + NESTED_EXAMPLES, 1)
    return head.rstrip("\n") + "\n" + NESTED_RULES + marker + tail


def compile_requirements(
    question: str,
    config: Config,
) -> tuple[list[Requirement], int]:
    """
    Decompose `question` into Requirement objects.

    Returns
    -------
    (requirements, llm_calls)
    """
    if getattr(config, "kinship_templates_enabled", True):
        reqs = kinship_requirements(question) or lifespan_requirements(question)
        if reqs:
            logger.info("Kinship template: %s -> %s", question[:60],
                        [r.information_need for r in reqs])
            return reqs, 0

    messages = [
        {"role": "system", "content": _system_prompt(config)},
        {"role": "user", "content": f"Question: {question}"},
    ]

    text, calls = call_llm(messages, config, json_mode=True, max_tokens=512)
    parsed = parse_json_response(text)

    if parsed and "requirements" in parsed:
        reqs = _parse_requirements(parsed["requirements"])
        if reqs and getattr(config, "title_step_collapse_enabled", True):
            reqs = collapse_title_steps(reqs)
        if reqs:
            logger.debug("Compiled %d requirements for: %s", len(reqs), question[:60])
            return reqs, calls

    # Fallback: single requirement mirrors the original question
    logger.warning("Requirement parsing failed; using single-requirement fallback")
    fallback = [Requirement(
        id="R1",
        information_need=question,
        depends_on=[],
        expected_answer_type="fact",
        criticality="required",
    )]
    return fallback, calls


# Two-generation kinship relations → chain of one-generation relations
_KIN_CHAINS = {
    "paternal grandfather": ["father", "father"],
    "paternal grandmother": ["father", "mother"],
    "maternal grandfather": ["mother", "father"],
    "maternal grandmother": ["mother", "mother"],
    "grandfather": ["father", "father"],
    "grandmother": ["mother", "mother"],
    "father-in-law": ["spouse", "father"],
    "mother-in-law": ["spouse", "mother"],
    "sibling-in-law": ["spouse", "sibling"],
    "brother-in-law": ["spouse", "brother"],
    "sister-in-law": ["spouse", "sister"],
    "son-in-law": ["child", "spouse"],
    "daughter-in-law": ["child", "spouse"],
    "stepmother": ["father", "spouse"],
    "stepfather": ["mother", "spouse"],
}
_KIN_ALT = "|".join(sorted((re.escape(k) for k in _KIN_CHAINS), key=len, reverse=True))
_KIN_OF = re.compile(rf"^\s*who\s+(?:is|was)\s+the\s+({_KIN_ALT})\s+of\s+(.+?)\s*\?*\s*$", re.I)
_KIN_POSS = re.compile(rf"^\s*who\s+(?:is|was)\s+(.+?)['’]s?\s+({_KIN_ALT})\s*\?*\s*$", re.I)


def kinship_requirements(question: str) -> list[Requirement]:
    """
    Deterministic decomposition of "Who is the <kin> of X?" / "Who is X's <kin>?"
    for two-generation relations, e.g. maternal grandfather → R1 mother of X,
    R2 father of [R1]. Returns [] when the question doesn't match.
    """
    m = _KIN_OF.match(question)
    if m:
        kin, subject = m.group(1).lower(), m.group(2)
    else:
        m = _KIN_POSS.match(question)
        if not m:
            return []
        subject, kin = m.group(1), m.group(2).lower()
    subject = subject.strip().rstrip("'’")
    chain = _KIN_CHAINS[kin]
    reqs = []
    for i, rel in enumerate(chain):
        rid = f"R{i + 1}"
        target = subject if i == 0 else f"[R{i}]"
        reqs.append(Requirement(
            id=rid,
            information_need=f"Who is the {rel} of {target}?",
            depends_on=[] if i == 0 else [f"R{i}"],
            expected_answer_type="person",
            criticality="required",
        ))
    return reqs


_LIFESPAN_Q = re.compile(
    r"^\s*who\s+(?:lived|had\s+lived|lives)\s+longer\s*,\s*(.+?)\s+or\s+(.+?)\s*\?*\s*$", re.I)


def lifespan_requirements(question: str) -> list[Requirement]:
    """
    "Who lived longer, A or B?" needs both birth and death dates of each
    person (death dates alone don't decide it): R1/R2 born/died A, R3/R4 born/died B.
    """
    m = _LIFESPAN_Q.match(question)
    if not m:
        return []
    reqs = []
    for i, person in enumerate((m.group(1).strip(), m.group(2).strip())):
        for j, (verb, need) in enumerate((("born", f"When was {person} born?"),
                                          ("died", f"When did {person} die?"))):
            reqs.append(Requirement(
                id=f"R{2 * i + j + 1}", information_need=need, depends_on=[],
                expected_answer_type="date", criticality="required",
            ))
    return reqs


_TITLE_STEP = re.compile(
    r"^\s*what\s+is\s+the\s+(?:full\s+|exact\s+)?(?:title|name)\s+of\s+(?:the\s+)?"
    r"(?:(?:film|movie|song|album|book|novel|show|series)\s+)?(.+?)\s*\?*\s*$", re.I)


def collapse_title_steps(reqs: list[Requirement]) -> list[Requirement]:
    """
    Remove root steps that only restate an entity the question already names
    ("What is the title of the film Love At Second Sight (1999 Film)?") and
    write that entity into the steps that depended on them.
    """
    drop: dict[str, str] = {}
    for r in reqs:
        m = _TITLE_STEP.match(r.information_need)
        if m and not r.depends_on and any(r.id in o.depends_on for o in reqs):
            drop[r.id] = m.group(1).strip()
    if not drop or len(drop) == len(reqs):
        return reqs
    out = []
    for r in reqs:
        if r.id in drop:
            continue
        need = r.information_need
        for rid, ent in drop.items():
            need = need.replace(f"[{rid}]", ent)
        out.append(Requirement(
            id=r.id, information_need=need,
            depends_on=[d for d in r.depends_on if d not in drop],
            expected_answer_type=r.expected_answer_type, criticality=r.criticality,
        ))
    logger.info("Collapsed title step(s) %s -> %s", list(drop), [r.information_need for r in out])
    return out


def _parse_requirements(raw: list[dict]) -> list[Requirement]:
    reqs = []
    valid_ids: set[str] = set()

    for item in raw:
        try:
            rid = str(item["id"])
            info_need = str(item["information_need"]).strip()
            depends_on = [str(d) for d in item.get("depends_on", [])
                          if str(d) in valid_ids]
            answer_type = str(item.get("expected_answer_type", "fact"))
            criticality = str(item.get("criticality", "required"))

            if not info_need:
                continue

            reqs.append(Requirement(
                id=rid,
                information_need=info_need,
                depends_on=depends_on,
                expected_answer_type=answer_type,
                criticality=criticality,
            ))
            valid_ids.add(rid)
        except (KeyError, TypeError) as exc:
            logger.warning("Skipping malformed requirement: %s — %s", item, exc)

    return reqs[:6]   # hard cap


def topological_order(requirements: list[Requirement]) -> list[Requirement]:
    """
    Return requirements in an order safe for sequential processing:
    all depends_on entries appear before the requirement that depends on them.
    """
    id_to_req = {r.id: r for r in requirements}
    order: list[Requirement] = []
    visited: set[str] = set()

    def visit(rid: str) -> None:
        if rid in visited:
            return
        visited.add(rid)
        req = id_to_req.get(rid)
        if req is None:
            return
        for dep in req.depends_on:
            visit(dep)
        order.append(req)

    for r in requirements:
        visit(r.id)

    return order
