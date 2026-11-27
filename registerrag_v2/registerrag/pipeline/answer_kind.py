"""
Answer-kind router (dataset-agnostic).

Classifies a question — or a requirement's information need — by the form of
answer it asks for, using only English question syntax:

  boolean      "Is/Does/Can/Will … ?", also after a leading context clause:
               "As a residential customer in Colorado, does X offer …?",
               "If I install …, can I …?", "I am a contractor. Will the program …?"
  choice       "Which/Who …, A or B?"  (pick one of the named options)
  description  asks for requirements / conditions / steps / an explanation:
               "What are the eligibility requirements …", "How does … work?",
               "What must a developer provide …", "Describe …"
  fact         everything else: who / when / where / how much / how many /
               "What is the <amount|limit|date|name> …"

Each kind sets
  binding_style  "span"   short answer phrase (≤12 words)           fact, choice
                 "clause" verbatim clause from the evidence (≤45)   boolean, description
  answer_format  the instruction for the generator's Answer line
  vote           whether self-consistency voting applies (closed answers)
"""
from __future__ import annotations
import re

BOOLEAN, CHOICE, DESCRIPTION, FACT = "boolean", "choice", "description", "fact"

_AUX = (r"(?:am|is|are|was|were|do|does|did|can|could|will|would|shall|should|may|might|"
        r"must|has|have|had)")
# An auxiliary that opens the question proper: at the start, or right after a
# context clause (", " or ". ") — followed by a subject-like word.
_BOOL = re.compile(
    rf"(?:^|[,;:]\s+|[.!?]\s+)(?:then\s+|so\s+|and\s+)?{_AUX}(?:n't)?\s+(?!\s*$)", re.I)

_POLITE_DESC = re.compile(
    r"^\s*(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:explain|describe|outline|tell|list|summari[sz]e|detail|walk)\b", re.I)
_OPTIONS = re.compile(r"^\s*(?:which|who)\b[^,]*,\s*(.+)\s+or\s+(.+?)\s*\?*\s*$", re.I)

_DESC_NOUNS = (
    r"requirements?|conditions?|criteria|steps?|process|procedures?|documents?|documentation|"
    r"standards?|rules?|obligations?|responsibilit(?:y|ies)|terms|provisions?|guidelines?|"
    r"restrictions?|limitations?|options|benefits|types|kinds|categories|components|elements|"
    r"factors|measures|actions|features|details|impact|role|purpose|goals?|objectives?|status|"
    r"information|types?|kinds?|services|activities|changes|implications"
)
# Start of the question proper: string start or after a context clause.
_Q0 = r"(?:^|[,;:.]\s+)"
# "what <0-2 words> <plural noun> are/do/can/…" asks for a list
_PLURAL_ASK = (r"\bwhat\s+(?:\w+\s+){0,2}?\w{3,}(?<!ss)(?<!us)(?<!is)s\s+"
               r"(?:are|were|do|does|did|can|could|should|must|will|would|have|has|is|need)\b")
_DESC = re.compile(
    rf"\b(?:describe|explain|outline|summari[sz]e|list)\b"
    rf"|{_Q0}how\s+(?:does|do|did|is|are|can|could|will|would|should|must|might)\b"
    rf"|{_Q0}why\b"
    rf"|{_Q0}what\s+(?:are|were)\b"
    rf"|{_PLURAL_ASK}"
    rf"|\bwhat\s+(?:\w+\s+){{0,3}}(?:{_DESC_NOUNS})\b"
    rf"|\bwhat\s+(?:must|should|needs?\s+to|has\s+to|have\s+to)\b"
    rf"|\bwhat\s+(?:happens|is\s+required|is\s+needed|is\s+involved)\b"
    rf"|\bwhat\s+(?:does|do|did)\s+.+?\s+(?:require|include|entail|cover|mean|involve|allow|provide|offer)\b",
    re.I)


def classify(text: str) -> str:
    """Return boolean | choice | description | fact for a question or information need."""
    t = " ".join((text or "").split())
    if not t:
        return FACT
    if _OPTIONS.match(t):
        return CHOICE
    # The sentence holding the question ("I am a contractor in RI. Will the program …?")
    last = re.split(r"(?<=[.!?])\s+(?=[A-Z])", t)[-1]
    # Boolean iff an auxiliary opens the sentence or opens a clause after a
    # context clause — "What is …" never matches (the auxiliary follows "What").
    if _POLITE_DESC.match(last):              # "Can you explain/describe …?"
        return DESCRIPTION
    if _BOOL.match(t) or _BOOL.search(last):
        return BOOLEAN
    if _DESC.search(last):
        return DESCRIPTION
    return FACT


def binding_style(kind: str) -> str:
    return "clause" if kind in (BOOLEAN, DESCRIPTION) else "span"


def votes(kind: str) -> bool:
    """Closed answers (yes/no, one of the options) are voted on under self-consistency."""
    return kind in (BOOLEAN, CHOICE)


ANSWER_FORMAT = {
    BOOLEAN: (
        "Answer: Yes or No — the first word must be Yes or No, followed by one short reason. "
        "Answer No only if the evidence contradicts it or rules it out; a detail the "
        "evidence does not mention is not a reason for No."
    ),
    DESCRIPTION: (
        "Answer: 1–3 sentences that directly answer the question with the specific facts "
        "from the evidence (amounts, eligibility, conditions, requirements, deadlines). "
        "Use only the evidence; do not add anything it does not say."
    ),
    CHOICE: "Answer: exactly one of the options named in the question.",
    # used instead of DESCRIPTION when config.description_complete_enabled
    "description_complete": (
        "Answer: a complete answer in 1–4 sentences: every requirement, condition, amount, "
        "deadline and exception the evidence states that bears on the question, and nothing "
        "the evidence does not say."
    ),
    FACT: "Answer: the final answer only — a name, number, date or short phrase.",
}
