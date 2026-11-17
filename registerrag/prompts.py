PROMPT_VERSION = "v1"   # default; select with --prompt

QA_SYSTEMS = {
    # v1: model may abstain
    "v1": ("Answer the question using only the provided evidence. "
           "Give the shortest possible answer (an entity, date, number, or yes/no) with no explanation. "
           "If the evidence is insufficient, answer exactly: unanswerable."),
    # v2: forced answer (standard baseline behaviour; abstention is evaluated separately for RegisterRAG)
    "v2": ("Answer the question using the provided evidence. "
           "Give the shortest possible answer (an entity, date, number, or yes/no) with no explanation. "
           "Always give your best answer, even if the evidence seems incomplete."),
}
SYSTEM_QA = ("Answer the question using only the provided evidence. "
             "Give the shortest possible answer (an entity, date, number, or yes/no) with no explanation. "
             "If the evidence is insufficient, answer exactly: unanswerable.")

SYSTEM_DECOMPOSE = (
    "Decompose the question into the minimal list of information requirements needed to answer it. "
    "Each requirement is one plain-English information need. If a requirement needs the result of an "
    "earlier one, list that id in depends_on. Use 2 to 4 requirements. "
    'Reply with JSON only: {"requirements":[{"id":"R1","need":"...","depends_on":[]}]}')


def qa_prompt(question: str, evidence_lines: list) -> str:
    ev = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(evidence_lines)) or "(no evidence)"
    return f"Evidence:\n{ev}\n\nQuestion: {question}\nAnswer:"
