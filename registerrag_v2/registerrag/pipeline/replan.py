"""
Re-plan on failure (dataset-agnostic).

After retrieval + gap fill, a step that is still NOT FOUND (and whose inputs
are known) is rewritten by one LLM call into 1-2 simpler sub-steps, given the
concrete bindings it builds on and the titles of the passages that did not
answer it.  Typical rewrites: a two-relation step split in two ("the city
where X was born's mayor" -> where was X born / who is mayor of [S1]); a step
whose wording doesn't match the corpus rephrased ("founder of" -> "who
established"); an unnamed entity made explicit.

The sub-steps run through the normal retrieve -> rerank -> bind path; if the
last one is supported, its binding becomes the original step's binding.
"""
from __future__ import annotations
import logging
import re

from registerrag.config import Config
from registerrag.llm_client import call_llm, parse_json_response
from registerrag.models import Requirement

logger = logging.getLogger(__name__)

_TYPES = {"person", "place", "university", "org", "date", "year", "number", "amount", "fact", "other"}

_SYSTEM = """A question-answering system broke a question into steps. One step could not be
answered from the passages it retrieved. Rewrite THAT STEP so it can be found.

Options:
- Split it into two simpler steps when it asks two things at once or describes an
  entity without naming it (find the entity first, then ask about it).
- Or rephrase it as ONE step in plainer words the source text is likely to use.

Rules:
- At most 2 steps. Step 2 may refer to the answer of step 1 as [S1].
- Use the concrete names given as known facts; never write placeholders like [R1].
- The last step must ask for exactly what the failed step asked for.
- "type": person | place | org | date | year | number | amount | fact | other

Examples (invented):
Failed step: "Who was the mayor of the city where Ana Ruiz was born?"
  -> {"steps": [{"need": "Where was Ana Ruiz born?", "type": "place"},
                {"need": "Who was the mayor of [S1]?", "type": "person"}]}
Failed step: "Who is the founder of Lumen Press?"
  -> {"steps": [{"need": "Who established the publishing house Lumen Press?", "type": "person"}]}

Return ONLY JSON: {"steps": [{"need": "...", "type": "..."}]}"""


def _concrete(need: str, register, req: Requirement) -> str:
    """Write known dependency bindings into the step text ([R1] -> its binding)."""
    out = need
    for d in req.depends_on:
        e = register.get(d)
        if e and e.binding:
            out = out.replace(f"[{d}]", e.binding)
    return out


def replan_step(
    question: str,
    req: Requirement,
    register: dict,
    passage_titles: list[str],
    config: Config,
) -> tuple[list[Requirement], int]:
    """-> (new sub-requirements, llm_calls). [] when the rewrite is unusable."""
    known = [f"  {d}: {register[d].binding}" for d in req.depends_on
             if d in register and register[d].binding]
    titles = ", ".join(dict.fromkeys(t for t in passage_titles if t))[:400] or "(none)"
    user = (f"Question: {question}\n"
            f"Failed step: \"{_concrete(req.information_need, register, req)}\"\n"
            f"Known facts:\n{chr(10).join(known) if known else '  (none)'}\n"
            f"Passages retrieved for it came from: {titles}")
    text, calls = call_llm([{"role": "system", "content": _SYSTEM},
                            {"role": "user", "content": user}],
                           config, json_mode=True, max_tokens=200)
    spec = parse_json_response(text) or {}
    steps = [s for s in spec.get("steps", []) if isinstance(s, dict) and str(s.get("need", "")).strip()]
    steps = steps[: max(1, getattr(config, "replan_max_steps", 2))]
    if not steps:
        return [], calls
    old = " ".join(_concrete(req.information_need, register, req).lower().split())
    if len(steps) == 1 and " ".join(steps[0]["need"].lower().split()) == old:
        return [], calls   # unchanged rewrite: nothing to gain
    out: list[Requirement] = []
    for i, s in enumerate(steps):
        rid = f"{req.id}s{i + 1}"
        need = re.sub(r"\[S(\d)\]", lambda m: f"[{req.id}s{m.group(1)}]", str(s["need"]).strip())
        deps = list(req.depends_on) + ([out[-1].id] if out else [])
        t = str(s.get("type", req.expected_answer_type)).lower()
        out.append(Requirement(
            id=rid, information_need=need, depends_on=deps,
            expected_answer_type=t if t in _TYPES else req.expected_answer_type,
            criticality=req.criticality,
        ))
    logger.info("Replan %s %r -> %s", req.id, req.information_need, [r.information_need for r in out])
    return out, calls
