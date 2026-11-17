"""Export draft gold requirement programs (Step 5). Output is a DRAFT: `needs_review` stays true until a human
has read the question and corrected requirement text / dependencies / gold evidence."""
from __future__ import annotations

import json


def export(examples, path):
    out = []
    for ex in examples:
        out.append({"qid": ex.qid, "dataset": ex.dataset, "question": ex.question, "answers": ex.answers,
                    "n_hops": ex.n_hops, "needs_review": True,
                    "requirements": [r.to_dict() for r in ex.gold_requirements]})
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
    heur = sum(1 for e in examples for r in e.gold_requirements if r.heuristic)
    nogold = sum(1 for e in examples for r in e.gold_requirements if not r.gold_doc_ids)
    return {"questions": len(out), "requirements": sum(len(o["requirements"]) for o in out),
            "heuristic_requirements": heur, "requirements_without_gold_doc": nogold}
