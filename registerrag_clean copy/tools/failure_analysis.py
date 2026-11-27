"""
Splits each failure class of a finished run into measurable sub-causes.

Usage (from registerrag_clean/):
  python tools/failure_analysis.py RUN_DIR DATASET QUESTIONS_JSON INDEX_DIR [DSIRE_CORPUS_DIR]
Needs RUN_DIR/failures.csv (tools/report.py).  Writes RUN_DIR/failure_subcauses.csv
and RUN_DIR/failure_subcauses.json.

Sub-causes (wrong answers only; same top-level classes as report.py)
  planner: fewer steps than gold evidence pieces (Wikipedia sets: one step per
           supporting article/paragraph is the minimum chain).  Reported as a
           flag on every failure, since it can cause any of the classes below.
  retrieval   first hop missed      no gold evidence in the first step's pool
              later hop (cascade)   first step found gold, a later piece is missing
  reranking   (as is)
  step not found
              final answer guessed  a step is NOT FOUND, the answer is still a value
              final answer refused  the answer itself says not found / not provided
  wrong value
              gold bound, lost      some step's binding matches the gold answer (F1>=0.5),
                                    the final answer does not
              partial / granularity answer overlaps the gold (0 < F1 < 0.5)
              wrong entity / value  no overlap at all (F1 = 0)
  yes/no flip (flag)  gold yes/no, the answer is the opposite; "with missing step" if
                      some step was NOT FOUND
"""
from __future__ import annotations
import csv
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import ROOT, load_questions, load_units, gold_docs, doc_hit, parse_trace  # noqa: E402
from registerrag.evaluation.metrics import token_f1  # noqa: E402

REFUSAL = re.compile(r"not\s+found|not\s+(?:provided|specified|mentioned|available)|does not (?:provide|mention|specify)|no (?:specific )?information", re.I)


def yn(s: str) -> str:
    m = re.match(r"\s*(yes|no)\b", s or "", re.I)
    return m.group(1).lower() if m else ""


def main():
    run, ds, qpath, index_dir = sys.argv[1:5]
    qs = load_questions(ds, qpath, sys.argv[5] if len(sys.argv) > 5 else None)
    units = load_units(index_dir)
    res = {json.loads(l)["question_id"]: json.loads(l)
           for l in open(os.path.join(run, "results.jsonl"), encoding="utf-8") if l.strip()}
    rows, sub, flags = [], Counter(), Counter()
    for r in csv.DictReader(open(os.path.join(run, "failures.csv"), encoding="utf-8-sig")):
        cat = r["category"]
        if cat in ("correct", "form"):
            continue
        q, rr = qs.get(r["id"], {}), res[r["id"]]
        steps = parse_trace(os.path.join(run, "traces", os.path.basename(rr["trace_path"])))
        golds = gold_docs(q)
        pred, gold = rr["predicted_answer"], rr["gold_answer"]
        f1 = float(r["f1"])
        if cat == "retrieval":
            first = [units.get(u) for u in (steps[0]["pool"] if steps else [])]
            s = ("first hop missed" if ds != "repolicyqa" and not any(doc_hit(g, u) for g in golds for u in first)
                 else "later hop (cascade)" if ds != "repolicyqa" else "gold snippet not retrieved")
        elif cat == "reranking":
            s = "retrieved, cut from top-10"
        elif cat == "step_not_found":
            s = "final answer refused" if REFUSAL.search(pred) else "final answer guessed"
        else:
            if any(st["binding"] and token_f1(st["binding"], gold) >= 0.5 for st in steps):
                s = "gold bound, lost"
            elif f1 > 0:
                s = "partial / granularity"
            else:
                s = "wrong entity / value"
        sub[f"{cat}: {s}"] += 1
        under = ds != "repolicyqa" and len(steps) < len(golds)
        flip = yn(gold) and yn(pred) and yn(gold) != yn(pred)
        missing = any(not st["binding"] for st in steps)
        if under:
            flags["planner: fewer steps than gold evidence"] += 1
        if flip:
            flags["yes/no flip" + (" (with missing step)" if missing else "")] += 1
        if flip and yn(gold) == "yes":
            flags["yes/no flip: gold yes -> no"] += 1
        rows.append({"id": r["id"], "type": r["type"], "category": cat, "subcause": s,
                     "steps": len(steps), "gold_pieces": len(golds), "planner_under": int(under),
                     "yesno_flip": int(bool(flip)), "f1": f1, "question": rr["question"],
                     "gold": gold, "predicted": pred,
                     "bindings": " | ".join(st["binding"] or "[NOT FOUND]" for st in steps)})
    with open(os.path.join(run, "failure_subcauses.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = {"failures": len(rows), "subcauses": dict(sub), "flags": dict(flags),
           "by_type": {t: dict(Counter(x["category"] for x in rows if x["type"] == t))
                       for t in sorted({x["type"] for x in rows})}}
    json.dump(out, open(os.path.join(run, "failure_subcauses.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
