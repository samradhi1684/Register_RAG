"""
One report for a finished run: answer quality, retrieval, cost, and where the
pipeline fails.

Usage (from registerrag_clean/):
  python tools/report.py RUN_DIR DATASET QUESTIONS_JSON INDEX_DIR [DSIRE_CORPUS_DIR]
  e.g. python tools/report.py runs/run1/2wiki/test100 2wiki \
          registerrag/qa-pairs/2wiki/2wiki_100.json runs/run1/2wiki/index
DATASET: 2wiki | hotpotqa | musique | repolicyqa

Needs RUN_DIR/results.jsonl, score_rows.jsonl (run.py score) and, optionally,
judge_rows.jsonl (tools/judge.py).  Writes RUN_DIR/report.md, RUN_DIR/failures.csv.

Definitions
-----------
Gold evidence: 2Wiki/HotpotQA = supporting-fact article titles; MuSiQue = the
  is_supporting paragraphs; RePolicyQA = the gold source document(s) for recall,
  and the gold context snippet for the failure classes: found when the units
  (taken together) cover >= 50% of its content words; with sentence units a
  snippet usually spans several units.
Doc recall@10: per question, the share of gold evidence found among the top-10
  passages (the SUPPORT units the LLM reads) of any reasoning step; averaged.
Pool recall: the same over every retrieved candidate (before the top-10 cut).
Wrong answer: F1 < 0.5 and not judged fully correct by the LLM judge.
Failure classes (wrong answers only):
  retrieval  - some gold evidence is in no step's retrieved pool
  reranking  - all gold evidence was retrieved, some never reached a step's top-10
  generation - all gold evidence reached a top-10, the answer is still wrong:
                 step not found = some step's binding ended NOT FOUND
                 wrong value    = every step bound a value, the value/answer is wrong
  answer form only - F1 < 0.5 but judged correct (right fact, other wording); not a failure
Note: a later step's "retrieval" failure can be caused by a wrong earlier binding
(the next query searches for the wrong entity).
"""
from __future__ import annotations
import csv
import json
import os
import pickle
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from registerrag.evaluation.scoring import evidence_recall  # noqa: E402

LOADER_FORMAT = {"2wiki": "2wiki", "hotpotqa": "hotpotqa", "musique": "musique", "repolicyqa": "repolicyqa"}


def load_questions(ds, path, corpus_dir=None):
    from registerrag import data_loaders as dl
    if ds == "2wiki":
        qs, _ = dl.load_2wiki(path)
    elif ds == "hotpotqa":
        qs, _ = dl.load_hotpotqa(path)
    elif ds == "musique":
        qs, _ = dl.load_musique(path)
    else:
        qs, _ = dl.load_repolicyqa(path, corpus_dir or os.path.join(ROOT, "registerrag", "data", "final"))
    return {q["id"]: q for q in qs}


def load_units(index_dir):
    u = pickle.load(open(os.path.join(index_dir, "units.pkl"), "rb"))
    u = list(u.values()) if isinstance(u, dict) else u
    out = {}
    for x in u:
        x = x if isinstance(x, dict) else vars(x)
        out[x["unit_id"]] = x
    return out


def gold_docs(q):
    """-> list of gold keys: ('title', t) or ('docs', {ids})."""
    out = []
    for sf in q.get("supporting_facts", []):
        if isinstance(sf, dict):
            out.append(("docs", frozenset(sf.get("doc_ids", []))))
        else:
            out.append(("title", sf[0]))
    return list(dict.fromkeys(out))


def doc_hit(g, unit):
    if unit is None:
        return False
    kind, v = g
    return unit["document_title"] == v if kind == "title" else unit["document_id"] in v


def parse_trace(path):
    """-> list of steps: dict(top10, pool, binding)."""
    t = open(path, encoding="utf-8", errors="ignore").read()
    end = t.find("\n## Working Register")
    t = t[: end if end > 0 else len(t)]
    steps = []
    for m in re.finditer(r"^### (R\w+): .*$", t, re.M):
        nxt = t.find("\n### ", m.end())
        block = t[m.end(): nxt if nxt > 0 else len(t)]
        top = re.findall(r"^\|\s*\d+\s*\|\s*`([^`]+)`", block, re.M)
        pm = re.search(r"\*\*Pool beyond top-10:\*\*(.*)$", block, re.M)
        pool = re.findall(r"`([^`]+)`", pm.group(1)) if pm else []
        b = re.search(r"\*\*Binding extracted:\*\*\s*`([^`]*)`", block)
        val = b.group(1).strip() if b else ""
        if "NOT FOUND" in val.upper() or len(val) <= 1:
            val = ""
        steps.append({"top10": top, "pool": top + pool, "binding": val})
    return steps


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def f(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def main():
    run, ds, qpath, index_dir = sys.argv[1:5]
    qs = load_questions(ds, qpath, sys.argv[5] if len(sys.argv) > 5 else None)
    units = load_units(index_dir)
    res = [json.loads(l) for l in open(os.path.join(run, "results.jsonl"), encoding="utf-8") if l.strip()]
    sc = {r["id"]: r for r in map(json.loads, open(os.path.join(run, "score_rows.jsonl"), encoding="utf-8"))}
    jp = os.path.join(run, "judge_rows.jsonl")
    judge = {r["id"]: r["judge"] for r in map(json.loads, open(jp, encoding="utf-8"))} if os.path.exists(jp) else {}

    rows, cls = [], {"correct": 0, "form": 0, "retrieval": 0, "reranking": 0, "step_not_found": 0, "wrong_value": 0}
    for r in res:
        qid = r["question_id"]
        q, s = qs.get(qid, {}), sc[qid]
        steps = parse_trace(os.path.join(run, "traces", os.path.basename(r["trace_path"])))
        golds = gold_docs(q)
        top = [units.get(u) for st in steps for u in st["top10"]]
        pool = [units.get(u) for st in steps for u in st["pool"]]
        rec10 = mean([1.0 if any(doc_hit(g, u) for u in top) else 0.0 for g in golds]) if golds else None
        recpool = mean([1.0 if any(doc_hit(g, u) for u in pool) else 0.0 for g in golds]) if golds else None

        # failure class
        if s["f1"] >= 0.5:
            cat = "correct"
        elif judge.get(qid) == 1.0:
            cat = "form"
        else:
            if ds == "repolicyqa" and q.get("gold_context"):
                gc = q["gold_context"]
                # sentence units: the snippet usually spans several units, so coverage is joint
                in_pool = evidence_recall(gc, [u["text"] for u in pool if u]) >= 0.5
                in_top = evidence_recall(gc, [u["text"] for u in top if u]) >= 0.5
            else:
                in_pool = all(any(doc_hit(g, u) for u in pool) for g in golds)
                in_top = all(any(doc_hit(g, u) for u in top) for g in golds)
            if not in_pool:
                cat = "retrieval"
            elif not in_top:
                cat = "reranking"
            elif any(not st["binding"] for st in steps):
                cat = "step_not_found"
            else:
                cat = "wrong_value"
        cls[cat] += 1
        rows.append({"id": qid, "type": s.get("type", ""), "em": s["em"], "f1": round(s["f1"], 3),
                     "judge": judge.get(qid, ""), "yesno_acc": s.get("yesno_acc"),
                     "recall_at_10": rec10, "pool_recall": recpool, "category": cat,
                     "steps": len(steps), "latency_s": round(r.get("latency_s", 0), 2),
                     "llm_calls": r.get("llm_calls"), "llm_requests": r.get("llm_requests"),
                     "prompt_tokens": r.get("prompt_tokens"), "completion_tokens": r.get("completion_tokens"),
                     "total_tokens": r.get("total_tokens"),
                     "question": r["question"], "gold": r["gold_answer"], "predicted": r["predicted_answer"]})

    with open(os.path.join(run, "failures.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    lat = sorted(x["latency_s"] for x in rows)
    tok = [x["total_tokens"] or 0 for x in rows]
    yn = [x["yesno_acc"] for x in rows if x["yesno_acc"] is not None]
    jv = [x["judge"] for x in rows if x["judge"] != ""]
    out = [f"# Report: {run} ({ds}, n={n})", "",
           "## Answer quality and retrieval", "",
           "| n | EM | F1 | LLM judge | Yes/no acc | Doc recall@10 | Pool recall |",
           "|---|---|---|---|---|---|---|",
           f"| {n} | {f(mean(x['em'] for x in rows))} | {f(mean(x['f1'] for x in rows))} | "
           f"{f(mean(jv)) if jv else '–'} | {f(mean(yn)) + f' (n={len(yn)})' if yn else '–'} | "
           f"{f(mean(x['recall_at_10'] for x in rows))} | {f(mean(x['pool_recall'] for x in rows))} |", "",
           "## Cost", "",
           "| Latency mean (s) | Latency p95 (s) | Total time (min) | LLM calls / q (pipeline count) | LLM requests / q (measured) | Prompt tokens / q | Completion tokens / q | Total tokens / q | Total tokens (run) |",
           "|---|---|---|---|---|---|---|---|---|",
           f"| {f(mean(lat), 1)} | {f(lat[int(0.95 * (n - 1))], 1)} | {f(sum(lat) / 60, 1)} | "
           f"{f(mean(x['llm_calls'] for x in rows), 1)} | {f(mean(x['llm_requests'] for x in rows), 1)} | "
           f"{f(mean(x['prompt_tokens'] for x in rows), 0)} | {f(mean(x['completion_tokens'] for x in rows), 0)} | "
           f"{f(mean(tok), 0)} | {sum(tok):,} |", "",
           "## By question type", "",
           "| Type | n | EM | F1 | Judge | Doc recall@10 |", "|---|---|---|---|---|---|"]
    for t in sorted({x["type"] for x in rows}):
        g = [x for x in rows if x["type"] == t]
        gj = [x["judge"] for x in g if x["judge"] != ""]
        out.append(f"| {t} | {len(g)} | {f(mean(x['em'] for x in g))} | {f(mean(x['f1'] for x in g))} | "
                   f"{f(mean(gj)) if gj else '–'} | {f(mean(x['recall_at_10'] for x in g))} |")
    fail = cls["retrieval"] + cls["reranking"] + cls["step_not_found"] + cls["wrong_value"]
    pc = lambda a: f"{a} ({100 * a / fail:.0f}%)" if fail else "0"
    out += ["", "## Where it fails (wrong answers only)", "",
            "| Correct (F1≥0.5) | Answer form only* | **Failures** | Retrieval | Reranking | Generation: step not found | Generation: wrong value |",
            "|---|---|---|---|---|---|---|",
            f"| {cls['correct']} | {cls['form']} | **{fail}** | {pc(cls['retrieval'])} | {pc(cls['reranking'])} | "
            f"{pc(cls['step_not_found'])} | {pc(cls['wrong_value'])} |", "",
            "*F1 < 0.5 but judged correct (right fact, other wording); not counted as a failure.",
            "Per-question rows: failures.csv"]
    text = "\n".join(out)
    open(os.path.join(run, "report.md"), "w", encoding="utf-8").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
