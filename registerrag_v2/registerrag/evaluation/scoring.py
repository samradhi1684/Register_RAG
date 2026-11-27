"""
Post-hoc scoring of a finished run, the same for every dataset.

Answer metrics (per question; max over gold answer + answer_aliases when given)
  em, f1      SQuAD-style exact match / token F1
  rouge_l     LCS-based F-measure over normalised tokens (abstractive answers)
  yesno_acc   only when the gold answer starts with yes/no: does the
              prediction's leading yes/no agree?  (A prediction with no
              leading yes/no counts as wrong.)
Evidence metrics
  doc_recall      copied from the run (gold document reached by retrieval)
  evidence_recall when the question carries a gold context snippet: share of
                  the snippet's content tokens found in the proof units;
                  evidence_hit = evidence_recall ≥ 0.5

Results are reported overall and per question type, and written to
<run>/score.json.  For yes/no groups the majority-class rate is shown, so
accuracy can be read against the "always answer Yes" baseline.
"""
from __future__ import annotations
import json
import os
import pickle
import re
from collections import Counter, defaultdict

from registerrag.evaluation.metrics import exact_match, normalize_answer, token_f1

_YN = re.compile(r"^\W*(yes|no)\b", re.I)
_STOP = frozenset("a an the of to in for and or is are be on by with as at from that this it its".split())


def yes_no(text: str):
    m = _YN.match(text or "")
    return m.group(1).lower() if m else None


def rouge_l(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    prev = [0] * (len(g) + 1)
    for pt in p:
        cur = [0]
        for j, gt in enumerate(g):
            cur.append(prev[j] + 1 if pt == gt else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    r, q = lcs / len(g), lcs / len(p)
    return round(2 * r * q / (r + q), 4)


def _content_tokens(text: str) -> list[str]:
    return [t for t in normalize_answer(text).split() if t not in _STOP]


def evidence_recall(gold_context: str, proof_texts: list[str]) -> float:
    gold = _content_tokens(gold_context)
    if not gold:
        return 0.0
    have = Counter(t for s in proof_texts for t in _content_tokens(s))
    need = Counter(gold)
    return round(sum(min(c, have[t]) for t, c in need.items()) / len(gold), 4)


def _load_unit_texts(index_dir: str) -> dict[str, str]:
    path = os.path.join(index_dir, "units.pkl")
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        units = pickle.load(f)
    out = {}
    for u in units:
        d = u if isinstance(u, dict) else u.__dict__
        out[d["unit_id"]] = d["text"]
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def score_run(run_dir: str, questions: list[dict], index_dir: str) -> dict:
    qmap = {q["id"]: q for q in questions}
    with open(os.path.join(run_dir, "results.jsonl"), encoding="utf-8") as f:
        results = [json.loads(l) for l in f if l.strip()]
    need_texts = any(qmap.get(r["question_id"], {}).get("gold_context") for r in results)
    texts = _load_unit_texts(index_dir) if need_texts else {}

    rows = []
    for r in results:
        q = qmap.get(r["question_id"], {})
        gold, pred = r.get("gold_answer", q.get("answer", "")), r.get("predicted_answer", "") or ""
        g_yn = yes_no(gold)
        golds = [gold] + [a for a in q.get("answer_aliases", []) if a]   # MuSiQue: max over aliases
        row = {
            "id": r["question_id"],
            "type": q.get("type", "") or "all",
            "em": max(exact_match(pred, g) for g in golds),
            "f1": max(token_f1(pred, g) for g in golds),
            "rouge_l": max(rouge_l(pred, g) for g in golds),
            "yesno_gold": g_yn,
            "yesno_acc": (1.0 if yes_no(pred) == g_yn else 0.0) if g_yn else None,
            "doc_recall": r.get("doc_recall"),
            "evidence_recall": None,
            "answer_evidence_recall": None,
        }
        if q.get("gold_context") and texts and "proof_unit_ids" in r:
            row["evidence_recall"] = evidence_recall(
                q["gold_context"], [texts.get(u, "") for u in r["proof_unit_ids"]])
        if q.get("gold_context") and texts and r.get("answer_unit_ids"):
            row["answer_evidence_recall"] = evidence_recall(
                q["gold_context"], [texts.get(u, "") for u in r["answer_unit_ids"]])
        rows.append(row)

    def agg(rs):
        yn = [x for x in rs if x["yesno_gold"]]
        maj = Counter(x["yesno_gold"] for x in yn).most_common(1)
        ev = [x["evidence_recall"] for x in rs if x["evidence_recall"] is not None]
        aev = [x["answer_evidence_recall"] for x in rs if x.get("answer_evidence_recall") is not None]
        return {
            "n": len(rs),
            "em": _mean(x["em"] for x in rs),
            "f1": _mean(x["f1"] for x in rs),
            "rouge_l": _mean(x["rouge_l"] for x in rs),
            "yesno_n": len(yn),
            "yesno_acc": _mean(x["yesno_acc"] for x in yn),
            "yesno_majority_rate": round(maj[0][1] / len(yn), 4) if yn else None,
            "doc_recall": _mean(x["doc_recall"] for x in rs),
            "evidence_recall": _mean(ev),
            "evidence_hit": _mean(1.0 if e >= 0.5 else 0.0 for e in ev) if ev else None,
            "answer_evidence_hit": _mean(1.0 if e >= 0.5 else 0.0 for e in aev) if aev else None,
        }

    by_type = defaultdict(list)
    for x in rows:
        by_type[x["type"]].append(x)
    report = {"overall": agg(rows), "by_type": {t: agg(v) for t, v in sorted(by_type.items())},
              "missing_questions": len(qmap) - len(rows)}
    with open(os.path.join(run_dir, "score.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(run_dir, "score_rows.jsonl"), "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    _print(report, run_dir)
    return report


def _fmt(v):
    return "  -  " if v is None else f"{v:.3f}"


def _print(report: dict, run_dir: str) -> None:
    cols = ["n", "em", "f1", "rouge_l", "yesno_acc", "yesno_majority_rate", "doc_recall",
            "evidence_hit", "answer_evidence_hit"]
    head = ["type", "n", "EM", "F1", "ROUGE-L", "YN-acc", "YN-major", "DocRec", "EvHit", "AnsEvHit"]
    print(f"\nScore: {run_dir}")
    print("  " + " | ".join(f"{h:>9}" if i else f"{h:<20}" for i, h in enumerate(head)))
    items = list(report["by_type"].items()) + [("OVERALL", report["overall"])]
    for name, a in items:
        cells = [f"{a['n']:>9}"] + [f"{_fmt(a[c]):>9}" for c in cols[1:]]
        print(f"  {name[:20]:<20} | " + " | ".join(cells))
    if report["missing_questions"]:
        print(f"  ({report['missing_questions']} questions in the data file have no result)")
