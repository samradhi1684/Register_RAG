"""
Append RegisterRAG rows (per dataset, per question type) to the baseline workbook
results_by_qtype.xlsx, using that workbook's metric definitions where they can be
reproduced from our runs.  Writes a NEW file; the input is not modified.

  python tools/append_registerrag_xlsx.py IN.xlsx OUT.xlsx

Definitions (matching the workbook):
  correct              = exact-match questions (the workbook's correct == em * n)
  failure classes      = among EM-wrong questions, first match of
                         retrieval (gold not in any step's pool) -> reranking (gold not in any
                         step's top-10) -> generation (everything else); DSIRE gold = the
                         gold context snippet, covered >= 50% by the units jointly
  em / f1              = max over gold answer + aliases (run.py score)
  *_ci95_lo/hi         = bootstrap percentile interval, 10,000 resamples, seed 0
  doc_recall@k         = share of gold documents with a unit in the union of every step's top-k
                         (k=20: top-10 + the next 10 of that step's ranked pool)
  sent_recall@k        = same for gold supporting sentences (2Wiki, HotpotQA only)
  unit_precision@k     = gold sentences / units in the union of every step's top-k (2Wiki, HotpotQA)
  ctx_*                = the same measures over the units passed to the answer generator
  ctx_all_gold_rate    = share of questions whose generator context holds every gold document
  latency_p50 / p95    = seconds per question;  total_tokens_mean = prompt + completion, all LLM requests
  calls_mean.compiler  = 1 (one planner call);  calls_mean.generator = every other LLM request
                         (binding extraction + answer) -- RegisterRAG has no separate counter for these
  Not recorded by RegisterRAG (left blank): peak_prompt_tokens_*, truncated_rate, gpu_peak_mb_max,
  calls_mean.retriever / reranker / verifier.
"""
from __future__ import annotations
import json
import os
import re
import sys

import numpy as np
import openpyxl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import ROOT, load_questions, load_units, gold_docs, doc_hit  # noqa: E402
from registerrag.evaluation.scoring import evidence_recall  # noqa: E402

RUN = os.path.join(ROOT, "runs", "run2_500ctx")
QA = os.path.join(ROOT, "registerrag", "qa-pairs")
RUNS = [  # (our dataset, workbook dataset, run name, generator, question file)
    ("2wiki", "2wiki", "all500_3b", "qwen2.5:3b", "2wiki_500.json"),
    ("2wiki", "2wiki", "all500", "qwen2.5:7b", "2wiki_500.json"),
    ("hotpotqa", "hotpot", "all500_3b", "qwen2.5:3b", "hotpot_500.json"),
    ("hotpotqa", "hotpot", "all500", "qwen2.5:7b", "hotpot_500.json"),
    ("musique", "musique", "all500", "qwen2.5:7b", "musique_500.json"),
    ("repolicyqa", "dsire", "all500", "qwen2.5:7b", "repolicyqa_500.json"),
]
DSIRE_CORPUS = os.path.join(ROOT, "registerrag", "data", "final_92docs")
LABEL, SYSTEM = "RegisterRAG (ours)", "registerrag"
VARIANT = "profile=v2, sentence units, corpus=500-question contexts, own 500q set"
norm = lambda s: " ".join(re.sub(r"[^\w\s]", " ", (s or "").lower()).split())


def parse_steps(path):
    t = open(path, encoding="utf-8", errors="ignore").read()
    end = t.find("\n## Working Register")
    t = t[: end if end > 0 else len(t)]
    steps = []
    for m in re.finditer(r"^### (R\w+): .*$", t, re.M):
        nxt = t.find("\n### ", m.end())
        blk = t[m.end(): nxt if nxt > 0 else len(t)]
        top = re.findall(r"^\|\s*\d+\s*\|\s*`([^`]+)`", blk, re.M)[:10]
        pm = re.search(r"\*\*Pool beyond top-10:\*\*(.*)$", blk, re.M)
        pool = re.findall(r"`([^`]+)`", pm.group(1)) if pm else []
        b = re.search(r"\*\*Binding extracted:\*\*\s*`([^`]*)`", blk)
        val = b.group(1).strip() if b else ""
        steps.append({"ranked": top + pool, "bound": bool(val) and "NOT FOUND" not in val.upper() and len(val) > 1})
    return steps


def gold_sentences(ds, qfile):
    """qid -> set of normalised gold supporting sentences (2Wiki / HotpotQA)."""
    raw = json.load(open(os.path.join(QA, qfile), encoding="utf-8"))
    out = {}
    for q in raw:
        if ds == "2wiki":
            ctx = {t: s for t, s in q["context"]}
            sf = [(t, i) for t, i in q["supporting_facts"]]
        else:
            ctx = dict(zip(q["context"]["title"], q["context"]["sentences"]))
            sf = list(zip(q["supporting_facts"]["title"], q["supporting_facts"]["sent_id"]))
        out[q["id"]] = {(t, norm(ctx[t][i])) for t, i in sf if t in ctx and i < len(ctx[t])}
    return out


def is_gold_sent(u, gs):
    return u is not None and (u["document_title"], norm(u["text"])) in gs


def boot(x, rng):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None, None
    idx = rng.integers(0, len(x), size=(10000, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def per_question(ds, run_name, qfile):
    run = os.path.join(RUN, ds, run_name)
    qs = load_questions(ds, os.path.join(QA, qfile), DSIRE_CORPUS if ds == "repolicyqa" else None)
    units = load_units(os.path.join(RUN, ds, "index"))
    sc = {r["id"]: r for r in map(json.loads, open(os.path.join(run, "score_rows.jsonl"), encoding="utf-8"))}
    gsent = gold_sentences(ds, qfile) if ds in ("2wiki", "hotpotqa") else {}
    out = []
    for line in open(os.path.join(run, "results.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        qid = r["question_id"]
        q, s = qs[qid], sc[qid]
        steps = parse_steps(os.path.join(run, "traces", os.path.basename(r["trace_path"])))
        golds = gold_docs(q)
        topk = {k: [units.get(u) for st in steps for u in st["ranked"][:k]] for k in (5, 10, 20)}
        pool = [units.get(u) for st in steps for u in st["ranked"]]
        ctx = [units.get(u) for u in r.get("answer_unit_ids") or []]
        drec = lambda us: (sum(any(doc_hit(g, u) for u in us) for g in golds) / len(golds)) if golds else None
        rec = {"em": s["em"], "f1": s["f1"], "latency": r["latency_s"], "tokens": r.get("total_tokens") or 0,
               "requests": r.get("llm_requests") or 0,
               "qtype": qid.split("__")[0] if ds == "musique" else s.get("type", ""),
               **{f"doc_recall@{k}": drec(v) for k, v in topk.items()},
               "ctx_doc_recall": drec(ctx),
               "ctx_all_gold_rate": float(all(any(doc_hit(g, u) for u in ctx) for g in golds)) if golds else None}
        gs = gsent.get(qid)
        if gs:
            for k, v in topk.items():
                uniq = {u["unit_id"]: u for u in v if u}
                rec[f"sent_recall@{k}"] = sum(any(is_gold_sent(u, {g}) for u in uniq.values()) for g in gs) / len(gs)
                if k in (5, 10):
                    rec[f"unit_precision@{k}"] = sum(is_gold_sent(u, gs) for u in uniq.values()) / max(1, len(uniq))
            cu = {u["unit_id"]: u for u in ctx if u}
            rec["ctx_sent_recall"] = sum(any(is_gold_sent(u, {g}) for u in cu.values()) for g in gs) / len(gs)
            rec["ctx_unit_precision"] = sum(is_gold_sent(u, gs) for u in cu.values()) / max(1, len(cu))
        # failure class on EM-wrong questions (workbook convention)
        if s["em"] >= 1.0:
            cat = "correct"
        else:
            if ds == "repolicyqa" and q.get("gold_context"):
                in_pool = evidence_recall(q["gold_context"], [u["text"] for u in pool if u]) >= 0.5
                in_top = evidence_recall(q["gold_context"], [u["text"] for u in topk[10] if u]) >= 0.5
            else:
                in_pool = all(any(doc_hit(g, u) for u in pool) for g in golds)
                in_top = all(any(doc_hit(g, u) for u in topk[10]) for g in golds)
            cat = "retrieval" if not in_pool else "reranking" if not in_top else "generation"
        rec["cat"] = cat
        out.append(rec)
    return out


def summarize(recs, rng):
    n = len(recs)
    mean = lambda k: (lambda v: sum(v) / len(v) if v else None)([x[k] for x in recs if x.get(k) is not None])
    lat = np.array([x["latency"] for x in recs])
    em_lo, em_hi = boot([x["em"] for x in recs], rng)
    f1_lo, f1_hi = boot([x["f1"] for x in recs], rng)
    c = lambda cat: float(sum(x["cat"] == cat for x in recs))
    m = {"generation_failures": c("generation"), "reranking_failures": c("reranking"),
         "retrieval_failures": c("retrieval"), "no_retrieval": 0.0, "system_errors": 0.0,
         "correct": c("correct"), "n": float(n), "em": mean("em"), "em_ci95_lo": em_lo, "em_ci95_hi": em_hi,
         "f1": mean("f1"), "f1_ci95_lo": f1_lo, "f1_ci95_hi": f1_hi,
         "latency_p50": float(np.percentile(lat, 50)), "latency_p95": float(np.percentile(lat, 95)),
         "total_tokens_mean": mean("tokens"), "calls_mean.compiler": 1.0,
         "calls_mean.generator": (mean("requests") or 1.0) - 1.0}
    for k in ("doc_recall@5", "doc_recall@10", "doc_recall@20", "sent_recall@5", "sent_recall@10", "sent_recall@20",
              "unit_precision@5", "unit_precision@10", "ctx_doc_recall", "ctx_sent_recall", "ctx_unit_precision",
              "ctx_all_gold_rate"):
        m[k] = mean(k)
    return m


def main():
    src, dst = sys.argv[1:3]
    wb = openpyxl.load_workbook(src)
    main_ws = wb.worksheets[0]
    header = [c.value for c in main_ws[1]]
    rng = np.random.default_rng(0)
    new_rows = []        # dicts keyed by main-sheet header
    for ds, wds, run_name, gen, qfile in RUNS:
        recs = per_question(ds, run_name, qfile)
        groups = [("ALL", recs)] + [(t, [x for x in recs if x["qtype"] == t]) for t in sorted({x["qtype"] for x in recs})]
        for qt, g in groups:
            m = summarize(g, rng)
            row = {"run_dir": f"registerrag_{wds}_{run_name}", "status": "complete", "generator": gen,
                   "system": SYSTEM, "label": LABEL, "variant": VARIANT, "dataset": wds, "qtype": qt, **m}
            new_rows.append(row)
        print(f"{wds} {gen}: {len(recs)} questions, {len(groups) - 1} types")

    # main sheet: rows are grouped in dataset blocks -> put each dataset's RegisterRAG rows
    # at the end of that dataset's block (rewrite the data rows in the new order)
    old = [list(r) for r in main_ws.iter_rows(min_row=2, values_only=True) if r[0] is not None]
    di = header.index("dataset")
    order, blocks = [], {}
    for r in old:
        if r[di] not in blocks:
            order.append(r[di])
            blocks[r[di]] = []
        blocks[r[di]].append(r)
    for row in new_rows:
        if row["dataset"] not in blocks:
            order.append(row["dataset"])
            blocks[row["dataset"]] = []
        blocks[row["dataset"]].append([row.get(h) for h in header])
    merged = [r for d in order for r in blocks[d]]
    for i, r in enumerate(merged, 2):
        for col, v in enumerate(r, 1):
            main_ws.cell(row=i, column=col, value=v)

    # pivot sheets: one row per run, metric x qtype columns; insert above the footnote
    for ws in wb.worksheets[1:]:
        metric_row = [c.value for c in ws[1]]
        qtype_row = [c.value for c in ws[2]]
        cur, cols = None, []
        for i, (mv, qv) in enumerate(zip(metric_row, qtype_row), 1):
            cur = mv if mv is not None else cur
            cols.append((i, cur, qv))
        data_end = max(i for i in range(3, ws.max_row + 1) if str(ws.cell(i, 1).value or "").count("_") >= 2)
        runs = [r for r in new_rows if r["dataset"] == ws.title]
        by_run = {}
        for r in runs:
            by_run.setdefault(r["run_dir"], {})[r["qtype"]] = r
        ws.insert_rows(data_end + 1, amount=len(by_run))
        for j, (rd, per_qt) in enumerate(by_run.items()):
            base = per_qt["ALL"]
            for i, metric, qt in cols:
                v = base.get(metric) if qt is None else per_qt.get(qt, {}).get(metric)
                ws.cell(row=data_end + 1 + j, column=i, value=v)
        foot = data_end + 1 + len(by_run)
        while foot <= ws.max_row and not ws.cell(foot, 1).value:
            foot += 1
        if foot <= ws.max_row:
            ws.cell(foot, 1, value=str(ws.cell(foot, 1).value) + " RegisterRAG rows: computed from our run2_500ctx runs "
                    "(tools/append_registerrag_xlsx.py); our 500-question set differs from the baselines' set; "
                    "correct = exact match, failures among EM-wrong questions; blank = not recorded by RegisterRAG.")
        print(f"sheet {ws.title}: +{len(by_run)} rows at row {data_end + 1}")
    wb.save(dst)
    print("wrote", dst, "| main sheet +", len(new_rows), "rows")


if __name__ == "__main__":
    main()
