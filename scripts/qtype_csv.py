"""Per-question-type breakdown of every run in runs_progressive/ (one row per run x qtype, plus qtype=ALL).

qtype comes from results.jsonl; MuSiQue has none, so its type is the qid prefix (2hop, 3hop1, 4hop2, ...).
Metrics are computed with registerrag.runner.summarize on the subset (same definitions as summary.json).
"""
import argparse, csv, glob, json, os, sys, tempfile
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from registerrag.runner import summarize
from registerrag import registry

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="runs_progressive")
ap.add_argument("--out", default=None)
a = ap.parse_args()
out = a.out or os.path.join(a.dir, "results_by_qtype.csv")
LABEL = {b.system: b.label for b in registry.implemented()}
ORDER = list(LABEL)
FAIL = {"generation_failures": "GENERATION_FAILURE", "reranking_failures": "RERANKING_FAILURE",
        "retrieval_failures": "RETRIEVAL_FAILURE", "no_retrieval": "NO_RETRIEVAL",
        "system_errors": "SYSTEM_ERROR", "correct": "CORRECT"}
SKIP = {"failure_counts", "by_hops", "per_hop_recall@10"}


def qtype(r):
    return r.get("qtype") or (r["qid"].split("__")[0] if "__" in r["qid"] else "unknown")


def summarize_recs(recs):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    try:
        return summarize(f.name)
    finally:
        os.unlink(f.name)


rows = []
for run in sorted(glob.glob(os.path.join(a.dir, "*", "results.jsonl"))):
    d = os.path.dirname(run)
    cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
    margs = dict(cfg.get("method_args") or {})
    if cfg.get("unit_mode"):
        margs["unit"] = cfg["unit_mode"]
    status = "complete" if os.path.exists(os.path.join(d, "summary.json")) else "incomplete"
    recs = [json.loads(l) for l in open(run, encoding="utf-8")]
    groups = defaultdict(list)
    for r in recs:
        groups[qtype(r)].append(r)
    for qt, sub in [("ALL", recs)] + sorted(groups.items()):
        S = summarize_recs(sub)
        fc = S["failure_counts"]
        row = {"run_dir": os.path.basename(d), "status": status, "generator": cfg["generator"],
               "system": cfg["system"], "label": LABEL.get(cfg["system"], cfg["system"]),
               "variant": ", ".join(f"{k}={v}" for k, v in margs.items()) or "-",
               "dataset": cfg["dataset"], "qtype": qt}
        row.update({k: fc.get(v, 0) for k, v in FAIL.items()})
        for k, v in S.items():
            if k in SKIP:
                continue
            if k.endswith("_ci95"):
                row[k + "_lo"], row[k + "_hi"] = v
            elif isinstance(v, dict):
                row.update({f"{k}.{kk}": vv for kk, vv in v.items()})
            else:
                row[k] = v
        rows.append(row)

rows.sort(key=lambda r: (r["dataset"], r["generator"], r["qtype"] != "ALL", r["qtype"],
                         ORDER.index(r["system"]) if r["system"] in ORDER else 99, r["variant"]))
cols = []
for r in rows:
    cols += [k for k in r if k not in cols]
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols, restval="")
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} rows x {len(cols)} cols -> {out}")
