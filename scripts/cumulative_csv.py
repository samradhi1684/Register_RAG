"""Flatten every run in runs_progressive/ into one cumulative CSV (one row per run).

All summary.json fields are flattened (CIs -> _lo/_hi, nested dicts -> prefix.key).
Runs without summary.json (still running) are summarized from results.jsonl and marked status=incomplete.
Failure counts get dedicated columns: generation_failures, reranking_failures, retrieval_failures.
"""
import argparse, csv, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from registerrag.runner import summarize
from registerrag import registry

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="runs_progressive")
ap.add_argument("--out", default=None)
a = ap.parse_args()
out = a.out or os.path.join(a.dir, "cumulative_results.csv")
LABEL = {b.system: b.label for b in registry.implemented()}


def flatten(d, prefix=""):
    o = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            o.update(flatten(v, key + "."))
        elif isinstance(v, (list, tuple)) and len(v) == 2 and key.endswith("_ci95"):
            o[key + "_lo"], o[key + "_hi"] = v
        elif isinstance(v, (list, tuple)):
            o[key] = json.dumps(v)
        else:
            o[key] = v
    return o


rows = []
for run in sorted(glob.glob(os.path.join(a.dir, "*", "results.jsonl"))):
    d = os.path.dirname(run)
    sp = os.path.join(d, "summary.json")
    if os.path.exists(sp):
        S, status = json.load(open(sp, encoding="utf-8")), "complete"
    else:
        S, status = summarize(run), "incomplete"
        cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
        S["data_sha256"] = cfg.pop("data_sha256", None)
        S["config"], S["config_hash"] = cfg, os.path.basename(d).rsplit("_", 1)[-1]
    cfg = S.get("config", {})
    margs = dict(cfg.get("method_args") or {})
    if cfg.get("unit_mode"):
        margs["unit"] = cfg["unit_mode"]
    fc = S.get("failure_counts", {})
    row = {"run_dir": os.path.basename(d), "status": status,
           "system": cfg.get("system"), "label": LABEL.get(cfg.get("system"), cfg.get("system")),
           "variant": ", ".join(f"{k}={v}" for k, v in margs.items()) or "-",
           "dataset": cfg.get("dataset"), "generator": cfg.get("generator"),
           "generation_failures": fc.get("GENERATION_FAILURE", 0),
           "reranking_failures": fc.get("RERANKING_FAILURE", 0),
           "retrieval_failures": fc.get("RETRIEVAL_FAILURE", 0)}
    S = dict(S)
    if "method_args" in cfg:
        S["config"] = {**cfg, "method_args": json.dumps(cfg["method_args"])}
    for k, v in flatten(S).items():
        row.setdefault(k, v)
    rows.append(row)

order = list(LABEL)
rows.sort(key=lambda r: (r["dataset"] or "", r["generator"] or "",
                         order.index(r["system"]) if r["system"] in order else 99, r["variant"]))
cols = []
for r in rows:
    cols += [k for k in r if k not in cols]
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        for c in cols:
            if c.startswith("failure_counts.") and c not in r:
                r[c] = 0
    w.writerows(rows)
print(f"wrote {len(rows)} rows x {len(cols)} cols -> {out}")
