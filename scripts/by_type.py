"""EM per 2Wiki question type (+ yes/no subset) for any run folders: shows how much EM is a guessing floor.
Usage: python scripts/by_type.py [dir ...]     (no dir -> all baselines)"""
import json
from collections import defaultdict
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from registerrag.paths import run_folders

folders = []
for a in (sys.argv[1:] or [None]):
    folders += run_folders(a)
rows, types = [], set()
for f in sorted(folders):
    if not (os.path.exists(os.path.join(f, "config.json")) and os.path.exists(os.path.join(f, "summary.json"))):
        print(f"skipping unfinished run (no summary.json): {f}   -> rerun its command to resume it")
        continue
    cfg = json.load(open(os.path.join(f, "config.json"), encoding="utf-8"))
    recs = [json.loads(l) for l in open(os.path.join(f, "results.jsonl"), encoding="utf-8")]
    g = defaultdict(list)
    for r in recs:
        g[r["qtype"] or "all"].append(r["em"])
        if r["gold_answers"][0].strip().lower() in ("yes", "no"):
            g["[yes/no]"].append(r["em"])
        g["ALL"].append(r["em"])
    types |= set(g)
    label = f"{cfg['system']}/k{cfg['top_k']}/b{cfg['budget_tokens']}/t{int(cfg['prepend_title'])}/{cfg['prompt_version']}"
    if cfg.get("unit_mode") or cfg.get("method_args"):
        label += "+" + ",".join([str(cfg.get("unit_mode") or "")] + [f"{k}={v}" for k, v in sorted((cfg.get("method_args") or {}).items())]).strip(",")
    rows.append((label, len(recs), g))
order = [t for t in sorted(types) if t not in ("ALL", "[yes/no]")] + ["[yes/no]", "ALL"]
print(f"{'run (system/topk/budget/title/prompt)':38s} {'n':>4s} " + " ".join(f"{t[:16]:>16s}" for t in order))
for label, n, g in rows:
    print(f"{label:38s} {n:4d} " + " ".join(
        (f"{sum(g[t])/len(g[t]):.2f} (n={len(g[t])})".rjust(16) if g.get(t) else "-".rjust(16)) for t in order))
