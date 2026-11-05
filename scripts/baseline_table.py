"""Builds the frozen baseline table from run folders (Step 3). Only runs with the frozen settings are used
(title on, prompt v2, top-k 5, no token budget). Writes baselines/baseline_table.md and prints it.
Usage: python scripts/baseline_table.py [dir] [--generator qwen2.5:3b]   (no dir -> all baselines)"""
import os, sys

import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from registerrag import registry
from registerrag.paths import run_folders

args = [a for a in sys.argv[1:] if not a.startswith("--")]
root = args[0] if args else None
gen = sys.argv[sys.argv.index("--generator") + 1] if "--generator" in sys.argv else None
LABEL = {b.system: b.label for b in registry.implemented()}      # registry order: B1, B2, ..., refs
ORDER = list(LABEL)
rows = []
for d in run_folders(root):
    p = os.path.join(d, "summary.json")
    if not os.path.exists(p):
        continue
    s = json.load(open(p, encoding="utf-8")); c = s["config"]
    if c["prepend_title"] and c["prompt_version"] == "v2" and c["budget_tokens"] is None and c["top_k"] == 5 \
            and c["system"] in LABEL and (gen is None or c["generator"] == gen) and not c.get("limit"):
        parts = [f"unit={c['unit_mode']}"] if c.get("unit_mode") else []
        parts += [f"{k}={v}" for k, v in sorted((c.get("method_args") or {}).items())]
        variant = ", ".join(parts) or "-"
        rows.append((c["dataset"], c["generator"], ORDER.index(c["system"]), variant, c, s, os.path.basename(d)))
if not rows:
    sys.exit("no run matches the frozen filter (title on, prompt v2, top-k 5, no token budget, no --limit"
             + (f", generator {gen}" if gen else "") + "). Run scripts/run_baselines.py without --limit first.")
rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
out = ["| dataset | generator | system | variant | n | EM [95% CI] | F1 | doc R@10 | all gold in prompt | peak prompt tok | total tok/q | latency p50 / p95 (s) | calls/q (retr, rerank, comp, gen) | run |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for ds, g, _, variant, c, s, name in rows:
    ci = s["em_ci95"]; k = s["calls_mean"]
    out.append(f"| {ds} | {g} | {LABEL[c['system']]} | {variant} | {s['n']} | {f(s['em'],2)} [{f(ci[0],2)}, {f(ci[1],2)}] | {f(s['f1'])} | "
               f"{f(s.get('doc_recall@10'),2)} | {f(s.get('ctx_all_gold_rate'),2)} | {s['peak_prompt_tokens_max']} | {s['total_tokens_mean']:.0f} | "
               f"{f(s['latency_p50'],2)} / {f(s['latency_p95'],2)} | {k.get('retriever',0):.1f}, {k.get('reranker',0):.1f}, "
               f"{k.get('compiler',0):.1f}, {k.get('generator',0):.1f} | {name} |")
txt = "\n".join(out)
open(os.path.join(registry.BASELINES_DIR, "baseline_table.md"), "w", encoding="utf-8").write(txt + "\n")
print(txt, "\n\nwritten to baselines/baseline_table.md")
