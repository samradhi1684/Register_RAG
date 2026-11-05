"""One table over every run folder: which settings differ, and the key metrics.
Usage: python scripts/summarize_runs.py [dir] [--hops]
  no dir -> every baseline under baselines/*/results (plus a legacy runs/ dir);
  dir    -> a results dir, a baseline folder such as baselines/b8_pyrag, or a single run folder."""
import json
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from registerrag.paths import run_folders

root = next((x for x in sys.argv[1:] if not x.startswith("--")), None)
rows = []
for d in run_folders(root):
    path = os.path.join(d, "summary.json")
    if os.path.exists(path):
        s = json.load(open(path, encoding="utf-8"))
        rows.append((os.path.basename(d), s["config"], s))
if not rows:
    sys.exit(f"no finished run (summary.json) found under {root or 'baselines/*/results'}")
keys = ["system", "dataset", "limit", "prepend_title", "prompt_version", "generator", "top_k", "lam",
        "budget_tokens", "corpus_scope", "embedder", "reranker", "tokenizer", "unit_mode", "method_args"]
varying = [k for k in keys if len({json.dumps(r[1].get(k)) for r in rows}) > 1]
print("settings that differ between runs:", varying or "none")
print("constant:", {k: rows[0][1].get(k) for k in keys if k not in varying})
hdr = f"{'run':26s} " + " ".join(f"{k[:12]:>12s}" for k in varying) + \
      f" {'n':>4s} {'EM':>6s} {'EM 95% CI':>13s} {'F1':>6s} {'sentR@10':>8s} {'allGold':>7s} {'peakTok':>7s}  failures"
print("\n" + hdr)
variants = [(n, c.get("method_args")) for n, c, _ in rows if c.get("method_args")]
for name, c, s in rows:
    ci = s.get("em_ci95") or [None, None]
    f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
    print(f"{name:26s} " + " ".join(f"{str(c.get(k))[:12]:>12s}" for k in varying) +
          f" {s['n']:4d} {f(s['em']):>6s} {'['+f(ci[0],2)+','+f(ci[1],2)+']':>13s} {f(s['f1']):>6s} "
          f"{f(s.get('sent_recall@10')):>8s} {f(s.get('ctx_all_gold_rate')):>7s} {s['peak_prompt_tokens_max']:>7d}  "
          f"{s['failure_counts']}")

if variants:
    print("\nmethod_args per run:")
    for n, m in variants:
        print(f"  {n}: {m}")
if "--hops" in sys.argv:
    hops = sorted({h for _, _, s in rows for h in s.get("by_hops", {})})
    print("\nEM by number of hops (n in brackets):")
    print(f"{'run':26s} " + " ".join(f"{h+'-hop':>14s}" for h in hops))
    for name, c, s in rows:
        print(f"{name:26s} " + " ".join(
            (f"{s['by_hops'][h]['em']:.2f} (n={s['by_hops'][h]['n']})".rjust(14) if h in s.get("by_hops", {}) else "-".rjust(14))
            for h in hops))
