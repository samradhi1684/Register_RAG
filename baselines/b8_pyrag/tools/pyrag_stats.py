"""How often did PyRAG actually run as PyRAG?  A high fallback rate means the small model could not write working
programs and the row is mostly plain RAG.  Usage: python baselines/b8_pyrag/tools/pyrag_stats.py [dir] [--show 2]   (default: this baseline's results/)"""
import json, os, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from registerrag.paths import run_folders

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("dir", nargs="?", default=str(Path(__file__).resolve().parents[1] / "results"))
ap.add_argument("--show", type=int, default=2)
root, show = ap.parse_args().dir, ap.parse_args().show
for p in [os.path.join(d, "results.jsonl") for d in run_folders(root) if os.path.basename(d).startswith("pyrag_")]:
    folder = os.path.dirname(p)
    cfg = json.load(open(os.path.join(folder, "config.json"), encoding="utf-8"))
    rs = [json.loads(l) for l in open(p, encoding="utf-8")]
    ex = [r.get("extra", {}) for r in rs]; n = len(rs)
    pct = lambda k: f"{k}/{n} ({100 * k / n:.0f}%)"
    errs = Counter((e.get("program_error") or "").split(":")[0] for e in ex if e.get("repairs"))
    em = lambda sel: (sum(r["em"] for r, e in zip(rs, ex) if sel(e)) / max(1, sum(1 for e in ex if sel(e))))
    print(f"\n{os.path.basename(folder)}  generator={cfg['generator']}  args={cfg.get('method_args')}  unit={cfg.get('unit_mode') or 'chunk'}")
    print(f"  plan model            : {Counter(e.get('plan_model') for e in ex).most_common(1)[0][0]}")
    print(f"  ran without any repair: {pct(sum(1 for e in ex if not e.get('repairs') and not e.get('fallback')))}")
    print(f"  needed >=1 repair     : {pct(sum(1 for e in ex if e.get('repairs')))}   error kinds of the last failure: {dict(errs)}")
    print(f"  FELL BACK to plain RAG: {pct(sum(1 for e in ex if e.get('fallback')))}")
    print(f"  adaptive retrieval    : {pct(sum(1 for e in ex if e.get('retried_with_topk')))}")
    print(f"  decompose failed      : {pct(sum(1 for e in ex if e.get('decompose_error')))}   plan failed: {pct(sum(1 for e in ex if e.get('plan_error')))}")
    print(f"  tool calls / question : {sum(e.get('n_tool_calls', 0) for e in ex) / n:.1f}   sub-queries: {sum(len(e.get('sub_queries', [])) for e in ex) / n:.1f}")
    print(f"  EM when program ran   : {em(lambda e: not e.get('fallback')):.2f}   EM when fell back: {em(lambda e: e.get('fallback')):.2f}")
    for r in rs[:show]:
        e = r.get("extra", {})
        print(f"  Q: {r['question'][:90]}\n     sub-queries: {e.get('sub_queries')}\n     program:\n" +
              "\n".join("       " + l for l in (e.get("code") or "").splitlines()[:8]) + f"\n     -> {r['prediction']!r}  (gold {r['gold_answers'][0]!r})")
