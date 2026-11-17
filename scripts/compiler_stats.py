"""Did Baseline 3's decomposition actually decompose?  Usage: python scripts/compiler_stats.py [dir] [--show 3]"""
import json
from collections import Counter
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from registerrag.paths import run_folders

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("dir", nargs="?", default=None)
ap.add_argument("--show", type=int, default=3)
root, show = ap.parse_args().dir, ap.parse_args().show
for p in [os.path.join(d, "results.jsonl") for d in run_folders(root) if os.path.basename(d).startswith("iterative_")]:
    rs = [json.loads(l) for l in open(p, encoding="utf-8")]
    ex = [r.get("extra", {}) for r in rs]
    nreq = Counter(len(e.get("requirements", [])) for e in ex)
    print(f"\n{os.path.basename(os.path.dirname(p))}  (n={len(rs)})")
    print(f"  JSON parsed ok : {sum(bool(e.get('compiler_parse_ok')) for e in ex)}/{len(rs)}")
    print(f"  compiler errors: {sum(bool(e.get('compiler_error')) for e in ex)}")
    print(f"  #requirements  : {dict(sorted(nreq.items()))}   (a '1' means no real decomposition)")
    for r in rs[:show]:
        print(f"  Q: {r['question'][:100]}")
        for q in r.get("extra", {}).get("requirements", []):
            print(f"       {q['id']} {q['need'][:100]}  depends_on={q['depends_on']}")
