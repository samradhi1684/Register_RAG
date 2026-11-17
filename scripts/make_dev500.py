"""Build a larger stratified dev file that CONTAINS your existing 100-question set (so earlier results stay comparable).
Needs the full dev files that the extract_*_qa_pairs.py scripts download (2wiki_dev.json / musique_dev.json[l]).
Usage:
  python scripts/make_dev500.py --dataset 2wiki   --dev datasets/wiki2_data/2wiki_dev.json     --base datasets/wiki2_data/2wiki_100.json     --out datasets/wiki2_data/2wiki_500.json
  python scripts/make_dev500.py --dataset musique --dev datasets/musique_data/musique_dev.json --base datasets/musique_data/musique_100.json --out datasets/musique_data/musique_500.json"""
import argparse, json, random
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["2wiki", "musique"], required=True)
ap.add_argument("--dev", required=True); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=500); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

def load(p):
    txt = open(p, encoding="utf-8").read()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]

key = lambda r: r.get("_id") or r.get("id")      # 2Wiki official uses _id, HF versions use id
stratum = (lambda r: r["type"]) if a.dataset == "2wiki" else (lambda r: len(r["question_decomposition"]))
dev = [r for r in load(a.dev) if a.dataset == "2wiki" or r.get("answerable", True)]
base = load(a.base)
strata = sorted({stratum(r) for r in dev})
quota = {s: a.n // len(strata) + (1 if i < a.n % len(strata) else 0) for i, s in enumerate(strata)}
have = defaultdict(int)
for r in base:
    have[stratum(r)] += 1
used = {key(r) for r in base}
pool = defaultdict(list)
for r in sorted(dev, key=key):
    if key(r) not in used:
        pool[stratum(r)].append(r)
rng, out = random.Random(a.seed), list(base)
for s in strata:
    need = quota[s] - have[s]
    if need < 0:
        raise SystemExit(f"base already has {have[s]} in stratum {s}, quota {quota[s]}")
    if need > len(pool[s]):
        raise SystemExit(f"stratum {s}: need {need}, pool has {len(pool[s])}")
    out += rng.sample(pool[s], need)
json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
print(f"wrote {len(out)} questions to {a.out}; per stratum:", {s: sum(1 for r in out if stratum(r) == s) for s in strata})
