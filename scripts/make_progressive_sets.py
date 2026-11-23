"""Builds the progressive-study files with exactly TOTAL questions (default 500): the existing 100-question set FIRST
(so the first 100 questions are the set used so far), then the first TOTAL-100 questions (file order) of the 500-question
file that are NOT in the 100-set (no overlap, no duplicates).
Output: datasets/wiki2_data/2wiki_prog500.json, datasets/musique_data/musique_prog500.json, datasets/dsire_data/repolicyqa_prog500.json
Usage: python scripts/make_progressive_sets.py [--total 500]"""
import json, os, sys

TOTAL = int(sys.argv[sys.argv.index("--total") + 1]) if "--total" in sys.argv else 500

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = [  # name, 100-file, 500-file, output, dedup key
    ("2wiki", "datasets/wiki2_data/2wiki_100.json", "datasets/wiki2_data/2wiki_500.json", "datasets/wiki2_data/2wiki_prog{T}.json", lambda r: r["id"]),
    ("musique", "datasets/musique_data/musique_100.json", "datasets/musique_data/musique_500.json", "datasets/musique_data/musique_prog{T}.json", lambda r: r["id"]),
    ("dsire", "datasets/dsire_data/repolicyqa_100.json", "datasets/dsire_data/repolicyqa_stratified.json", "datasets/dsire_data/repolicyqa_prog{T}.json",
     lambda r: (r["document_id"], r["question"].strip())),
]
for name, a, b, out, key in SETS:
    first = json.load(open(os.path.join(ROOT, a), encoding="utf-8"))
    more = json.load(open(os.path.join(ROOT, b), encoding="utf-8"))
    seen, recs = set(), []
    for r in first + more:
        k = key(r)
        if k not in seen:
            seen.add(k); recs.append(r)
    if len(recs) < TOTAL:
        raise SystemExit(f"{name}: only {len(recs)} unique questions, need {TOTAL}")
    recs = recs[:TOTAL]
    out = out.format(T=TOTAL)
    json.dump(recs, open(os.path.join(ROOT, out), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{name}: {len(first)} + {len(more)} -> {len(recs)} questions (100 + {len(recs)-100} non-overlapping) -> {out}")
