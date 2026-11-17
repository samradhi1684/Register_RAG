"""Counts the manual labels you typed into failures_to_label.md (Step 3 failure review).
Usage: python scripts/tally_labels.py failures_to_label.md
Type after 'Manual label ...:' one of: retrieval | reranking | generation (or any note; unknown -> OTHER)."""
import re, sys
from collections import Counter

text = open(sys.argv[1], encoding="utf-8").read()
auto = Counter(re.findall(r"auto=([A-Z_]+)", text))
cnt = Counter()
for m in re.finditer(r"Manual label[^\n]*?:[ \t]*([^\n]*)", text):
    v = m.group(1).strip().lower()
    if not v: cnt["UNLABELED"] += 1
    elif "rerank" in v: cnt["RERANKING"] += 1
    elif "retriev" in v: cnt["RETRIEVAL"] += 1
    elif v.startswith(("gen", "reason")) or "generat" in v or "reasoning" in v: cnt["GENERATION"] += 1
    else: cnt["OTHER"] += 1
n = sum(cnt.values()); lab = n - cnt["UNLABELED"]
print(f"{n} failures in file, {lab} labelled\nmanual : {dict(cnt)}\nauto   : {dict(auto)}")
if lab:
    g = cnt["GENERATION"] / lab
    print(f"\nreasoning/generation share of labelled failures: {g:.0%}")
    print("roadmap rule: > 50% -> 'context is the bottleneck' is weaker than expected (report it either way)."
          if g > 0.5 else "not above 50%: retrieval/selection is the larger share (consistent with the core assumption).")
