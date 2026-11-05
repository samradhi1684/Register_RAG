"""Scan EVERY sentence unit for likely splitting errors and for context-dependence.
Usage:  python scripts/audit_splits.py --dataset musique --data datasets/musique_data/musique_100.json
"""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from registerrag.data import load_examples
from registerrag.units import make_store

ABBR = re.compile(r"\b(Dr|Mr|Mrs|Ms|St|Jr|Sr|Mt|No|Nos|vs|Inc|Ltd|Co|Corp|Gen|Col|Lt|Capt|Sgt|Prof|Rev|Gov|Sen|Rep|Fr|Ft|Vol|Dept|Univ|U\.S|U\.K|a\.m|p\.m|e\.g|i\.e|etc|approx|ca|c)\.$")
PRON = re.compile(r"^(He|She|His|Her|Hers|It|Its|They|Their|Them|This|These|That|Those|The film|The album|The song|The company|The band|However|Also|Later|Then|Both)\b")

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--show", type=int, default=6)
a = ap.parse_args()
_, docs, _ = load_examples(a.data, a.dataset)
units = make_store(docs, "sentence").units
checks = {
  "ends_with_abbreviation (likely split after 'Dr.', 'St.' ...)": lambda t: bool(ABBR.search(t)),
  "starts_lowercase (likely mid-sentence fragment)": lambda t: t[:1].islower(),
  "very_short (<20 chars)": lambda t: len(t) < 20,
  "citation_marker like [3]": lambda t: bool(re.search(r"\[\d+\]", t)),
  "no_terminal_punctuation": lambda t: t[-1] not in '.!?"\')”',
}
print(f"{len(units)} sentence units")
for name, fn in checks.items():
    hits = [u for u in units if fn(u.text)]
    print(f"\n{name}: {len(hits)} ({100*len(hits)/len(units):.1f}%)")
    for u in hits[: a.show]:
        print(f"   [{u.unit_id}] ({u.title}) {u.text[:140]}")
dep = [u for u in units if PRON.match(u.text)]
print(f"\nCONTEXT-DEPENDENT openers (pronoun / 'The film' / 'However' ...): {len(dep)} ({100*len(dep)/len(units):.1f}%)"
      "  -> these need the page title to be retrievable (see --prepend-title)")
