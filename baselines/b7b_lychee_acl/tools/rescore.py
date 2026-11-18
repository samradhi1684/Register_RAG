"""Re-score B7b runs from the stored raw outputs (no model call, no GPU). Writes <run>/rescore.json next to each run.

Why: the released checkpoint answers in full sentences (no answer-format instruction exists for it: no template, no eval
script), so STRICT exact match is ~0 by construction. The strict EM/F1 stored in summary.json are left untouched and
remain the numbers every other row is compared with. This tool adds a clearly different, LENIENT metric:

  contains      the normalised gold answer occurs as a whole-word sequence anywhere in the normalised raw output
                (for yes/no gold: the FIRST word of the output must be that word - 'no' would otherwise match 'not')
  says_no_info  the output states that the memory/context does not contain the answer (a refusal, counted wrong)

`contains` is an UPPER-LEANING lenient accuracy (a long output can mention the gold by chance). Never put it in the same
column as EM; report it as 'B7b contains-acc'.

Usage: python baselines/b7b_lychee_acl/tools/rescore.py [results_dir ...]   (default: baselines/b7b_lychee_acl/results)
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
NO_INFO = re.compile(r"does not (?:contain|provide|mention|include)|do not (?:contain|provide)|no information|"
                     r"not (?:enough|sufficient) information|cannot (?:be )?(?:determine|answer|find)", re.I)


def norm(s: str) -> list:
    s = re.sub(r"\b(a|an|the)\b", " ", (s or "").lower())
    return re.sub(r"[^a-z0-9 ]", " ", s).split()


def contains(gold: str, raw: str) -> bool:
    g, o = norm(gold), norm(raw)
    if not g:
        return False
    if g in (["yes"], ["no"]):
        return bool(o) and o[0] == g[0]
    n = len(g)
    return any(o[i:i + n] == g for i in range(len(o) - n + 1))


def rescore_run(folder: str) -> dict:
    rows = [json.loads(l) for l in open(os.path.join(folder, "results.jsonl"), encoding="utf-8") if l.strip()]
    hit = refuse = 0
    for r in rows:
        raw = (r.get("extra") or {}).get("raw_output") or r.get("prediction") or ""
        hit += any(contains(g, raw) for g in r["gold_answers"])
        refuse += bool(NO_INFO.search(raw))
    s = json.load(open(os.path.join(folder, "summary.json"), encoding="utf-8"))
    out = {"n": len(rows), "strict_em": s["em"], "strict_f1": s["f1"], "contains_acc": hit / len(rows),
           "says_no_info": refuse / len(rows), "note": "contains_acc is a lenient metric; do not compare with EM"}
    json.dump(out, open(os.path.join(folder, "rescore.json"), "w", encoding="utf-8"), indent=2)
    return out


if __name__ == "__main__":
    roots = sys.argv[1:] or [os.path.join(ROOT, "baselines", "b7b_lychee_acl", "results")]
    for root in roots:
        for res in sorted(glob.glob(os.path.join(root, "*", "results.jsonl"))):
            d = os.path.dirname(res)
            try:
                o = rescore_run(d)
                print(f"{os.path.basename(d):32s} n={o['n']} strict EM={o['strict_em']:.2f} F1={o['strict_f1']:.3f} "
                      f"| contains-acc={o['contains_acc']:.2f} | refusals={o['says_no_info']:.2f}")
            except Exception as e:  # noqa: BLE001
                print(os.path.basename(d), "failed:", e)
