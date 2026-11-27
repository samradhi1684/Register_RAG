"""
LLM-judge correctness for finished runs (post hoc; does not touch the pipeline).

For each question: does the predicted answer convey the gold answer's facts?
  1   correct: same facts (wording, length and extra true detail don't matter)
  0.5 partly correct: some of the gold facts, or right but missing a key condition
  0   wrong, contradicting, or "not found"
Yes/no gold: only the polarity (and, if the gold states one, the condition) counts.

The judge is the local qwen2.5:7b through the same LLM cache (deterministic).
Bias note: the judge is the same model family as the generator; use it for
comparing runs with each other, not as an absolute score.

Usage:  python tools/judge.py RUN_DIR        (the folder holding results.jsonl)
Writes  RUN_DIR/judge_rows.jsonl and prints the mean per dataset/type.
"""
from __future__ import annotations
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("PYTHONHASHSEED", "0")

from registerrag import llm_cache  # noqa: E402
from registerrag.config import Config  # noqa: E402
from registerrag.llm_client import call_llm  # noqa: E402

SYSTEM = """You grade answers to questions against a reference answer.

Score:
1   = the answer states the same facts as the reference (different wording, a shorter or
      longer answer, or extra correct detail are all fine)
0.5 = partly correct: some of the reference's facts, or correct but missing a condition
      or qualifier that the reference treats as essential
0   = wrong, contradicts the reference, answers a different question, or says the
      information is not available

For yes/no questions the polarity must match; if the reference adds a condition, a
matching polarity without it is 0.5.

Return ONLY JSON: {"score": 0 | 0.5 | 1}"""


def judge(cfg, q, gold, pred):
    user = f"Question: {q}\nReference answer: {gold}\nAnswer to grade: {pred}"
    text, _ = call_llm([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                       cfg, json_mode=True, max_tokens=20)
    m = re.search(r"\"score\"\s*:\s*\"?([01](?:\.5)?)", text or "")
    return float(m.group(1)) if m else 0.0


def main():
    run = sys.argv[1]
    sets = sys.argv[2:] or [""]
    cfg = Config()
    cfg.llm_provider, cfg.llm_model, cfg.llm_temperature = "ollama", "qwen2.5:7b", 0.0
    llm_cache.configure(os.path.join(ROOT, "cache", "judge-qwen2.5-7b.sqlite"))
    for ds in sets:
        d = os.path.join(run, ds)
        rp = os.path.join(d, "results.jsonl")
        if not os.path.exists(rp):
            continue
        types = {}
        sp = os.path.join(d, "score_rows.jsonl")
        if os.path.exists(sp):
            types = {r["id"]: r["type"] for r in map(json.loads, open(sp, encoding="utf-8"))}
        rows, by = [], defaultdict(list)
        for r in map(json.loads, open(rp, encoding="utf-8")):
            s = judge(cfg, r["question"], r["gold_answer"], r["predicted_answer"])
            rows.append({"id": r["question_id"], "type": types.get(r["question_id"], ""), "judge": s})
            by[types.get(r["question_id"], "")].append(s)
        with open(os.path.join(d, "judge_rows.jsonl"), "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x) + "\n")
        allv = [x["judge"] for x in rows]
        print(f"{ds}: judge {sum(allv) / len(allv):.3f} (n={len(allv)})  " +
              "  ".join(f"{t}={sum(v) / len(v):.2f}" for t, v in sorted(by.items())))


if __name__ == "__main__":
    main()
