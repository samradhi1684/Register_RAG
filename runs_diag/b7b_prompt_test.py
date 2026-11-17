"""B7b diagnostic: does the answer format (not the model) explain EM=0?  Same released checkpoint, same contexts, same
greedy decoding; only the question string changes.  Variants:
  raw     the authors' example (what the B7b rows used) - read from the stored results, not re-run
  suffix  question + " Answer with a short phrase only."
  chatml  Qwen chat template (user turn = question + the same instruction, assistant turn opened)
Resumable: finished (variant, qid) pairs in runs_diag/b7b_prompt_test.jsonl are skipped."""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from registerrag.data import load_examples
from registerrag.metrics import exact_match, f1_score
from baselines.b7b_lychee_acl.lychee_acl import HERE, HF_REVISION, Worker, build_context, extract_answer

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
OUT = os.path.join(ROOT, "runs_diag", "b7b_prompt_test.jsonl")
INSTR = " Answer with a short phrase only."
VARIANTS = {
    "suffix": lambda q: q + INSTR,
    "chatml": lambda q: f"<|im_start|>user\n{q}{INSTR}<|im_end|>\n<|im_start|>assistant\n",
}

examples, docs, _ = load_examples(os.path.join(ROOT, "datasets", "wiki2_data", "2wiki_100.json"), "2wiki", N)
done = set()
if os.path.exists(OUT):
    for l in open(OUT, encoding="utf-8"):
        if l.strip():
            r = json.loads(l); done.add((r["variant"], r["qid"]))

py = os.path.join(os.path.dirname(ROOT), ".venv_lychee", "Scripts", "python.exe")
cmd = [py, os.path.join(HERE, "worker.py"), "--vendor", os.path.join(HERE, "vendor"), "--model", "lerverson/LycheeMemory-7B",
       "--attn", "sdpa", "--stream_chunk_size", "4096", "--ratio", "4", "--window", "2048", "--stride", "2048",
       "--max_new_tokens", "256", "--revision", HF_REVISION]
w = Worker(cmd, 3600)
for ex in examples:
    ctx = build_context([docs[d] for d in ex.doc_ids])
    for name, fmt in VARIANTS.items():
        if (name, ex.qid) in done:
            continue
        for attempt in (1, 2):
            try:
                r = w.ask({"id": f"{name}:{ex.qid}", "context": ctx, "question": fmt(ex.question)}); break
            except (TimeoutError, OSError) as e:
                w.kill()
                if attempt == 2:
                    raise
        raw = r.get("answer", "")
        pred = extract_answer(raw)
        rec = {"variant": name, "qid": ex.qid, "gold": ex.answers,
               "raw": raw, "pred": pred, "seconds": r.get("seconds"), "error": r.get("error")}
        rec["em"] = exact_match(pred, rec["gold"]) if rec["gold"] else None
        rec["f1"] = f1_score(pred, rec["gold"]) if rec["gold"] else None
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
        print(f"[{name}] {ex.qid[:8]} EM={rec['em']} F1={rec['f1']} {r.get('seconds')}s :: {pred[:80]!r}", flush=True)
w.kill()
print("##### B7B PROMPT TEST DONE", flush=True)
