"""Stand-in for worker.py (same protocol, no model). Modes via argv[1]: ok | boxed | err | die_once."""
import json, os, sys

mode = sys.argv[1] if len(sys.argv) > 1 else "ok"
marker = sys.argv[2] if len(sys.argv) > 2 else ""
print("library chatter that must be ignored", flush=True)
print("@@RESULT " + json.dumps({"id": "__ready__", "error": None}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if mode == "die_once" and not os.path.exists(marker):
        open(marker, "w").write("x"); sys.exit(3)            # simulates a crash (e.g. native access violation)
    if mode == "err":
        out = {"id": req["id"], "answer": "", "error": "CUDA out of memory", "seconds": 0.1}
    else:
        ans = "reasoning...\n\\boxed{Paris}" if mode == "boxed" else "Answer: Paris\nextra"
        out = {"id": req["id"], "answer": ans, "text_tokens": len(req["context"].split()), "question_tokens": 5,
               "new_tokens": 3, "seconds": 0.1, "peak_mem_mb": 1234, "error": None}
    print("noise", flush=True)
    print("@@RESULT " + json.dumps(out), flush=True)
