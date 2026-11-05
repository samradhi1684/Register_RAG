"""Worker for B7b. Runs in the ISOLATED venv (.venv_lychee: transformers==4.53.2), never in the main venv.

Protocol (stdin/stdout, one JSON object per line; library chatter on stdout is ignored because every reply is prefixed):
  in : {"id": ..., "context": "...", "question": "..."}
  out: @@RESULT {"id": ..., "answer": "...", "text_tokens": n, "question_tokens": n, "new_tokens": n, "seconds": s,
                  "peak_mem_mb": m, "error": null|"..."}
Inference follows the authors' example.py exactly (raw text, no chat template, memory_mode="recurrent", greedy decoding).
"""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--vendor", required=True, help="checkout of github.com/owoakuma/LycheeMemory (provides src/)")
ap.add_argument("--model", default="lerverson/LycheeMemory-7B")
ap.add_argument("--revision", default=None)
ap.add_argument("--attn", default="sdpa")
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--stream_chunk_size", type=int, default=4096)
ap.add_argument("--ratio", type=int, default=4)
ap.add_argument("--window", type=int, default=2048)
ap.add_argument("--stride", type=int, default=2048)
ap.add_argument("--max_new_tokens", type=int, default=1024)
a = ap.parse_args()
sys.path.insert(0, os.path.abspath(a.vendor))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from transformers import AutoTokenizer
from src.modeling_qwen2 import Qwen2ForCausalLM     # authors' code (vendor checkout)


def reply(obj):
    sys.stdout.write("@@RESULT " + json.dumps(obj) + "\n"); sys.stdout.flush()


tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = Qwen2ForCausalLM.from_pretrained(a.model, revision=a.revision, trust_remote_code=True,
                                         torch_dtype=getattr(torch, a.dtype), attn_implementation=a.attn,
                                         low_cpu_mem_usage=False)
model.memory.set(lychee_memory_ratio=[a.ratio], lychee_memory_window=a.window, lychee_memory_stride=a.stride)
meta = [n for n, p in model.named_parameters() if p.is_meta]
if meta:
    raise RuntimeError(f"model still has meta tensors: {meta[:5]}")
model = model.to("cuda").eval()
reply({"id": "__ready__", "attn": a.attn, "dtype": a.dtype, "revision": a.revision, "error": None})

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    t0 = time.perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats()
        t = tok(req["context"], return_tensors="pt", add_special_tokens=False)
        q = tok(req["question"], return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            out = model.generate(input_ids=q["input_ids"].cuda(), attention_mask=q["attention_mask"].cuda(),
                                 text_ids=t["input_ids"].cuda(), text_attention_mask=t["attention_mask"].cuda(),
                                 stream_chunk_size=a.stream_chunk_size, max_new_tokens=a.max_new_tokens,
                                 do_sample=False, memory_mode="recurrent", tokenizer=tok,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        ans_ids = out[:, q["input_ids"].shape[1]:]
        reply({"id": req["id"], "answer": tok.decode(ans_ids[0], skip_special_tokens=True).strip(),
               "text_tokens": int(t["input_ids"].shape[1]), "question_tokens": int(q["input_ids"].shape[1]),
               "new_tokens": int(ans_ids.shape[1]), "seconds": round(time.perf_counter() - t0, 2),
               "peak_mem_mb": int(torch.cuda.max_memory_allocated() / 2**20), "error": None})
    except Exception as e:  # noqa: BLE001  (e.g. CUDA OOM: report it, free memory, keep serving)
        import traceback
        torch.cuda.empty_cache()
        sys.stderr.write(traceback.format_exc()); sys.stderr.flush()          # full trace to the log, short error to the record
        reply({"id": req["id"], "answer": "", "error": repr(e)[:500], "seconds": round(time.perf_counter() - t0, 2)})
