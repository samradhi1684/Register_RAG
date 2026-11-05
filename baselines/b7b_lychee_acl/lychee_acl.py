"""B7b - LycheeMemory (ACL 2026) wrapper. The model runs in a separate venv (baselines/b7b_lychee_acl/worker.py);
this module only builds the long context, talks to the worker over JSON lines, and records everything."""
from __future__ import annotations

import json, os, re, subprocess, sys, threading

from registerrag.pipelines import BasePipeline, PipelineResult

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
HF_REVISION = "17bb642d067328d6c54b6f4e12bf70528c8c5c22"      # lerverson/LycheeMemory-7B (pinned)
PREFIX = "@@RESULT "
BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def build_context(docs) -> str:
    """Same layout as the authors' test.json: 'Document i:\\n<title>\\n<text>' blocks."""
    return "\n\n".join(f"Document {i}:\n{d.title}\n{d.text}" for i, d in enumerate(docs, 1))


def extract_answer(raw: str) -> str:
    """The checkpoint's output format is unverified (no eval script is released): take the last \\boxed{...} if present,
    else the text after a final 'Answer:', else the whole stripped output. The raw text is always kept in `extra`."""
    b = BOXED.findall(raw)
    if b:
        return b[-1].strip()
    m = re.findall(r"(?is)\banswer\s*(?:is)?\s*:\s*(.+)", raw)
    return (m[-1] if m else raw).strip().splitlines()[0].strip() if raw.strip() else ""


class Worker:
    """One persistent model process. Restarted (once per question) if it dies or exceeds the timeout."""

    def __init__(self, cmd, timeout):
        self.cmd, self.timeout, self.p = cmd, timeout, None

    @staticmethod
    def free_gpu_from_ollama():
        """The 7B fills a 16 GB card: unload whatever Ollama still holds (the harness preflight loads the 3B)."""
        import urllib.request
        try:
            base = "http://127.0.0.1:11434"
            for m in json.load(urllib.request.urlopen(base + "/api/ps", timeout=5)).get("models", []):
                req = urllib.request.Request(base + "/api/generate", method="POST", headers={"Content-Type": "application/json"},
                                             data=json.dumps({"model": m["name"], "keep_alive": 0}).encode())
                urllib.request.urlopen(req, timeout=30).read()
        except Exception:                                  # Ollama not running: nothing to free
            pass

    def start(self):
        self.free_gpu_from_ollama()
        # the parent may run with CUDA_VISIBLE_DEVICES="" (CPU-only harness side); the worker must see the GPU
        self.p = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True,
                                  encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1", "CUDA_VISIBLE_DEVICES": "0"}, cwd=ROOT)
        ready = self._read("__ready__", 1800)           # model load can take minutes the first time (8 GB download)
        if ready.get("error"):
            raise RuntimeError(f"worker failed to start: {ready['error']}")

    def _read(self, want, timeout):
        box = {}

        def reader():
            for line in self.p.stdout:
                if line.startswith(PREFIX):
                    obj = json.loads(line[len(PREFIX):])
                    if obj.get("id") == want:
                        box["r"] = obj; return
            box["dead"] = True

        t = threading.Thread(target=reader, daemon=True); t.start(); t.join(timeout)
        if box.get("dead"):
            code = self.p.poll(); self.kill(); raise OSError(f"worker exited (code {code})")
        if "r" not in box:
            self.kill(); raise TimeoutError(f"worker silent for {timeout}s")
        return box["r"]

    def ask(self, req):
        if self.p is None or self.p.poll() is not None:
            self.start()
        self.p.stdin.write(json.dumps(req) + "\n"); self.p.stdin.flush()
        return self._read(req["id"], self.timeout)

    def kill(self):
        if self.p and self.p.poll() is None:
            self.p.kill()
        self.p = None


class LycheeACL(BasePipeline):
    name = "lycheeacl"
    _worker = None                                     # shared by all instances in the process

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        m = self.margs = dict(self.cfg.method_args or {})
        self.timeout = float(m.get("timeout_s", 3600))
        if "worker_cmd" in m:                          # tests: a fake worker
            self.cmd = json.loads(m["worker_cmd"]) if isinstance(m["worker_cmd"], str) else list(m["worker_cmd"])
        else:
            py = m.get("python", os.path.join(os.path.dirname(ROOT), ".venv_lychee", "Scripts", "python.exe"))
            self.cmd = [py, os.path.join(HERE, "worker.py"), "--vendor", str(m.get("vendor", os.path.join(HERE, "vendor"))),
                        "--model", str(m.get("model", "lerverson/LycheeMemory-7B")), "--attn", str(m.get("attn", "sdpa")),
                        "--stream_chunk_size", str(m.get("stream_chunk_size", 4096)), "--ratio", str(m.get("ratio", 4)),
                        "--window", str(m.get("window", 2048)), "--stride", str(m.get("stride", 2048)),
                        "--max_new_tokens", str(m.get("max_new_tokens", 1024))]
            if not os.path.exists(py) or not os.path.isdir(self.cmd[3]):
                raise RuntimeError("B7b is not set up: run  powershell baselines/b7b_lychee_acl/tools/setup_env.ps1  "
                                   f"(missing {py} or the vendor checkout {self.cmd[3]})")
            self.cmd += ["--revision", str(m.get("revision", HF_REVISION))]
        if LycheeACL._worker is None or LycheeACL._worker.cmd != self.cmd:
            LycheeACL._worker = Worker(self.cmd, self.timeout)

    def run(self, ex, ledger):
        docs = [self.store.docs[d] for d in ex.doc_ids]
        ctx = build_context(docs)
        r = None
        for attempt in (1, 2):                         # a dead / hung worker is restarted once, then the question errors
            try:
                r = LycheeACL._worker.ask({"id": ex.qid, "context": ctx, "question": ex.question}); break
            except (TimeoutError, OSError, ValueError) as e:
                LycheeACL._worker.kill()
                if attempt == 2:
                    raise RuntimeError(f"[generator] worker failed twice: {e!r}") from e
        if r.get("error"):
            raise RuntimeError(f"[generator] {r['error']}")
        raw = r["answer"]
        ledger.lm_call("generator", r["text_tokens"] + r["question_tokens"], r["new_tokens"])
        extra = {"raw_output": raw, "text_tokens": r["text_tokens"], "n_context_docs": len(docs),
                 "worker_seconds": r["seconds"], "worker_peak_mem_mb": r.get("peak_mem_mb"), "note":
                 "retrieval metrics are not applicable: the model receives the question's whole context, no retrieval"}
        return PipelineResult(extract_answer(raw), [], [], [], [], False, extra)
