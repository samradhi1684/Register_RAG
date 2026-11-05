from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional


class TokenCounter:
    """tiktoken (cl100k) if usable, else a regex word/punctuation estimate. The choice is recorded in the
    run config because every token number in the paper depends on it."""

    def __init__(self):
        self.name = "approx_regex"
        self._enc = None
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding("cl100k_base")
            self.name = "tiktoken_cl100k"
        except Exception:
            pass

    def count(self, text: str) -> int:
        if self._enc is not None:
            return len(self._enc.encode(text))
        return len(re.findall(r"\w+|[^\w\s]", text))


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class OllamaGenerator:
    def __init__(self, model="qwen2.5:14b", host="http://127.0.0.1:11434", temperature=0.0, seed=0,
                 num_ctx=8192, max_new_tokens=64, timeout=120):
        self.name, self.host, self.temperature, self.seed = model, host.rstrip("/"), temperature, seed
        self.num_ctx, self.max_new_tokens, self.timeout = num_ctx, max_new_tokens, timeout

    def generate(self, system: str, user: str, json_mode: bool = False, max_new_tokens: Optional[int] = None) -> GenResult:
        import requests
        t0 = time.perf_counter()
        body = {"model": self.name, "stream": False, "keep_alive": "30m",
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": self.temperature, "seed": self.seed,
                            "num_ctx": self.num_ctx, "num_predict": max_new_tokens or self.max_new_tokens}}
        if json_mode:
            body["format"] = "json"          # Ollama constrains decoding to valid JSON
        last = None
        for attempt in range(2):
            try:
                r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
                r.raise_for_status()
                j = r.json()
                return GenResult(j["message"]["content"].strip(), j.get("prompt_eval_count", 0),
                                 j.get("eval_count", 0), time.perf_counter() - t0)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"ollama failed: {last}")

    def ping(self):
        """Preflight: server reachable, model installed, one tiny generation works. Raises with a clear message."""
        import requests
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Cannot reach Ollama at {self.host}: {e}. Is Ollama running?") from e
        names = {m["name"] for m in r.json().get("models", [])}
        if self.name not in names and f"{self.name}:latest" not in names:
            raise RuntimeError(f"Model '{self.name}' not installed. Installed: {sorted(names)}")
        t0 = time.perf_counter()
        out = self.generate("Reply with one word.", "Say OK")
        return f"ollama ok: {self.name} answered {out.text!r} in {time.perf_counter() - t0:.1f}s"


class OpenAIGenerator:
    def __init__(self, model="gpt-4o-mini", temperature=0.0, seed=0, max_new_tokens=64):
        from openai import OpenAI
        self.client, self.name = OpenAI(), model
        self.temperature, self.seed, self.max_new_tokens = temperature, seed, max_new_tokens

    def generate(self, system: str, user: str, json_mode: bool = False, max_new_tokens: Optional[int] = None) -> GenResult:
        t0 = time.perf_counter()
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        r = self.client.chat.completions.create(
            model=self.name, temperature=self.temperature, seed=self.seed,
            max_tokens=max_new_tokens or self.max_new_tokens, **extra,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        u = r.usage
        return GenResult(r.choices[0].message.content.strip(), u.prompt_tokens, u.completion_tokens,
                         time.perf_counter() - t0)


class HFTokenCounter:
    """Exact token counts from the generator's own tokenizer (preferred for the budget experiments)."""

    def __init__(self, tokenizer, name):
        self._tok, self.name = tokenizer, name

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))


class HFGenerator:
    """Runs a chat model locally with transformers (no Ollama). Use  --generator hf:Qwen/Qwen2.5-3B-Instruct
    bf16 weights: 3B ~6 GB, 7B ~15 GB (too tight for 16 GB VRAM -> use 3B, or quantise)."""

    def __init__(self, model="Qwen/Qwen2.5-3B-Instruct", temperature=0.0, seed=0, max_new_tokens=64,
                 dtype="bfloat16", device="cuda"):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name, self.temperature, self.seed = f"hf:{model}", temperature, seed
        self.max_new_tokens, self.device, self._torch = max_new_tokens, device, torch
        self.tok = AutoTokenizer.from_pretrained(model)
        ver = tuple(int(x) for x in transformers.__version__.split(".")[:2] if x.isdigit())
        key = "dtype" if ver >= (4, 56) else "torch_dtype"      # kwarg was renamed in transformers 4.56
        self.model = AutoModelForCausalLM.from_pretrained(model, **{key: getattr(torch, dtype)}).to(device).eval()
        self.counter = HFTokenCounter(self.tok, f"hf_tokenizer:{model}")

    def generate(self, system: str, user: str, json_mode: bool = False, max_new_tokens: Optional[int] = None) -> GenResult:
        torch = self._torch
        t0 = time.perf_counter()
        text = self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        kw = {"do_sample": False} if self.temperature <= 0 else {"do_sample": True, "temperature": self.temperature}
        if self.temperature > 0:
            torch.manual_seed(self.seed)
        with torch.inference_mode():
            out = self.model.generate(**ids, max_new_tokens=max_new_tokens or self.max_new_tokens,
                                      pad_token_id=self.tok.eos_token_id, **kw)
        new = out[0, ids["input_ids"].shape[1]:]
        return GenResult(self.tok.decode(new, skip_special_tokens=True).strip(), int(ids["input_ids"].shape[1]),
                         int(new.shape[0]), time.perf_counter() - t0)

    def ping(self):
        t0 = time.perf_counter()
        out = self.generate("Reply with one word.", "Say OK")
        return f"hf ok: {self.name} answered {out.text!r} in {time.perf_counter() - t0:.1f}s"


class StubGenerator:
    """Offline stub. Returns a one-requirement decomposition for compiler prompts, else 'stub'. TEST ONLY."""
    name = "stub"

    def generate(self, system: str, user: str, json_mode: bool = False, max_new_tokens: Optional[int] = None) -> GenResult:
        out = "stub"
        if "question decomposition agent" in system:
            q = user.split("Original question:", 1)[-1].split("\n\nExample:", 1)[0].strip()
            out = json.dumps([q])
        elif "planning agent that writes Python code" in system and "runtime error" not in user:
            q = user.split("Original question:\n", 1)[-1].split("\n\nSub-queries", 1)[0].strip()
            out = f"```python\ndocs = retrieve({q!r})\nfinal_answer = answer({q!r}, docs)\n```"
        elif "Decompose" in system:
            q = user.split("Question:", 1)[-1].strip()
            out = json.dumps({"requirements": [{"id": "R1", "need": q, "depends_on": []}]})
        return GenResult(out, len(user.split()), 1, 0.0)


def parse_requirements(text: str, question: str):
    """Robustly parse the compiler JSON. Returns (requirements, parse_ok)."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        reqs = json.loads(m.group(0))["requirements"]
        out = [{"id": r.get("id", f"R{i+1}"), "need": str(r["need"]), "depends_on": list(r.get("depends_on", []))}
               for i, r in enumerate(reqs)]
        if out:
            return out, True
    except Exception:  # noqa: BLE001
        pass
    return [{"id": "R1", "need": question, "depends_on": []}], False


def build_prompt(counter: TokenCounter, system: str, question: str, units: list, index_text,
                 budget: Optional[int]):
    """Add evidence units in rank order while the whole prompt (system + evidence + question) stays within
    `budget` tokens. Returns (user_prompt, included_units, truncated)."""
    from .prompts import qa_prompt
    kept = []
    fixed = counter.count(system) + counter.count(qa_prompt(question, []))
    used = fixed
    for u in units:
        t = counter.count(index_text(u)) + 4
        if budget is not None and used + t > budget:
            return qa_prompt(question, [index_text(x) for x in kept]), kept, True
        kept.append(u)
        used += t
    return qa_prompt(question, [index_text(x) for x in kept]), kept, False
