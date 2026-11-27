"""
Persistent LLM response cache — makes runs reproducible.

Why: on the local GPU server, an identical temperature-0 request occasionally
returns a different completion (floating-point non-determinism at near-tied
tokens; measured ~1 in 8 repeats on some planner prompts).  A fixed `seed`
does not help.  That noise changes the planner's step list on 9–11 of 100
2Wiki questions between otherwise identical runs, so a ±1–2 question
difference between two runs cannot be attributed to a code change.

With the cache on, every completion is stored under a key built from
(provider, model, messages, temperature, max_tokens, json_mode).  A rerun
replays stored completions; only calls whose prompt changed reach the model.
Two runs therefore differ only where the code made them differ.

Sampled calls (temperature > 0, e.g. self-consistency votes) repeat the same
request k times on purpose; each repeat gets its occurrence number in the key,
so the k samples stay distinct and are replayed in the same order.

Storage: one SQLite file (safe across interrupted runs).  Off unless
config.llm_cache_path is set (CLI: --llm-cache PATH).

Usage accounting: every request (cached or not) is counted in USAGE —
requests, prompt_tokens, completion_tokens (as reported by the server).  The
server's token counts are stored next to the completion, so a replayed run
reports the same tokens as the run that made the calls.  Entries written before
this column existed replay with 0 tokens (counted in USAGE["unknown_usage"]).
"""
from __future__ import annotations
import hashlib
import json
import logging
import sqlite3
import threading
from collections import Counter
from types import SimpleNamespace
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# process-wide request/token counters (read deltas around a unit of work)
USAGE: Counter = Counter()
_USAGE_LOCK = threading.Lock()


def record_usage(usage) -> dict:
    """OpenAI-style usage object/dict -> {"prompt_tokens", "completion_tokens"}; adds it to USAGE."""
    if usage is None:
        u = None
    elif isinstance(usage, dict):
        u = usage
    else:
        u = {"prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
             "completion_tokens": getattr(usage, "completion_tokens", 0) or 0}
    with _USAGE_LOCK:
        USAGE["requests"] += 1
        if u is None:
            USAGE["unknown_usage"] += 1
            return {}
        USAGE["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        USAGE["completion_tokens"] += int(u.get("completion_tokens") or 0)
    return u


def usage_snapshot() -> dict:
    with _USAGE_LOCK:
        return dict(USAGE)


def _split(result):
    """call() may return text or (text, usage)."""
    return result if isinstance(result, tuple) else (result, None)


class LLMCache:
    def __init__(self, path: str):
        self.path = path
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS completions (k TEXT PRIMARY KEY, text TEXT)")
        cols = [r[1] for r in self._db.execute("PRAGMA table_info(completions)")]
        if "usage" not in cols:
            self._db.execute("ALTER TABLE completions ADD COLUMN usage TEXT")
        self._db.commit()
        self._lock = threading.Lock()
        self._seen: Counter = Counter()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(params: dict, occurrence: int) -> str:
        blob = json.dumps({**params, "_occ": occurrence}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get_or_call(self, params: dict, call: Callable[[], object]) -> str:
        """call() returns text or (text, usage)."""
        base = json.dumps(params, sort_keys=True, ensure_ascii=False)
        with self._lock:
            occ = 0
            if float(params.get("temperature") or 0.0) > 0.0:
                occ = self._seen[base]
                self._seen[base] += 1
            key = self._key(params, occ)
            row = self._db.execute("SELECT text, usage FROM completions WHERE k=?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            record_usage(json.loads(row[1]) if row[1] else None)
            return row[0]
        text, usage = _split(call())
        u = record_usage(usage)
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO completions (k, text, usage) VALUES (?, ?, ?)",
                             (key, text, json.dumps(u) if u else None))
            self._db.commit()
        self.misses += 1
        return text

    def stats(self) -> str:
        return f"LLM cache {self.path}: {self.hits} hits, {self.misses} misses"


_CACHE: Optional[LLMCache] = None


def configure(path: Optional[str]) -> Optional[LLMCache]:
    """Open (or close, with path=None/'') the process-wide cache."""
    global _CACHE
    _CACHE = LLMCache(path) if path else None
    if _CACHE:
        logger.info("LLM response cache: %s", path)
    return _CACHE


def active() -> Optional[LLMCache]:
    return _CACHE


class CachedChatClient:
    """
    Wraps an OpenAI-style client so `client.chat.completions.create(...)`
    goes through the cache.  Only `.choices[0].message.content` is returned.
    """

    def __init__(self, client, provider: str):
        self._client = client
        self._provider = provider
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        cache = active()
        if cache is None:
            r = self._client.chat.completions.create(**kw)
            record_usage(getattr(r, "usage", None))
            return r
        params = {
            "provider": self._provider,
            "model": kw.get("model"),
            "messages": kw.get("messages"),
            "temperature": kw.get("temperature", 0.0),
            "max_tokens": kw.get("max_tokens"),
            "json_mode": bool(kw.get("response_format")),
        }

        def call():
            r = self._client.chat.completions.create(**kw)
            return r.choices[0].message.content or "", getattr(r, "usage", None)

        text = cache.get_or_call(params, call)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
