"""
Thin LLM adapter — OpenAI and Anthropic behind one interface.
Returns (text, calls_made) so callers can track LLM budget.
"""
from __future__ import annotations
import json
import logging
from typing import Any, Optional

from registerrag.config import Config

logger = logging.getLogger(__name__)


def call_llm(
    messages: list[dict],
    config: Config,
    json_mode: bool = False,
    max_tokens: int = 1024,
) -> tuple[str, int]:
    """
    Send messages to the configured LLM.

    Returns
    -------
    (response_text, 1)   — always 1 call; callers accumulate
    """
    from registerrag import llm_cache
    cache = llm_cache.active()
    if cache is not None:
        params = {"provider": config.llm_provider, "model": config.llm_model,
                  "messages": messages, "temperature": config.llm_temperature,
                  "max_tokens": max_tokens, "json_mode": json_mode}
        text = cache.get_or_call(params, lambda: _dispatch_with_usage(messages, config, json_mode, max_tokens))
        return text, 1
    text, usage = _dispatch_with_usage(messages, config, json_mode, max_tokens)
    llm_cache.record_usage(usage)
    return text, 1


_LAST_USAGE = None   # usage of the most recent provider response (set by _call_*)


def _dispatch_with_usage(messages, config, json_mode, max_tokens):
    """-> (text, usage dict or None)."""
    global _LAST_USAGE
    _LAST_USAGE = None
    text = _dispatch(messages, config, json_mode, max_tokens)[0]
    return text, _LAST_USAGE


def _remember(prompt_tokens, completion_tokens):
    global _LAST_USAGE
    _LAST_USAGE = {"prompt_tokens": int(prompt_tokens or 0), "completion_tokens": int(completion_tokens or 0)}


def _dispatch(messages, config, json_mode, max_tokens) -> tuple[str, int]:
    if config.llm_provider == "openai":
        return _call_openai(messages, config, json_mode, max_tokens)
    elif config.llm_provider == "anthropic":
        return _call_anthropic(messages, config, max_tokens)
    elif config.llm_provider == "ollama":
        return _call_ollama(messages, config, json_mode, max_tokens)
    else:
        raise ValueError(f"Unknown LLM provider: {config.llm_provider!r}")


def _call_openai(
    messages: list[dict],
    config: Config,
    json_mode: bool,
    max_tokens: int,
) -> tuple[str, int]:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI(api_key=config.openai_api_key)
    kwargs: dict[str, Any] = {
        "model": config.llm_model,
        "messages": messages,
        "temperature": config.llm_temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    if resp.usage:
        _remember(resp.usage.prompt_tokens, resp.usage.completion_tokens)
    logger.debug("OpenAI call: %d prompt + %d completion tokens",
                 resp.usage.prompt_tokens, resp.usage.completion_tokens)
    return text, 1


def _call_anthropic(
    messages: list[dict],
    config: Config,
    max_tokens: int,
) -> tuple[str, int]:
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    # Anthropic separates system from human/assistant turns
    system_text = ""
    user_messages: list[dict] = []
    for m in messages:
        if m["role"] == "system":
            system_text = m["content"]
        else:
            user_messages.append({"role": m["role"], "content": m["content"]})

    kwargs: dict[str, Any] = {
        "model": config.llm_model,
        "max_tokens": max_tokens,
        "messages": user_messages,
    }
    if system_text:
        kwargs["system"] = system_text

    resp = client.messages.create(**kwargs)
    text = resp.content[0].text if resp.content else ""
    if getattr(resp, "usage", None):
        _remember(resp.usage.input_tokens, resp.usage.output_tokens)
    return text, 1


def _call_ollama(
    messages: list[dict],
    config: Config,
    json_mode: bool,
    max_tokens: int,
) -> tuple[str, int]:
    """
    Ollama via its OpenAI-compatible endpoint at /v1.
    Requires Ollama running locally (or at OLLAMA_BASE_URL).
    Model must already be pulled:  ollama pull qwen2.5:7b
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai  (used as the Ollama client)")

    base = config.ollama_base_url.rstrip("/")
    client = OpenAI(base_url=f"{base}/v1", api_key="ollama")

    kwargs: dict = {
        "model": config.llm_model,
        "messages": messages,
        "temperature": config.llm_temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        # Ollama supports format="json" via options, but the OpenAI-compat
        # endpoint honours response_format for recent versions
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    if resp.usage:
        _remember(resp.usage.prompt_tokens, resp.usage.completion_tokens)
    if resp.usage and resp.usage.prompt_tokens and resp.usage.prompt_tokens > 2900:
        # the local server silently drops the start of prompts beyond its window
        logger.warning("Long prompt: %d tokens — may be truncated by the server",
                       resp.usage.prompt_tokens)
    logger.debug("Ollama call (%s): %d tokens out", config.llm_model,
                 resp.usage.completion_tokens if resp.usage else -1)
    return text, 1


def parse_json_response(text: str) -> Optional[dict]:
    """Extract JSON from LLM response (handles markdown code fences)."""
    # strip ```json … ``` fences if present
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        stripped = inner.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        logger.warning("JSON parse failed for LLM response: %s…", text[:200])
        return None
