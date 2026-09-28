"""Model-agnostic LLM layer.

The rest of the app calls exactly one function, complete(). Which vendor
answers is decided by the model registry in src/config.py, so switching from
Claude to an open-source model (or adding a new provider) never touches the
assistant, the app, or the eval code.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Callable

import requests

from src import config


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    seconds: float = 0.0

    @property
    def cost_usd(self) -> float | None:
        price = config.MODELS.get(self.model, {}).get("price")
        if not price:
            return None
        inp, out = price
        # cache reads bill at 0.1x input, cache writes at 1.25x input (5-minute cache)
        return (
            self.input_tokens * inp
            + self.cache_read_tokens * inp * 0.1
            + self.cache_write_tokens * inp * 1.25
            + self.output_tokens * out
        ) / 1_000_000


# A provider takes (spec, system_blocks, messages, max_tokens) and returns an LLMResponse.
# spec is the model's entry from config.MODELS.
ProviderFn = Callable[[dict, list[str], list[dict], int], LLMResponse]
_PROVIDERS: dict[str, ProviderFn] = {}


def register_provider(name: str, fn: ProviderFn) -> None:
    """Add a provider (used by tests for a fake model, and for future vendors)."""
    _PROVIDERS[name] = fn


def complete(system: str | list[str], messages: list[dict], model: str = config.DEFAULT_MODEL,
             max_tokens: int = 4096) -> LLMResponse:
    """The single entry point for every model call in the project.

    system:   one string, or a list of strings. With a list, the first block is the
              static part (schema, rules) and is marked for prompt caching where the
              provider supports it; later blocks are the per-question context.
    messages: [{"role": "user" | "assistant", "content": str}, ...]
    model:    a key from config.MODELS, e.g. "claude-haiku", "claude-sonnet", "qwen-coder"
    """
    if model not in config.MODELS:
        raise ValueError(f"Unknown model '{model}'. Options: {list(config.MODELS)}")
    spec = config.MODELS[model]
    provider = _PROVIDERS.get(spec["provider"])
    if provider is None:
        raise ValueError(f"No provider registered for '{spec['provider']}'")
    blocks = [system] if isinstance(system, str) else list(system)
    start = time.perf_counter()
    resp = provider(spec, blocks, messages, max_tokens)
    resp.model = model
    resp.seconds = time.perf_counter() - start
    return resp


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
_anthropic_client = None


def _anthropic(spec: dict, system_blocks: list[str], messages: list[dict], max_tokens: int) -> LLMResponse:
    global _anthropic_client
    import anthropic

    if _anthropic_client is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        _anthropic_client = anthropic.Anthropic(max_retries=3)

    system = []
    for i, block in enumerate(system_blocks):
        if not block:
            continue
        item = {"type": "text", "text": block}
        if i == 0:
            item["cache_control"] = {"type": "ephemeral"}  # schema + rules: identical every call
        system.append(item)

    kwargs = {"model": spec["model_id"], "max_tokens": max_tokens, "system": system, "messages": messages}
    if spec.get("temperature") is not None:
        kwargs["temperature"] = spec["temperature"]
    msg = _anthropic_client.messages.create(**kwargs)
    # keep only the final answer text (skips any thinking blocks)
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    u = msg.usage
    return LLMResponse(
        text=text,
        model=spec["model_id"],
        input_tokens=u.input_tokens or 0,
        output_tokens=u.output_tokens or 0,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
    )


def _ollama(spec: dict, system_blocks: list[str], messages: list[dict], max_tokens: int) -> LLMResponse:
    payload = {
        "model": spec["model_id"],
        "messages": [{"role": "system", "content": "\n\n".join(b for b in system_blocks if b)}] + messages,
        "stream": False,
        "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 16384},
    }
    try:
        r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=300)
    except requests.ConnectionError as e:
        raise RuntimeError(f"Can't reach Ollama at {config.OLLAMA_URL}. Is it running? (ollama serve)") from e
    r.raise_for_status()
    data = r.json()
    return LLMResponse(
        text=data["message"]["content"],
        model=spec["model_id"],
        input_tokens=data.get("prompt_eval_count", 0),
        output_tokens=data.get("eval_count", 0),
    )


def _groq(spec: dict, system_blocks: list[str], messages: list[dict], max_tokens: int) -> LLMResponse:
    """Groq speaks the OpenAI chat-completions format, so this works for any OpenAI-compatible API."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    payload = {
        "model": spec["model_id"],
        "messages": [{"role": "system", "content": "\n\n".join(b for b in system_blocks if b)}] + messages,
        "max_completion_tokens": max_tokens,
    }
    if spec.get("temperature") is not None:
        payload["temperature"] = spec["temperature"]
    headers = {"Authorization": f"Bearer {key}"}

    # The free tier allows only a few thousand tokens per minute, so wait and retry on 429.
    for attempt in range(6):
        r = requests.post(f"{config.GROQ_URL}/chat/completions", json=payload, headers=headers, timeout=120)
        if r.status_code != 429:
            break
        wait = float(r.headers.get("retry-after") or 0) or min(5 * 2 ** attempt, 60)
        time.sleep(min(wait, 60))
    if r.status_code >= 400:
        try:
            detail = r.json().get("error", {}).get("message", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(f"Groq API error {r.status_code}: {detail[:300]}")
    data = r.json()
    text = data["choices"][0]["message"].get("content") or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  # some models show their reasoning inline
    u = data.get("usage") or {}
    return LLMResponse(
        text=text,
        model=spec["model_id"],
        input_tokens=u.get("prompt_tokens", 0) or 0,
        output_tokens=u.get("completion_tokens", 0) or 0,
    )


register_provider("anthropic", _anthropic)
register_provider("groq", _groq)
register_provider("ollama", _ollama)
