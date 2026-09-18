"""Thin Claude client used by agents and by tools that embed model calls.

Pattern note (important for this project's design story): the news sentiment
tool calls Claude *inside an MCP tool*, i.e. one agent's tool invokes the model
as a subroutine. This file is the single seam for that, so it can be mocked in
tests and so the offline-fallback policy lives in exactly one place.

Offline mode: if ANTHROPIC_API_KEY is unset, `available` is False and
complete() raises ModelUnavailableError. Server/agent code decides per-call
whether to (a) fail with an explicit envelope, or (b) fall back to a
deterministic, clearly-labeled heuristic. Which choice was made is always
recorded in the trace and visible in memos (no silent mode switching).
"""

from __future__ import annotations

from core.config import get_settings


class ModelUnavailableError(RuntimeError):
    """Raised when a model call is attempted without an API key."""


def available() -> bool:
    return bool(get_settings().anthropic_api_key)


def client():
    """Construct an Anthropic client from env config (never hardcoded keys)."""
    s = get_settings()
    if not s.anthropic_api_key:
        raise ModelUnavailableError(
            "ANTHROPIC_API_KEY is not set — no model calls available. "
            "Copy .env.example to .env and add your key for live Claude mode."
        )
    import anthropic  # imported lazily so offline mode never needs the SDK key

    return anthropic.Anthropic(api_key=s.anthropic_api_key)


def complete(prompt: str, system: str | None = None, max_tokens: int | None = None) -> str:
    """One-shot completion. Raises ModelUnavailableError in offline mode."""
    s = get_settings()
    if not available():
        raise ModelUnavailableError("ANTHROPIC_API_KEY is not set")
    c = client()
    kwargs: dict = {
        "model": s.model,
        "max_tokens": max_tokens or s.max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    resp = c.messages.create(**kwargs)
    return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")


def complete_json(prompt: str, system: str | None = None, max_tokens: int | None = None) -> str:
    """Completion instructed to return pure JSON (caller parses + validates)."""
    sys = (system or "") + "\nRespond with a single JSON object and nothing else."
    return complete(prompt, system=sys, max_tokens=max_tokens)
