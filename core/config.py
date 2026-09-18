"""Central configuration.

Secrets and knobs come from environment variables (optionally via a .env file).
No default API keys are ever hard-coded here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env if present; real values live there, never in code

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
TRACE_DIR = DATA_DIR / "traces"


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    # Model settings. If ANTHROPIC_API_KEY is unset the system runs in
    # "offline" mode: deterministic heuristic stand-ins for model calls so the
    # pipeline is testable without secrets. This is printed in memos/trace.
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    model: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"))
    max_tokens: int = field(default_factory=lambda: _int_env("ANTHROPIC_MAX_TOKENS", 4000))

    # News: optional NewsAPI key; GDELT and RSS fallbacks need no key.
    newsapi_key: str = field(default_factory=lambda: os.environ.get("NEWS_API_KEY", ""))

    # Cache / rate limiting
    cache_ttl_seconds: int = field(default_factory=lambda: _int_env("CACHE_TTL_SECONDS", 900))
    rate_limit_per_min: int = field(default_factory=lambda: _int_env("RATE_LIMIT_PER_MIN", 30))
    data_dir: Path = field(default_factory=lambda: DATA_DIR)

    # Orchestrator
    max_verification_retries: int = field(default_factory=lambda: _int_env("MAX_VERIFICATION_RETRIES", 1))
    lookback_days: int = field(default_factory=lambda: _int_env("DEFAULT_LOOKBACK_DAYS", 90))


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
