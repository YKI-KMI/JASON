"""Tests for core infrastructure: TTL cache, rate limiter, trace logger."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from core.cache import TTLCache
from core.ratelimit import RateLimiter
from core.trace import TraceLogger, read_trace


class TestTTLCache:
    def test_set_then_get_roundtrip(self, tmp_path: Path) -> None:
        cache = TTLCache(tmp_path / "cache.db", ttl_seconds=60)
        cache.set("ns", {"ticker": "AAPL"}, {"close": 123.4})
        assert cache.get("ns", {"ticker": "AAPL"}) == {"close": 123.4}

    def test_missing_key_returns_none(self, tmp_path: Path) -> None:
        cache = TTLCache(tmp_path / "cache.db", ttl_seconds=60)
        assert cache.get("ns", {"nope": 1}) is None

    def test_ttl_expiry(self, tmp_path: Path) -> None:
        cache = TTLCache(tmp_path / "cache.db", ttl_seconds=0)
        cache.set("ns", {"k": 1}, "v", ttl_seconds=0.05)
        assert cache.get("ns", {"k": 1}) == "v"
        time.sleep(0.08)
        assert cache.get("ns", {"k": 1}) is None

    def test_namespaces_are_isolated(self, tmp_path: Path) -> None:
        cache = TTLCache(tmp_path / "cache.db", ttl_seconds=60)
        cache.set("a", {"k": 1}, "from-a")
        cache.set("b", {"k": 1}, "from-b")
        assert cache.get("a", {"k": 1}) == "from-a"
        assert cache.get("b", {"k": 1}) == "from-b"

    def test_overwrite_same_key(self, tmp_path: Path) -> None:
        cache = TTLCache(tmp_path / "cache.db", ttl_seconds=60)
        cache.set("ns", {"k": 1}, "first")
        cache.set("ns", {"k": 1}, "second")
        assert cache.get("ns", {"k": 1}) == "second"

    def test_persists_across_instances(self, tmp_path: Path) -> None:
        db = tmp_path / "cache.db"
        c1 = TTLCache(db, ttl_seconds=60)
        c1.set("ns", {"k": 1}, {"v": [1, 2, 3]})
        c1.close()
        c2 = TTLCache(db, ttl_seconds=60)
        assert c2.get("ns", {"k": 1}) == {"v": [1, 2, 3]}


class TestRateLimiter:
    def test_first_call_is_immediate(self) -> None:
        rl = RateLimiter(calls_per_minute=600)
        start = time.monotonic()
        rl.wait()
        assert time.monotonic() - start < 0.1

    def test_enforces_min_interval(self) -> None:
        rl = RateLimiter(calls_per_minute=1200)  # min interval = 50ms
        t0 = time.monotonic()
        rl.wait()
        rl.wait()
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.045  # allow small scheduling slack


class TestTraceLogger:
    def test_writes_jsonl_lines(self, tmp_path: Path) -> None:
        tl = TraceLogger("testrun", trace_dir=tmp_path)
        tl.agent_step("planner", "decompose", detail="x")
        tl.tool_call("quant_agent", "quant_signals", "compute_momentum", {"lookback_days": 20}, {"score": 0.1})
        events = read_trace(tl.path)
        assert len(events) == 2
        assert events[0]["event_type"] == "agent_step"
        assert events[0]["agent"] == "planner"
        assert events[1]["event_type"] == "tool_call"
        assert events[1]["args"] == {"lookback_days": 20}

    def test_events_have_timestamp_and_run_id(self, tmp_path: Path) -> None:
        tl = TraceLogger("run-abc", trace_dir=tmp_path)
        tl.log("custom", extra=1)
        ev = read_trace(tl.path)[0]
        assert ev["run_id"] == "run-abc"
        assert "timestamp" in ev
        assert ev["extra"] == 1

    def test_lines_are_valid_json(self, tmp_path: Path) -> None:
        tl = TraceLogger("j", trace_dir=tmp_path)
        for i in range(5):
            tl.log("evt", i=i)
        lines = tl.path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5
        for line in lines:
            assert json.loads(line)["event_type"] == "evt"
