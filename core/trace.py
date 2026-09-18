"""Structured JSONL trace logging.

Every agent step and MCP tool call is appended as one JSON object per line to
data/traces/<run_id>.jsonl. The trace is the explainability artifact: each memo
number can be traced back to a tool call with args and result.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import get_settings

_lock = threading.Lock()


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


class TraceLogger:
    def __init__(self, run_id: str, trace_dir: Path | None = None) -> None:
        self.run_id = run_id
        # Default under the configured data dir so tests can isolate via settings.
        self.dir = Path(trace_dir) if trace_dir else get_settings().data_dir / "traces"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{run_id}.jsonl"

    def log(self, event_type: str, **fields: Any) -> dict[str, Any]:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event_type": event_type,
            **fields,
        }
        line = json.dumps(event, default=str)
        with _lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return event

    def agent_step(self, agent: str, action: str, **fields: Any) -> dict[str, Any]:
        return self.log("agent_step", agent=agent, action=action, **fields)

    def tool_call(self, agent: str, server: str, tool: str, args: dict[str, Any], result: Any, error: str | None = None, duration_ms: float | None = None) -> dict[str, Any]:
        return self.log(
            "tool_call",
            agent=agent,
            server=server,
            tool=tool,
            args=args,
            result=result,
            error=error,
            duration_ms=duration_ms,
        )


def read_trace(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
