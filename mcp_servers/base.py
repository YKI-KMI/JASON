"""Shared helpers for the four authored MCP servers.

Each server module follows the same pattern:
- tools are thin, schema-typed functions registered with FastMCP;
- pure logic lives in sibling modules (signals.py, verification.py) or in
  the wrapped data client, keeping I/O and math testable without MCP;
- every tool returns a uniform envelope: {"ok": True, "result": ...} or
  {"ok": False, "error": "..."} so agents can handle "no data" explicitly
  instead of guessing.

Run any server standalone with:  uv run python -m mcp_servers.<name>
"""

from __future__ import annotations

from typing import Any


def ok(result: Any) -> dict[str, Any]:
    """Successful tool result envelope."""
    return {"ok": True, "result": result}


def no_data(reason: str) -> dict[str, Any]:
    """Explicit 'no data' envelope.

    Used everywhere a source returns nothing: callers must surface this
    honestly rather than inventing numbers.
    """
    return {"ok": False, "error": reason, "no_data": True}


def tool_error(reason: str) -> dict[str, Any]:
    """Error envelope for genuine failures (bad args, upstream outage)."""
    return {"ok": False, "error": reason}
