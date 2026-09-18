"""MCP client: spawn authored servers as subprocesses and call their tools.

Two transports:
- "stdio": real MCP over JSON-RPC stdio (StdioServerParameters + ClientSession).
  This is the production path — the orchestrator genuinely talks MCP to
  separately-running server processes.
- "inproc": calls the server modules' tool functions directly (no subprocess).
  Used by the end-to-end tests so they are hermetic and fast; the tool
  functions are the exact functions the MCP protocol exposes.

Every call is recorded to the TraceLogger with agent, server, tool, args,
result, error, and duration — that trace is the run's explainability artifact.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, Literal

from core.config import PROJECT_ROOT, get_settings
from core.trace import TraceLogger

TransportKind = Literal["stdio", "inproc"]

# Modules for each authored server (module path under mcp_servers.*).
SERVER_MODULES: dict[str, str] = {
    "market_data": "mcp_servers.market_data_server",
    "news_sentiment": "mcp_servers.news_sentiment_server",
    "quant_signals": "mcp_servers.quant_signals_server",
    "verification": "mcp_servers.verification_server",
}

# Env vars that must reach server subprocesses (get_default_environment()
# filters most things, so we forward what the servers actually need).
_FORWARD_ENV = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_MAX_TOKENS",
    "NEWS_API_KEY",
    "CACHE_TTL_SECONDS",
    "RATE_LIMIT_PER_MIN",
]


class MCPClientPool:
    """Owns connections to all four servers; call any tool by server+name."""

    def __init__(self, transport: TransportKind = "stdio", trace: TraceLogger | None = None) -> None:
        self.transport_kind = transport
        self.trace = trace
        self._sessions: dict[str, Any] = {}
        self._exit_stacks: dict[str, Any] = {}
        self._inproc_modules: dict[str, Any] = {}

    async def start(self) -> None:
        if self.transport_kind == "inproc":
            import importlib

            for name, module_path in SERVER_MODULES.items():
                self._inproc_modules[name] = importlib.import_module(module_path)
            return
        await asyncio.gather(*(self._start_stdio(name) for name in SERVER_MODULES))

    async def _start_stdio(self, name: str) -> None:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        settings = get_settings()
        env = {k: os.environ[k] for k in _FORWARD_ENV if os.environ.get(k)}
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", SERVER_MODULES[name]],
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        stack = context_stack()
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions[name] = session
        self._exit_stacks[name] = stack
        if self.trace:
            self.trace.agent_step("orchestrator", "server_started", server=name, transport="stdio")

    async def call_tool(self, server: str, tool: str, args: dict[str, Any], agent: str) -> dict[str, Any]:
        started = time.monotonic()
        error: str | None = None
        result: dict[str, Any] | None = None
        try:
            if self.transport_kind == "inproc":
                result = self._call_inproc(server, tool, args)
            else:
                result = await self._call_stdio(server, tool, args)
            return result
        except Exception as exc:  # transport/spawn/protocol errors
            error = f"{type(exc).__name__}: {exc}"
            return {"ok": False, "error": error, "transport_error": True}
        finally:
            duration_ms = (time.monotonic() - started) * 1000.0
            if self.trace:
                self.trace.tool_call(
                    agent=agent, server=server, tool=tool, args=args, result=_safe(result), error=error, duration_ms=round(duration_ms, 2)
                )

    def _call_inproc(self, server: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        module = self._inproc_modules.get(server)
        if module is None:
            return {"ok": False, "error": f"server '{server}' not started"}
        fn = getattr(module, tool, None)
        if fn is None or not callable(fn):
            return {"ok": False, "error": f"tool '{tool}' not found on server '{server}'"}
        result = fn(**args)
        return result if isinstance(result, dict) else {"value": result}

    async def _call_stdio(self, server: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        session = self._sessions.get(server)
        if session is None:
            return {"ok": False, "error": f"server '{server}' not started"}
        result = await session.call_tool(tool, args)
        return _result_to_dict(result)

    async def stop(self) -> None:
        for stack in self._exit_stacks.values():
            try:
                await stack.aclose()
            except Exception:
                pass
        self._exit_stacks.clear()
        self._sessions.clear()


def _result_to_dict(result: Any) -> dict[str, Any]:
    """Normalize an MCP CallToolResult into the servers' ok/no_data envelopes."""
    if getattr(result, "is_error", False):
        text = "".join(getattr(c, "text", "") for c in (result.content or []))
        return {"ok": False, "error": text or "tool call failed (MCP is_error)"}
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        # SDKs may wrap payloads under "result"; unwrap when shape demands it.
        inner = structured.get("result")
        if isinstance(inner, dict) and ("ok" in inner or "error" in inner):
            return inner
        if "ok" in structured or "error" in structured:
            return structured
        return structured
    text = "".join(getattr(c, "text", "") for c in (result.content or []))
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return {"ok": False, "error": f"unparseable tool result: {text[:200]}"}


def _safe(value: Any) -> Any:
    """Best-effort JSON-safety for trace logging."""
    try:
        json.dumps(value, default=str)
        return value
    except Exception:
        return {"repr": repr(value)[:500]}


def context_stack():
    from contextlib import AsyncExitStack

    return AsyncExitStack()
