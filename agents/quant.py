"""Stage 4: quant agent — run requested analyses over gathered price data.

Calls the quant_signals MCP server; every insufficient-data case becomes a
note (honest gap) rather than an invented number. Metric keys follow
`<TICKER>_<signal>` so the verifier can bind claim numbers to metrics.
"""

from __future__ import annotations

import asyncio

from agents.schemas import GatheredData, QuantOutput, TaskPlan
from core.mcp_client import MCPClientPool
from core.trace import TraceLogger


class QuantAgent:
    name = "quant_agent"

    def __init__(self, pool: MCPClientPool, trace: TraceLogger | None = None) -> None:
        self.pool = pool
        self.trace = trace

    async def compute(self, plan: TaskPlan, gathered: GatheredData) -> QuantOutput:
        analyses = set(plan.analyses) or {"momentum"}
        metrics: dict[str, dict] = {}
        notes: list[str] = []

        # Tickers with usable price rows (>= 2 closes) go to the signal calls.
        usable: dict[str, list[float]] = {}
        for ticker, env in gathered.prices.items():
            if not env.get("ok"):
                continue
            rows = env.get("result", {}).get("rows") or []
            closes = [r["close"] for r in rows if r.get("close") is not None]
            if len(closes) >= 2:
                usable[ticker] = closes
            else:
                notes.append(f"{ticker}: price data had too few closes for signals")

        for ticker, closes in usable.items():
            if "momentum" in analyses:
                res = await self.pool.call_tool(
                    "quant_signals", "compute_momentum",
                    {"price_series": closes, "lookback_days": min(plan.lookback_days, len(closes) - 1)},
                    agent=self.name,
                )
                _store(metrics, ticker, "momentum", res, notes)
            if "volatility" in analyses:
                window = min(20, len(closes) - 1)
                res = await self.pool.call_tool(
                    "quant_signals", "compute_volatility",
                    {"price_series": closes, "window": window},
                    agent=self.name,
                )
                _store(metrics, ticker, "volatility", res, notes)

        correlation = None
        if "correlation" in analyses and len(usable) >= 2:
            res = await self.pool.call_tool(
                "quant_signals", "compute_correlation_matrix",
                {"tickers": list(usable), "price_data": usable, "min_overlap": 10},
                agent=self.name,
            )
            if res.get("ok"):
                correlation = res["result"]
            else:
                notes.append(f"correlation unavailable: {res.get('error')}")

        # gathered.no_data and gathered.notes are rendered by the writer's
        # Data Gaps section; only ADD quant-specific notes here to avoid dupes.
        notes.extend(gathered.notes)

        out = QuantOutput(metrics=metrics, correlation=correlation, notes=notes)
        if self.trace:
            self.trace.agent_step(
                self.name, "quant_complete",
                n_metrics=len(metrics), n_notes=len(notes), analyses=sorted(analyses),
            )
        return out


def _store(metrics: dict, ticker: str, signal: str, envelope: dict, notes: list[str]) -> None:
    if envelope.get("ok"):
        r = dict(envelope["result"])
        r.setdefault("window_days", r.get("n_observations"))
        metrics[f"{ticker}_{signal}"] = r
    else:
        notes.append(f"{ticker} {signal}: not computed — {envelope.get('error', 'unknown error')}")
