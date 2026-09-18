"""Stage 2/3 worker agents.

- DataAgent: per-ticker price history + fundamentals via the market_data MCP
  server. Failures become entries in `no_data` — the run continues with honest
  gaps instead of crashing.
- NewsAgent: headlines + sentiment via the news_sentiment MCP server.

Orchestrator runs DataAgent and NewsAgent CONCURRENTLY with asyncio.gather
(parallel branches over the sequential spine), which is why both only touch
their own server and return plain pydantic results.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from agents.schemas import GatheredData, TaskPlan
from core.mcp_client import MCPClientPool
from core.trace import TraceLogger


class DataAgent:
    name = "data_agent"

    def __init__(self, pool: MCPClientPool, trace: TraceLogger | None = None) -> None:
        self.pool = pool
        self.trace = trace

    async def gather(self, plan: TaskPlan) -> GatheredData:
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=plan.lookback_days)
        tickers = list(plan.tickers)
        sector_note: str | None = None
        if not tickers and plan.sector:
            # Sector-scoped request without explicit tickers: resolve members
            # through the market_data server's curated map.
            res = await self.pool.call_tool(
                "market_data", "get_sector_tickers", {"sector_name": plan.sector}, agent=self.name
            )
            if res.get("ok"):
                tickers = res["result"]["tickers"][:6]  # cap to keep runs bounded
                sector_note = (
                    f"sector '{res['result']['sector']}' resolved to {len(res['result']['tickers'])} "
                    f"representative tickers ({', '.join(res['result']['tickers'])}); "
                    f"analyzing first {len(tickers)}: {', '.join(tickers)}"
                )
            elif self.trace:
                self.trace.agent_step(self.name, "sector_resolution_failed", sector=plan.sector, reason=res.get("error"))
        if not tickers:
            return GatheredData(no_data={"(none)": "plan contained no tickers and no resolvable sector"})

        async def one(t: str) -> tuple[str, dict, dict | None, str | None]:
            hist = await self.pool.call_tool(
                "market_data", "get_price_history",
                {"ticker": t, "start_date": start.isoformat(), "end_date": end.isoformat(), "interval": "1d"},
                agent=self.name,
            )
            fund = await self.pool.call_tool(
                "market_data", "get_fundamentals", {"ticker": t}, agent=self.name
            )
            return t, hist, fund, None

        results = await asyncio.gather(*(one(t) for t in tickers), return_exceptions=True)

        prices: dict[str, dict] = {}
        fundamentals: dict[str, dict] = {}
        no_data: dict[str, str] = {}
        for t, res in zip(tickers, results):
            if isinstance(res, Exception):
                no_data[t] = f"data agent task failed: {res}"
                continue
            _, hist, fund, _ = res
            prices[t] = hist
            if isinstance(fund, dict) and fund.get("ok"):
                fundamentals[t] = fund
            if not hist.get("ok"):
                reason = hist.get("error", "unknown market data failure")
                no_data[t] = reason
                if self.trace:
                    self.trace.agent_step(self.name, "no_data_recorded", ticker=t, reason=reason)

        gathered = GatheredData(prices=prices, fundamentals=fundamentals, no_data=no_data)
        if sector_note:
            gathered.notes.append(sector_note)
            if self.trace:
                self.trace.agent_step(self.name, "sector_expanded", note=sector_note)
        if self.trace:
            self.trace.agent_step(
                self.name, "gather_complete",
                tickers=tickers,
                ok=[t for t in tickers if t not in no_data],
                no_data=list(no_data),
            )
        return gathered


class NewsAgent:
    name = "news_agent"

    def __init__(self, pool: MCPClientPool, trace: TraceLogger | None = None) -> None:
        self.pool = pool
        self.trace = trace

    async def gather(self, plan: TaskPlan) -> dict | None:
        query = plan.news_query or plan.sector or (plan.tickers[0] if plan.tickers else None)
        if not query:
            return None
        headlines = await self.pool.call_tool(
            "news_sentiment", "get_recent_headlines",
            {"query": query, "days_back": 7, "max_items": 10},
            agent=self.name,
        )
        sentiment: dict | None = None
        if headlines.get("ok") and headlines.get("result", {}).get("items"):
            titles = [item["title"] for item in headlines["result"]["items"]]
            sentiment = await self.pool.call_tool(
                "news_sentiment", "summarize_sentiment",
                {"headlines": titles, "context": plan.request_summary},
                agent=self.name,
            )
        out = {"headlines": headlines, "sentiment": sentiment}
        if self.trace:
            self.trace.agent_step(
                self.name, "news_complete",
                query=query,
                found=bool(headlines.get("ok")),
                sentiment_mode=(sentiment or {}).get("result", {}).get("mode") if sentiment else None,
            )
        return out
