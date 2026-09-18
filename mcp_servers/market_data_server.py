"""MCP server: market data tools (yfinance wrapper).

Tools
-----
- get_price_history(ticker, start_date, end_date, interval="1d") -> OHLCV rows
- get_fundamentals(ticker) -> P/E, market cap, sector, ...
- get_sector_tickers(sector_name) -> representative tickers (curated static map)

Design notes
------------
- All external calls go through a SQLite TTL cache + rate limiter so re-running
  during dev doesn't get blocked by Yahoo Finance.
- "No data" is an explicit envelope (ok=False, no_data=True) — the server never
  invents prices. The data client lives in mcp_servers/market_data.py so it is
  testable without MCP.
- Uses the MCP Python SDK 2.x API (MCPServer, plain @mcp.tool decorator).
- Run standalone:  uv run python -m mcp_servers.market_data_server
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from mcp_servers import market_data

mcp = MCPServer("market-data")


@mcp.tool()
def get_price_history(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> dict:
    """OHLCV price history for a ticker between ISO dates (e.g. 2024-01-01).

    Returns rows of {date, open, high, low, close, volume}, or an explicit
    no-data envelope if the source returns nothing.
    """
    return market_data.get_price_history(ticker, start_date, end_date, interval)


@mcp.tool()
def get_fundamentals(ticker: str) -> dict:
    """Selected fundamentals for a ticker: P/E, market cap, sector, industry,
    EPS, beta, dividend yield, 52-week range. Missing fields are reported."""
    return market_data.get_fundamentals(ticker)


@mcp.tool()
def get_sector_tickers(sector_name: str) -> dict:
    """Representative tickers for a sector name (e.g. 'semiconductors').

    Uses a curated static map — deterministic and honest about being a
    hand-picked sample. Unknown sectors return the list of known sectors.
    """
    return market_data.get_sector_tickers(sector_name)


if __name__ == "__main__":
    mcp.run()
