"""MCP server: quant signal tools (pure math over numpy/pandas).

Tools
-----
- compute_momentum(price_series, lookback_days, skip_days=0)
- compute_volatility(price_series, window=20)
- compute_correlation_matrix(tickers, price_data, min_overlap=10)
- compute_zscore(current_value, historical_series)

Design notes
------------
- The math lives in mcp_servers/signals.py, fully unit-tested against
  hand-computed values. This module only adapts JSON-friendly inputs to
  pandas and re-raises insufficient-data errors as explicit envelopes.
- Uses the MCP Python SDK 2.x API (MCPServer, plain @mcp.tool decorator).
- Run standalone:  uv run python -m mcp_servers.quant_signals_server
"""

from __future__ import annotations

import pandas as pd
from mcp.server.mcpserver import MCPServer

from mcp_servers.base import no_data, ok, tool_error
from mcp_servers.signals import (
    compute_correlation_matrix as _correlation_matrix,
)
from mcp_servers.signals import (
    compute_momentum as _momentum,
)
from mcp_servers.signals import (
    compute_volatility as _volatility,
)
from mcp_servers.signals import (
    compute_zscore as _zscore,
)

mcp = MCPServer("quant-signals")


@mcp.tool()
def compute_momentum(price_series: list[float], lookback_days: int, skip_days: int = 0) -> dict:
    """Total return over `lookback_days` (optionally skipping the most recent
    `skip_days`). Returns the momentum score with n_observations."""
    try:
        prices = pd.Series(price_series, dtype=float)
        res = _momentum(prices, lookback_days, skip_days)
        return ok(res.to_dict())
    except ValueError as exc:
        if "insufficient data" in str(exc) or "no valid" in str(exc):
            return no_data(str(exc))
        return tool_error(str(exc))


@mcp.tool()
def compute_volatility(price_series: list[float], window: int = 20) -> dict:
    """Annualized realized volatility (std of daily log returns * sqrt(252))
    over the last `window` returns."""
    try:
        prices = pd.Series(price_series, dtype=float)
        res = _volatility(prices, window)
        return ok(res.to_dict())
    except ValueError as exc:
        if "insufficient data" in str(exc) or "no valid" in str(exc):
            return no_data(str(exc))
        return tool_error(str(exc))


@mcp.tool()
def compute_zscore(current_value: float, historical_series: list[float]) -> dict:
    """Z-score of `current_value` against the historical sample
    (mean/std with ddof=1)."""
    try:
        hist = pd.Series(historical_series, dtype=float)
        res = _zscore(current_value, hist)
        return ok(res.to_dict())
    except ValueError as exc:
        if "at least 2" in str(exc) or "zero variance" in str(exc) or "no valid" in str(exc):
            return no_data(str(exc))
        return tool_error(str(exc))


@mcp.tool()
def compute_correlation_matrix(tickers: list[str], price_data: dict[str, list[float]], min_overlap: int = 10) -> dict:
    """Pairwise Pearson correlation of daily log returns over overlapping
    dates. Tickers with < min_overlap shared dates get null (never guessed)."""
    try:
        series_map = {t: pd.Series(v, dtype=float) for t, v in price_data.items()}
        result = _correlation_matrix(tickers, series_map, min_overlap)
        return ok(result)
    except Exception as exc:  # defensive: never crash the server on bad args
        return tool_error(f"correlation computation failed: {exc}")


if __name__ == "__main__":
    # stdio transport: the orchestrator spawns this process and talks MCP.
    mcp.run()
