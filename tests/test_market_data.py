"""Tests for the market data client (yfinance mocked) and MCP server schema.

yfinance is monkeypatched so these tests never touch the network.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from mcp_servers import market_data
from mcp_servers.market_data_server import mcp as market_mcp


# --------------------------------------------------------------------------
# Fake yfinance
# --------------------------------------------------------------------------
class FakeTicker:
    """Stands in for yf.Ticker in tests."""

    history_frames: dict[str, pd.DataFrame] = {}
    info_payloads: dict[str, dict[str, Any]] = {}
    calls: int = 0

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker

    def history(self, start: str, end: str, interval: str = "1d", auto_adjust: bool = True) -> pd.DataFrame:
        FakeTicker.calls += 1
        frame = FakeTicker.history_frames.get(self.ticker)
        return frame if frame is not None else pd.DataFrame()

    @property
    def info(self) -> dict[str, Any]:
        FakeTicker.calls += 1
        return FakeTicker.info_payloads.get(self.ticker, {})


@pytest.fixture()
def fake_yf(monkeypatch: pytest.MonkeyPatch) -> type[FakeTicker]:
    FakeTicker.calls = 0
    FakeTicker.history_frames = {
        "NVDA": pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.5],
                "Volume": [1_000_000, 1_100_000],
            },
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]),
        ),
        "EMPTY": pd.DataFrame(),
    }
    FakeTicker.info_payloads = {
        "NVDA": {
            "trailingPE": 60.0,
            "forwardPE": 40.0,
            "marketCap": 2_000_000_000_000,
            "sector": "Technology",
            "industry": "Semiconductors",
            "currency": "USD",
        },
        "EMPTYINFO": {},
    }
    monkeypatch.setattr(market_data.yf, "Ticker", FakeTicker)
    return FakeTicker


@pytest.fixture(autouse=True)
def temp_cache(tmp_path: pd.DataFrame.__class__.__mro__, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    """Point the settings data dir at a temp dir so tests never touch real cache."""
    from core.config import get_settings

    s = get_settings()
    monkeypatch.setattr(s, "data_dir", tmp_path / "data")
    return s


# --------------------------------------------------------------------------
# Price history
# --------------------------------------------------------------------------
class TestGetPriceHistory:
    def test_returns_ohlcv_rows(self, fake_yf: type[FakeTicker]) -> None:
        out = market_data.get_price_history("NVDA", "2024-01-01", "2024-01-31")
        assert out["ok"] is True
        r = out["result"]
        assert r["ticker"] == "NVDA"
        assert r["row_count"] == 2
        assert r["rows"][0] == {"date": "2024-01-02", "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1_000_000}
        assert r["source"] == "yahoo_finance"

    def test_second_call_served_from_cache(self, fake_yf: type[FakeTicker]) -> None:
        market_data.get_price_history("NVDA", "2024-01-01", "2024-01-31")
        calls_after_first = FakeTicker.calls
        out2 = market_data.get_price_history("NVDA", "2024-01-01", "2024-01-31")
        assert out2["ok"] is True
        assert out2["from_cache"] is True
        assert FakeTicker.calls == calls_after_first  # no new upstream call

    def test_different_args_not_served_from_cache(self, fake_yf: type[FakeTicker]) -> None:
        market_data.get_price_history("NVDA", "2024-01-01", "2024-01-31")
        out2 = market_data.get_price_history("NVDA", "2024-02-01", "2024-02-28")
        assert out2["from_cache"] is False

    def test_empty_frame_is_explicit_no_data(self, fake_yf: type[FakeTicker]) -> None:
        out = market_data.get_price_history("EMPTY", "2024-01-01", "2024-01-31")
        assert out["ok"] is False
        assert out["no_data"] is True
        assert "not fabricating" in out["error"]

    def test_bad_interval_rejected(self, fake_yf: type[FakeTicker]) -> None:
        out = market_data.get_price_history("NVDA", "2024-01-01", "2024-01-31", interval="17m")
        assert out["ok"] is False and "interval" in out["error"]

    def test_blank_ticker_rejected(self, fake_yf: type[FakeTicker]) -> None:
        out = market_data.get_price_history("  ", "2024-01-01", "2024-01-31")
        assert out["ok"] is False

    def test_ticker_uppercased(self, fake_yf: type[FakeTicker]) -> None:
        out = market_data.get_price_history("nvda", "2024-01-01", "2024-01-31")
        assert out["result"]["ticker"] == "NVDA"


# --------------------------------------------------------------------------
# Fundamentals
# --------------------------------------------------------------------------
class TestGetFundamentals:
    def test_selects_known_fields(self, fake_yf: type[FakeTicker]) -> None:
        out = market_data.get_fundamentals("NVDA")
        assert out["ok"] is True
        f = out["result"]["fields"]
        assert f["trailing_pe"] == 60.0
        assert f["sector"] == "Technology"
        assert f["market_cap"] == 2_000_000_000_000
        # currency was present; fields yfinance did not provide are listed
        assert "beta" in out["result"]["unavailable_fields"]
        assert "trailing_pe" not in out["result"]["unavailable_fields"]

    def test_no_info_is_explicit_no_data(self, fake_yf: type[FakeTicker]) -> None:
        out = market_data.get_fundamentals("EMPTYINFO")
        assert out["ok"] is False and out["no_data"] is True

    def test_cached_on_second_call(self, fake_yf: type[FakeTicker]) -> None:
        market_data.get_fundamentals("NVDA")
        out2 = market_data.get_fundamentals("NVDA")
        assert out2["from_cache"] is True


# --------------------------------------------------------------------------
# Sector map
# --------------------------------------------------------------------------
class TestGetSectorTickers:
    def test_semiconductors_resolves(self) -> None:
        out = market_data.get_sector_tickers("semiconductors")
        assert out["ok"] is True
        assert "NVDA" in out["result"]["tickers"]
        assert out["result"]["source"] == "curated_static_map"

    def test_alias_resolves(self) -> None:
        assert market_data.get_sector_tickers("Semis")["ok"] is True
        assert market_data.get_sector_tickers("chip stocks")["ok"] is True
        assert market_data.get_sector_tickers("Tech")["result"]["sector"] == "technology"

    def test_unknown_sector_lists_available(self) -> None:
        out = market_data.get_sector_tickers("underwater basket weaving")
        assert out["ok"] is False and out["no_data"] is True
        assert "semiconductors" in out["available_sectors"]


# --------------------------------------------------------------------------
# MCP server schema
# --------------------------------------------------------------------------
class TestMarketDataServerSchema:
    def test_registers_three_tools(self) -> None:
        import asyncio

        tools = asyncio.run(market_mcp.list_tools())
        names = {t.name for t in tools}
        assert {"get_price_history", "get_fundamentals", "get_sector_tickers"} <= names

    def test_price_history_tool_schema(self) -> None:
        import asyncio

        tools = asyncio.run(market_mcp.list_tools())
        tool = next(t for t in tools if t.name == "get_price_history")
        props = tool.input_schema["properties"]
        assert set(props) == {"ticker", "start_date", "end_date", "interval"}
        assert props["ticker"]["type"] == "string"
        assert tool.input_schema["required"] == ["ticker", "start_date", "end_date"]

    def test_sector_tool_schema(self) -> None:
        import asyncio

        tools = asyncio.run(market_mcp.list_tools())
        tool = next(t for t in tools if t.name == "get_sector_tickers")
        assert set(tool.input_schema["properties"]) == {"sector_name"}
