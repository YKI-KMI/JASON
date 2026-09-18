"""yfinance client for the market data MCP server.

Kept separate from the server wrapper so tests can mock `yf` here and so the
cache/rate-limit policy is visible in one place. All functions return
JSON-serializable dicts and never raise for "no data" — callers get explicit
empty/no-data results instead.

Caching: every external call goes through a SQLite TTL cache keyed by
(tool, args) so repeated dev/demo runs don't hammer Yahoo Finance.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf

from core.cache import TTLCache
from core.config import get_settings
from core.ratelimit import RateLimiter

ALLOWED_INTERVALS = ("1d", "1h", "5d", "1wk", "1mo")

_rate_limiter = RateLimiter(get_settings().rate_limit_per_min)

# Curated, static representative tickers per sector. Deliberately NOT a live
# index scraper: it is deterministic, testable, and honest about being a
# hand-picked sample (the memo reports it as such).
SECTOR_TICKERS: dict[str, dict[str, Any]] = {
    "semiconductors": {
        "tickers": ["NVDA", "AMD", "INTC", "TSM", "AVGO", "QCOM", "TXN", "MU", "MRVL", "ON"],
        "note": "hand-picked representative semis (designers, foundries, equipment-adjacent)",
    },
    "technology": {
        "tickers": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "CRM", "ORCL", "ADBE"],
        "note": "large-cap tech sample",
    },
    "communication services": {
        "tickers": ["GOOGL", "META", "NFLX", "DIS", "TMUS", "VZ", "T"],
        "note": "large-cap communication services sample",
    },
    "financials": {
        "tickers": ["JPM", "BAC", "GS", "MS", "WFC", "BLK", "SCHW", "AXP"],
        "note": "large-cap banks and asset managers",
    },
    "healthcare": {
        "tickers": ["UNH", "LLY", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT"],
        "note": "large-cap pharma/medtech sample",
    },
    "consumer discretionary": {
        "tickers": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TJX", "BKNG"],
        "note": "large-cap consumer discretionary sample",
    },
    "consumer staples": {
        "tickers": ["WMT", "COST", "PG", "KO", "PEP", "MDLZ", "CL"],
        "note": "large-cap consumer staples sample",
    },
    "energy": {
        "tickers": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "OXY"],
        "note": "large-cap energy sample",
    },
    "industrials": {
        "tickers": ["CAT", "BA", "HON", "UNP", "GE", "RTX", "DE", "LMT"],
        "note": "large-cap industrials sample",
    },
    "utilities": {
        "tickers": ["NEE", "DUK", "SO", "D", "AEP", "EXC"],
        "note": "large-cap utilities sample",
    },
    "materials": {
        "tickers": ["LIN", "SHW", "APD", "ECL", "NEM", "FCX"],
        "note": "large-cap materials sample",
    },
    "real estate": {
        "tickers": ["PLD", "AMT", "EQIX", "SPG", "O", "PSA"],
        "note": "large-cap REITs sample",
    },
}

# Common synonyms so "semis"/"chip stocks" resolve without a network call.
_SECTOR_ALIASES: dict[str, str] = {
    "semis": "semiconductors",
    "semiconductor": "semiconductors",
    "chips": "semiconductors",
    "chipstocks": "semiconductors",
    "tech": "technology",
    "info tech": "technology",
    "informationtechnology": "technology",
    "communicationservices": "communication services",
    "telecom": "communication services",
    "banks": "financials",
    "finance": "financials",
    "health care": "healthcare",
    "pharma": "healthcare",
    "consumerdiscretionary": "consumer discretionary",
    "consumerstaples": "consumer staples",
    "reits": "real estate",
    "energy": "energy",
    "utilities": "utilities",
    "materials": "materials",
    "industrials": "industrials",
}


def _normalize_sector(name: str) -> str | None:
    key = " ".join(name.lower().split())
    if key in SECTOR_TICKERS:
        return key
    compact = key.replace(" ", "")
    if key in _SECTOR_ALIASES:
        return _SECTOR_ALIASES[key]
    if compact in _SECTOR_ALIASES:
        return _SECTOR_ALIASES[compact]
    return None


def _cache() -> TTLCache:
    s = get_settings()
    return TTLCache(s.data_dir / "cache.db", ttl_seconds=s.cache_ttl_seconds)


def _rate_wait() -> None:
    _rate_limiter.wait()


def get_price_history(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> dict[str, Any]:
    """OHLCV rows for a ticker between ISO dates (inclusive start, exclusive end).

    Returns {"ticker", "interval", "rows": [{date, open, high, low, close,
    volume}...], "row_count"} or a no_data-style dict with ok=False.
    """
    ticker = _clean_ticker(ticker)
    if ticker is None:
        return {"ok": False, "error": "invalid ticker (empty)"}
    if interval not in ALLOWED_INTERVALS:
        return {"ok": False, "error": f"interval must be one of {ALLOWED_INTERVALS}, got '{interval}'"}

    cache = _cache()
    key_payload = {"tool": "get_price_history", "ticker": ticker, "start": start_date, "end": end_date, "interval": interval}
    cached = cache.get("market_data", key_payload)
    if cached is not None:
        cached["from_cache"] = True
        return cached

    _rate_wait()
    try:
        df = yf.Ticker(ticker).history(start=start_date, end=end_date, interval=interval, auto_adjust=True)
    except Exception as exc:  # upstream outage / bad symbol handling
        return {"ok": False, "error": f"price history fetch failed for {ticker}: {exc}"}

    if df is None or len(df) == 0:
        return {
            "ok": False,
            "error": f"no price data returned for ticker '{ticker}' "
            f"({start_date}..{end_date}, interval={interval}) — not fabricating values",
            "no_data": True,
        }

    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        ts = pd.Timestamp(idx)
        date_str = ts.strftime("%Y-%m-%d") if interval in ("1d", "5d", "1wk", "1mo") else ts.isoformat()
        rows.append(
            {
                "date": date_str,
                "open": _f(row.get("Open")),
                "high": _f(row.get("High")),
                "low": _f(row.get("Low")),
                "close": _f(row.get("Close")),
                "volume": _i(row.get("Volume")),
            }
        )

    result: dict[str, Any] = {
        "ok": True,
        "result": {
            "ticker": ticker,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "row_count": len(rows),
            "rows": rows,
            "source": "yahoo_finance",
        },
    }
    cache.set("market_data", key_payload, result)
    result["from_cache"] = False
    return result


def get_fundamentals(ticker: str) -> dict[str, Any]:
    """Selected fundamentals for a ticker (P/E, market cap, sector, etc.)."""
    ticker = _clean_ticker(ticker)
    if ticker is None:
        return {"ok": False, "error": "invalid ticker (empty)"}

    cache = _cache()
    key_payload = {"tool": "get_fundamentals", "ticker": ticker}
    cached = cache.get("market_data", key_payload)
    if cached is not None:
        cached["from_cache"] = True
        return cached

    _rate_wait()
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        return {"ok": False, "error": f"fundamentals fetch failed for {ticker}: {exc}"}

    fields = {
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "market_cap": info.get("marketCap"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "trailing_eps": info.get("trailingEps"),
        "beta": info.get("beta"),
        "dividend_yield": info.get("dividendYield"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "currency": info.get("currency"),
    }
    available = {k: v for k, v in fields.items() if v is not None}
    if not available:
        return {
            "ok": False,
            "error": f"no fundamentals returned for ticker '{ticker}' — not fabricating values",
            "no_data": True,
        }

    result = {
        "ok": True,
        "result": {
            "ticker": ticker,
            "fields": available,
            "unavailable_fields": sorted(set(fields) - set(available)),
            "source": "yahoo_finance",
        },
    }
    cache.set("market_data", key_payload, result)
    result["from_cache"] = False
    return result


def get_sector_tickers(sector_name: str) -> dict[str, Any]:
    """Representative tickers for a sector from the curated static map."""
    matched = _normalize_sector(sector_name)
    if matched is None:
        return {
            "ok": False,
            "error": f"unknown sector '{sector_name}'",
            "no_data": True,
            "available_sectors": sorted(SECTOR_TICKERS),
        }
    entry = SECTOR_TICKERS[matched]
    return {
        "ok": True,
        "result": {
            "sector": matched,
            "requested": sector_name,
            "tickers": list(entry["tickers"]),
            "note": entry["note"],
            "source": "curated_static_map",
        },
    }


def available_sectors() -> list[str]:
    return sorted(SECTOR_TICKERS)


def _clean_ticker(ticker: str) -> str | None:
    if not ticker or not str(ticker).strip():
        return None
    return str(ticker).strip().upper()


def _f(value: Any) -> float | None:
    try:
        f = float(value)
        return f if pd.notna(f) else None
    except (TypeError, ValueError):
        return None


def _i(value: Any) -> int | None:
    try:
        f = float(value)
        return int(f) if pd.notna(f) else None
    except (TypeError, ValueError):
        return None
