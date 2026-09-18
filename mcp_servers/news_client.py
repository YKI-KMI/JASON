"""News client for the news/sentiment MCP server.

Source chain (first one that returns results wins):
  1. NewsAPI (only if NEWS_API_KEY is set)
  2. GDELT DOC 2.0 API (keyless)
  3. Google News RSS (keyless)

HARD RULE — never fabricate: headlines are passed through verbatim from the
source. If every source returns zero usable items, the caller gets an explicit
no-data envelope; nothing is ever invented to fill the gap.

All external calls are rate-limited and cached (TTL) via core helpers.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import httpx

from core.cache import TTLCache
from core.config import get_settings
from core.ratelimit import RateLimiter

_rate_limiter = RateLimiter(get_settings().rate_limit_per_min)
_gdelt_rate = RateLimiter(15)  # GDELT asks for <=1 req/5s per doc

_USER_AGENT = "Mozilla/5.0 (compatible; MarketResearchAgents/0.1; research-educational)"


def _cache() -> TTLCache:
    s = get_settings()
    return TTLCache(s.data_dir / "cache.db", ttl_seconds=s.cache_ttl_seconds)


def _cache_wrap(namespace: str, payload: dict, fn) -> dict:
    cache = _cache()
    cached = cache.get(namespace, payload)
    if cached is not None:
        cached["from_cache"] = True
        return cached
    _rate_limiter.wait()
    result = fn()
    if result.get("ok"):
        cache.set(namespace, payload, result)
    result.setdefault("from_cache", False)
    return result


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").replace("&amp;", "&").replace("&#39;", "'").replace("&quot;", '"').strip()


def get_recent_headlines(query: str, days_back: int = 7, max_items: int = 12) -> dict[str, Any]:
    """Headlines mentioning `query` from the last `days_back` days.

    Tries NewsAPI (if key), then GDELT, then Google News RSS. Returns
    {ok: True, result: {query, window_days, items: [{title, source, published,
    url}], source_used, item_count}} or an explicit no-data envelope.
    """
    query = (query or "").strip()
    if not query:
        return {"ok": False, "error": "empty query"}
    days_back = max(1, min(int(days_back), 30))
    max_items = max(1, min(int(max_items), 25))

    payload = {"tool": "get_recent_headlines", "query": query, "days_back": days_back, "max_items": max_items}
    return _cache_wrap("news", payload, lambda: _fetch_headlines(query, days_back, max_items))


def _fetch_headlines(query: str, days_back: int, max_items: int) -> dict[str, Any]:
    errors: list[str] = []

    if get_settings().newsapi_key:
        try:
            items = _from_newsapi(query, days_back, max_items)
            if items:
                return _headlines_ok(query, days_back, items, "newsapi", errors)
        except Exception as exc:
            errors.append(f"newsapi: {exc}")

    try:
        items = _from_gdelt(query, days_back, max_items)
        if items:
            return _headlines_ok(query, days_back, items, "gdelt", errors)
    except Exception as exc:
        errors.append(f"gdelt: {exc}")

    try:
        items = _from_rss(query, max_items)
        if items:
            return _headlines_ok(query, days_back, items, "google_news_rss", errors)
    except Exception as exc:
        errors.append(f"rss: {exc}")

    return {
        "ok": False,
        "error": (
            f"no headlines found for '{query}' in the last {days_back} days "
            f"across any configured source ({'; '.join(errors) if errors else 'no sources returned items'}) "
            "— returning no data rather than fabricating headlines"
        ),
        "no_data": True,
    }


def _headlines_ok(query: str, days_back: int, items: list[dict], source: str, errors: list[str]) -> dict:
    return {
        "ok": True,
        "result": {
            "query": query,
            "window_days": days_back,
            "item_count": len(items),
            "items": items,
            "source_used": source,
            "source_errors": errors,  # transparent about skipped sources
        },
    }


def _http_get(url: str, timeout: float = 15.0) -> str:
    _gdelt_rate.wait()
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed scheme https
        return resp.read().decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# Source 1: NewsAPI (optional key)
# --------------------------------------------------------------------------
def _from_newsapi(query: str, days_back: int, max_items: int) -> list[dict]:
    key = get_settings().newsapi_key
    frm = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = (
        "https://newsapi.org/v2/everything?"
        + urllib.parse.urlencode(
            {
                "q": query,
                "from": frm,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": max_items,
                "apiKey": key,
            }
        )
    )
    data = httpx.get(url, timeout=15.0).json()
    if data.get("status") != "ok":
        raise RuntimeError(f"newsapi status={data.get('status')}: {data.get('message', 'unknown')}")
    out = []
    for art in data.get("articles", [])[:max_items]:
        title = _strip_html(art.get("title") or "")
        if not title:
            continue
        out.append(
            {
                "title": title,
                "source": (art.get("source") or {}).get("name") or "newsapi",
                "published": art.get("publishedAt"),
                "url": art.get("url"),
            }
        )
    return out


# --------------------------------------------------------------------------
# Source 2: GDELT DOC 2.0 (keyless)
# --------------------------------------------------------------------------
def _from_gdelt(query: str, days_back: int, max_items: int) -> list[dict]:
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc?"
        + urllib.parse.urlencode(
            {
                "query": f"{query} sourcelang:english",
                "mode": "artlist",
                "maxrecords": str(max_items),
                "timespan": f"{days_back}d",
                "format": "json",
                "sort": "datedesc",
            }
        )
    )
    raw = _http_get(url)
    if not raw.strip():
        # GDELT returns an empty body when it has nothing (common for narrow
        # queries) — treat as zero results, not an error.
        return []
    return _parse_gdelt(raw, max_items)


def _parse_gdelt(raw: str, max_items: int) -> list[dict]:
    import json as _json

    data = _json.loads(raw)
    out = []
    for art in (data.get("articles") or [])[:max_items]:
        title = _strip_html(art.get("title") or "")
        if not title:
            continue
        out.append(
            {
                "title": title,
                "source": art.get("domain") or "gdelt",
                "published": art.get("seendate"),
                "url": art.get("url"),
            }
        )
    return out


# --------------------------------------------------------------------------
# Source 3: Google News RSS (keyless)
# --------------------------------------------------------------------------
def _from_rss(query: str, max_items: int) -> list[dict]:
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    )
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=15.0) as resp:  # noqa: S310 - fixed scheme https
        parsed = feedparser.parse(resp.read())
    out = []
    for entry in parsed.entries[:max_items]:
        title = _strip_html(entry.get("title") or "")
        if not title:
            continue
        src = ""
        if entry.get("source") and isinstance(entry["source"], dict):
            src = entry["source"].get("title", "")
        out.append(
            {
                "title": title,
                "source": src or "google_news_rss",
                "published": entry.get("published"),
                "url": entry.get("link"),
            }
        )
    return out
