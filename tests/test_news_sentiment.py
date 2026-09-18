"""Tests for news client + sentiment (sources mocked, zero network).

Key behaviors under test:
- explicit no-data when every source fails/returns nothing (NEVER fabricated)
- source chain order (NewsAPI -> GDELT -> RSS) with transparent source_errors
- headline passthrough verbatim
- sentiment: empty input refused; Claude mode; transparent offline fallback
- MCP tool schemas registered with expected parameters
"""

from __future__ import annotations

import pytest

from core.config import get_settings
from mcp_servers import news_client, sentiment
from mcp_servers.news_sentiment_server import mcp as news_mcp


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN001
    """Temp cache dir + no API keys by default."""
    s = get_settings()
    monkeypatch.setattr(s, "data_dir", tmp_path / "data")
    monkeypatch.setattr(s, "newsapi_key", "")
    return s


# --------------------------------------------------------------------------
# Headlines: source chain
# --------------------------------------------------------------------------
class TestHeadlineSourceChain:
    def test_newsapi_used_when_key_and_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(get_settings(), "newsapi_key", "test-key")
        monkeypatch.setattr(
            news_client, "_from_newsapi",
            lambda q, d, m: [{"title": "Chips beat expectations", "source": "X", "published": "2026-09-10", "url": "u1"}],
        )
        out = news_client.get_recent_headlines("semiconductors", days_back=7)
        assert out["ok"] is True
        assert out["result"]["source_used"] == "newsapi"
        assert out["result"]["items"][0]["title"] == "Chips beat expectations"

    def test_falls_through_to_gdelt_when_newsapi_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(get_settings(), "newsapi_key", "test-key")
        monkeypatch.setattr(news_client, "_from_newsapi", lambda q, d, m: [])
        monkeypatch.setattr(
            news_client, "_from_gdelt",
            lambda q, d, m: [{"title": "Chip rally continues", "source": "example.com", "published": "2026-09-11", "url": "u2"}],
        )
        out = news_client.get_recent_headlines("semiconductors", days_back=7)
        assert out["ok"] is True
        assert out["result"]["source_used"] == "gdelt"

    def test_rss_used_when_others_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(news_client, "_from_gdelt", lambda q, d, m: [])
        monkeypatch.setattr(
            news_client, "_from_rss",
            lambda q, m: [{"title": "TSMC expansion", "source": "Google News", "published": None, "url": "u3"}],
        )
        out = news_client.get_recent_headlines("chips", days_back=5)
        assert out["ok"] is True
        assert out["result"]["source_used"] == "google_news_rss"

    def test_all_sources_empty_is_explicit_no_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(news_client, "_from_gdelt", lambda q, d, m: [])
        monkeypatch.setattr(news_client, "_from_rss", lambda q, m: [])
        out = news_client.get_recent_headlines("obscure query xyz", days_back=7)
        assert out["ok"] is False
        assert out["no_data"] is True
        assert "fabricat" in out["error"].lower()

    def test_all_sources_error_is_no_data_with_transparent_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*a, **k):  # noqa: ANN002, ANN003
            raise RuntimeError("upstream down")

        monkeypatch.setattr(news_client, "_from_gdelt", boom)
        monkeypatch.setattr(news_client, "_from_rss", boom)
        out = news_client.get_recent_headlines("anything", days_back=7)
        assert out["ok"] is False and out["no_data"] is True
        assert "gdelt: upstream down" in out["error"]
        assert "rss: upstream down" in out["error"]

    def test_headlines_passed_through_verbatim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        raw_title = "Chip Stocks: Surging &quot;Demand&quot; — Q3   beat!"
        monkeypatch.setattr(news_client, "_from_gdelt", lambda q, d, m: [{"title": raw_title, "source": "d", "published": None, "url": "u"}])
        out = news_client.get_recent_headlines("q", days_back=7)
        assert out["result"]["items"][0]["title"] == raw_title

    def test_empty_query_rejected(self) -> None:
        assert news_client.get_recent_headlines("   ")["ok"] is False

    def test_days_back_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict = {}

        def spy(q, d, m):  # noqa: ANN001
            seen["days"] = d
            return [{"title": "t", "source": "s", "published": None, "url": None}]

        monkeypatch.setattr(news_client, "_from_gdelt", spy)
        news_client.get_recent_headlines("q", days_back=999)
        assert seen["days"] == 30  # clamped, not passed through raw

    def test_result_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(news_client, "_from_gdelt", lambda q, d, m: [{"title": "t", "source": "s", "published": None, "url": None}])
        first = news_client.get_recent_headlines("cache me", days_back=7)
        second = news_client.get_recent_headlines("cache me", days_back=7)
        assert first["from_cache"] is False
        assert second["from_cache"] is True

    def test_no_data_not_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(news_client, "_from_gdelt", lambda q, d, m: [])
        monkeypatch.setattr(news_client, "_from_rss", lambda q, m: [])
        news_client.get_recent_headlines("nothing here", days_back=7)
        monkeypatch.setattr(news_client, "_from_gdelt", lambda q, d, m: [{"title": "found later", "source": "s", "published": None, "url": None}])
        out = news_client.get_recent_headlines("nothing here", days_back=7)
        assert out["ok"] is True  # retry can succeed: failures aren't cached


# --------------------------------------------------------------------------
# Sentiment
# --------------------------------------------------------------------------
class TestSentiment:
    def test_empty_headlines_refused(self) -> None:
        out = sentiment.summarize_sentiment([])
        assert out["ok"] is False and out["no_data"] is True
        assert "refusing" in out["error"].lower()

    def test_whitespace_only_refused(self) -> None:
        assert sentiment.summarize_sentiment(["  ", ""])["no_data"] is True

    def test_lexicon_mode_labeled_and_scored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sentiment.model_client, "available", lambda: False)
        out = sentiment.summarize_sentiment(
            ["Chipmaker beats record revenue", "Demand plunges amid glut", "Company files routine filing"]
        )
        assert out["ok"] is True
        r = out["result"]
        assert r["mode"] == "offline_lexicon"
        assert r["confidence"] == "low"
        assert r["n_headlines"] == 3
        labels = {h["label"] for h in r["headline_labels"]}
        assert "pos" in labels and "neg" in labels

    def test_lexicon_reasoning_mentions_offline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sentiment.model_client, "available", lambda: False)
        out = sentiment.summarize_sentiment(["Strong demand lifts shares"])
        assert "offline" in out["result"]["reasoning"].lower()

    def test_claude_mode_used_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sentiment.model_client, "available", lambda: True)
        monkeypatch.setattr(
            sentiment.model_client, "complete_json",
            lambda prompt, system=None, max_tokens=None: (
                '{"score": 0.6, "label": "positive", "confidence": "medium", '
                '"reasoning": "Mostly upbeat headlines.", "headline_labels": []}'
            ),
        )
        out = sentiment.summarize_sentiment(["Chips rally on strong demand"])
        assert out["result"]["mode"] == "claude"
        assert out["result"]["score"] == pytest.approx(0.6)
        assert out["result"]["confidence"] == "medium"

    def test_claude_failure_degrades_to_lexicon_transparently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sentiment.model_client, "available", lambda: True)

        def boom(*a, **k):  # noqa: ANN002, ANN003
            raise RuntimeError("api down")

        monkeypatch.setattr(sentiment.model_client, "complete_json", boom)
        out = sentiment.summarize_sentiment(["Shares rise on earnings beat"])
        assert out["result"]["mode"] == "offline_lexicon"
        assert "api down" in out["result"]["claude_error"]

    def test_claude_unparseable_json_degrades_to_lexicon(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sentiment.model_client, "available", lambda: True)
        monkeypatch.setattr(sentiment.model_client, "complete_json", lambda *a, **k: "not json at all")
        out = sentiment.summarize_sentiment(["Shares rise"])
        assert out["result"]["mode"] == "offline_lexicon"
        assert "unparseable" in out["result"]["claude_error"]

    def test_score_is_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sentiment.model_client, "available", lambda: True)
        monkeypatch.setattr(
            sentiment.model_client, "complete_json",
            lambda *a, **k: '{"score": 5.0, "reasoning": "", "headline_labels": []}',
        )
        out = sentiment.summarize_sentiment(["x"])
        assert -1.0 <= out["result"]["score"] <= 1.0


# --------------------------------------------------------------------------
# MCP server schema
# --------------------------------------------------------------------------
class TestNewsServerSchema:
    def test_registers_two_tools(self) -> None:
        import asyncio

        tools = asyncio.run(news_mcp.list_tools())
        names = {t.name for t in tools}
        assert {"get_recent_headlines", "summarize_sentiment"} <= names

    def test_headlines_tool_schema(self) -> None:
        import asyncio

        tools = asyncio.run(news_mcp.list_tools())
        tool = next(t for t in tools if t.name == "get_recent_headlines")
        props = tool.input_schema["properties"]
        assert set(props) == {"query", "days_back", "max_items"}
        assert props["days_back"]["type"] == "integer"
        assert tool.input_schema["required"] == ["query"]

    def test_sentiment_tool_schema(self) -> None:
        import asyncio

        tools = asyncio.run(news_mcp.list_tools())
        tool = next(t for t in tools if t.name == "summarize_sentiment")
        assert set(tool.input_schema["properties"]) == {"headlines", "context"}
        assert tool.input_schema["properties"]["headlines"]["type"] == "array"
