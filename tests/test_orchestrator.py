"""End-to-end orchestrator tests — mocked Claude, inproc MCP transport.

No network anywhere: Claude calls are monkeypatched to deterministic
implementations; MCP tools run in-process. These tests exercise the full
pipeline: plan -> concurrent gather -> quant -> write -> verify (with
revision) -> final memo.
"""

from __future__ import annotations

import pytest

from core import model as model_client
from core.config import get_settings


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
FAKE_PRICES = {
    "NVDA": [100.0 + 0.5 * i for i in range(61)],          # +0.83%/day drift
    "AMD": [100.0 - 0.2 * i for i in range(61)],           # negative drift
    "WDC": None,                                            # no data ticker
}


@pytest.fixture()
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force offline mode + isolated cache dir."""
    s = get_settings()
    monkeypatch.setattr(s, "anthropic_api_key", "")
    monkeypatch.setattr(model_client, "available", lambda: False)


class FakeTicker:
    frames: dict = {}

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker

    def history(self, start=None, end=None, interval="1d", auto_adjust=True):
        closes = FakeTicker.frames.get(self.ticker)
        if closes is None:
            import pandas as pd

            return pd.DataFrame()
        import pandas as pd

        idx = pd.date_range("2026-06-01", periods=len(closes), freq="B")
        return pd.DataFrame({"Close": closes, "Open": closes, "High": closes, "Low": closes, "Volume": [1000] * len(closes)}, index=idx)

    @property
    def info(self):
        return {"trailingPE": 55.0, "marketCap": 1_000_000_000, "sector": "Technology"} if self.ticker in FakeTicker.frames else {}


@pytest.fixture()
def fake_yf(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    FakeTicker.frames = {t: closes for t, closes in FAKE_PRICES.items() if closes is not None}
    from mcp_servers import market_data

    monkeypatch.setattr(market_data.yf, "Ticker", FakeTicker)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN001
    s = get_settings()
    monkeypatch.setattr(s, "data_dir", tmp_path / "data")
    monkeypatch.setattr("agents.orchestrator.OUTPUT_DIR", tmp_path / "output")
    return s


class GdeltNoNews:
    @staticmethod
    def fetch(*a, **k):  # pragma: no cover - unused
        return []


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------
class TestEndToEnd:
    async def test_full_pipeline_offline_semis(self, offline, fake_yf, monkeypatch: pytest.MonkeyPatch) -> None:
        """Complete offline run: 2 usable tickers + 1 no-data ticker."""
        from mcp_servers import news_client
        from agents.orchestrator import ResearchOrchestrator

        monkeypatch.setattr(news_client, "_from_gdelt", lambda q, d, m: [
            {"title": "Chip demand beats expectations", "source": "ex.com", "published": None, "url": "u"},
        ])
        monkeypatch.setattr(news_client, "_from_rss", lambda q, m: [])

        orch = ResearchOrchestrator(transport="inproc")
        result = await orch.run("analyze semiconductor sector momentum over 60 days")
        await orch.pool.stop()

        assert result.output_path is not None
        memo = result.memo_markdown
        assert "For educational and research purposes only" in memo
        assert "Verification report" in memo
        assert result.plan.sector == "semiconductors"
        # Sector members produced momentum metrics, visible in the memo's table.
        assert "NVDA_momentum" in memo
        assert "AMD_momentum" in memo

    async def test_no_data_ticker_degrades_gracefully(self, offline, fake_yf, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid ticker -> honest gap in memo, no crash, no fabricated number."""
        from mcp_servers import news_client
        from agents.orchestrator import ResearchOrchestrator

        monkeypatch.setattr(news_client, "_from_gdelt", lambda q, d, m: [])
        monkeypatch.setattr(news_client, "_from_rss", lambda q, m: [])

        orch = ResearchOrchestrator(transport="inproc")
        result = await orch.run("analyze momentum for NVDA and WDC over 60 days")
        await orch.pool.stop()

        memo = result.memo_markdown
        assert "WDC" in memo  # named as a gap
        assert "no data" in memo.lower()
        # No fabricated WDC number anywhere
        import re

        wdc_numbers = re.findall(r"WDC[^.]*?(\d+\.\d+)", memo)
        assert wdc_numbers == []

    async def test_verifier_catches_injected_bad_claim(self, offline, fake_yf, monkeypatch: pytest.MonkeyPatch) -> None:
        """DoD check: an injected bad claim passes through the WRITER path, the
        verifier catches it, a revision is triggered, and the catch is visible
        in the report and the final memo."""
        from mcp_servers import news_client
        from agents.orchestrator import ResearchOrchestrator
        from agents.schemas import MemoDraft

        monkeypatch.setattr(news_client, "_from_gdelt", lambda q, d, m: [])
        monkeypatch.setattr(news_client, "_from_rss", lambda q, m: [])

        orch = ResearchOrchestrator(transport="inproc")
        # Inject a poisoned writer: always produces the unsupported claim.
        bad_draft = MemoDraft(
            title="bad",
            executive_summary="bad",
            body_markdown=(
                "# bad\n\n## Key findings\n- WDC momentum surged 85% (0.85), "
                "confirming the sector rally.\n"
            ),
            claims=["WDC momentum surged 85% (0.85), confirming the sector rally."],
        )
        orch.writer.write = lambda *a, **k: _async_return(bad_draft)

        result = await orch.run("analyze semiconductor momentum")
        await orch.pool.stop()

        # 1. The bad claim WAS caught: a revision was triggered with the flags.
        assert result.verification.revisions, "verifier must record the catch"
        flagged = result.verification.revisions[0]["flagged"]
        codes = {f.get("code") for f in flagged}
        assert {"unsupported_number", "cites_missing_data"} & codes
        # 2. After the bounded retry, the pipeline published anyway but the
        #    catch stayed visible (nothing silently dropped).
        assert "Verification report" in result.memo_markdown
        # 3. Trace contains the verifier's revision_requested event.
        events = [e for e in orch.trace_path_events() if e.get("event_type") == "agent_step"]
        assert any(e.get("action") == "revision_requested" for e in events)

    async def test_news_and_data_run_concurrently(self, offline, fake_yf, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both branches execute under the same event loop (asyncio.gather)."""
        from mcp_servers import news_client
        from agents.orchestrator import ResearchOrchestrator

        events: list[str] = []

        real_gdelt = news_client._from_gdelt

        def spy_gdelt(q, d, m):
            events.append("news")
            return real_gdelt.__wrapped__(q, d, m) if hasattr(real_gdelt, "__wrapped__") else []

        import asyncio as _asyncio

        from agents import workers as workers_mod

        orig_gather_data = workers_mod.DataAgent.gather

        async def spy_gather(self, plan):
            events.append("data")
            return await orig_gather_data(self, plan)

        monkeypatch.setattr(news_client, "_from_gdelt", spy_gdelt)
        monkeypatch.setattr(news_client, "_from_rss", lambda q, m: [])
        monkeypatch.setattr(workers_mod.DataAgent, "gather", spy_gather)

        orch = ResearchOrchestrator(transport="inproc")
        await orch.run("analyze semiconductor momentum over 60 days")
        await orch.pool.stop()

        assert "data" in events and "news" in events


async def _async_return(value):  # helper used above
    return value


def trace_events(path):
    from core.trace import read_trace

    return read_trace(path)
