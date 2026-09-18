"""Stage 1: planner agent — decompose the user request into a TaskPlan.

Live mode: Claude returns JSON validated against the pydantic schema (with one
repair retry). Offline mode (or unparseable model output): a transparent
keyword heuristic produces the plan; the fallback is recorded in the plan notes
and the trace, never silently.
"""

from __future__ import annotations

import re

from agents.base import ask_structured, mode_label
from agents.schemas import TaskPlan
from core.trace import TraceLogger

_KNOWN_SECTORS = [
    "semiconductors", "technology", "communication services", "financials",
    "healthcare", "consumer discretionary", "consumer staples", "energy",
    "industrials", "utilities", "materials", "real estate",
]
_SECTOR_ALIASES = {
    "semis": "semiconductors",
    "semi": "semiconductors",
    "chips": "semiconductors",
    "chip": "semiconductors",
    "semiconductor": "semiconductors",
    "tech": "technology",
    "financial": "financials",
    "material": "materials",
    "industrial": "industrials",
    "healthcare": "healthcare",
    "health care": "healthcare",
}

_ANALYSIS_KEYWORDS = {
    "momentum": "momentum",
    "trend": "momentum",
    "volatility": "volatility",
    "vol": "volatility",
    "risk": "volatility",
    "valuation": "valuation",
    "cheap": "valuation",
    "expensive": "valuation",
    "sentiment": "sentiment",
    "news": "sentiment",
    "correlation": "correlation",
    "diversification": "correlation",
}
_TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")
_DAYS_RE = re.compile(r"(\d+)\s*[- ]?day", re.IGNORECASE)


class PlannerAgent:
    name = "planner"

    def __init__(self, trace: TraceLogger | None = None) -> None:
        self.trace = trace

    async def plan(self, request: str) -> TaskPlan:
        if _claude_available():
            try:
                plan = self._plan_with_claude(request)
                self._trace(plan, mode="claude")
                return plan
            except Exception as exc:
                if self.trace:
                    self.trace.agent_step(self.name, "planner_fallback", reason=str(exc))
                plan = self._heuristic_plan(request)
                plan.notes = [f"planner fell back to keyword heuristic: {exc}"]
                self._trace(plan, mode="offline_heuristic_fallback")
                return plan
        plan = self._heuristic_plan(request)
        plan.notes = ["planner ran in offline heuristic mode (no ANTHROPIC_API_KEY)"]
        self._trace(plan, mode="offline_heuristic")
        return plan

    def _plan_with_claude(self, request: str) -> TaskPlan:
        system = (
            "You decompose market-research requests into a JSON task plan. "
            "Map sector words to one of: " + ", ".join(_KNOWN_SECTORS) + ". "
            "Only include explicit ticker symbols the user named. Choose a "
            "lookback window (days) appropriate to the ask (default 90). "
            "analyses must be drawn from: momentum, volatility, valuation, "
            "sentiment, correlation. news_query: a good web-news query string."
        )
        prompt = (
            f"User request: {request!r}\n\n"
            'Return JSON: {"request_summary": str, "sector": str|null, '
            '"tickers": [str], "lookback_days": int, "analyses": [str], '
            '"news_query": str|null, "notes": str|null}'
        )
        return ask_structured(prompt, TaskPlan, system=system)

    def _heuristic_plan(self, request: str) -> TaskPlan:
        text = request.lower().replace("-", " ")
        sector = None
        for alias, canonical in _SECTOR_ALIASES.items():
            if re.search(rf"\b{alias}s?\b", text):
                sector = canonical
                break
        if sector is None:
            for known in _KNOWN_SECTORS:
                # match plural or singular ("semiconductors"/"semiconductor")
                if known in text or (known.endswith("s") and known[:-1] in text):
                    sector = known
                    break

        tickers = [t for t in _TICKER_RE.findall(request) if t not in _STOPWORDS]
        # Symbol-like tokens that are NOT uppercase (e.g. 'ZZZINVALID') are
        # ignored by the regex; surface that instead of silently dropping.
        skipped = [w for w in re.findall(r"\b[A-Za-z]{2,}\b", request)
                   if w.isupper() and w not in tickers and w not in _STOPWORDS]
        skipped_notes = [f"ignored symbol-like token '{w}' (not a plausible ticker)" for w in skipped]
        analyses: list[str] = []
        seen: set[str] = set()
        for kw, analysis in _ANALYSIS_KEYWORDS.items():
            if re.search(rf"\b{kw}\b", text) and analysis not in seen:
                analyses.append(analysis)
                seen.add(analysis)
        if not analyses:
            analyses = ["momentum"]  # a research tool needs a default ask

        days_match = _DAYS_RE.search(request)
        lookback = int(days_match.group(1)) if days_match else 90
        lookback = max(5, min(lookback, 730))

        news_query = f"{sector} stocks" if sector else (f"{tickers[0]} stock" if tickers else request)

        return TaskPlan(
            request_summary=request.strip(),
            sector=sector,
            tickers=tickers,
            lookback_days=lookback,
            analyses=analyses,
            news_query=news_query,
            notes=skipped_notes or None,
        )

    def _trace(self, plan: TaskPlan, mode: str) -> None:
        if self.trace:
            self.trace.agent_step(
                self.name, "plan_created", mode=mode,
                plan=plan.model_dump(),
            )


_STOPWORDS = {
    "A", "I", "AN", "THE", "AND", "OR", "OF", "IN", "ON", "FOR", "TO", "IS",
    "ARE", "VS", "WITH", "AT", "BY", "US", "UK", "EU", "IT", "AS", "BE",
    "WAS", "DAY", "DAYS", "WEEK", "WEEKS", "MONTH", "MONTHS", "YEAR", "YEARS",
    "MEAN", "NVD", "DOS",
}


def _claude_available() -> bool:
    from core import model as model_client

    return model_client.available()


_mode_label = mode_label  # re-export for orchestrator convenience
