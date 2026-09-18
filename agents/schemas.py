"""Pydantic schemas enforcing structured agent I/O.

The orchestrator pipeline passes these between stages; schema violations are
hard failures (with one bounded repair retry) rather than silent corruption.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TaskPlan(BaseModel):
    """Stage 1 output: the decomposed request."""

    request_summary: str = Field(description="one-sentence restatement of the user request")
    sector: str | None = Field(default=None, description="sector name if the request is sector-scoped")
    tickers: list[str] = Field(default_factory=list, description="explicit tickers to analyze")
    lookback_days: int = Field(default=90, ge=5, le=730, description="price history window in days")
    analyses: list[str] = Field(
        default_factory=list,
        description="requested analysis types, e.g. momentum, volatility, valuation, sentiment, correlation",
    )
    news_query: str | None = Field(default=None, description="news query derived from the request")
    notes: list[str] | None = Field(default=None, description="caveats, fallback notices, clarifications")


class GatheredData(BaseModel):
    """Stage 2/3 outputs: raw tool results, kept verbatim for traceability."""

    prices: dict[str, dict] = Field(default_factory=dict, description="ticker -> get_price_history envelope")
    fundamentals: dict[str, dict] = Field(default_factory=dict, description="ticker -> get_fundamentals envelope")
    headlines: dict | None = Field(default=None, description="get_recent_headlines envelope")
    news: dict | None = Field(default=None, description="full news agent output: headlines + sentiment envelopes")
    no_data: dict[str, str] = Field(default_factory=dict, description="ticker -> honest no-data reason")
    notes: list[str] = Field(default_factory=list, description="methodology notes, e.g. how a sector was expanded to tickers")


class QuantOutput(BaseModel):
    """Stage 4 output: computed signals with sample sizes."""

    metrics: dict[str, dict] = Field(
        default_factory=dict,
        description="name like NVDA_momentum -> {value, n_observations, window_days, ...}",
    )
    correlation: dict | None = Field(default=None)
    notes: list[str] = Field(default_factory=list, description="honest notes about gaps/limitations")


class MemoDraft(BaseModel):
    """Stage 5 output: the writer's memo with extractable claims."""

    title: str
    executive_summary: str
    body_markdown: str
    claims: list[str] = Field(default_factory=list, description="atomic factual claims for verification")


class VerificationReport(BaseModel):
    """Stage 6 output: per-claim verdicts plus lookahead checks."""

    claim_reports: list[dict] = Field(default_factory=list)
    lookahead: dict = Field(default_factory=dict)
    unresolved_flags: list[dict] = Field(default_factory=list)
    revisions: list[dict] = Field(default_factory=list, description="history of verify->revise cycles: what was flagged and when")
    passed: bool = Field(default=True)


class RunResult(BaseModel):
    """Final pipeline output."""

    memo_markdown: str
    plan: TaskPlan
    verification: VerificationReport
    output_path: str | None = None
    model_mode: str = Field(description="'claude' or 'offline_heuristic'")
