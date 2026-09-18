"""Sentiment analysis over headlines.

PATTERN NOTE — model-as-a-tool: `summarize_sentiment` is an MCP tool whose
implementation calls Claude (via core.model) as a subroutine. So the writer
agent's data-gathering toolchain embeds its own model call: one agent's tool
invokes the model directly, independent of the orchestrator loop. This is a
deliberate, documented composition pattern — the trace log records these
embedded calls with agent="news_sentiment_server(tool-internal)" so the whole
run remains explainable.

Offline mode (no ANTHROPIC_API_KEY): falls back to a transparent lexicon
heuristic. The result is clearly labeled mode="offline_lexicon" (vs
mode="claude") so no one mistakes it for a model judgment. It never fabricates
headlines and it still refuses to score an empty list.
"""

from __future__ import annotations

import re
from typing import Any

from core import model as model_client
from core.config import get_settings

# Tiny, transparent lexicon for the offline fallback only. Deliberately simple:
# its job is to keep the pipeline runnable without secrets, not to be SOTA.
_POSITIVE = {
    "beat", "beats", "record", "surge", "surges", "soar", "soars", "rally",
    "rallies", "jump", "jumps", "gain", "gains", "strong", "growth", "upgrade",
    "upgraded", "outperform", "expand", "expands", "expansion", "demand",
    "profit", "profits", "win", "wins", "boost", "boosts", "optimism",
    "bullish", "tops", "exceeds", "raises", "hikes",
}
_NEGATIVE = {
    "miss", "misses", "fall", "falls", "drop", "drops", "plunge", "plunges",
    "slump", "slumps", "weak", "weakness", "downgrade", "downgraded",
    "underperform", "cuts", "cut", "layoffs", "lawsuit", "probe", "recall",
    "shortage", "glut", "warns", "warning", "bearish", "loss", "losses",
    "decline", "declines", "slides", "tumbles", "fears", "concerns",
}
_WORD_RE = re.compile(r"[a-z']+")


def summarize_sentiment(headlines: list[str], context: str | None = None) -> dict[str, Any]:
    """Score headline sentiment with reasoning; refuses empty input."""
    titles = [h.strip() for h in headlines if h and h.strip()]
    if not titles:
        return {
            "ok": False,
            "error": "no headlines provided to summarize — refusing to invent sentiment",
            "no_data": True,
        }

    if model_client.available():
        return _summarize_with_claude(titles, context)
    return _summarize_lexicon(titles, context)


def _summarize_with_claude(titles: list[str], context: str | None) -> dict[str, Any]:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(titles))
    prompt = (
        "You are a careful research analyst. Score the aggregate news sentiment "
        "of these headlines.\n\n"
        f"Headlines:\n{numbered}\n\n"
        + (f"Context: {context}\n\n" if context else "")
        + "Return JSON exactly like: "
        '{"score": <float in [-1, 1]>, "label": "<very negative|negative|neutral|positive|very positive>", '
        '"confidence": "<low|medium|high>", "reasoning": "<2-3 sentences citing specific headlines>", '
        '"headline_labels": [{"title": "<verbatim title>", "label": "<pos|neg|neutral>"}]}'
    )
    system = (
        "You assess news sentiment for a research tool. Be calibrated: if "
        "evidence is mixed or thin, lower the confidence. Use only the "
        "headlines given; never invent additional headlines or facts."
    )
    try:
        raw = model_client.complete_json(prompt, system=system)
    except Exception as exc:
        # Model call failed (network/quota) — degrade to lexicon, transparently.
        out = _summarize_lexicon(titles, context)
        out["result"]["claude_error"] = str(exc)
        return out

    import json as _json

    try:
        parsed = _json.loads(raw)
    except _json.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, dict) or "score" not in parsed:
        # Unparseable model output — degrade to lexicon, transparently.
        out = _summarize_lexicon(titles, context)
        out["result"]["claude_error"] = "model returned unparseable JSON; used fallback"
        return out

    score = max(-1.0, min(1.0, float(parsed.get("score", 0.0))))
    return {
        "ok": True,
        "result": {
            "mode": "claude",
            "model": _model_name(),
            "score": score,
            "label": parsed.get("label", _label_for(score)),
            "confidence": parsed.get("confidence", "low"),
            "reasoning": parsed.get("reasoning", ""),
            "headline_labels": parsed.get("headline_labels", []),
            "n_headlines": len(titles),
        },
    }


def _summarize_lexicon(titles: list[str], context: str | None) -> dict[str, Any]:
    per: list[dict] = []
    for t in titles:
        words = _WORD_RE.findall(t.lower())
        pos = sum(1 for w in words if w in _POSITIVE)
        neg = sum(1 for w in words if w in _NEGATIVE)
        label = "pos" if pos > neg else "neg" if neg > pos else "neutral"
        per.append({"title": t, "label": label})

    n_pos = sum(1 for p in per if p["label"] == "pos")
    n_neg = sum(1 for p in per if p["label"] == "neg")
    n_neu = len(per) - n_pos - n_neg
    score = (n_pos - n_neg) / len(per)

    reasoning = (
        f"Offline lexicon heuristic (no ANTHROPIC_API_KEY configured): {n_pos} positive, "
        f"{n_neg} negative, {n_neu} neutral of {len(per)} headlines. Words are matched "
        "against a small fixed lexicon; this is a placeholder, not a model judgment."
    )
    return {
        "ok": True,
        "result": {
            "mode": "offline_lexicon",
            "score": round(score, 4),
            "label": _label_for(score),
            "confidence": "low",
            "reasoning": reasoning,
            "headline_labels": per,
            "n_headlines": len(per),
        },
    }


def _label_for(score: float) -> str:
    if score <= -0.5:
        return "very negative"
    if score < -0.1:
        return "negative"
    if score < 0.1:
        return "neutral"
    if score < 0.5:
        return "positive"
    return "very positive"


def _model_name() -> str:
    return get_settings().model
