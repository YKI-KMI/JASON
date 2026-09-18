"""Deterministic verification logic for research claims.

Why deterministic: an LLM verifier would just be another model opinion. The
verification server instead runs explainable, unit-tested checks:

  - numeric grounding: every number in a claim must match a metric in the
    supporting data (with percentage/magnitude conversions and hedging-aware
    tolerances); unmatched numbers -> flag "unsupported_number";
  - directional consistency: language like "positive"/"declined"/"strong" must
    agree with the sign of the matched metric — checked within the claim
    clause containing the matched number, not across the whole claim;
  - certainty overclaim: unhedged certainty words, or certainty language on a
    weak/short sample (flag "certainty_overclaim");
  - overstated strength: "strong/significant" wording on a near-zero metric;
  - window mismatch: "90-day momentum" claims where the metric used another
    window;
  - missing-data honesty: claims referencing metrics known to have no data.

check_lookahead_bias scans analysis steps for future-dated data usage and
forward-looking wording relative to the run's as-of date.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

_ABSOLUTE_TOL = 0.02        # absolute tolerance when |expected| < 1
_RELATIVE_TOL = 0.02        # 2% relative tolerance otherwise

_CERTAINTY_WORDS = [
    "guaranteed", "guarantee", "certainly", "undoubtedly", "definitively",
    "will rise", "will fall", "will rally", "will crash", "proven",
    "unquestionably", "undeniably", "assuredly", "inevitably",
    "confirms", "confirming", "confirms that", "proves",
]
_CERTAINTY_SAMPLE_RE = re.compile(r"\b(definitive|certain|clear|unambiguous|proven)\b")

_HEDGES = ("about", "approximately", "around", "roughly", "nearly", "close to", "almost")

_DIRECTIONAL = {
    "positive": (">", 0.0),
    "negative": ("<", 0.0),
    "rose": (">", 0.0),
    "gained": (">", 0.0),
    "increased": (">", 0.0),
    "declined": ("<", 0.0),
    "fell": ("<", 0.0),
    "dropped": ("<", 0.0),
    "strengthened": (">", 0.0),
    "weakened": ("<", 0.0),
    "accelerated": (">", 0.0),
    "decelerated": ("<", 0.0),
    "expanded": (">", 0.0),
    "contracted": ("<", 0.0),
    "outperformed": (">", 0.0),
    "underperformed": ("<", 0.0),
    "elevated": (">", 0.0),
}

_STRENGTH_WORDS = {"strong": 0.05, "significant": 0.05, "substantial": 0.05, "sharply": 0.05}
_MIN_N_FOR_CERTAINTY = 20


@dataclass
class Flag:
    code: str
    message: str
    metric: str | None = None

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "metric": self.metric}


@dataclass
class Verdict:
    supported: bool
    flags: list[Flag] = field(default_factory=list)
    explanation: str = ""
    checks: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "supported": self.supported,
            "flags": [f.to_dict() for f in self.flags],
            "explanation": self.explanation,
            "checks": self.checks,
        }


def check_claim_against_data(claim_text: str, supporting_data: dict[str, Any]) -> dict[str, Any]:
    """Verify a claim against the metric payload it cites.

    supporting_data shape (produced by the orchestrator's context builder):
      {
        "as_of": "2026-09-14",
        "metrics": {
          "NVDA_momentum": {"value": 0.234, "n_observations": 89, "window_days": 90},
        },
        "no_data": {"WDC": "no price data returned for ticker 'WDC'"}
      }
    Returns {ok: True, result: Verdict.to_dict()}; bad *inputs* return error
    envelopes — claim-content issues become flags, never exceptions.
    """
    if not isinstance(claim_text, str) or not claim_text.strip():
        return {"ok": False, "error": "claim_text must be a non-empty string"}
    if not isinstance(supporting_data, dict):
        return {"ok": False, "error": "supporting_data must be a dict with a 'metrics' mapping"}

    metrics = supporting_data.get("metrics") or {}
    flags: list[Flag] = []
    checks: list[dict] = []
    clauses = _clauses(claim_text)

    numbers = _numbers_in(claim_text)
    search_from = 0
    for num, raw in numbers:
        pos = claim_text.find(raw, search_from)
        if pos >= 0:
            search_from = pos + 1
        clause = _clause_at(clauses, pos if pos >= 0 else 0)
        check = {"number": num, "as_written": raw, "matched": None, "status": "unmatched"}
        m = _match_metric(num, claim_text, metrics)
        if m is None:
            # Bare small integers are usually ordinals ("3 of 5 tickers"), not data.
            if not (num == int(num) and 0 <= num <= 12 and not raw.endswith("%")):
                flags.append(Flag(
                    "unsupported_number",
                    f"number {raw} in the claim does not match any metric in the supporting data",
                ))
        else:
            name, expected, md, ambiguous_with = m
            check["matched"] = name
            check["status"] = "matched"
            if ambiguous_with:
                flags.append(Flag(
                    "ambiguous_number",
                    f"number {raw} matches multiple metrics ({name} and {ambiguous_with}) "
                    "equally — the claim should name its subject",
                    metric=name,
                ))
            flags.extend(_direction_flags(clause, name, expected))
            flags.extend(_calibration_flags(claim_text, clause, name, expected, md))
        checks.append(check)

    flags.extend(_certainty_flags(claim_text))

    for key, reason in (supporting_data.get("no_data") or {}).items():
        if key.lower() in claim_text.lower():
            flags.append(Flag("cites_missing_data", f"claim references '{key}' which has no data: {reason}", metric=key))

    supported = not any(
        f.code in {"unsupported_number", "direction_mismatch", "cites_missing_data"} for f in flags
    )
    explanation = _explain(checks, flags)
    verdict = Verdict(supported=supported, flags=flags, explanation=explanation, checks=checks)
    return {"ok": True, "result": verdict.to_dict()}


# --------------------------------------------------------------------------
# numeric extraction + matching
# --------------------------------------------------------------------------
_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(%|percent|basis points|bps)?", re.IGNORECASE)
_WINDOW_RE = re.compile(r"\b\d+(?:\.\d+)?\s*[- ]?\s*(?:days?|weeks?|months?|years?)\b", re.IGNORECASE)


def _numbers_in(text: str) -> list[tuple[float, str]]:
    """Extract (value, as_written) pairs.

    Skips: bare years 1900-2099, and "N day/week/month/year" window phrases
    (those are checked separately by the window-mismatch rule, not as data).
    """
    text = _WINDOW_RE.sub(" ", text)
    out: list[tuple[float, str]] = []
    for m in _NUM_RE.finditer(text):
        raw = m.group(0)
        val = float(m.group(1))
        unit = (m.group(2) or "").lower()
        if not unit and 1900 <= val <= 2099 and val == int(val):
            continue
        if unit in ("basis points", "bps"):
            val = val / 10000.0
        elif unit in ("%", "percent"):
            val = val / 100.0
        out.append((val, raw.strip()))
    return out


def _match_metric(num: float, claim: str, metrics: dict) -> tuple[str, float, dict, str | None] | None:
    """Best-matching metric for a number, or None.

    A metric is a candidate only if the claim mentions its TYPE token (the
    last name segment, e.g. 'momentum' in NVDA_momentum) — so a momentum
    number can't silently match a volatility metric that merely shares the
    ticker. Among candidates the highest token-affinity wins; if two same-
    type candidates tie, the match is ambiguous and the caller flags it.
    """
    claim_l = claim.lower()
    candidates: list[tuple[str, float, dict, float]] = []
    for name, spec in metrics.items():
        if not isinstance(spec, dict) or "value" not in spec:
            continue
        tokens = [t for t in re.split(r"[_\s\-.]+", name.lower()) if t]
        if not tokens:
            continue
        type_token = tokens[-1]
        if type_token not in claim_l:
            continue
        affinity = sum(1 for t in tokens if t in claim_l) / len(tokens)
        if _values_agree(num, float(spec["value"]), name):
            candidates.append((name, float(spec["value"]), spec, affinity))
    if not candidates:
        return None
    candidates.sort(key=lambda c: -c[3])
    best = candidates[0]
    ambiguous = next((c[0] for c in candidates[1:] if c[3] == best[3] and c[0] != best[0]), None)
    return best[0], best[1], best[2], ambiguous


def _name_affinity(claim: str, metric_name: str) -> float:
    tokens = [t for t in re.split(r"[_\s\-.]+", metric_name.lower()) if t]
    claim_l = claim.lower()
    return sum(1 for t in tokens if t in claim_l) / max(1, len(tokens))


def _values_agree(a: float, b: float, metric_name: str) -> bool:
    if a == b:
        return True
    tol = _ABSOLUTE_TOL if abs(b) < 1 else abs(b) * _RELATIVE_TOL
    if abs(a - b) <= tol:
        return True
    # Percentage-vs-ratio confusion: claim says -12%, metric stores -0.12.
    for factor in (100.0, 0.01):
        if abs(a * factor - b) <= max(_ABSOLUTE_TOL, abs(b) * _RELATIVE_TOL):
            return True
    return False


# --------------------------------------------------------------------------
# language checks
# --------------------------------------------------------------------------
_CLAUSE_SPLIT_RE = re.compile(r"[;.!]")


def _clauses(text: str) -> list[tuple[int, int, str]]:
    """(start, end, text) spans split on sentence/clause punctuation."""
    spans: list[tuple[int, int, str]] = []
    start = 0
    for m in _CLAUSE_SPLIT_RE.finditer(text):
        spans.append((start, m.start(), text[start:m.start()]))
        start = m.end()
    spans.append((start, len(text), text[start:]))
    return spans


def _clause_at(clauses: list[tuple[int, int, str]], pos: int) -> str:
    for start, end, text in clauses:
        if start <= pos <= end:
            return text
    return clauses[0][2] if clauses else ""


def _direction_flags(clause: str, metric_name: str, value: float) -> list[Flag]:
    """Directional words in the SAME clause as the matched number must agree
    with the metric's sign."""
    flags: list[Flag] = []
    clause_l = clause.lower()
    for word, (op, ref) in _DIRECTIONAL.items():
        if word not in clause_l:
            continue
        ok = value > ref if op == ">" else value < ref
        if not ok:
            flags.append(Flag(
                "direction_mismatch",
                f"claim says '{word}' but {metric_name}={value:.4g} contradicts that",
                metric=metric_name,
            ))
        break  # one directional word per clause is enough
    return flags


def _calibration_flags(claim: str, clause: str, metric_name: str, value: float, spec: dict) -> list[Flag]:
    flags: list[Flag] = []
    claim_l = claim.lower()
    clause_l = clause.lower()
    n_obs = int(spec.get("n_observations") or 0)

    # Strength words on a near-zero metric.
    for strong, thresh in _STRENGTH_WORDS.items():
        if strong in clause_l and abs(value) < thresh:
            flags.append(Flag(
                "overstated_strength",
                f"'{strong}' claim but {metric_name}={value:.4g} is small; prefer calibrated language",
                metric=metric_name,
            ))
            break

    # Certainty language on a thin sample.
    if n_obs and n_obs < _MIN_N_FOR_CERTAINTY and _CERTAINTY_SAMPLE_RE.search(claim_l):
        flags.append(Flag(
            "certainty_overclaim",
            f"certainty language with only n={n_obs} observations for {metric_name}",
            metric=metric_name,
        ))

    # Window-size mismatch: "90-day momentum" but metric used another window.
    window_words = re.findall(r"(\d+)\s*[- ]?\s*day", clause_l)
    if window_words and spec.get("window_days"):
        for w in window_words:
            if abs(int(w) - int(spec["window_days"])) > 2:
                flags.append(Flag(
                    "window_mismatch",
                    f"claim says {w}-day window but {metric_name} used {spec['window_days']}",
                    metric=metric_name,
                ))
                break
    return flags


def _certainty_flags(claim: str) -> list[Flag]:
    flags: list[Flag] = []
    claim_l = claim.lower()
    for w in _CERTAINTY_WORDS:
        if w in claim_l:
            flags.append(Flag(
                "certainty_overclaim",
                f"unhedged certainty language ('{w}') — research memos must be calibrated",
            ))
    return flags


def _explain(checks: list[dict], flags: list[Flag]) -> str:
    n_match = sum(1 for c in checks if c["status"] == "matched")
    n_num = len(checks)
    parts = [f"{n_match}/{n_num} numeric reference(s) matched to supporting metrics."]
    if flags:
        parts.append("Flags: " + "; ".join(f"{f.code}: {f.message}" for f in flags))
    else:
        parts.append("No consistency flags.")
    return " ".join(parts)


# --------------------------------------------------------------------------
# lookahead bias
# --------------------------------------------------------------------------
_FUTURE_WORDS = re.compile(
    r"\b(next (?:week|month|quarter|year|days)|going forward|forecast(?:ed|ing)?|"
    r"predict(?:s|ed|ing)?|will (?:be|rise|fall|grow|earn))\b",
    re.IGNORECASE,
)


def check_lookahead_bias(analysis_steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Flag analysis steps that could use data not available at as_of time.

    Each step: {"step": str, "data_end_date": "YYYY-MM-DD" | null,
                "as_of": "YYYY-MM-DD"}  (as_of falls back to the first step's)
    """
    if not isinstance(analysis_steps, list) or not analysis_steps:
        return {"ok": False, "error": "analysis_steps must be a non-empty list"}

    default_as_of = analysis_steps[0].get("as_of")
    flags: list[dict] = []
    for i, step in enumerate(analysis_steps):
        name = str(step.get("step", f"step_{i}"))
        as_of = step.get("as_of") or default_as_of
        end = step.get("data_end_date")
        if as_of and end:
            d_as_of = _parse_date(as_of)
            d_end = _parse_date(end)
            if d_as_of and d_end and d_end > d_as_of:
                flags.append({
                    "step": name,
                    "code": "future_data_used",
                    "message": f"step uses data ending {d_end.isoformat()} which is after as_of {d_as_of.isoformat()}",
                })
        desc = str(step.get("step", ""))
        m = _FUTURE_WORDS.search(desc)
        if m:
            flags.append({
                "step": name,
                "code": "forward_looking_wording",
                "message": f"step contains forward-looking wording ('{m.group(0)}') — describe data in the past tense or label as scenario",
            })
    return {
        "ok": True,
        "result": {"n_steps": len(analysis_steps), "n_flags": len(flags), "flags": flags},
    }


def _parse_date(s: Any) -> date | None:
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
