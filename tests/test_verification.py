"""Tests for the verification server: claim grounding, direction/overclaim
checks, lookahead bias, MCP schema — including the DEFINITION-OF-DONE check
where a deliberately-injected unsupported claim gets flagged."""

from __future__ import annotations

import pytest

from mcp_servers import verification
from mcp_servers.verification_server import mcp as verif_mcp


GOOD_METRICS = {
    "metrics": {
        "NVDA_momentum": {"value": 0.234, "n_observations": 89, "window_days": 90},
        "NVDA_volatility": {"value": 0.412, "n_observations": 20, "window_days": 20},
        "AMD_momentum": {"value": -0.052, "n_observations": 89, "window_days": 90},
    }
}


# --------------------------------------------------------------------------
# Numeric grounding
# --------------------------------------------------------------------------
class TestNumericGrounding:
    def test_exact_number_passes(self) -> None:
        out = verification.check_claim_against_data(
            "NVDA 90-day momentum was 0.234.", GOOD_METRICS
        )
        assert out["ok"] is True
        assert out["result"]["supported"] is True
        assert out["result"]["flags"] == []

    def test_percent_form_of_ratio_matches(self) -> None:
        # Claim writes 23.4%, metric stores 0.234 -> must match.
        out = verification.check_claim_against_data(
            "NVDA 90-day momentum was 23.4%.", GOOD_METRICS
        )
        assert out["result"]["supported"] is True

    def test_hedged_rounding_within_tolerance(self) -> None:
        out = verification.check_claim_against_data(
            "NVDA momentum was roughly 0.23 over the window.", GOOD_METRICS
        )
        assert out["result"]["supported"] is True

    def test_wrong_number_is_flagged(self) -> None:
        out = verification.check_claim_against_data(
            "NVDA 90-day momentum was 0.4.", GOOD_METRICS
        )
        flags = [f["code"] for f in out["result"]["flags"]]
        assert "unsupported_number" in flags
        assert out["result"]["supported"] is False

    def test_fabricated_number_with_no_metric(self) -> None:
        out = verification.check_claim_against_data(
            "NVDA price-to-book ratio is 47.3.", GOOD_METRICS
        )
        assert out["result"]["supported"] is False
        assert any(f["code"] == "unsupported_number" for f in out["result"]["flags"])

    def test_bare_small_integers_are_not_flagged(self) -> None:
        # "3 of 5 tickers" style ordinals aren't data citations.
        out = verification.check_claim_against_data(
            "Momentum was 0.234; 3 of 5 tickers declined.", GOOD_METRICS
        )
        assert out["result"]["supported"] is True

    def test_check_details_recorded(self) -> None:
        out = verification.check_claim_against_data(
            "NVDA volatility was 0.412.", GOOD_METRICS
        )
        checks = out["result"]["checks"]
        assert len(checks) == 1
        assert checks[0]["matched"] == "NVDA_volatility"


# --------------------------------------------------------------------------
# Direction / calibration
# --------------------------------------------------------------------------
class TestDirectionAndCalibration:
    def test_direction_mismatch_flagged(self) -> None:
        out = verification.check_claim_against_data(
            "AMD momentum is positive at -0.052 over 90 days.", GOOD_METRICS
        )
        codes = [f["code"] for f in out["result"]["flags"]]
        assert "direction_mismatch" in codes
        assert out["result"]["supported"] is False

    def test_direction_agreement_passes(self) -> None:
        out = verification.check_claim_against_data(
            "NVDA momentum is positive at 0.234.", GOOD_METRICS
        )
        codes = [f["code"] for f in out["result"]["flags"]]
        assert "direction_mismatch" not in codes

    def test_certainty_word_flagged(self) -> None:
        out = verification.check_claim_against_data(
            "NVDA momentum of 0.234 will rise further.", GOOD_METRICS
        )
        codes = [f["code"] for f in out["result"]["flags"]]
        assert "certainty_overclaim" in codes

    def test_thin_sample_flagged_for_certainty(self) -> None:
        thin = {"metrics": {"XYZ_momentum": {"value": 0.9, "n_observations": 8, "window_days": 10}}}
        out = verification.check_claim_against_data(
            "The evidence is definitive: XYZ momentum is 0.9.", thin
        )
        codes = [f["code"] for f in out["result"]["flags"]]
        assert "certainty_overclaim" in codes
        assert "n=8" in out["result"]["explanation"]

    def test_overstated_strength_flagged(self) -> None:
        weak = {"metrics": {"ABC_volatility": {"value": 0.01, "n_observations": 30, "window_days": 30}}}
        out = verification.check_claim_against_data(
            "ABC volatility is significantly elevated at 0.01.", weak
        )
        codes = [f["code"] for f in out["result"]["flags"]]
        assert "overstated_strength" in codes

    def test_calibrated_language_passes(self) -> None:
        # The target style: positive but weak, n stated, no certainty words.
        out = verification.check_claim_against_data(
            "Momentum signal is positive but weak (0.05, n=30 days); treat as indicative only.",
            {"metrics": {"SECTOR_momentum": {"value": 0.05, "n_observations": 30, "window_days": 30}}},
        )
        assert out["result"]["supported"] is True
        assert out["result"]["flags"] == []


# --------------------------------------------------------------------------
# Missing data honesty
# --------------------------------------------------------------------------
class TestMissingData:
    def test_claim_citing_no_data_metric_flagged(self) -> None:
        data = {
            "metrics": {"NVDA_momentum": {"value": 0.2, "n_observations": 60}},
            "no_data": {"WDC": "no price data returned for ticker 'WDC'"},
        }
        out = verification.check_claim_against_data(
            "NVDA momentum was 0.2 and WDC momentum was strong.", data
        )
        codes = [f["code"] for f in out["result"]["flags"]]
        assert "cites_missing_data" in codes
        assert out["result"]["supported"] is False

    def test_no_data_not_mentioned_is_fine(self) -> None:
        data = {
            "metrics": {"NVDA_momentum": {"value": 0.2, "n_observations": 60}},
            "no_data": {"WDC": "no price data"},
        }
        out = verification.check_claim_against_data("NVDA momentum was 0.2.", data)
        assert out["result"]["supported"] is True


# --------------------------------------------------------------------------
# Lookahead bias
# --------------------------------------------------------------------------
class TestLookaheadBias:
    def test_future_data_use_flagged(self) -> None:
        steps = [
            {"step": "gather prices", "data_end_date": "2026-09-01", "as_of": "2026-08-15"},
        ]
        out = verification.check_lookahead_bias(steps)
        codes = [f["code"] for f in out["result"]["flags"]]
        assert "future_data_used" in codes

    def test_clean_steps_pass(self) -> None:
        steps = [
            {"step": "gather prices through 2026-08-01", "data_end_date": "2026-08-01", "as_of": "2026-08-15"},
            {"step": "compute 30-day momentum", "data_end_date": "2026-08-01", "as_of": "2026-08-15"},
        ]
        out = verification.check_lookahead_bias(steps)
        assert out["result"]["n_flags"] == 0

    def test_forward_looking_wording_flagged(self) -> None:
        steps = [{"step": "predict next quarter earnings", "as_of": "2026-08-15"}]
        out = verification.check_lookahead_bias(steps)
        codes = [f["code"] for f in out["result"]["flags"]]
        assert "forward_looking_wording" in codes

    def test_empty_steps_error(self) -> None:
        assert verification.check_lookahead_bias([])["ok"] is False


# --------------------------------------------------------------------------
# DEFINITION OF DONE: deliberately-injected bad claim
# --------------------------------------------------------------------------
class TestInjectedBadClaim:
    def test_injected_unsupported_claim_is_caught(self) -> None:
        """Simulates the writer agent hallucinating a number: the claim cites
        'WDC momentum of 0.85' but the supporting data contains no WDC metric
        and its only numbers don't match. The verifier MUST flag it."""
        injected_claim = "WDC momentum surged 85% (0.85), confirming the sector rally."
        supporting = {
            "metrics": {
                "NVDA_momentum": {"value": 0.234, "n_observations": 89, "window_days": 90},
                "AMD_momentum": {"value": -0.052, "n_observations": 89, "window_days": 90},
            },
            "no_data": {"WDC": "no price data returned for ticker 'WDC'"},
        }
        out = verification.check_claim_against_data(injected_claim, supporting)
        res = out["result"]
        codes = [f["code"] for f in res["flags"]]
        assert res["supported"] is False
        assert "unsupported_number" in codes      # 0.85 matches nothing
        assert "certainty_overclaim" in codes     # "confirming" style certainty via 'surged'
        assert "cites_missing_data" in codes      # WDC is a known no-data key
        assert "0.85" in res["explanation"] or any("0.85" in f["message"] for f in res["flags"])


# --------------------------------------------------------------------------
# Input validation + MCP schema
# --------------------------------------------------------------------------
class TestValidationAndSchema:
    def test_empty_claim_error(self) -> None:
        assert verification.check_claim_against_data("   ", GOOD_METRICS)["ok"] is False

    def test_non_dict_supporting_data_error(self) -> None:
        assert verification.check_claim_against_data("claim", "not a dict")["ok"] is False

    def test_registers_two_tools(self) -> None:
        import asyncio

        tools = asyncio.run(verif_mcp.list_tools())
        assert {t.name for t in tools} == {"check_claim_against_data", "check_lookahead_bias"}

    def test_claim_tool_schema(self) -> None:
        import asyncio

        tools = asyncio.run(verif_mcp.list_tools())
        tool = next(t for t in tools if t.name == "check_claim_against_data")
        assert set(tool.input_schema["properties"]) == {"claim_text", "supporting_data"}
        assert tool.input_schema["properties"]["supporting_data"]["type"] == "object"
