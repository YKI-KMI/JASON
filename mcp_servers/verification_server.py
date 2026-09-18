"""MCP server: verification tools (the differentiator).

Tools
-----
- check_claim_against_data(claim_text, supporting_data)
    Deterministic, explainable verdict on whether a claim is actually
    supported by the cited numbers: numeric grounding, directional
    consistency, hedging-aware tolerances, certainty-language overclaim flags.
- check_lookahead_bias(analysis_steps)
    Flags steps that could use data not available at the analysis as-of date.

Design notes
------------
- Deliberately DETERMINISTIC (no model calls): an LLM verifier would be just
  another opinion. These checks are unit-testable and explainable; the
  orchestrator's verifier agent adds a model-based review on top.
- This server exists to catch the writer agent overstating or misrepresenting
  what the data shows — see tests/test_verification.py for the documented
  behaviors, including the deliberately-injected unsupported-claim scenario.
- Uses the MCP Python SDK 2.x API. Run standalone:
    uv run python -m mcp_servers.verification_server
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from mcp_servers import verification

mcp = MCPServer("verification")


@mcp.tool()
def check_claim_against_data(claim_text: str, supporting_data: dict) -> dict:
    """Verify a research claim against the metrics it cites.

    supporting_data: {"metrics": {name: {value, n_observations, window_days}},
                      "no_data": {key: reason}, "as_of": "YYYY-MM-DD"}
    Returns {supported, flags[], explanation, checks[]}.
    """
    return verification.check_claim_against_data(claim_text, supporting_data)


@mcp.tool()
def check_lookahead_bias(analysis_steps: list[dict]) -> dict:
    """Flag analysis steps that could use data not available at the as-of
    date (future-dated data or forward-looking wording in data steps)."""
    return verification.check_lookahead_bias(analysis_steps)


if __name__ == "__main__":
    mcp.run()
