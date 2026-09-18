"""Stage 6: verifier agent — check the writer's claims BEFORE publishing.

Flow:
1. Every claim from the memo draft goes to the verification server's
   check_claim_against_data with the quant metrics as supporting data
   (plus the run's no_data map so missing-data references get flagged).
2. Analysis steps go through check_lookahead_bias with the run as-of date.
3. Failed claims + flags go back to the writer for ONE revision (bounded
   retry, per the project spec). The revised claims are re-verified.
4. Whatever still fails is returned as `unresolved_flags` and rendered
   visibly in the final memo — never silently dropped.

In offline mode the deterministic template's claims are verified the same way
(verification is deterministic itself, so this works identically with no key).
"""

from __future__ import annotations

from datetime import datetime, timezone

from agents.base import ask_structured
from agents.schemas import GatheredData, MemoDraft, QuantOutput, TaskPlan, VerificationReport
from core.mcp_client import MCPClientPool
from core.trace import TraceLogger

_FATAL_CODES = {"unsupported_number", "direction_mismatch", "cites_missing_data"}


class VerifierAgent:
    name = "verifier_agent"

    def __init__(self, pool: MCPClientPool, trace: TraceLogger | None = None, max_retries: int = 1) -> None:
        self.pool = pool
        self.trace = trace
        self.max_retries = max_retries

    async def verify_and_revise(
        self,
        plan: TaskPlan,
        gathered: GatheredData,
        quant: QuantOutput,
        draft: MemoDraft,
        revise_fn=None,
    ) -> tuple[MemoDraft, VerificationReport]:
        """Verify draft claims; if unsupported, call revise_fn once; re-verify.

        revise_fn: async callable (draft, failed_claims, flags) -> MemoDraft.
        Returns the (possibly revised) draft and the final report.
        """
        report = await self._verify(plan, gathered, quant, draft)
        revisions: list[dict] = []
        attempts = 0
        while not report.passed and attempts < self.max_retries and revise_fn is not None:
            attempts += 1
            revisions.append({
                "attempt": attempts,
                "trigger": "failed_verification",
                "flagged": [
                    {"claim": f.get("claim"), "code": f.get("code"), "message": f.get("message")}
                    for f in report.unresolved_flags
                ],
            })
            if self.trace:
                self.trace.agent_step(
                    self.name, "revision_requested",
                    attempt=attempts,
                    n_failed=len(report.unresolved_flags),
                )
            draft = await revise_fn(draft, report.unresolved_flags)
            report = await self._verify(plan, gathered, quant, draft)
        report.revisions = revisions
        if self.trace:
            self.trace.agent_step(
                self.name, "verification_complete",
                passed=report.passed,
                revisions=attempts,
                n_unresolved=len(report.unresolved_flags),
            )
        return draft, report

    async def _verify(self, plan: TaskPlan, gathered: GatheredData, quant: QuantOutput, draft: MemoDraft) -> VerificationReport:
        as_of = datetime.now(timezone.utc).date().isoformat()
        supporting = {
            "as_of": as_of,
            "metrics": quant.metrics,
            "no_data": gathered.no_data,
        }

        claim_reports: list[dict] = []
        unresolved: list[dict] = []
        claims = draft.claims or _extract_claim_sentences(draft.body_markdown)
        for claim in claims:
            res = await self.pool.call_tool(
                "verification", "check_claim_against_data",
                {"claim_text": claim, "supporting_data": supporting},
                agent=self.name,
            )
            entry = {"claim": claim, "verdict": res.get("result", res)}
            claim_reports.append(entry)
            if not res.get("ok") or not res.get("result", {}).get("supported", False):
                for f in res.get("result", {}).get("flags", []) if res.get("ok") else [{"code": "verifier_error", "message": res.get("error", "unknown")}]:
                    if f.get("code") in _FATAL_CODES or not res.get("ok"):
                        unresolved.append({"claim": claim, **f})
                        break

        steps = _steps_from_trace(plan, quant, as_of)
        lookahead_res = await self.pool.call_tool(
            "verification", "check_lookahead_bias",
            {"analysis_steps": steps},
            agent=self.name,
        )
        lookahead = lookahead_res.get("result", lookahead_res)
        for f in lookahead.get("flags", []):
            unresolved.append({"claim": "(pipeline step)", **f})

        return VerificationReport(
            claim_reports=claim_reports,
            lookahead=lookahead,
            unresolved_flags=unresolved,
            passed=len(unresolved) == 0,
        )


def _extract_claim_sentences(body_markdown: str) -> list[str]:
    """Fallback claim extraction: sentences from Key Findings bullets."""
    claims: list[str] = []
    in_findings = False
    for line in body_markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            in_findings = "key findings" in stripped.lower()
            continue
        if in_findings and stripped.startswith("- "):
            claims.append(stripped[2:])
    return claims


def _steps_from_trace(plan: TaskPlan, quant: QuantOutput, as_of: str) -> list[dict]:
    steps = [
        {"step": f"planned analysis: {plan.request_summary}", "as_of": as_of},
        {"step": f"gathered price history and fundamentals for {', '.join(plan.tickers) or 'no tickers'}", "as_of": as_of},
    ]
    for name, m in quant.metrics.items():
        steps.append({
            "step": f"computed {name} over a {m.get('window_days', '?')}-day window ending on or before the as-of date",
            "as_of": as_of,
        })
    return steps
