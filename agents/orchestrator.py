"""Orchestrator: sequential-with-branches pipeline over the four MCP servers.

Pipeline (v1 — deliberately not a graph framework):
  1. PlannerAgent        decompose request -> TaskPlan (pydantic-validated)
  2. DataAgent  ┐
     NewsAgent  ┘        run CONCURRENTLY via asyncio.gather (parallel branches)
  3. QuantAgent          signals over gathered prices
  4. WriterAgent         memo draft with numeric citations
  5. VerifierAgent       every claim through the verification server BEFORE
                         publishing; one bounded revision; unresolved flags
                         rendered visibly in the final memo (never dropped)

Every tool call is traced to data/traces/<run_id>.jsonl (agent, server, tool,
args, result, error, timestamp, duration) — the explainability artifact.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from agents.base import mode_label
from agents.planner import PlannerAgent
from agents.quant import QuantAgent
from agents.schemas import (
    GatheredData,
    MemoDraft,
    QuantOutput,
    RunResult,
    TaskPlan,
    VerificationReport,
)
from agents.verifier import VerifierAgent
from agents.workers import DataAgent, NewsAgent
from agents.writer import WriterAgent
from core.config import OUTPUT_DIR
from core.mcp_client import MCPClientPool
from core.trace import TraceLogger, new_run_id


class ResearchOrchestrator:
    def __init__(self, transport: str = "stdio", trace: TraceLogger | None = None, output_dir: Path | None = None) -> None:
        self.run_id = new_run_id()
        self.trace = trace or TraceLogger(self.run_id)
        # Injectable so tests can keep artifacts out of the real output/ dir.
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.pool = MCPClientPool(transport=transport, trace=self.trace)  # type: ignore[arg-type]
        self.planner = PlannerAgent(trace=self.trace)
        self.data_agent = DataAgent(self.pool, trace=self.trace)
        self.news_agent = NewsAgent(self.pool, trace=self.trace)
        self.quant_agent = QuantAgent(self.pool, trace=self.trace)
        self.writer = WriterAgent(trace=self.trace)
        self.verifier = VerifierAgent(
            self.pool, trace=self.trace,
            max_retries=_max_verification_retries(),
        )

    async def run(self, request: str) -> RunResult:
        await self.pool.start()
        try:
            return await self._run_inner(request)
        finally:
            await self.pool.stop()

    async def _run_inner(self, request: str) -> RunResult:
        self.trace.log("run_started", request=request, transport=self.pool.transport_kind)

        # 1. Plan
        plan: TaskPlan = await self.planner.plan(request)

        # 2+3. Gather market data and news CONCURRENTLY (parallel branches).
        data_task = self.data_agent.gather(plan)
        news_task = self.news_agent.gather(plan)
        gathered_raw, news_out = await asyncio.gather(data_task, news_task, return_exceptions=True)

        gathered = await self._resolve_gathered(gathered_raw, news_out, plan)

        # 4. Quant signals
        quant: QuantOutput = await self.quant_agent.compute(plan, gathered)

        # 5. Writer draft
        draft: MemoDraft = await self.writer.write(plan, gathered, quant)

        # 6. Verify + one bounded revision, then publish with visible flags.
        draft, report = await self.verifier.verify_and_revise(
            plan, gathered, quant, draft, revise_fn=self._make_revise_fn(plan, gathered, quant)
        )

        memo_markdown = self._render_final_memo(draft, report, plan)
        output_path = self._save_memo(memo_markdown)

        self.trace.log(
            "run_completed",
            output_path=str(output_path),
            verification_passed=report.passed,
            n_unresolved=len(report.unresolved_flags),
        )
        return RunResult(
            memo_markdown=memo_markdown,
            plan=plan,
            verification=report,
            output_path=str(output_path),
            model_mode=mode_label(),
        )

    # ------------------------------------------------------------------
    def trace_path_events(self) -> list[dict]:
        """Read back this run's full trace (for tests and post-run analysis)."""
        from core.trace import read_trace

        return read_trace(self.trace.path)

    async def _resolve_gathered(self, gathered_raw, news_out, plan: TaskPlan) -> GatheredData:
        """Fold concurrent branch results (or their exceptions) into GatheredData."""
        if isinstance(gathered_raw, Exception):
            gathered = GatheredData(no_data={"(data agent)": f"concurrent data gathering failed: {gathered_raw}"})
        else:
            gathered = gathered_raw
        if isinstance(news_out, Exception):
            gathered.news = {"headlines": {"ok": False, "error": f"concurrent news gathering failed: {news_out}"}}
        else:
            gathered.news = news_out
        if self.trace:
            self.trace.agent_step(
                "orchestrator", "concurrent_gather_complete",
                data_ok=not gathered.no_data,
                news_ok=bool((gathered.news or {}).get("headlines", {}).get("ok")),
            )
        return gathered

    def _make_revise_fn(self, plan: TaskPlan, gathered: GatheredData, quant: QuantOutput):
        async def revise(draft: MemoDraft, unresolved_flags: list[dict]) -> MemoDraft:
            self.trace.agent_step(
                self.writer.name, "revising",
                n_flags=len(unresolved_flags),
                flags=[f.get("code") for f in unresolved_flags],
            )
            if _claude_available():
                try:
                    return self._revise_with_claude(draft, unresolved_flags)
                except Exception as exc:
                    self.trace.agent_step(self.writer.name, "revision_fallback", reason=str(exc))
            return _deterministic_revise(draft, unresolved_flags)

        return revise

    def _revise_with_claude(self, draft: MemoDraft, unresolved_flags: list[dict]) -> MemoDraft:
        from agents.base import ask_structured

        system = (
            "You revise research memos to satisfy a deterministic verifier. "
            "Fix ONLY the flagged claims: ground every number in the original "
            "metrics, hedge appropriately, or remove the numeric claim and say "
            "the data was insufficient. Keep everything else unchanged. No "
            "recommendations, no predictions."
        )
        prompt = (
            f"Memo draft:\n{draft.body_markdown}\n\n"
            f"Claims: {draft.claims}\n\n"
            f"Verification flags to fix: {unresolved_flags}\n\n"
            "Return JSON: {title, executive_summary, body_markdown, claims}."
        )
        return ask_structured(prompt, MemoDraft, system=system)

    def _render_final_memo(self, draft: MemoDraft, report: VerificationReport, plan: TaskPlan) -> str:
        parts = [draft.body_markdown.rstrip()]
        parts.append("\n## Verification report\n")
        n_claims = len(report.claim_reports)
        n_ok = sum(1 for c in report.claim_reports if c.get("verdict", {}).get("supported"))
        parts.append(
            f"- {n_ok}/{n_claims} claim(s) verified against the supporting tool data."
        )
        if report.unresolved_flags:
            parts.append("\n### Unresolved flags (surfaced, not dropped)\n")
            for f in report.unresolved_flags:
                parts.append(f"- **{f.get('code')}** — {f.get('message')}  \n  > _claim: {f.get('claim')}_")
        else:
            parts.append("- No unresolved flags: all claims passed deterministic verification.")
        parts.append(
            f"\n*Run ID: `{self.run_id}` · model mode: `{mode_label()}` · "
            f"generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} · "
            "For educational and research purposes only. Not investment advice.*"
        )
        return "\n".join(parts) + "\n"

    def _save_memo(self, memo_markdown: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = Path(self.output_dir) / f"memo_{self.run_id}.md"
        path.write_text(memo_markdown, encoding="utf-8")
        return path


def _deterministic_revise(draft: MemoDraft, unresolved_flags: list[dict]) -> MemoDraft:
    """Offline revision: restate flagged claims without the unverifiable numbers.

    The claim is replaced with an explicitly-unverified restatement; anything
    still unfixable stays out of the claims list and shows up in the final
    memo's 'Unresolved flags' section — nothing is silently dropped.
    """
    flagged_claims = {f.get("claim") for f in unresolved_flags if f.get("claim")}
    revised_claims: list[str] = []
    for claim in draft.claims:
        if claim in flagged_claims:
            revised_claims.append(
                "[Unverified — flagged by the verifier; restated without numbers] "
                "A requested metric could not be confirmed against the supporting "
                "data, so no numeric claim is made here."
            )
        else:
            revised_claims.append(claim)
    body = draft.body_markdown
    return MemoDraft(
        title=draft.title,
        executive_summary=draft.executive_summary,
        body_markdown=body,
        claims=revised_claims,
    )


def _claude_available() -> bool:
    from core import model as model_client

    return model_client.available()


def _max_verification_retries() -> int:
    from core.config import get_settings

    return get_settings().max_verification_retries
