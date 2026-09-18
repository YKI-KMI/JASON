#!/usr/bin/env python
"""CLI entrypoint: run a research query end-to-end and write a markdown memo.

Usage:
    python run_research.py "analyze semiconductor sector momentum"
    python run_research.py "compare momentum for NVDA and AMD over 60 days" --json
    python run_research.py "..." --transport inproc   # in-process servers (tests)

Environment: copy .env.example to .env and set ANTHROPIC_API_KEY for Claude
mode. Without a key the run completes in clearly-labeled offline mode.
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run_research",
        description="Multi-agent market research memo generator (educational/research use only).",
    )
    parser.add_argument("request", nargs="+", help="the research request, e.g. 'analyze semiconductor sector momentum'")
    parser.add_argument("--transport", choices=["stdio", "inproc"], default="stdio",
                        help="stdio: run the four MCP servers as subprocesses (default). inproc: in-process.")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON summary at the end")
    args = parser.parse_args()
    request = " ".join(args.request)

    # Imports after argparse so --help is fast and import errors are readable.
    from agents.base import mode_label
    from agents.orchestrator import ResearchOrchestrator

    print(f"[run_research] request: {request!r}")
    print(f"[run_research] model mode: {mode_label()}" + (" (set ANTHROPIC_API_KEY for Claude mode)" if mode_label() == "offline_heuristic" else ""))
    print(f"[run_research] transport: {args.transport} — spawning MCP servers...")

    orch = ResearchOrchestrator(transport=args.transport)
    try:
        result = asyncio.run(orch.run(request))
    except KeyboardInterrupt:
        print("\n[run_research] interrupted", file=sys.stderr)
        return 130

    print(f"[run_research] memo written to: {result.output_path}")
    print(f"[run_research] trace log:       {orch.trace.path}")
    print(f"[run_research] verification:    {'passed' if result.verification.passed else 'FLAGGED (see memo)'}"
          f" — {len(result.verification.claim_reports)} claim(s) checked, "
          f"{len(result.verification.unresolved_flags)} unresolved flag(s)")

    if args.json:
        import json

        print(json.dumps({
            "run_id": orch.run_id,
            "output_path": result.output_path,
            "trace_path": str(orch.trace.path),
            "model_mode": result.model_mode,
            "verification_passed": result.verification.passed,
            "n_claims": len(result.verification.claim_reports),
            "unresolved_flags": result.verification.unresolved_flags,
            "plan": result.plan.model_dump(),
        }, indent=2, default=str))

    print("\n--- Memo preview (first 40 lines) ---")
    for line in result.memo_markdown.splitlines()[:40]:
        print(line)
    print("\nDisclaimer: for educational and research purposes only. Not investment advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
