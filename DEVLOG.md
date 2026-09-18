# DEVLOG — building this with an agentic coding assistant (Claude Code)

Honest notes on how this repo was actually built, for interview discussion.
Everything below refers to real events from the build session and its git
history (commits are phase-scoped: read `git log --oneline` top-down).

## How the build was run

The whole system was specified up front in a single structured prompt
(architecture, deliverables, definition-of-done, self-verification checklist)
with instructions to work phase by phase, run each phase's tests before
committing, and keep going without asking for approval. The assistant made the
reasonable-call-without-stopping decisions below and documented them here.

## Judgment calls the agent made (and why)

1. **`uv` instead of Poetry.** The machine had `uv` on PATH but no Poetry.
   uv is faster, is lockfile-first via `pyproject.toml` + `uv.lock`, and
   exports a plain `requirements.txt` for non-uv users. Documented tradeoff,
   not a limitation.
2. **MCP Python SDK 2.x.** The first `uv sync` pulled `mcp` 2.x, where
   `FastMCP` was renamed `MCPServer` (`mcp.server.mcpserver`) and tool schemas
   expose `input_schema` (snake_case) instead of `inputSchema`. The agent
   migrated to 2.x rather than pinning `mcp<2`, since the point of the project
   is current-API server authoring. Migration friction (below) was real.
3. **Keyless-first news chain: NewsAPI (optional) -> GDELT -> Google News RSS.**
   The spec allowed "a free news API or web search". GDELT DOC 2.0 and Google
   News RSS need no key, so the demo works out of the box; NewsAPI is used
   first only if `NEWS_API_KEY` is set. Failed/empty sources are reported in
   the tool result (`source_errors`) instead of being swallowed.
4. **Curated static sector map instead of scraping index memberships.**
   Deterministic, testable, and honest: the memo labels it a hand-picked
   representative sample. Scraping Wikipedia S&P 500 tables is brittle and
   adds a fabrication-adjacent failure mode for zero research value in v1.
5. **Offline mode as a first-class feature.** `ANTHROPIC_API_KEY` was not set
   in the build environment. Rather than halting (or worse, stubbing silent
   fake "model" output), the design makes model availability explicit:
   `core/model.available()`, `ModelUnavailableError`, planner/writer/sentiment
   fallbacks that are *labeled in the artifacts* (`mode` fields, memo mode
   notes, trace events). Verifiability does not depend on the model: the
   verification server is deterministic by design.

## Where the tests caught real bugs (the honest list)

These are all in the git history; each was found by running the phase's tests,
not by hoping:

- **Hand-computed vol variance was wrong in the test, not the code.** The
  alternate ±1% return series has mean 0.002, so deviations are ±0.008/0.012
  and the ddof=1 variance is 0.00012 (std ≈ 0.010954), not 0.01. The suite
  forced a redo of the arithmetic on paper — exactly what "test the MATH"
  is supposed to do.
- **"Opposite prices" is not "negative return correlation".** A first
  correlation test used two price series that fall/rise over time; their
  *returns* were co-trending, so corr ≈ +0.9997 was the correct output and the
  test was wrong. Also: mirroring log returns needs `ln(1.01) ≠ -ln(0.99)`
  care; the fixed test builds B by exactly negating A's log returns.
- **Min-overlap defaults.** Tiny synthetic series with `min_overlap=10`
  correctly produced `None` (refuse to guess) — the test had to pass explicit
  small thresholds.
- **MCP 2.x migration mishaps.** The assistant's first mechanical rename
  produced `@mcp.tool` + indented `def` (syntax error), then
  `@mcp.tool()def` (eaten newline), and finally learned this SDK wants
  `@mcp.tool()` with parentheses (it *raises* on the bare form). Caught by
  immediate test/import runs after each edit.
- **Verification subtleties, caught one by one:**
  - "90-day" was being extracted as a data claim (number 90) — windows are
    now stripped from numeric extraction and checked by a dedicated
    window-mismatch rule.
  - Certainty/thin-sample checks were initially nested inside a directional
    word loop, so they never ran for claims without directional words.
  - Direction words needed *clause* scoping ("AMD momentum is positive at
    -0.052" must flag; "NVDA is positive. AMD declined." must not mis-flag).
  - Ticker-only affinity let a claim's "0.4" silently match a 0.412
    volatility metric; the matcher now requires the metric's *type token*
    ("momentum", "volatility", …) to appear in the claim and flags
    ambiguous equal-affinity matches.
- **The orchestrator never started its MCP servers.** The first e2e run
  silently "worked" for the no-data test only because everything failed
  gracefully. The poisoned-writer test exposed it: `pool.start()` was never
  called. Now `run()` owns server lifecycle with try/finally.
- **Sector plans produced zero tickers.** The planner mapped
  "semiconductor sector momentum" to `sector="semiconductors"` but nothing
  expanded the sector into tickers, so the memo had no data table. The e2e
  test asserted metrics existed and failed; the data agent now resolves
  sectors via `get_sector_tickers` (capped to 6 tickers, methodology noted
  in the memo).
- **Schema drift:** `TaskPlan.notes` typed `str | None` but assigned a list —
  pydantic caught it the moment the planner fallback ran.
- **Singular/plural sector matching:** "semiconductor" (singular) missed the
  plural-keyed map; aliases + singular fallback matching added after the e2e
  failure.

## Deliberately injected bad claim (DoD requirement)

`tests/test_orchestrator.py::test_verifier_catches_injected_bad_claim`
poisons the writer agent to emit *"WDC momentum surged 85% (0.85),
confirming the sector rally."* while the supporting data has no WDC metric
and WDC is a known no-data key. Asserted outcomes:

- the verifier records a revision whose flagged codes include
  `unsupported_number` and `cites_missing_data`;
- the final memo contains a "Verification report" section (flags are
  surfaced, never dropped);
- the run's trace contains the verifier's `revision_requested` event.

The same check exists at the unit level in
`tests/test_verification.py::TestInjectedBadClaim` against the verification
server directly.

## What the agent did well / where a human had to stay in the loop

**Well:** phase-scoped commits; running tests between every edit; refusing to
let "no data" become fabricated data anywhere in the stack; making every
fallback visible in artifacts rather than silent; reading the installed SDK's
actual source instead of trusting training-memory APIs.

**Human-in-the-loop value:** the spec itself (scope discipline — "don't build
a graph framework" prevented over-engineering); judgment on which failures
were *test bugs* vs *code bugs* (the verifier work was genuinely iterative);
and the decision to treat offline mode as a feature to show rather than a
hole to hide. An unattended agent would plausibly have "fixed" the
±0.9997-correlation test by weakening the assertion instead of understanding
the math.

## Things known to be imperfect (next-time list)

- The Windows console emitted mojibake for non-ASCII in some test output
  (cosmetic; files are UTF-8).
- `feedparser` pulls in extra transitively; trimming deps is possible.
- The writer's Claude prompt asks for claims + citations, but claim
  extraction from free-form memos (`_extract_claim_sentences`) is heuristic;
  a structured claim-extraction pass would tighten it.
- Traces are JSONL, good for review; a proper trace-viewer page (stretch
  goal) would make the artifact demo-able in interviews.
- No CI configured yet; `pytest` locally is the gate (the final self-check
  runs the full suite green).
