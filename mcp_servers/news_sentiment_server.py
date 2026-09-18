"""MCP server: news + sentiment tools.

Tools
-----
- get_recent_headlines(query, days_back=7, max_items=12)
- summarize_sentiment(headlines, context=None)

Design notes
------------
- Headlines come verbatim from a keyless-first source chain (NewsAPI if a key
  is configured, then GDELT, then Google News RSS). Zero results -> explicit
  no-data envelope; headlines are NEVER fabricated.
- summarize_sentiment embeds a Claude call inside the tool ("model-as-a-tool"
  pattern — see mcp_servers/sentiment.py) with a transparent offline fallback.
- Uses the MCP Python SDK 2.x API. Run standalone:
    uv run python -m mcp_servers.news_sentiment_server
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from mcp_servers import news_client, sentiment

mcp = MCPServer("news-sentiment")


@mcp.tool()
def get_recent_headlines(query: str, days_back: int = 7, max_items: int = 12) -> dict:
    """Recent news headlines (title, source, published, url) for a query.

    Sources are tried in order: NewsAPI (if NEWS_API_KEY set), GDELT, Google
    News RSS. If nothing is found the tool returns an explicit no-data
    envelope — it never fabricates headlines.
    """
    return news_client.get_recent_headlines(query, days_back=days_back, max_items=max_items)


@mcp.tool()
def summarize_sentiment(headlines: list[str], context: str | None = None) -> dict:
    """Aggregate sentiment over headline titles with reasoning.

    Uses Claude when ANTHROPIC_API_KEY is set (model-as-a-tool pattern);
    otherwise a clearly-labeled lexicon heuristic. Empty input is refused.
    """
    return sentiment.summarize_sentiment(headlines, context)


if __name__ == "__main__":
    mcp.run()
