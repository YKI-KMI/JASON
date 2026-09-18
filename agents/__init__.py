"""Agent implementations for the research pipeline.

Each agent is a thin, purpose-specific wrapper around one structured model
call (or a deterministic procedure) that uses the MCP client pool for tools.
Model calls share one seam (core.model) with a clearly-labeled offline mode.
"""
