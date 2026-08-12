"""A2A agents over the truthclf MCP servers.

Four deployable units: an orchestrator and three specialists (zero-shot
predictor, fine-tuned predictor, explainer). Agents talk to each other over A2A
and reach every capability through MCP tool calls.

No agent imports truthclf. The MCP servers are the only processes that do.
"""

__all__ = ["cards", "common", "mcp_client", "pooling"]
