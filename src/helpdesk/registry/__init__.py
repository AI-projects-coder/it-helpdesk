"""Tool registry and AI (agent) registry."""
from helpdesk.registry.ai_registry import AgentSpec, AIRegistry, UnknownAgentError
from helpdesk.registry.tool_registry import ToolRegistry, ToolSpec, UnknownToolError

__all__ = ["AgentSpec", "AIRegistry", "UnknownAgentError", "ToolRegistry", "ToolSpec", "UnknownToolError"]
