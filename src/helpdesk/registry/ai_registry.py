"""Loads config/agents.yaml; catalog of registered agents."""
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from helpdesk.config import CONFIG_DIR


class UnknownAgentError(KeyError):
    pass


class AgentSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    description: str
    capabilities: list[str]
    allowed_tools: list[str]


class AIRegistry:
    def __init__(self, agents: list[AgentSpec]) -> None:
        self._agents = {a.name: a for a in agents}

    @classmethod
    def from_yaml(cls, path: Path = CONFIG_DIR / "agents.yaml") -> "AIRegistry":
        return cls([AgentSpec(**a) for a in yaml.safe_load(path.read_text())["agents"]])

    def get(self, name: str) -> AgentSpec:
        try:
            return self._agents[name]
        except KeyError:
            raise UnknownAgentError(name) from None

    def all(self) -> list[AgentSpec]:
        return list(self._agents.values())

    def find_by_capability(self, capability: str) -> list[AgentSpec]:
        return [a for a in self._agents.values() if capability in a.capabilities]

    def can_use(self, agent: str, tool: str) -> bool:
        return agent in self._agents and tool in self._agents[agent].allowed_tools
