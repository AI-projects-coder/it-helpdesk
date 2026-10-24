"""Evaluates config/policy.yaml before every tool call: allow, deny, or require approval."""
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

from helpdesk.config import CONFIG_DIR
from helpdesk.registry import AIRegistry, ToolRegistry

Effect = Literal["allow", "require_approval", "deny"]


class PolicyDecision(BaseModel):
    effect: Effect
    rule_id: str
    reason: str

    @property
    def allowed(self) -> bool:
        return self.effect == "allow"


class PolicyEngine:
    def __init__(self, config: dict, tools: ToolRegistry, agents: AIRegistry) -> None:
        self.version = config.get("version", "0")
        self.default_effect: Effect = config.get("default_effect", "deny")
        self.role_permissions: dict[str, list[str]] = config.get("role_permissions", {})
        self.rules: list[dict] = config.get("rules", [])
        self.tools = tools
        self.agents = agents

    @classmethod
    def from_yaml(
        cls, tools: ToolRegistry, agents: AIRegistry, path: Path = CONFIG_DIR / "policy.yaml"
    ) -> "PolicyEngine":
        return cls(yaml.safe_load(path.read_text()), tools, agents)

    def evaluate(
        self, tool: str, agent: str, role: str, args: dict[str, Any], facts: dict[str, Any] | None = None
    ) -> PolicyDecision:
        if tool not in self.tools:
            return PolicyDecision(effect="deny", rule_id="unknown-tool", reason=f"tool {tool!r} is not registered")
        if not self.agents.can_use(agent, tool):
            return PolicyDecision(
                effect="deny", rule_id="least-privilege", reason=f"agent {agent!r} may not call {tool!r}"
            )
        permitted = self.role_permissions.get(role, [])
        if "*" not in permitted and tool not in permitted:
            return PolicyDecision(
                effect="deny", rule_id="role-permission", reason=f"role {role!r} may not call {tool!r}"
            )

        spec = self.tools.get(tool)
        facts = {
            "account_found": None,  # unknown unless the session looked the account up
            **(facts or {}),
            "tool": tool, "risk": spec.risk, "role": role, "approval_required": spec.approval_required,
        }
        for rule in self.rules:
            if _matches(rule.get("when", {}), facts, args):
                return PolicyDecision(
                    effect=rule["effect"], rule_id=rule["id"], reason=rule.get("description", rule["id"])
                )
        return PolicyDecision(effect=self.default_effect, rule_id="default", reason="no rule matched")


def _matches(when: dict, facts: dict, args: dict) -> bool:
    for key, expected in when.items():
        if key == "args_gt":
            for field, threshold in expected.items():
                value = args.get(field)
                if not isinstance(value, (int, float)) or value <= threshold:
                    return False
            continue
        options = expected if isinstance(expected, list) else [expected]
        if facts.get(key) not in options:
            return False
    return True
