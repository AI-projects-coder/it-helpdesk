import pytest
import yaml
from pydantic import BaseModel

from helpdesk.registry import AIRegistry, ToolRegistry, UnknownAgentError, UnknownToolError
from helpdesk.registry.tool_registry import ToolSpec

EXPECTED_TOOLS = {
    "search_kb", "get_account_status", "create_ticket",
    "reset_credentials", "update_security_settings", "issue_refund",
}


def test_loads_all_tools(tools):
    assert {t.name for t in tools.all()} == EXPECTED_TOOLS


def test_tool_lookup_by_name(tools):
    spec = tools.get("reset_credentials")
    assert spec.approval_required is True
    assert spec.risk == "high"
    assert callable(spec.handler)
    assert issubclass(spec.input_model, BaseModel)


def test_unknown_tool_raises(tools):
    with pytest.raises(UnknownToolError):
        tools.get("delete_everything")


def test_every_high_risk_tool_requires_approval(tools):
    for t in tools.all():
        if t.risk == "high":
            assert t.approval_required, t.name


def test_describe_exposes_json_schema(tools):
    d = tools.get("search_kb").describe()
    assert d["parameters"]["properties"]["query"]["type"] == "string"
    assert d["version"] == "1.0.0"


def test_duplicate_registration_rejected(tools):
    reg = ToolRegistry()
    spec = tools.get("search_kb")
    reg.register(spec)
    with pytest.raises(ValueError):
        reg.register(spec)


def test_bad_schema_name_rejected(tmp_path):
    bad = {"tools": [{
        "name": "x", "version": "1", "description": "d", "risk": "low", "approval_required": False,
        "handler": "helpdesk.tools.actions:create_ticket",
        "input_schema": "NoSuchModel", "output_schema": "ActionOutput",
    }]}
    path = tmp_path / "tools.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="NoSuchModel"):
        ToolRegistry.from_yaml(path)


def test_tool_spec_rejects_invalid_risk(tools):
    base = tools.get("search_kb").model_dump()
    base["risk"] = "extreme"
    with pytest.raises(ValueError):
        ToolSpec(**base)


# --- AI registry ---

def test_agents_loaded(agents):
    names = {a.name for a in agents.all()}
    assert {"planner", "knowledge_search", "access_check", "reasoning", "reflection", "action_ticket"} <= names


def test_find_by_capability(agents):
    assert [a.name for a in agents.find_by_capability("semantic_search")] == ["knowledge_search"]


def test_unknown_agent_raises(agents):
    with pytest.raises(UnknownAgentError):
        agents.get("ghost")


def test_agent_allowed_tools_exist_in_tool_registry(agents, tools):
    """Config consistency: agents.yaml must only reference registered tools."""
    for a in agents.all():
        for t in a.allowed_tools:
            assert t in tools, f"{a.name} references unknown tool {t}"


def test_can_use(agents):
    assert agents.can_use("knowledge_search", "search_kb")
    assert not agents.can_use("knowledge_search", "reset_credentials")
    assert not agents.can_use("ghost", "search_kb")


def test_ai_registry_from_custom_yaml(tmp_path):
    path = tmp_path / "agents.yaml"
    path.write_text(yaml.safe_dump({"agents": [{
        "name": "a", "version": "1", "description": "d", "capabilities": ["c"], "allowed_tools": [],
    }]}))
    assert AIRegistry.from_yaml(path).get("a").capabilities == ["c"]
