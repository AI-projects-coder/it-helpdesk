import time

import pytest

from helpdesk.policy import PolicyEngine
from helpdesk.registry import AIRegistry, ToolRegistry
from helpdesk.registry.ai_registry import AgentSpec
from helpdesk.registry.tool_registry import ToolSpec
from helpdesk.tools.executor import MAX_ATTEMPTS, CallContext, ToolExecutor, TransientToolError
from helpdesk.tools.schemas import CreateTicketInput, CreateTicketOutput

TICKET_ARGS = {"customer_id": "CUST_05883", "summary": "Login fails", "issue_type": "account_access"}


# --- against the real config ---

def test_ticket_needs_approval_then_runs(executor, action_ctx):
    pending = executor.execute("create_ticket", TICKET_ARGS, action_ctx)
    assert pending.status == "needs_approval" and pending.decision.rule_id == "approval-required-tools"
    r = executor.execute("create_ticket", TICKET_ARGS, action_ctx, approved=True)
    assert r.ok
    assert r.output["ticket_id"].startswith("TKT-")
    assert r.attempts == 1


def test_escalation_ticket_runs_without_approval(executor):
    ctx = CallContext(session_id="t", agent="action_ticket", facts={"escalation": True})
    r = executor.execute("create_ticket", TICKET_ARGS, ctx)
    assert r.ok and r.decision.rule_id == "escalation-ticket"


def test_approval_required_blocks_without_approval(executor, action_ctx):
    r = executor.execute("reset_credentials", {"customer_id": "CUST_05883", "reason": "locked out"}, action_ctx)
    assert r.status == "needs_approval"
    assert r.output is None
    assert r.attempts == 0


def test_approved_action_runs(executor, action_ctx):
    r = executor.execute(
        "reset_credentials", {"customer_id": "CUST_05883", "reason": "locked out"}, action_ctx, approved=True
    )
    assert r.ok
    assert r.output["reference_id"].startswith("RST-")


def test_approval_cannot_override_deny(executor, action_ctx):
    args = {"customer_id": "CUST_05883", "amount": 5000, "reason": "big refund"}
    r = executor.execute("issue_refund", args, action_ctx, approved=True)
    assert r.status == "denied"
    assert r.decision.rule_id == "refund-limit"


def test_wrong_agent_denied(executor):
    ctx = CallContext(session_id="t", agent="knowledge_search")
    r = executor.execute("create_ticket", TICKET_ARGS, ctx)
    assert r.status == "denied"


def test_invalid_input_rejected(executor, action_ctx):
    r = executor.execute("create_ticket", {**TICKET_ARGS, "customer_id": "not-an-id"}, action_ctx, approved=True)
    assert r.status == "invalid_input"
    assert "customer_id" in r.error


def test_extra_fields_rejected(executor, action_ctx):
    r = executor.execute("create_ticket", {**TICKET_ARGS, "sudo": True}, action_ctx, approved=True)
    assert r.status == "invalid_input"


def test_account_lookup(executor):
    ctx = CallContext(session_id="t", agent="access_check")
    found = executor.execute("get_account_status", {"customer_id": "CUST_05883"}, ctx)
    missing = executor.execute("get_account_status", {"customer_id": "CUST_99999"}, ctx)
    assert found.ok and found.output["found"] and found.output["sla_plan"] == "standard"
    assert missing.ok and not missing.output["found"]


def test_locked_account_from_scenario(executor):
    import json

    from helpdesk.config import get_settings

    golden = json.loads(get_settings().golden_path.read_text())
    locked = next(t for t in golden if t["scenario_key"] == "access_locked")
    ctx = CallContext(session_id="t", agent="access_check")
    r = executor.execute("get_account_status", {"customer_id": locked["customer_id"]}, ctx)
    assert r.output["account_status"] == "locked"


def test_ticket_ids_are_deterministic(executor, action_ctx):
    a = executor.execute("create_ticket", TICKET_ARGS, action_ctx, approved=True)
    b = executor.execute("create_ticket", TICKET_ARGS, action_ctx, approved=True)
    c = executor.execute("create_ticket", {**TICKET_ARGS, "summary": "Something else"}, action_ctx, approved=True)
    assert a.output["ticket_id"] == b.output["ticket_id"] != c.output["ticket_id"]


# --- retry / timeout with synthetic tools ---

def _make_executor(handler, timeout_s: float = 2) -> ToolExecutor:
    tools = ToolRegistry()
    tools.register(ToolSpec(
        name="flaky", version="1", description="test", risk="low", approval_required=False,
        timeout_s=timeout_s, input_model=CreateTicketInput, output_model=CreateTicketOutput, handler=handler,
    ))
    tester = AgentSpec(name="tester", version="1", description="", capabilities=[], allowed_tools=["flaky"])
    agents = AIRegistry([tester])
    policy = PolicyEngine(
        {"role_permissions": {"user": ["*"]}, "rules": [{"id": "allow", "when": {}, "effect": "allow"}]}, tools, agents
    )
    return ToolExecutor(tools, policy, retry_wait_s=0)


CTX = CallContext(session_id="t", agent="tester")


def test_transient_failure_retried_then_succeeds():
    calls = []

    def handler(inp):
        calls.append(1)
        if len(calls) < 3:
            raise TransientToolError("blip")
        return CreateTicketOutput(ticket_id="TKT-1")

    r = _make_executor(handler).execute("flaky", TICKET_ARGS, CTX)
    assert r.ok
    assert r.attempts == 3


def test_transient_failure_gives_up_after_max_attempts():
    calls = []

    def handler(inp):
        calls.append(1)
        raise TransientToolError("down")

    r = _make_executor(handler).execute("flaky", TICKET_ARGS, CTX)
    assert r.status == "error"
    assert len(calls) == MAX_ATTEMPTS
    assert "after 3 attempts" in r.error


def test_permanent_failure_not_retried():
    calls = []

    def handler(inp):
        calls.append(1)
        raise ValueError("bad state")

    r = _make_executor(handler).execute("flaky", TICKET_ARGS, CTX)
    assert r.status == "error"
    assert len(calls) == 1


def test_timeout_is_retried_and_reported():
    def handler(inp):
        time.sleep(0.3)
        return CreateTicketOutput(ticket_id="TKT-1")

    r = _make_executor(handler, timeout_s=0.05).execute("flaky", TICKET_ARGS, CTX)
    assert r.status == "error"
    assert "ToolTimeoutError" in r.error
    assert r.attempts == MAX_ATTEMPTS


def test_bad_output_caught():
    r = _make_executor(lambda inp: {"ticket_id": 123, "status": "closed"}).execute("flaky", TICKET_ARGS, CTX)
    assert r.status == "error"
    assert "bad tool output" in r.error


@pytest.mark.integration
def test_search_kb_live(executor):
    ctx = CallContext(session_id="t", agent="knowledge_search")
    r = executor.execute("search_kb", {"query": "I was charged twice this month"}, ctx)
    assert r.ok
    assert r.output["hits"][0]["issue_type"] == "billing_problem"
