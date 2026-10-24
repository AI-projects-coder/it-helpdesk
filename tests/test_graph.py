"""End-to-end graph tests with a scripted LLM and a stubbed KB (no API calls)."""
import time

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from helpdesk import telemetry
from helpdesk.app import HelpdeskApp
from helpdesk.session_store import SessionStore
from helpdesk.tools import kb
from tests.fakes import CUSTOMER, GROUNDED, KB_HITS, UNGROUNDED, FakeLLM, draft, plan


@pytest.fixture(scope="session")
def spans_exporter():
    exporter = InMemorySpanExporter()
    telemetry.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


@pytest.fixture(autouse=True)
def stub_kb(monkeypatch):
    monkeypatch.setattr(kb, "search", lambda query, top_k=3: [dict(h) for h in KB_HITS[:top_k]])


@pytest.fixture
def make_app(tmp_path):
    def _make(responses: dict, **deps_kwargs):
        llm = FakeLLM(responses)
        app = HelpdeskApp(llm=llm, store=SessionStore(tmp_path), state_dir=tmp_path / "state", **deps_kwargs)
        return app, llm

    return _make


def flow(result) -> list[str]:
    return [e["agent"] for e in result.state["events"]]


def tools_called(result) -> list[tuple[str, str]]:
    return [(r["tool"], r["status"]) for r in result.state["tool_results"]]


# --- happy paths ---

def test_how_to_resolved_without_action(make_app):
    app, _ = make_app({"planner": plan(), "reasoning": draft(), "reflection": GROUNDED})
    r = app.start("How do I export my data?")
    assert r.status == "completed"
    assert flow(r) == ["intake", "planner", "knowledge_search", "reasoning", "reflection", "policy_gate", "respond"]
    assert r.state["outcome"] == "resolved"
    assert r.state["final_response"] == "Here is what to do next."


def test_account_check_skipped_without_customer_id(make_app):
    app, _ = make_app({"planner": plan("account_access", needs_account=True),
                       "reasoning": draft(cited=["KB_ACCOUNT_ACCESS"]), "reflection": GROUNDED})
    r = app.start("I can't log in")
    assert "access_check" not in flow(r)


def test_account_data_not_fetched_for_non_account_issues(make_app):
    app, _ = make_app({"planner": plan("how_to", needs_account=True), "reasoning": draft(), "reflection": GROUNDED})
    r = app.start("How do I export?", customer_id=CUSTOMER)
    assert "access_check" not in flow(r)
    assert r.state["plan"]["needs_account_check"] is False


def test_account_check_runs_with_customer_id(make_app):
    app, llm = make_app({"planner": plan("account_access", needs_account=True),
                         "reasoning": draft(cited=["KB_ACCOUNT_ACCESS"]), "reflection": GROUNDED})
    r = app.start("I can't log in", customer_id=CUSTOMER)
    assert "access_check" in flow(r)
    assert ("get_account_status", "ok") in tools_called(r)
    reasoning_prompt = next(u for n, u in llm.calls if n == "reasoning")
    assert "ACCOUNT STATUS" in reasoning_prompt and CUSTOMER in reasoning_prompt


def test_bug_creates_ticket_without_approval(make_app):
    app, _ = make_app({"planner": plan("bug"), "reasoning": draft("create_ticket", cited=["KB_BUG"]),
                       "reflection": GROUNDED})
    r = app.start("The app crashes", customer_id=CUSTOMER)
    assert r.status == "completed"
    assert r.state["outcome"] == "ticket_created"
    assert r.state["action_result"]["ticket_id"].startswith("TKT-")
    assert "human_approval" not in flow(r)


# --- HITL ---

def _reset_app(make_app):
    return make_app({"planner": plan("account_access", needs_account=True),
                     "reasoning": draft("reset_credentials", cited=["KB_ACCOUNT_ACCESS"]), "reflection": GROUNDED})


def test_hitl_pauses_before_high_risk_action(make_app):
    app, _ = _reset_app(make_app)
    r = app.start("I'm locked out", customer_id=CUSTOMER)
    assert r.status == "awaiting_approval"
    assert r.interrupt["tool"] == "reset_credentials"
    assert r.interrupt["args"]["customer_id"] == CUSTOMER
    assert r.interrupt["policy_rule"] == "approval-required-tools"
    assert "reset_credentials" not in [t for t, _ in tools_called(r)]
    assert app.store.load(r.session_id)["status"] == "awaiting_approval"


def test_hitl_approve_executes_action(make_app):
    app, _ = _reset_app(make_app)
    paused = app.start("I'm locked out", customer_id=CUSTOMER)
    r = app.resume(paused.session_id, approved=True, approver="alice")
    assert r.status == "completed"
    assert r.state["outcome"] == "action_completed"
    assert ("reset_credentials", "ok") in tools_called(r)
    assert r.state["approval"] == {"approved": True, "approver": "alice", "comment": ""}
    assert flow(r)[-3:] == ["human_approval", "action_ticket", "respond"]


def test_hitl_reject_escalates_without_running_action(make_app):
    app, _ = _reset_app(make_app)
    paused = app.start("I'm locked out", customer_id=CUSTOMER)
    r = app.resume(paused.session_id, approved=False, approver="bob", comment="verify identity first")
    assert r.state["outcome"] == "escalated"
    assert "reset_credentials" not in [t for t, _ in tools_called(r)]
    assert r.state["escalation"]["ticket_id"].startswith("TKT-")
    assert "verify identity first" in r.state["escalation"]["reason"]


def test_resume_requires_paused_session(make_app):
    app, _ = make_app({"planner": plan(), "reasoning": draft(), "reflection": GROUNDED})
    r = app.start("How do I export?")
    with pytest.raises(ValueError, match="not awaiting approval"):
        app.resume(r.session_id, approved=True)


# --- policy ---

def test_refund_over_limit_denied_and_escalated(make_app):
    app, _ = make_app({"planner": plan("billing_problem", needs_account=True),
                       "reasoning": draft("issue_refund", cited=["KB_HOW_TO"], refund_amount=900),
                       "reflection": GROUNDED})
    r = app.start("Refund 900 please", customer_id=CUSTOMER)
    assert r.status == "completed"  # denied outright, no human asked
    assert r.state["policy_decision"]["rule_id"] == "refund-limit"
    assert r.state["outcome"] == "escalated"
    assert "issue_refund" not in [t for t, _ in tools_called(r)]


def test_refund_without_amount_escalates_before_approval(make_app):
    app, _ = make_app({"planner": plan("billing_problem"),
                       "reasoning": draft("issue_refund", cited=["KB_HOW_TO"]), "reflection": GROUNDED})
    r = app.start("I was double charged", customer_id=CUSTOMER)
    assert r.status == "completed"
    assert "human_approval" not in flow(r)
    assert "amount" in r.state["escalation"]["reason"]


def test_action_on_unknown_account_denied_even_with_approval(make_app):
    app, _ = make_app({"planner": plan("account_access", needs_account=True),
                       "reasoning": draft("reset_credentials", cited=["KB_ACCOUNT_ACCESS"]), "reflection": GROUNDED})
    r = app.start("I'm locked out", customer_id="CUST_99999")
    assert r.status == "completed"  # denied at the gate; no human is asked to approve
    assert r.state["policy_decision"]["rule_id"] == "account-must-exist"
    assert "reset_credentials" not in [t for t, _ in tools_called(r)]


def test_viewer_role_cannot_run_actions(make_app):
    app, _ = make_app({"planner": plan("bug"), "reasoning": draft("create_ticket", cited=["KB_BUG"]),
                       "reflection": GROUNDED})
    r = app.start("The app crashes", customer_id=CUSTOMER, role="viewer")
    assert r.state["policy_decision"]["rule_id"] == "role-permission"
    assert r.state["outcome"] == "escalated"


# --- reflection ---

def test_reflection_retry_then_grounded(make_app):
    app, llm = make_app({"planner": plan(), "reasoning": draft(), "reflection": [UNGROUNDED, GROUNDED]})
    r = app.start("How do I export?")
    assert r.state["outcome"] == "resolved"
    assert llm.count("reasoning") == 2
    second_prompt = [u for n, u in llm.calls if n == "reasoning"][1]
    assert "remove the invented claim" in second_prompt


def test_ungrounded_twice_escalates(make_app):
    app, llm = make_app({"planner": plan(), "reasoning": draft(), "reflection": GROUNDED | {"grounded": False}})
    r = app.start("How do I export?")
    assert r.state["outcome"] == "escalated"
    assert llm.count("reasoning") == 2


def test_invented_citation_caught_without_llm(make_app):
    app, llm = make_app({"planner": plan(), "reasoning": [draft(cited=["KB_MADE_UP"]), draft()],
                         "reflection": GROUNDED})
    r = app.start("How do I export?")
    assert r.state["outcome"] == "resolved"
    assert llm.count("reflection") == 1  # first check was deterministic


# --- resilience ---

def test_llm_failure_escalates(make_app):
    app, _ = make_app({"planner": RuntimeError("provider down")})
    r = app.start("How do I export?", customer_id=CUSTOMER)
    assert r.state["outcome"] == "escalated"
    assert any("provider down" in e for e in r.state["errors"])
    assert r.state["escalation"]["ticket_id"]


def test_node_timeout_escalates(make_app):
    def slow(_):
        time.sleep(0.5)
        return draft()

    app, _ = make_app({"planner": plan(), "reasoning": slow, "reflection": GROUNDED}, node_timeout_s=0.1)
    r = app.start("How do I export?")
    assert r.state["outcome"] == "escalated"
    assert any("node timeout" in e for e in r.state["errors"])


# --- PII, sessions, telemetry ---

def test_pii_redacted_before_llm_and_storage(make_app):
    app, llm = make_app({"planner": plan(), "reasoning": draft(), "reflection": GROUNDED})
    r = app.start("How do I export? Email me at jane@corp.com")
    assert all("jane@corp.com" not in u for _, u in llm.calls)
    assert "jane@corp.com" not in app.store.path(r.session_id).read_text()
    assert r.state["events"][0]["decision"] == "redacted"


def test_session_capture(make_app):
    app, _ = make_app({"planner": plan("bug"), "reasoning": draft("create_ticket", cited=["KB_BUG"]),
                       "reflection": GROUNDED})
    r = app.start("The app crashes", customer_id=CUSTOMER)
    saved = app.store.load(r.session_id)
    assert saved["status"] == "completed"
    assert [t["tool"] for t in saved["tool_results"]] == ["search_kb", "create_ticket"]
    assert saved["messages"][0]["role"] == "user" and saved["messages"][-1]["role"] == "assistant"
    assert saved["trace_id"] == r.state["trace_id"]


def test_spans_cover_every_step_in_one_trace(make_app, spans_exporter):
    spans_exporter.clear()
    app, _ = _reset_app(make_app)
    paused = app.start("I'm locked out", customer_id=CUSTOMER)
    app.resume(paused.session_id, approved=True)

    spans = spans_exporter.get_finished_spans()
    names = {s.name for s in spans}
    assert {"helpdesk.session", "helpdesk.resume", "agent.planner", "llm.planner",
            "tool.search_kb", "agent.human_approval", "tool.reset_credentials"} <= names
    assert len({s.context.trace_id for s in spans}) == 1  # HITL resume continues the same trace

    agent_spans = [s for s in spans if s.name.startswith("agent.")]
    assert all(s.attributes["session_id"] == paused.session_id for s in agent_spans)
    assert all("latency_ms" in s.attributes and "decision" in s.attributes for s in agent_spans)
    tool_span = next(s for s in spans if s.name == "tool.reset_credentials")
    assert tool_span.attributes["decision"] == "require_approval"
    assert tool_span.attributes["approved"] is True
