import copy
import json

import pytest
from streamlit.testing.v1 import AppTest

from helpdesk.app import HelpdeskApp
from helpdesk.config import ROOT
from helpdesk.session_store import SessionStore
from helpdesk.tools import kb
from helpdesk.ui import dashboard, views
from tests.fakes import CUSTOMER, GROUNDED, KB_HITS, FakeLLM, draft, plan

# --- views (pure) ---

SPANS = [
    {"name": "helpdesk.session", "span_id": "a", "parent_span_id": None, "start_ns": 0, "end_ns": 3_000_000,
     "duration_ms": 3.0, "status": "UNSET", "attributes": {"session_id": "s"}},
    {"name": "agent.planner", "span_id": "b", "parent_span_id": "a", "start_ns": 1_000_000, "end_ns": 2_000_000,
     "duration_ms": 1.0, "status": "UNSET", "attributes": {"decision": "issue_type=bug"}},
    {"name": "llm.planner", "span_id": "c", "parent_span_id": "b", "start_ns": 1_100_000, "end_ns": 1_900_000,
     "duration_ms": 0.8, "status": "UNSET", "attributes": {"token_count": 42}},
    {"name": "tool.search_kb", "span_id": "d", "parent_span_id": "a", "start_ns": 2_000_000, "end_ns": 2_500_000,
     "duration_ms": 0.5, "status": "ERROR", "attributes": {"tool_status": "error"}},
]


def test_spans_frame_tree_order_depth_and_offsets():
    df = views.spans_frame(list(reversed(SPANS)))  # input order must not matter
    assert df["name"].tolist() == ["helpdesk.session", "agent.planner", "llm.planner", "tool.search_kb"]
    assert df["depth"].tolist() == [0, 1, 2, 1]
    assert df["kind"].tolist() == ["session", "agent", "llm", "tool"]
    assert df.loc[1, "start_ms"] == 1.0 and df.loc[1, "end_ms"] == 2.0
    assert df.loc[3, "decision"] == "error"


def test_trace_stats():
    stats = views.trace_stats(views.spans_frame(SPANS))
    assert stats == {"duration_ms": 3.0, "spans": 4, "llm_calls": 1, "tool_calls": 1, "tokens": 42, "errors": 1}
    assert views.trace_stats(views.spans_frame([]))["spans"] == 0


def test_state_frames():
    state = {
        "events": [{"agent": "planner", "decision": "issue_type=bug", "latency_ms": 5.0}],
        "tool_results": [{"tool": "create_ticket", "status": "ok", "args": {"a": 1}, "output": {"ticket_id": "T"},
                          "decision": {"effect": "allow", "rule_id": "r"}, "attempts": 1, "latency_ms": 1.0}],
    }
    assert views.events_frame(state).iloc[0]["agent"] == "planner"
    row = views.tools_frame(state).iloc[0]
    assert (row["tool"], row["policy"], row["output"]) == ("create_ticket", "allow", '{"ticket_id": "T"}')


def test_recent_sessions(tmp_path):
    (tmp_path / "x.json").write_text(json.dumps({"session_id": "x", "status": "completed", "outcome": "resolved",
                                                 "query": "q"}))
    (tmp_path / "broken.json").write_text("{")
    assert [s["session_id"] for s in views.recent_sessions(tmp_path)] == ["x"]


def test_kind_palette_matches_kinds():
    for mode in ("light", "dark"):
        assert len(views.KIND_COLORS[mode]) == len(views.SPAN_KINDS)


# --- dashboard (Streamlit AppTest, fake LLM) ---

def scripted_llm() -> FakeLLM:
    def planner(user):
        return plan("account_access", needs_account=True) if "locked" in user else plan("how_to")

    def reasoning(user):
        return draft("reset_credentials", cited=["KB_ACCOUNT_ACCESS"]) if "locked" in user else draft()

    return FakeLLM({"planner": planner, "reasoning": reasoning, "reflection": GROUNDED})


@pytest.fixture
def ui(tmp_path, monkeypatch):
    monkeypatch.setattr(kb, "search", lambda query, top_k=3: copy.deepcopy(KB_HITS[:top_k]))
    app = HelpdeskApp(llm=scripted_llm(), store=SessionStore(tmp_path), state_dir=tmp_path / "state")
    monkeypatch.setattr(dashboard, "get_app", lambda: app)
    monkeypatch.setattr(dashboard, "api_key_problem", lambda: None)
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    return at


def texts(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_dashboard_renders_all_tabs(ui):
    assert [t.label for t in ui.tabs] == ["💬 Chat", "🔭 Traces", "📊 Evaluation", "🗂️ Registry"]
    assert "AI registry" in texts(ui)


def test_chat_resolves_simple_question(ui):
    ui.chat_input[0].set_value("How do I export my data?").run()
    assert not ui.exception
    assert "Here is what to do next." in texts(ui)
    assert any("Resolved" in c.value for c in ui.caption)


def test_chat_hitl_approve(ui):
    ui.selectbox(key="customer").set_value(CUSTOMER).run()
    ui.chat_input[0].set_value("I am locked out of my account").run()
    assert any("Approval required" in w.value for w in ui.warning)
    assert ui.chat_input[0].disabled  # no new questions while an action waits for a human

    ui.button(key="approve_0").click().run()
    assert not ui.exception
    assert "Credentials reset and account unlocked" in texts(ui)
    assert not ui.chat_input[0].disabled


def test_chat_hitl_reject_escalates(ui):
    ui.selectbox(key="customer").set_value(CUSTOMER).run()
    ui.chat_input[0].set_value("I am locked out of my account").run()
    ui.text_input(key="comment_0").input("verify identity first")
    ui.button(key="reject_0").click().run()
    assert "escalated your request to a human" in texts(ui)


def test_missing_api_key_shows_clear_error(monkeypatch):
    monkeypatch.setattr(dashboard, "api_key_problem", lambda: "OPENAI_API_KEY in .env is missing")
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30)
    at.run()
    assert "OPENAI_API_KEY" in at.error[0].value
    assert not at.tabs  # nothing else renders until the key is fixed


def test_conversation_list_reopen_and_url(ui):
    ui.chat_input[0].set_value("How do I export my data?").run()
    cid = ui.session_state["conversation_id"]
    assert cid and ui.query_params["c"] == cid

    ui.button(key=f"conv_{cid}").click().run()  # reopen from the sidebar list
    assert "Here is what to do next." in texts(ui)

    next(b for b in ui.button if b.label == "➕ New conversation").click().run()
    assert ui.session_state["conversation_id"] is None
    assert "Here is what to do next." not in texts(ui)
    assert ui.button(key=f"conv_{cid}")  # old conversation still listed


def test_follow_up_in_ui_uses_history(ui):
    ui.chat_input[0].set_value("How do I export my data?").run()
    ui.chat_input[0].set_value("It still does not work").run()
    assert any("used 1 earlier turn" in c.value for c in ui.caption)
