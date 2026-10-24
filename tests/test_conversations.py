import copy

import pytest

from helpdesk.app import HISTORY_TURNS, HelpdeskApp
from helpdesk.conversation_store import ConversationStore
from helpdesk.session_store import SessionStore
from helpdesk.tools import kb
from tests.fakes import CUSTOMER, GROUNDED, KB_HITS, FakeLLM, draft, plan

OTHER_CUSTOMER = "CUST_02280"


@pytest.fixture(autouse=True)
def stub_kb(monkeypatch):
    monkeypatch.setattr(kb, "search", lambda query, top_k=3: copy.deepcopy(KB_HITS[:top_k]))


def scripted() -> FakeLLM:
    def planner(user):
        return plan("account_access", needs_account=True) if "locked" in user.lower() else plan("how_to")

    def reasoning(user):
        last = user.rsplit("Request:", 1)[-1].lower()
        return draft("reset_credentials", cited=["KB_ACCOUNT_ACCESS"]) if "locked" in last else draft()

    return FakeLLM({"planner": planner, "reasoning": reasoning, "reflection": GROUNDED})


@pytest.fixture
def make_app(tmp_path):
    def _make(llm=None):
        llm = llm or scripted()
        return HelpdeskApp(llm=llm, store=SessionStore(tmp_path / "sessions"), state_dir=tmp_path / "state"), llm

    return _make


# --- store ---

def test_store_create_turns_and_recent(tmp_path):
    store = ConversationStore(tmp_path / "c.sqlite")
    a = store.create("How do I export my data to CSV? " + "x" * 100, CUSTOMER)
    b = store.create("Second", None)
    store.add_turn(a, "s1", "How do I export", CUSTOMER, "user")
    store.add_turn(a, "s2", "Follow up", CUSTOMER, "user")
    store.update_turn("s1", "completed", "resolved")
    store.update_turn("s2", "awaiting_approval", None)

    conv = store.get(a)
    assert [t.seq for t in conv.turns] == [1, 2]
    assert conv.turns[0].outcome == "resolved"
    assert conv.pending is True
    assert len(conv.title) == 60
    recent = store.recent()
    assert {c.id for c in recent} == {a, b}
    assert next(c for c in recent if c.id == a).turn_count == 2
    assert store.get("missing") is None
    store.update_turn("unknown-session", "completed", None)  # no-op, no error


# --- app ---

def test_first_turn_creates_conversation(make_app):
    app, llm = make_app()
    r = app.start("How do I export?", customer_id=CUSTOMER)
    assert r.conversation_id
    conv = app.conversations.get(r.conversation_id)
    assert [t.session_id for t in conv.turns] == [r.session_id]
    assert conv.turns[0].status == "completed" and conv.turns[0].outcome == "resolved"
    assert "EARLIER IN THIS CONVERSATION" not in llm.calls[0][1]
    assert app.store.load(r.session_id)["conversation_id"] == r.conversation_id


def test_follow_up_sees_earlier_turn(make_app):
    app, llm = make_app()
    first = app.start("How do I export my data?", customer_id=CUSTOMER)
    second = app.start("It still does not work", customer_id=CUSTOMER, conversation_id=first.conversation_id)

    assert second.conversation_id == first.conversation_id
    assert second.state["history"][0]["query"] == "How do I export my data?"
    planner_prompt = [u for n, u in llm.calls if n == "planner"][1]
    reasoning_prompt = [u for n, u in llm.calls if n == "reasoning"][1]
    for prompt in (planner_prompt, reasoning_prompt):
        assert "EARLIER IN THIS CONVERSATION" in prompt and "How do I export my data?" in prompt


def test_history_never_crosses_customers(make_app):
    app, _ = make_app()
    first = app.start("How do I export my data?", customer_id=CUSTOMER)
    second = app.start("Another question", customer_id=OTHER_CUSTOMER, conversation_id=first.conversation_id)
    assert second.state["history"] == []


def test_history_is_bounded(make_app):
    app, _ = make_app()
    cid = app.start("question 0", customer_id=CUSTOMER).conversation_id
    for i in range(1, 5):
        app.start(f"question {i}", customer_id=CUSTOMER, conversation_id=cid)
    h = app.history(cid, CUSTOMER)
    assert [t["query"] for t in h] == [f"question {i}" for i in range(5 - HISTORY_TURNS, 5)]


def test_history_carries_action_reference(make_app):
    app, _ = make_app()
    paused = app.start("I am locked out", customer_id=CUSTOMER)
    app.resume(paused.session_id, approved=True)
    h = app.history(paused.conversation_id, CUSTOMER)
    assert h[0]["action"] == "reset_credentials" and h[0]["reference"].startswith("RST-")


def test_pending_turn_excluded_from_history(make_app):
    app, _ = make_app()
    paused = app.start("I am locked out", customer_id=CUSTOMER)
    assert app.history(paused.conversation_id, CUSTOMER) == []


def test_untracked_runs_create_no_conversation(make_app):
    app, _ = make_app()
    r = app.start("How do I export?", track_conversation=False)
    assert r.conversation_id is None
    assert app.conversations.recent() == []


def test_unknown_conversation_rejected(make_app):
    app, _ = make_app()
    with pytest.raises(ValueError, match="unknown conversation"):
        app.start("hi", conversation_id="nope")


def test_pending_approval_survives_restart(make_app):
    app1, _ = make_app()
    paused = app1.start("I am locked out", customer_id=CUSTOMER)
    assert paused.status == "awaiting_approval"

    app2, _ = make_app()  # new process: same SQLite state dir, fresh memory
    restored = app2.get_run(paused.session_id)
    assert restored.status == "awaiting_approval"
    assert restored.interrupt["tool"] == "reset_credentials"
    assert app2.conversations.get(paused.conversation_id).pending

    done = app2.resume(paused.session_id, approved=True, approver="alice")
    assert done.state["outcome"] == "action_completed"
    assert done.conversation_id == paused.conversation_id
    assert not app2.conversations.get(paused.conversation_id).pending
    assert app2.store.load(paused.session_id)["conversation_id"] == paused.conversation_id


def test_get_run_falls_back_to_session_record(make_app, tmp_path):
    app, _ = make_app()
    r = app.start("How do I export?")
    fresh = HelpdeskApp(llm=scripted(), store=app.store, state_dir=tmp_path / "other-state")  # no checkpoint
    loaded = fresh.get_run(r.session_id)
    assert loaded.status == "completed"
    assert loaded.state["final_response"] == r.state["final_response"]
