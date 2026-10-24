"""Application facade used by the CLI, the dashboard, and the eval runner.

start() runs a query until it finishes or pauses for approval; resume()
continues a paused session with the human's decision. Both run under one
OpenTelemetry trace per session (resume re-attaches to the original root span).

Turns are grouped into conversations. Graph checkpoints live in SQLite, so a
paused approval survives a restart and can be resumed by another process.
"""
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from helpdesk.agents.base import Deps
from helpdesk.config import get_settings
from helpdesk.conversation_store import ConversationStore
from helpdesk.graph import build_graph
from helpdesk.llm import LLMClient, get_llm
from helpdesk.policy import PolicyEngine
from helpdesk.registry import AIRegistry, ToolRegistry
from helpdesk.session_store import SessionStore
from helpdesk.telemetry import current_ids, parent_context, span
from helpdesk.tools.executor import ToolExecutor

HISTORY_TURNS = 3   # earlier turns given to the agents as context
HISTORY_CHARS = 300  # per field, keeps the context bounded


@dataclass
class RunResult:
    session_id: str
    status: Literal["completed", "awaiting_approval"]
    state: dict[str, Any]
    interrupt: dict[str, Any] | None = None
    conversation_id: str | None = None


class HelpdeskApp:
    def __init__(
        self,
        llm: LLMClient | None = None,
        tools: ToolRegistry | None = None,
        agents: AIRegistry | None = None,
        store: SessionStore | None = None,
        state_dir: Path | None = None,
        **deps_kwargs: Any,
    ) -> None:
        self.tools = tools or ToolRegistry.from_yaml()
        self.agents = agents or AIRegistry.from_yaml()
        self.policy = PolicyEngine.from_yaml(self.tools, self.agents)
        self.executor = ToolExecutor(self.tools, self.policy)
        self.deps = Deps(llm=llm or get_llm(), executor=self.executor, **deps_kwargs)

        state_dir = state_dir or get_settings().state_dir
        state_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(state_dir / "checkpoints.sqlite", check_same_thread=False)
        self.graph = build_graph(self.deps, checkpointer=SqliteSaver(conn))
        self.conversations = ConversationStore(state_dir / "conversations.sqlite")
        self.store = store or SessionStore()

    def _config(self, session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}}

    def start(
        self,
        query: str,
        customer_id: str | None = None,
        role: str = "user",
        session_id: str | None = None,
        conversation_id: str | None = None,
        track_conversation: bool = True,
    ) -> RunResult:
        """Run one turn. Continues `conversation_id` if given, else starts a new conversation
        (unless track_conversation=False, used by eval/smoke runs to stay out of the history)."""
        session_id = session_id or uuid.uuid4().hex[:12]
        history: list[dict] = []
        if conversation_id:
            history = self.history(conversation_id, customer_id)
        elif track_conversation:
            conversation_id = self.conversations.create(query, customer_id)
        if conversation_id:
            self.conversations.add_turn(conversation_id, session_id, query, customer_id, role)

        with span("helpdesk.session", session_id=session_id, conversation_id=conversation_id, role=role,
                  customer_id=customer_id, history_turns=len(history)):
            trace_id, root_span_id = current_ids()
            self.graph.invoke(
                {
                    "session_id": session_id, "conversation_id": conversation_id,
                    "trace_id": trace_id, "root_span_id": root_span_id,
                    "query": query, "customer_id": customer_id, "role": role, "history": history,
                },
                self._config(session_id),
            )
        return self._result(session_id)

    def resume(self, session_id: str, approved: bool, approver: str = "cli-user", comment: str = "") -> RunResult:
        snapshot = self.graph.get_state(self._config(session_id))
        if not snapshot.interrupts:
            raise ValueError(f"session {session_id} is not awaiting approval")
        values = snapshot.values
        parent = parent_context(values["trace_id"], values["root_span_id"])
        with span("helpdesk.resume", parent=parent, session_id=session_id, approved=approved, approver=approver):
            self.graph.invoke(
                Command(resume={"approved": approved, "approver": approver, "comment": comment}),
                self._config(session_id),
            )
        return self._result(session_id)

    def get_run(self, session_id: str) -> RunResult:
        """Current state of a turn: from the checkpoint if present (resumable), else the saved session record."""
        snapshot = self.graph.get_state(self._config(session_id))
        if snapshot.values:
            state = dict(snapshot.values)
            cid = state.get("conversation_id")
            if snapshot.interrupts:
                return RunResult(session_id, "awaiting_approval", state, snapshot.interrupts[0].value, cid)
            return RunResult(session_id, "completed", state, conversation_id=cid)
        record = self.store.load(session_id)
        return RunResult(session_id, record["status"], {"session_id": session_id, **record}, record.get("interrupt"),
                         record.get("conversation_id"))

    def history(self, conversation_id: str, customer_id: str | None) -> list[dict]:
        """Summaries of the last completed turns for the same customer (never another customer's data)."""
        conv = self.conversations.get(conversation_id)
        if conv is None:
            raise ValueError(f"unknown conversation {conversation_id}")
        items = []
        for t in conv.turns:
            if t.status != "completed" or t.customer_id != customer_id or t.outcome == "out_of_scope":
                continue
            try:
                rec = self.store.load(t.session_id)  # already PII-redacted
            except FileNotFoundError:
                continue
            draft, action, esc = rec.get("draft") or {}, rec.get("action_result") or {}, rec.get("escalation") or {}
            items.append({
                "query": (rec.get("query") or "")[:HISTORY_CHARS],
                "resolution": (draft.get("resolution_summary") or "")[:HISTORY_CHARS],
                "action": draft.get("action"),
                "outcome": rec.get("outcome"),
                "reference": action.get("ticket_id") or action.get("reference_id") or esc.get("ticket_id"),
            })
        return items[-HISTORY_TURNS:]

    def _result(self, session_id: str) -> RunResult:
        snapshot = self.graph.get_state(self._config(session_id))
        state = dict(snapshot.values)
        conversation_id = state.get("conversation_id")
        if snapshot.interrupts:
            payload = snapshot.interrupts[0].value
            status, outcome = "awaiting_approval", None
            self.store.save(session_id, state, status, payload, conversation_id)
        else:
            payload, status, outcome = None, "completed", state.get("outcome")
            self.store.save(session_id, state, status, conversation_id=conversation_id)
        self.conversations.update_turn(session_id, status, outcome)
        return RunResult(session_id, status, state, payload, conversation_id)
