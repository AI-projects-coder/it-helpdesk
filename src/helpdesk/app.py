"""Application facade used by the CLI (and later Streamlit and the eval runner).

start() runs a query until it finishes or pauses for approval; resume()
continues a paused session with the human's decision. Both run under one
OpenTelemetry trace per session (resume re-attaches to the original root span).
"""
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.types import Command

from helpdesk.agents.base import Deps
from helpdesk.graph import build_graph
from helpdesk.llm import LLMClient, get_llm
from helpdesk.policy import PolicyEngine
from helpdesk.registry import AIRegistry, ToolRegistry
from helpdesk.session_store import SessionStore
from helpdesk.telemetry import current_ids, parent_context, span
from helpdesk.tools.executor import ToolExecutor


@dataclass
class RunResult:
    session_id: str
    status: Literal["completed", "awaiting_approval"]
    state: dict[str, Any]
    interrupt: dict[str, Any] | None = None


class HelpdeskApp:
    def __init__(
        self,
        llm: LLMClient | None = None,
        tools: ToolRegistry | None = None,
        agents: AIRegistry | None = None,
        store: SessionStore | None = None,
        **deps_kwargs: Any,
    ) -> None:
        self.tools = tools or ToolRegistry.from_yaml()
        self.agents = agents or AIRegistry.from_yaml()
        self.policy = PolicyEngine.from_yaml(self.tools, self.agents)
        self.executor = ToolExecutor(self.tools, self.policy)
        self.deps = Deps(llm=llm or get_llm(), executor=self.executor, **deps_kwargs)
        self.graph = build_graph(self.deps)
        self.store = store or SessionStore()

    def _config(self, session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}}

    def start(
        self, query: str, customer_id: str | None = None, role: str = "user", session_id: str | None = None
    ) -> RunResult:
        session_id = session_id or uuid.uuid4().hex[:12]
        with span("helpdesk.session", session_id=session_id, role=role, customer_id=customer_id):
            trace_id, root_span_id = current_ids()
            self.graph.invoke(
                {
                    "session_id": session_id, "trace_id": trace_id, "root_span_id": root_span_id,
                    "query": query, "customer_id": customer_id, "role": role,
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

    def _result(self, session_id: str) -> RunResult:
        snapshot = self.graph.get_state(self._config(session_id))
        state = dict(snapshot.values)
        if snapshot.interrupts:
            payload = snapshot.interrupts[0].value
            self.store.save(session_id, state, "awaiting_approval", payload)
            return RunResult(session_id, "awaiting_approval", state, payload)
        self.store.save(session_id, state, "completed")
        return RunResult(session_id, "completed", state)
