"""Shared agent plumbing: dependencies, the node wrapper, and context formatting.

`node()` wraps every graph node with:
  - an OpenTelemetry span (agent_name, session_id, decision, latency)
  - a per-node timeout and an overall graph budget (supervisor timeouts)
  - failure capture: an exception or timeout marks the state `failed`,
    and the routers send failed runs to the escalate node (fallback to a human)
"""
import contextvars
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any

from langgraph.errors import GraphBubbleUp

from helpdesk.llm import LLMClient
from helpdesk.state import HelpdeskState
from helpdesk.telemetry import set_attrs, span
from helpdesk.tools.executor import CallContext, ToolExecutor

NodeFn = Callable[[HelpdeskState, "Deps"], dict[str, Any]]


@dataclass
class Deps:
    llm: LLMClient
    executor: ToolExecutor
    node_timeout_s: float = 60
    graph_budget_s: float = 180
    max_reflections: int = 1

    def ctx(self, state: HelpdeskState, agent: str) -> CallContext:
        acct = latest_tool_output(state, "get_account_status")
        return CallContext(
            session_id=state["session_id"], agent=agent, role=state.get("role", "user"),
            facts={"account_found": acct["found"] if acct else None},
        )


_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="node")


def node(name: str, fn: NodeFn, deps: Deps, *, timeout: bool = True) -> Callable[[HelpdeskState], dict]:
    """Wrap a node function. It may return `_decision` (a short string) to record on the span."""

    def run(state: HelpdeskState) -> dict[str, Any]:
        start = time.perf_counter()
        with span(f"agent.{name}", session_id=state["session_id"], agent_name=name, passthrough=(GraphBubbleUp,)) as sp:
            try:
                if time.time() - state.get("started_at", time.time()) > deps.graph_budget_s:
                    raise TimeoutError(f"graph budget of {deps.graph_budget_s}s exceeded before {name}")
                if timeout:
                    # copy_context keeps the current span as parent inside the worker thread
                    future = _pool.submit(contextvars.copy_context().run, fn, state, deps)
                    try:
                        update = future.result(timeout=deps.node_timeout_s)
                    except FutureTimeout:
                        raise TimeoutError(f"{name} exceeded node timeout of {deps.node_timeout_s}s") from None
                else:
                    update = fn(state, deps)
            except GraphBubbleUp:
                set_attrs(sp, decision="interrupted")
                raise
            except Exception as e:
                error = f"{name}: {type(e).__name__}: {e}"
                set_attrs(sp, decision="failed", error=error)
                update = {"failed": True, "errors": [error], "escalation_reason": error, "_decision": "failed"}

            decision = update.pop("_decision", "done")
            set_attrs(sp, decision=decision)
            latency = round((time.perf_counter() - start) * 1000, 2)
            return {
                **update,
                "current_agent": name,
                "events": [{"agent": name, "decision": decision, "latency_ms": latency}],
            }

    run.__name__ = name
    return run


def latest_tool_output(state: HelpdeskState, tool: str) -> dict | None:
    for r in reversed(state.get("tool_results", [])):
        if r["tool"] == tool and r["status"] == "ok":
            return r["output"]
    return None


def format_history(state: HelpdeskState) -> str:
    """Earlier turns of this conversation, or "" for a first message."""
    turns = state.get("history") or []
    if not turns:
        return ""
    lines = ["EARLIER IN THIS CONVERSATION (oldest first):"]
    for i, t in enumerate(turns, 1):
        ref = f", reference {t['reference']}" if t.get("reference") else ""
        lines.append(f"{i}. user: {t['query']}\n   resolution: {t['resolution']} "
                     f"(action {t['action']}, outcome {t['outcome']}{ref})")
    return "\n".join(lines)


def format_context(state: HelpdeskState) -> str:
    """Tool evidence shown to the reasoning and reflection agents."""
    parts = []
    kb = latest_tool_output(state, "search_kb")
    if kb and kb["hits"]:
        parts.append("KNOWLEDGE BASE ARTICLES:")
        for h in kb["hits"]:
            parts.append(
                f"- [{h['article_id']}] {h['title']} (score {h['score']:.2f}; issue_type={h['issue_type']})\n"
                f"  Resolution: {h['resolution']}"
            )
    else:
        parts.append("KNOWLEDGE BASE ARTICLES: none retrieved")
    acct = latest_tool_output(state, "get_account_status")
    if acct:
        parts.append("ACCOUNT STATUS:\n" + json.dumps(acct, indent=2))
    return "\n".join(parts)
