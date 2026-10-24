"""LangGraph StateGraph wiring: routing, reflection, HITL interrupt.

    intake -> planner -> [knowledge_search] -> [access_check] -> reasoning -> reflection
                                                                   ^             |
                                                                   +-- retry ----+
    reflection -> policy_gate -> respond                      (no action)
                              -> action_ticket -> respond     (allowed)
                              -> human_approval -> action_ticket | escalate   (HITL)
                              -> escalate -> respond          (denied)
    Any failed node (exception / timeout) routes to escalate.
"""
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from helpdesk.agents.access import access_check
from helpdesk.agents.action import action_ticket
from helpdesk.agents.base import Deps, node
from helpdesk.agents.frontdesk import intake, respond
from helpdesk.agents.governance import escalate, human_approval, policy_gate
from helpdesk.agents.knowledge import knowledge_search
from helpdesk.agents.planner import planner
from helpdesk.agents.reasoning import reasoning
from helpdesk.agents.reflection import reflection
from helpdesk.state import HelpdeskState


def after_planner(state: HelpdeskState) -> str:
    if state.get("failed"):
        return "escalate"
    plan = state["plan"]
    if plan["needs_kb"]:
        return "knowledge_search"
    return "access_check" if plan["needs_account_check"] else "reasoning"


def after_knowledge(state: HelpdeskState) -> str:
    if state.get("failed"):
        return "escalate"
    return "access_check" if state["plan"]["needs_account_check"] else "reasoning"


def after_step(state: HelpdeskState, nxt: str) -> str:
    return "escalate" if state.get("failed") else nxt


def make_after_reflection(max_reflections: int):
    def after_reflection(state: HelpdeskState) -> str:
        if state.get("failed"):
            return "escalate"
        if state["reflection"]["grounded"]:
            return "policy_gate"
        # one retry with feedback; still ungrounded afterwards -> a human takes over
        return "reasoning" if state["reflection_attempts"] <= max_reflections else "escalate"

    return after_reflection


def after_policy(state: HelpdeskState) -> str:
    if state.get("failed"):
        return "escalate"
    decision = state.get("policy_decision")
    if decision is None:
        return "escalate" if state.get("escalation_reason") else "respond"
    return {"allow": "action_ticket", "require_approval": "human_approval", "deny": "escalate"}[decision["effect"]]


def after_approval(state: HelpdeskState) -> str:
    if state.get("failed"):
        return "escalate"
    return "action_ticket" if state["approval"]["approved"] else "escalate"


def after_action(state: HelpdeskState) -> str:
    return "respond" if state.get("action_result") else "escalate"


def build_graph(deps: Deps, checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    g = StateGraph(HelpdeskState)
    for name, fn in [
        ("intake", intake), ("planner", planner), ("knowledge_search", knowledge_search),
        ("access_check", access_check), ("reasoning", reasoning), ("reflection", reflection),
        ("policy_gate", policy_gate), ("action_ticket", action_ticket),
        ("escalate", escalate), ("respond", respond),
    ]:
        g.add_node(name, node(name, fn, deps))
    # no timeout on the human step: it waits for a person, not a model
    g.add_node("human_approval", node("human_approval", human_approval, deps, timeout=False))

    g.add_edge(START, "intake")
    g.add_edge("intake", "planner")
    g.add_conditional_edges("planner", after_planner, ["knowledge_search", "access_check", "reasoning", "escalate"])
    g.add_conditional_edges("knowledge_search", after_knowledge, ["access_check", "reasoning", "escalate"])
    g.add_conditional_edges("access_check", lambda s: after_step(s, "reasoning"), ["reasoning", "escalate"])
    g.add_conditional_edges("reasoning", lambda s: after_step(s, "reflection"), ["reflection", "escalate"])
    g.add_conditional_edges(
        "reflection", make_after_reflection(deps.max_reflections), ["reasoning", "policy_gate", "escalate"]
    )
    g.add_conditional_edges("policy_gate", after_policy, ["respond", "action_ticket", "human_approval", "escalate"])
    g.add_conditional_edges("human_approval", after_approval, ["action_ticket", "escalate"])
    g.add_conditional_edges("action_ticket", after_action, ["respond", "escalate"])
    g.add_edge("escalate", "respond")
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
