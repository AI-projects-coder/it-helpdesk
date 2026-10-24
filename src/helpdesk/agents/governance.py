"""Orchestrator governance nodes: policy gate, human approval (HITL), escalation."""
import time
from dataclasses import replace

from langgraph.types import interrupt
from pydantic import ValidationError

from helpdesk.agents.base import Deps
from helpdesk.state import HelpdeskState

ACTION_AGENT = "action_ticket"

# Argument problems the user can fix themselves: ask for the value instead of escalating.
USER_SUPPLIED_FIELDS = {"customer_id", "amount"}


def build_action_args(state: HelpdeskState) -> dict:
    """Tool arguments are built in code from state, not free-form by the LLM."""
    draft, plan = state["draft"], state["plan"]
    tool, cid = draft["action"], state.get("customer_id")
    if tool == "reset_credentials":
        return {"customer_id": cid, "reason": draft["action_reason"][:300]}
    if tool == "update_security_settings":
        return {"customer_id": cid, "changes": draft.get("security_changes") or ["revoke_sessions"]}
    if tool == "issue_refund":
        return {"customer_id": cid, "amount": draft.get("refund_amount"), "reason": draft["action_reason"][:300]}
    if tool == "create_ticket":
        return {"customer_id": cid, "summary": plan["summary"][:500], "issue_type": plan["issue_type"],
                "priority": plan["urgency"]}
    raise ValueError(f"no argument builder for action {tool!r}")


def policy_gate(state: HelpdeskState, deps: Deps) -> dict:
    tool = state["draft"]["action"]
    if tool == "none":
        return {"pending_action": None, "policy_decision": None, "needs_approval": False, "_decision": "no_action"}
    args = build_action_args(state)
    # validate before asking a human: an approver should never see a malformed request
    try:
        deps.executor.tools.get(tool).input_model.model_validate(args)
    except ValidationError as e:
        bad = {str(x["loc"][0]) for x in e.errors() if x["loc"]}
        if bad and bad <= USER_SUPPLIED_FIELDS:
            return {"pending_action": {"tool": tool, "args": args}, "policy_decision": None, "needs_approval": False,
                    "missing_info": sorted(bad), "_decision": "needs_info:" + ",".join(sorted(bad))}
        reason = f"cannot run {tool}: " + "; ".join(f"{'.'.join(map(str, x['loc']))} {x['msg']}" for x in e.errors())
        return {"pending_action": {"tool": tool, "args": args}, "policy_decision": None, "needs_approval": False,
                "escalation_reason": reason, "_decision": "invalid_action_args"}
    ctx = deps.ctx(state, ACTION_AGENT)
    decision = deps.executor.policy.evaluate(tool, ACTION_AGENT, ctx.role, args, ctx.facts)
    update = {
        "pending_action": {"tool": tool, "args": args},
        "policy_decision": decision.model_dump(),
        "needs_approval": decision.effect == "require_approval",
        "_decision": f"{decision.effect}:{decision.rule_id}",
    }
    if decision.effect == "deny":
        update["escalation_reason"] = f"policy denied {tool}: {decision.reason}"
    return update


def human_approval(state: HelpdeskState, deps: Deps) -> dict:
    """Pauses the graph (LangGraph interrupt) until a human approves or rejects.

    Re-runs from the top on resume, so nothing before interrupt() may have side effects.
    """
    pending = state["pending_action"]
    answer = interrupt({
        "session_id": state["session_id"],
        "tool": pending["tool"],
        "args": pending["args"],
        "reason": state["draft"]["action_reason"],
        "policy_rule": state["policy_decision"]["rule_id"],
    })
    approval = {
        "approved": bool(answer.get("approved")),
        "approver": answer.get("approver", "unknown"),
        "comment": answer.get("comment", ""),
    }
    update = {
        "approval": approval,
        "started_at": time.time(),
        "_decision": "approved" if approval["approved"] else "rejected",
    }
    if not approval["approved"]:
        reason = f"{pending['tool']} rejected by {approval['approver']}: {approval['comment']}"
        update["escalation_reason"] = reason.strip(": ")
    return update


def escalate(state: HelpdeskState, deps: Deps) -> dict:
    """Fallback path: hand the case to a human, with a ticket when we know the customer."""
    reason = state.get("escalation_reason") or "unable to resolve automatically"
    escalation = {"reason": reason, "ticket_id": None}
    update: dict = {"escalation": escalation}
    if state.get("customer_id"):
        plan = state.get("plan") or {}
        args = {
            "customer_id": state["customer_id"],
            "summary": f"[ESCALATED] {plan.get('summary') or state['query']}"[:500],
            "issue_type": plan.get("issue_type", "other"),
            "priority": "high",
        }
        ctx = deps.ctx(state, ACTION_AGENT)
        ctx = replace(ctx, facts={**ctx.facts, "escalation": True})  # policy rule escalation-ticket
        r = deps.executor.execute("create_ticket", args, ctx)
        update["tool_results"] = [r.model_dump()]
        if r.ok:
            escalation["ticket_id"] = r.output["ticket_id"]
    update["_decision"] = "escalated"
    return update
