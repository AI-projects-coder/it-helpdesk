"""Intake (first node) and respond (last node)."""
import time

from helpdesk.agents.base import Deps
from helpdesk.policy import redact
from helpdesk.state import HelpdeskState

OUT_OF_SCOPE_REPLY = "Sorry, I can only help with IT support questions."

ACTION_LABELS = {
    "reset_credentials": "reset your password",
    "update_security_settings": "update your account's security settings",
    "issue_refund": "process your refund",
    "create_ticket": "create a support ticket",
}


def needs_info_reply(tool: str, missing: list[str]) -> str:
    asks = []
    if "customer_id" in missing:
        asks.append("your Customer ID: please enter it in the sidebar (for example CUST_12345)")
    if "amount" in missing:
        asks.append("the refund amount: please include it in your message (for example \"refund 40 dollars\")")
    label = ACTION_LABELS.get(tool, "continue")
    return f"To {label} I need {' and '.join(asks)}. Then send your request again."


def intake(state: HelpdeskState, deps: Deps) -> dict:
    query = redact(state["query"].strip())
    return {
        "query": query,
        "started_at": time.time(),
        "reflection_attempts": 0,
        "failed": False,
        "messages": [{"role": "user", "content": query}],
        "_decision": "redacted" if query != state["query"].strip() else "accepted",
    }


def respond(state: HelpdeskState, deps: Deps) -> dict:
    draft = state.get("draft") or {}
    escalation = state.get("escalation")
    action = state.get("action_result")

    if not (state.get("plan") or {}).get("in_scope", True) and not state.get("failed"):
        outcome = "out_of_scope"
        text = OUT_OF_SCOPE_REPLY
    elif state.get("missing_info") and not state.get("failed"):
        outcome = "needs_info"
        text = needs_info_reply((state.get("pending_action") or {}).get("tool", ""), state["missing_info"])
    elif escalation or state.get("failed"):
        outcome = "escalated"
        ticket = f" (ticket {escalation['ticket_id']})" if (escalation or {}).get("ticket_id") else ""
        text = f"I've escalated your request to a human support engineer{ticket}. They will follow up shortly."
        if draft.get("customer_reply") and not state.get("failed"):
            text = f"{draft['customer_reply']}\n\n{text}"
    elif action:
        outcome = "ticket_created" if action["tool"] == "create_ticket" else "action_completed"
        detail = f"Ticket {action['ticket_id']} created." if outcome == "ticket_created" else action["detail"]
        text = f"{draft['customer_reply']}\n\n{detail}"
    else:
        outcome = "resolved"
        text = draft["customer_reply"]

    return {
        "final_response": text,
        "outcome": outcome,
        "messages": [{"role": "assistant", "content": text}],
        "_decision": outcome,
    }
