"""Intake (first node) and respond (last node)."""
import time

from helpdesk.agents.base import Deps
from helpdesk.policy import redact
from helpdesk.state import HelpdeskState


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

    if escalation or state.get("failed"):
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
