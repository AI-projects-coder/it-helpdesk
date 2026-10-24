"""Action/Ticket agent: executes the approved action or creates a ticket."""
from helpdesk.agents.base import Deps
from helpdesk.agents.governance import ACTION_AGENT
from helpdesk.state import HelpdeskState


def action_ticket(state: HelpdeskState, deps: Deps) -> dict:
    pending = state["pending_action"]
    approved = bool((state.get("approval") or {}).get("approved"))
    # the executor re-checks policy, so an unapproved high-risk call can't slip through
    r = deps.executor.execute(pending["tool"], pending["args"], deps.ctx(state, ACTION_AGENT), approved=approved)
    update = {"tool_results": [r.model_dump()], "_decision": f"{pending['tool']}:{r.status}"}
    if r.ok:
        update["action_result"] = {"tool": pending["tool"], **r.output}
    else:
        update["escalation_reason"] = f"{pending['tool']} {r.status}: {r.error}"
    return update
