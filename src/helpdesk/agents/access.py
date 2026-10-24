"""Access-Check agent."""
from helpdesk.agents.base import Deps
from helpdesk.state import HelpdeskState


def access_check(state: HelpdeskState, deps: Deps) -> dict:
    r = deps.executor.execute(
        "get_account_status", {"customer_id": state["customer_id"]}, deps.ctx(state, "access_check")
    )
    decision = f"account_status={r.output['account_status']}" if r.ok else f"lookup_{r.status}"
    return {"tool_results": [r.model_dump()], "_decision": decision}
