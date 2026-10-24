"""Knowledge Search agent."""
from helpdesk.agents.base import Deps
from helpdesk.config import get_settings
from helpdesk.state import HelpdeskState


def knowledge_search(state: HelpdeskState, deps: Deps) -> dict:
    r = deps.executor.execute(
        "search_kb", {"query": state["query"], "top_k": get_settings().kb_top_k}, deps.ctx(state, "knowledge_search")
    )
    decision = f"hits={len(r.output['hits'])}" if r.ok else f"search_{r.status}"
    return {"tool_results": [r.model_dump()], "_decision": decision}
