"""Reflection: checks the draft is grounded in tool results.

Two layers: a deterministic citation check (cited ids must have been
retrieved), then an LLM judgment of whether the claims are supported.
"""

from pydantic import BaseModel

from helpdesk.agents.base import Deps, format_context, format_history, latest_tool_output
from helpdesk.state import HelpdeskState

SYSTEM = """You review a helpdesk draft for groundedness. Given the user's REQUEST, the tool
CONTEXT, and the DRAFT, decide whether the draft's claims are supported.

The PROPOSED ACTION is executed by the system right after your review, so saying it
will happen is supported.

Supported:
- restating the user's request, and anything in the context or in earlier turns
- the proposed action as a next step, and its normal consequence (create_ticket -> the
  engineering/support team will investigate and follow up)
- courtesy and follow-up phrases ("we'll keep you informed", "let us know if...")
Not supported:
- facts absent from request and context (invented causes, versions, settings, timelines)
- promising a specific outcome the action can't guarantee ("a fix will be deployed shortly")
- claiming the action is already complete
- an action that does not address the request

If not grounded, list the unsupported claims and give one sentence of feedback."""


class Reflection(BaseModel):
    grounded: bool
    unsupported_claims: list[str]
    feedback: str


def reflection(state: HelpdeskState, deps: Deps) -> dict:
    draft = state["draft"]
    attempts = state.get("reflection_attempts", 0) + 1

    kb = latest_tool_output(state, "search_kb") or {"hits": []}
    retrieved = {h["article_id"] for h in kb["hits"]}
    bad_citations = [c for c in draft["cited_articles"] if c not in retrieved]
    if bad_citations or (retrieved and not draft["cited_articles"]):
        result = Reflection(
            grounded=False,
            unsupported_claims=bad_citations,
            feedback=f"Cite only retrieved article ids: {sorted(retrieved)}.",
        )
    else:
        # resolution_summary is the article's resolution label (used for eval), not a claim to the customer
        user = (
            f"{format_history(state) + chr(10) * 2 if state.get('history') else ''}"
            f"REQUEST:\n{state['query']}\n\nCONTEXT:\n{format_context(state)}\n\n"
            f"PROPOSED ACTION: {draft['action']} ({draft['action_reason']})\n\n"
            f"DRAFT REPLY:\n{draft['customer_reply']}"
        )
        result = deps.llm.structured(Reflection, SYSTEM, user, name="reflection").parsed

    update = {
        "reflection": result.model_dump(),
        "reflection_attempts": attempts,
        "_decision": "grounded" if result.grounded else "not_grounded",
    }
    if not result.grounded and attempts > deps.max_reflections:
        update["escalation_reason"] = f"draft not grounded after {attempts} reviews: {result.feedback}"
    return update
