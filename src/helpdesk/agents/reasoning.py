"""Reasoning agent: drafts resolution and proposed action."""
from typing import Literal

from pydantic import BaseModel, Field

from helpdesk.agents.base import Deps, format_context
from helpdesk.state import ActionName, HelpdeskState

SYSTEM = """You are the reasoning agent of an IT helpdesk. Using ONLY the context provided
(knowledge base articles and account status), draft how the request is resolved.

- resolution_summary: one sentence, past tense, describing the resolution, based on the
  best-matching article's resolution text.
- customer_reply: 2-4 sentences to the user explaining what happens next. The action has
  NOT run yet (it may need approval), so describe it as a next step, never as done.
  Only mention the chosen action; don't invent work in progress (e.g. "a fix is being prepared")
  and don't promise outcomes or timelines. For create_ticket, say the ticket goes to the
  engineering/support team, who will investigate and follow up.
- cited_articles: the article ids you relied on (must come from the context).
- action, choose one:
    reset_credentials         user cannot log in, is locked out, or 2FA is broken
    update_security_settings  suspicious activity or a vulnerability affecting the account
    issue_refund              incorrect or duplicate charge
    create_ticket             needs engineering/product follow-up (bugs, performance, feature requests)
    none                      instructions or guidance are enough
- refund_amount: only for issue_refund, the amount the user mentions, otherwise null.
- security_changes: only for update_security_settings.

Do not state facts that are not in the context."""


class Draft(BaseModel):
    resolution_summary: str
    customer_reply: str
    cited_articles: list[str]
    action: ActionName
    action_reason: str = Field(description="Why this action, in one sentence")
    refund_amount: float | None = Field(
        default=None,
        description="For issue_refund: the amount the user states, e.g. 'refund the extra 40 dollars' -> 40",
    )
    security_changes: list[Literal["revoke_sessions", "force_mfa", "rotate_api_keys", "enable_audit_log"]] | None = None


def reasoning(state: HelpdeskState, deps: Deps) -> dict:
    plan = state["plan"]
    user = (
        f"Request: {state['query']}\n"
        f"Planner: issue_type={plan['issue_type']}, urgency={plan['urgency']}, summary={plan['summary']}\n\n"
        f"{format_context(state)}"
    )
    feedback = (state.get("reflection") or {}).get("feedback")
    if feedback:
        user += f"\n\nA reviewer rejected your previous draft. Fix this: {feedback}"
    draft = deps.llm.structured(Draft, SYSTEM, user, name="reasoning").parsed
    return {"draft": draft.model_dump(), "_decision": f"action={draft.action}"}
