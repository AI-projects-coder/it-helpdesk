"""Shared LangGraph state definition.

Nodes return partial updates. Fields annotated with `add` are append-only
(each node's items are concatenated); everything else is last-write-wins.
"""
from operator import add
from typing import Annotated, Any, Literal, TypedDict

IssueType = Literal[
    "account_access", "billing_problem", "security_concern", "bug",
    "performance", "how_to", "feature_request", "other",
]

ActionName = Literal["none", "create_ticket", "reset_credentials", "update_security_settings", "issue_refund"]

Outcome = Literal["resolved", "action_completed", "ticket_created", "escalated", "out_of_scope", "needs_info"]


class HelpdeskState(TypedDict, total=False):
    # identity / correlation
    session_id: str
    conversation_id: str | None
    trace_id: str
    root_span_id: str
    started_at: float  # per-invocation budget clock, reset after HITL resume

    # request (query is PII-redacted at intake)
    query: str
    customer_id: str | None
    role: str

    # earlier turns of the same conversation (redacted, same customer only), oldest first
    history: list[dict[str, Any]]

    # conversation + audit trail
    messages: Annotated[list[dict[str, Any]], add]
    events: Annotated[list[dict[str, Any]], add]       # agent flow: one entry per node run
    tool_results: Annotated[list[dict[str, Any]], add]  # every ToolResult, in order
    errors: Annotated[list[str], add]

    current_agent: str
    plan: dict[str, Any]
    draft: dict[str, Any]
    reflection: dict[str, Any]
    reflection_attempts: int

    # governance
    pending_action: dict[str, Any] | None  # {"tool": ..., "args": {...}}
    policy_decision: dict[str, Any] | None
    needs_approval: bool
    missing_info: list[str] | None         # action blocked only by user-suppliable fields (customer_id, amount)
    approval: dict[str, Any] | None        # {"approved": bool, "approver": str, "comment": str}

    action_result: dict[str, Any] | None
    escalation_reason: str | None
    escalation: dict[str, Any] | None
    failed: bool

    final_response: str
    outcome: Outcome
