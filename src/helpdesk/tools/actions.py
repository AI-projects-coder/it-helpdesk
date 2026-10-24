"""Mock actions: create_ticket, reset_credentials, update_security_settings, issue_refund.

Reference ids are a hash of the input, so the same request always yields the
same id. That keeps runs deterministic for replay and evaluation.
"""
import hashlib

from pydantic import BaseModel

from helpdesk.tools.schemas import (
    ActionOutput,
    CreateTicketInput,
    CreateTicketOutput,
    RefundInput,
    ResetCredentialsInput,
    SecurityUpdateInput,
)


def _ref(prefix: str, inp: BaseModel) -> str:
    digest = hashlib.sha256(inp.model_dump_json().encode()).hexdigest()[:8].upper()
    return f"{prefix}-{digest}"


def create_ticket(inp: CreateTicketInput) -> CreateTicketOutput:
    return CreateTicketOutput(ticket_id=_ref("TKT", inp))


def reset_credentials(inp: ResetCredentialsInput) -> ActionOutput:
    return ActionOutput(
        action="reset_credentials",
        status="completed",
        reference_id=_ref("RST", inp),
        detail=f"Credentials reset and account unlocked for {inp.customer_id}; reset link sent.",
    )


def update_security_settings(inp: SecurityUpdateInput) -> ActionOutput:
    return ActionOutput(
        action="update_security_settings",
        status="completed",
        reference_id=_ref("SEC", inp),
        detail=f"Applied {', '.join(inp.changes)} for {inp.customer_id}.",
    )


def issue_refund(inp: RefundInput) -> ActionOutput:
    return ActionOutput(
        action="issue_refund",
        status="completed",
        reference_id=_ref("RFD", inp),
        detail=f"Refund of {inp.amount:.2f} issued to {inp.customer_id}.",
    )
