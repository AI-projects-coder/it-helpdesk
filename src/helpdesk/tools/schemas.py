"""Pydantic input/output contracts for every tool."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CustomerId = Field(pattern=r"^CUST_\d{5}$", description="Customer id, e.g. CUST_05883")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- search_kb ---

class SearchKBInput(_Strict):
    query: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=8)


class KBHit(_Strict):
    article_id: str
    issue_type: str
    title: str
    resolution: str
    score: float


class SearchKBOutput(_Strict):
    hits: list[KBHit]


# --- get_account_status ---

class AccountStatusInput(_Strict):
    customer_id: str = CustomerId


class AccountStatusOutput(_Strict):
    customer_id: str
    found: bool
    account_status: Literal["active", "locked", "unknown"] = "unknown"
    sla_plan: str | None = None
    segment: str | None = None
    mfa_enabled: bool | None = None


# --- create_ticket ---

class CreateTicketInput(_Strict):
    customer_id: str = CustomerId
    summary: str = Field(min_length=5, max_length=500)
    issue_type: str
    priority: Literal["low", "medium", "high", "urgent"] = "medium"


class CreateTicketOutput(_Strict):
    ticket_id: str
    status: Literal["open"] = "open"


# --- state-changing actions (HITL) ---

class ResetCredentialsInput(_Strict):
    customer_id: str = CustomerId
    reason: str = Field(min_length=3, max_length=300)


class SecurityUpdateInput(_Strict):
    customer_id: str = CustomerId
    changes: list[Literal["revoke_sessions", "force_mfa", "rotate_api_keys", "enable_audit_log"]] = Field(
        min_length=1
    )


class RefundInput(_Strict):
    customer_id: str = CustomerId
    amount: float = Field(gt=0, le=10_000)
    reason: str = Field(min_length=3, max_length=300)


class ActionOutput(_Strict):
    action: str
    status: Literal["completed"]
    reference_id: str
    detail: str
