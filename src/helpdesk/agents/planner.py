"""Planner: classifies intent and plans which agents/tools to run."""
from typing import Literal

from pydantic import BaseModel, Field

from helpdesk.agents.base import Deps
from helpdesk.state import HelpdeskState, IssueType

SYSTEM = """You are the planning agent of an IT helpdesk. Classify the user's request and plan the work.

Issue types:
- account_access: cannot log in, password rejected, account locked, 2FA problems
- billing_problem: incorrect or duplicate charge, invoice problems, refunds, plan billing
- security_concern: concrete signs of compromise (suspicious logins, unknown devices), reported
  vulnerabilities, or security/compliance questions
- bug: errors, crashes, changes not saving, a feature that is broken
- performance: slowness, timeouts, degraded performance
- how_to: the user asks how to set up, configure, or export something
- feature_request: the user asks for a new capability or more options
- other: anything that fits none of the above, including vague requests ("something seems off",
  "not sure which category") that give no concrete sign of a problem

Classify by what is wrong, not by which product area is mentioned
("the billing page is slow" is performance, not billing_problem).
Set needs_account_check to true only for account_access, security_concern, or billing_problem."""

# Data minimization: customer account data is only fetched for issue types where it
# matters, whatever the LLM asks for. The LLM may still skip the lookup for these.
ACCOUNT_ISSUE_TYPES = {"account_access", "security_concern", "billing_problem"}


class Plan(BaseModel):
    issue_type: IssueType
    summary: str = Field(description="One-sentence restatement of the problem")
    urgency: Literal["low", "medium", "high", "urgent"]
    needs_kb: bool = Field(default=True, description="Search the knowledge base")
    needs_account_check: bool
    steps: list[str] = Field(description="2-4 short planned steps")


def planner(state: HelpdeskState, deps: Deps) -> dict:
    user = f"Request: {state['query']}\nCustomer id available: {bool(state.get('customer_id'))}"
    plan = deps.llm.structured(Plan, SYSTEM, user, name="planner").parsed
    plan.needs_account_check = (
        plan.needs_account_check and plan.issue_type in ACCOUNT_ISSUE_TYPES and bool(state.get("customer_id"))
    )
    return {"plan": plan.model_dump(), "_decision": f"issue_type={plan.issue_type}"}
