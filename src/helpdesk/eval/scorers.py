"""Correctness, groundedness, tool appropriateness, latency.

Deterministic scorers (classification, resolution match, tool use) cannot
drift. The LLM judges use a pinned model at temperature 0 and a versioned
prompt, recorded with every run, so a score change can be traced to its cause.
"""
import json
from typing import Literal

from pydantic import BaseModel, Field

from helpdesk.llm import LLMClient

JUDGE_PROMPT_VERSION = "1.1"

# Which proposed actions are appropriate per issue type (scenario overrides win).
ACCEPTABLE_ACTIONS: dict[str, set[str]] = {
    "account_access": {"reset_credentials"},
    "security_concern": {"update_security_settings", "create_ticket"},
    "billing_problem": {"issue_refund", "create_ticket"},
    "bug": {"create_ticket"},
    "performance": {"create_ticket"},
    "feature_request": {"create_ticket"},
    "how_to": {"none"},
    "other": {"none", "create_ticket"},
}
SCENARIO_OVERRIDES: dict[str, set[str]] = {
    "security_compliance": {"none", "create_ticket"},  # an information request, not an incident
}
NEEDS_ACCOUNT_LOOKUP = {"account_access", "security_concern", "billing_problem"}


def acceptable_actions(ticket: dict) -> set[str]:
    return SCENARIO_OVERRIDES.get(ticket["scenario_key"], ACCEPTABLE_ACTIONS[ticket["issue_type"]])


# --- deterministic ---

def score_classification(ticket: dict, state: dict) -> float:
    return float((state.get("plan") or {}).get("issue_type") == ticket["issue_type"])


def score_resolution_match(ticket: dict, state: dict) -> float:
    """Did the agent pick the KB article whose resolution is the golden one?"""
    draft = state.get("draft") or {}
    hits = {h["article_id"]: h for r in state.get("tool_results", [])
            if r["tool"] == "search_kb" and r["status"] == "ok" for h in r["output"]["hits"]}
    cited = [hits[c]["resolution"] for c in draft.get("cited_articles", []) if c in hits]
    return float(bool(cited) and cited[0] == ticket["resolution_summary"])


def score_tool_use(ticket: dict, state: dict) -> tuple[float, str]:
    """Mean of: proposed action is appropriate; account lookup done when needed (and only then)."""
    action = (state.get("draft") or {}).get("action", "none")
    action_ok = action in acceptable_actions(ticket)
    looked_up = any(r["tool"] == "get_account_status" for r in state.get("tool_results", []))
    lookup_ok = looked_up == (ticket["issue_type"] in NEEDS_ACCOUNT_LOOKUP)
    notes = []
    if not action_ok:
        notes.append(f"action {action} not in {sorted(acceptable_actions(ticket))}")
    if not lookup_ok:
        notes.append("missing account lookup" if not looked_up else "unneeded account lookup")
    return (action_ok + lookup_ok) / 2, "; ".join(notes)


# --- LLM judges ---

Score = Literal[0.0, 0.5, 1.0]


class CorrectnessVerdict(BaseModel):
    score: Score = Field(description="1.0 same resolution, 0.5 partially matches, 0.0 different or wrong")
    reason: str


class GroundednessVerdict(BaseModel):
    score: Score = Field(description="1.0 fully supported, 0.5 minor unsupported details, 0.0 major unsupported claims")
    reason: str


CORRECTNESS_SYSTEM = """You grade an IT helpdesk agent. Compare the agent's resolution with the
REFERENCE resolution written by a human support engineer. Judge whether the agent arrived at the
same kind of resolution (same fix / same handling), not wording.

The agent is a helpdesk: it cannot deploy code fixes or change infrastructure itself. When the
reference describes engineering work (patching a bug, optimizing queries, adding capacity),
routing the issue to engineering with a ticket that captures the problem counts as the same
resolution.

1.0 = same resolution; 0.5 = partially (right direction, missing or extra key step); 0.0 = different."""

GROUNDEDNESS_SYSTEM = """You grade whether an IT helpdesk reply is grounded in the evidence the agent
had: the user's request, the tool results (knowledge base articles, account data, action results).
1.0 = every claim supported; 0.5 = minor unsupported details; 0.0 = major claims not supported."""


class Judge:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def correctness(self, ticket: dict, state: dict) -> CorrectnessVerdict:
        draft = state.get("draft") or {}
        user = (
            f"USER REQUEST: {ticket['initial_message']}\n\n"
            f"REFERENCE RESOLUTION: {ticket['resolution_summary']}\n\n"
            f"AGENT RESOLUTION: {draft.get('resolution_summary', '(none)')}\n"
            f"AGENT ACTION: {draft.get('action', 'none')}\n"
            f"AGENT REPLY: {state.get('final_response', '')}"
        )
        return self.llm.structured(CorrectnessVerdict, CORRECTNESS_SYSTEM, user, name="judge_correctness").parsed

    def groundedness(self, ticket: dict, state: dict) -> GroundednessVerdict:
        evidence = [{"tool": r["tool"], "status": r["status"], "output": r["output"]}
                    for r in state.get("tool_results", [])]
        user = (
            f"USER REQUEST: {state.get('query', ticket['initial_message'])}\n\n"
            f"TOOL RESULTS:\n{json.dumps(evidence, indent=1)}\n\n"
            f"AGENT REPLY: {state.get('final_response', '')}"
        )
        return self.llm.structured(GroundednessVerdict, GROUNDEDNESS_SYSTEM, user, name="judge_groundedness").parsed
