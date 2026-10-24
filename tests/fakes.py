"""Shared test doubles: a scripted LLM and canned KB hits / agent outputs."""
from pydantic import BaseModel

from helpdesk import telemetry
from helpdesk.llm import LLMResult

CUSTOMER = "CUST_05883"
GROUNDED = {"grounded": True, "unsupported_claims": [], "feedback": ""}
UNGROUNDED = {"grounded": False, "unsupported_claims": ["invented claim"], "feedback": "remove the invented claim"}

KB_HITS = [
    {"article_id": "KB_ACCOUNT_ACCESS", "issue_type": "account_access", "title": "Access",
     "resolution": "Reset account credentials and confirmed successful login.", "score": 0.8},
    {"article_id": "KB_HOW_TO", "issue_type": "how_to", "title": "How-to",
     "resolution": "Provided step-by-step instructions and relevant documentation.", "score": 0.6},
    {"article_id": "KB_BUG", "issue_type": "bug", "title": "Bugs",
     "resolution": "Patched the bug and deployed a fix to production.", "score": 0.5},
]


class FakeLLM:
    """Returns scripted responses per agent name. A list is consumed in order;
    a callable gets the user prompt; an Exception is raised."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def structured(self, schema: type[BaseModel], system: str, user: str, *, name: str) -> LLMResult:
        with telemetry.span(f"llm.{name}", llm_model="fake", token_count=15):
            self.calls.append((name, user))
            r = self.responses[name]
            if isinstance(r, list):
                r = r.pop(0)
            if callable(r):
                r = r(user)
            if isinstance(r, Exception):
                raise r
            return LLMResult(parsed=schema.model_validate(r), model="fake", input_tokens=10, output_tokens=5)

    def count(self, name: str) -> int:
        return sum(1 for n, _ in self.calls if n == name)


def plan(issue_type="how_to", needs_account=False, urgency="medium"):
    return {"issue_type": issue_type, "summary": f"User has a {issue_type} issue", "urgency": urgency,
            "needs_kb": True, "needs_account_check": needs_account, "steps": ["search", "answer"]}


def draft(action="none", cited=("KB_HOW_TO",), **kw):
    return {"resolution_summary": "Provided step-by-step instructions and relevant documentation.",
            "customer_reply": "Here is what to do next.", "cited_articles": list(cited),
            "action": action, "action_reason": "because the user asked", **kw}
