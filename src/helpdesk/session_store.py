"""Captures messages and tool I/O per session_id.

One JSON file per session under data/sessions/. The record holds everything
needed to audit a run and, later, to replay it with mocked tool responses.
"""
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from helpdesk.config import get_settings
from helpdesk.policy import redact_obj

CAPTURED_FIELDS = [
    "query", "customer_id", "role", "messages", "events", "tool_results", "plan", "draft",
    "reflection", "pending_action", "policy_decision", "approval", "action_result",
    "escalation", "errors", "final_response", "outcome",
]


class SessionStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or get_settings().sessions_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def save(self, session_id: str, state: dict[str, Any], status: str, interrupt: dict | None = None) -> Path:
        record = {
            "session_id": session_id,
            "trace_id": state.get("trace_id"),
            "status": status,
            "saved_at": datetime.now(UTC).isoformat(),
            "interrupt": interrupt,
            **{k: state.get(k) for k in CAPTURED_FIELDS},
        }
        p = self.path(session_id)
        p.write_text(json.dumps(redact_obj(record), indent=2, default=str))
        return p

    def load(self, session_id: str) -> dict[str, Any]:
        return json.loads(self.path(session_id).read_text())

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))
