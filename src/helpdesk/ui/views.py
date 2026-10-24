"""Pure data shaping for the dashboard (no Streamlit imports, unit-testable)."""
import json
from pathlib import Path

import pandas as pd

# Span kinds -> categorical slots 1-4 of the validated palette (fixed order, never cycled).
SPAN_KINDS = ["session", "agent", "llm", "tool"]
KIND_COLORS = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500"],
}
SERIES_1 = {"light": "#2a78d6", "dark": "#3987e5"}

def span_kind(name: str) -> str:
    prefix = name.split(".", 1)[0]
    return {"helpdesk": "session", "agent": "agent", "llm": "llm", "tool": "tool"}.get(prefix, "agent")


def spans_frame(spans: list[dict]) -> pd.DataFrame:
    """One row per span, in tree order, with depth and offsets from the trace start (ms)."""
    if not spans:
        return pd.DataFrame(columns=["label", "name", "kind", "depth", "start_ms", "end_ms", "duration_ms",
                                     "decision", "status", "tokens", "order"])
    ids = {s["span_id"] for s in spans}
    children: dict[str | None, list[dict]] = {}
    for s in sorted(spans, key=lambda s: s["start_ns"]):
        parent = s["parent_span_id"] if s["parent_span_id"] in ids else None
        children.setdefault(parent, []).append(s)
    t0 = min(s["start_ns"] for s in spans)

    rows: list[dict] = []

    def walk(parent: str | None, depth: int) -> None:
        for s in children.get(parent, []):
            a = s["attributes"]
            rows.append({
                "label": f"{len(rows) + 1:02d} {'  ' * depth}{s['name']}",
                "name": s["name"],
                "kind": span_kind(s["name"]),
                "depth": depth,
                "start_ms": round((s["start_ns"] - t0) / 1e6, 1),
                "end_ms": round((s["end_ns"] - t0) / 1e6, 1),
                "duration_ms": s["duration_ms"],
                "decision": a.get("decision") or a.get("tool_status") or "",
                "status": s["status"],
                "tokens": a.get("token_count", 0),
                "order": len(rows),
            })
            walk(s["span_id"], depth + 1)

    walk(None, 0)
    return pd.DataFrame(rows)


def trace_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"duration_ms": 0, "spans": 0, "llm_calls": 0, "tool_calls": 0, "tokens": 0, "errors": 0}
    return {
        "duration_ms": float(df["end_ms"].max() - df["start_ms"].min()),
        "spans": len(df),
        "llm_calls": int((df["kind"] == "llm").sum()),
        "tool_calls": int((df["kind"] == "tool").sum()),
        "tokens": int(df["tokens"].sum()),
        "errors": int((df["status"] == "ERROR").sum()),
    }


def eval_tickets_frame(run: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"ticket": t["ticket_id"], "issue_type": t["issue_type"], "predicted": t["predicted_issue_type"],
         "action": t["action"], "outcome": t["outcome"], "hitl": t["hitl"], **t["scores"],
         "latency_ms": t["latency_ms"], "tokens": t["tokens"], "session_id": t["session_id"],
         "correctness_note": t["notes"]["correctness"]}
        for t in run["tickets"]
    ])


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def recent_sessions(sessions_dir: Path, limit: int = 30) -> list[dict]:
    files = sorted(sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out = []
    for p in files:
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        out.append({"session_id": d["session_id"], "status": d.get("status"), "outcome": d.get("outcome"),
                    "query": (d.get("query") or "")[:70], "saved_at": d.get("saved_at")})
    return out
