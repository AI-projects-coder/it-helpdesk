"""Batch runner over the frozen golden subset.

Each ticket's initial_message runs through the real graph. HITL interrupts
are auto-approved (recorded as approver "eval-auto") so the full path is
exercised. Token counts and latency come from the run's OpenTelemetry spans.
"""
import json
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from helpdesk import telemetry
from helpdesk.app import HelpdeskApp
from helpdesk.config import get_settings
from helpdesk.eval import scorers
from helpdesk.llm import LLMClient

_span_exporter: InMemorySpanExporter | None = None


def _spans() -> InMemorySpanExporter:
    global _span_exporter
    if _span_exporter is None:
        _span_exporter = InMemorySpanExporter()
        telemetry.add_span_processor(SimpleSpanProcessor(_span_exporter))
    return _span_exporter


def load_golden(split: str = "all") -> list[dict]:
    tickets = json.loads(get_settings().golden_path.read_text())
    return tickets if split == "all" else [t for t in tickets if t["eval_split"] == split]


def _trace_usage(trace_id: str) -> dict:
    spans = [s for s in _spans().get_finished_spans() if format(s.context.trace_id, "032x") == trace_id]
    agent_llm = [s for s in spans if s.name.startswith("llm.") and not s.name.startswith("llm.judge")]
    return {
        "tokens": sum(s.attributes.get("token_count", 0) for s in agent_llm),
        "llm_calls": len(agent_llm),
        "spans": len(spans),
    }


def run_ticket(app: HelpdeskApp, judge: scorers.Judge, ticket: dict) -> dict:
    start = time.perf_counter()
    result = app.start(ticket["initial_message"], customer_id=ticket["customer_id"], role="user",
                       track_conversation=False)
    hitl = result.status == "awaiting_approval"
    if hitl:
        result = app.resume(result.session_id, approved=True, approver="eval-auto")
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    state = result.state

    tool_score, tool_notes = scorers.score_tool_use(ticket, state)
    correctness = judge.correctness(ticket, state)
    groundedness = judge.groundedness(ticket, state)
    draft = state.get("draft") or {}
    return {
        "ticket_id": ticket["ticket_id"],
        "session_id": result.session_id,
        "trace_id": state.get("trace_id"),
        "issue_type": ticket["issue_type"],
        "predicted_issue_type": (state.get("plan") or {}).get("issue_type"),
        "reference": ticket["resolution_summary"],
        "resolution": draft.get("resolution_summary"),
        "action": draft.get("action"),
        "outcome": state.get("outcome"),
        "hitl": hitl,
        "flow": [e["agent"] for e in state.get("events", [])],
        "tools": [f"{r['tool']}:{r['status']}" for r in state.get("tool_results", [])],
        "errors": state.get("errors", []),
        "latency_ms": latency_ms,
        **_trace_usage(state.get("trace_id", "")),
        "scores": {
            "correctness": correctness.score,
            "groundedness": groundedness.score,
            "tool_use": tool_score,
            "classification": scorers.score_classification(ticket, state),
            "resolution_match": scorers.score_resolution_match(ticket, state),
        },
        "notes": {"correctness": correctness.reason, "groundedness": groundedness.reason, "tool_use": tool_notes},
    }


def summarize(rows: list[dict]) -> dict:
    def mean(key: str) -> float:
        return round(statistics.mean(r["scores"][key] for r in rows), 4)

    latencies = sorted(r["latency_ms"] for r in rows)
    outcomes: dict[str, int] = {}
    for r in rows:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    by_type: dict[str, list[float]] = {}
    for r in rows:
        by_type.setdefault(r["issue_type"], []).append(r["scores"]["correctness"])
    return {
        "tickets": len(rows),
        **{k: mean(k) for k in ("correctness", "groundedness", "tool_use", "classification", "resolution_match")},
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "latency_p95_ms": round(latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))], 1),
        "tokens_avg": round(statistics.mean(r["tokens"] for r in rows), 1),
        "error_rate": round(sum(bool(r["errors"]) for r in rows) / len(rows), 4),
        "hitl_rate": round(sum(r["hitl"] for r in rows) / len(rows), 4),
        "outcomes": outcomes,
        "correctness_by_issue_type": {k: round(statistics.mean(v), 3) for k, v in sorted(by_type.items())},
    }


def run_eval(
    tickets: list[dict],
    app: HelpdeskApp | None = None,
    judge_llm: LLMClient | None = None,
    concurrency: int = 4,
    on_result: Callable[[dict], None] | None = None,
) -> dict:
    from helpdesk.llm import LangChainLLM

    s = get_settings()
    _spans().clear()
    app = app or HelpdeskApp()
    judge = scorers.Judge(judge_llm or LangChainLLM(s.llm_provider, s.judge_model))

    def one(ticket: dict) -> dict:
        row = run_ticket(app, judge, ticket)
        if on_result:
            on_result(row)
        return row

    started = datetime.now(UTC)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(one, tickets))

    return {
        "run_id": started.strftime("%Y%m%dT%H%M%SZ"),
        "started_at": started.isoformat(),
        "config": {
            "llm_provider": s.llm_provider, "llm_model": s.llm_model, "judge_model": s.judge_model,
            "embedding_model": s.embedding_model, "kb_top_k": s.kb_top_k,
            "judge_prompt_version": scorers.JUDGE_PROMPT_VERSION,
        },
        "summary": summarize(rows),
        "tickets": rows,
    }


def save_run(run: dict) -> str:
    d = get_settings().eval_dir
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"run_{run['run_id']}.json"
    text = json.dumps(run, indent=2)
    path.write_text(text)
    (d / "latest.json").write_text(text)
    return str(path)
