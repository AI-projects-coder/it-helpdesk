"""OpenTelemetry setup and span helpers.

Every graph node, tool call, and LLM call opens a span through `span()`.
Attribute values are PII-redacted and truncated before they are recorded.
Spans are exported to data/traces/traces.jsonl (read back by the CLI) and
optionally to the console.
"""
import json
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
    SpanProcessor,
)
from opentelemetry.trace import NonRecordingSpan, SpanContext, Status, StatusCode, TraceFlags

from helpdesk.policy.redaction import redact

SERVICE_NAME = "it-helpdesk"
MAX_ATTR_LEN = 2000

_provider: TracerProvider | None = None


class JsonlSpanExporter(SpanExporter):
    """Appends one JSON object per finished span."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock, self.path.open("a") as f:
            for s in spans:
                f.write(json.dumps(span_to_dict(s)) + "\n")
        return SpanExportResult.SUCCESS


def span_to_dict(s: ReadableSpan) -> dict[str, Any]:
    return {
        "name": s.name,
        "trace_id": format(s.context.trace_id, "032x"),
        "span_id": format(s.context.span_id, "016x"),
        "parent_span_id": format(s.parent.span_id, "016x") if s.parent else None,
        "start_ns": s.start_time,
        "end_ns": s.end_time,
        "duration_ms": round((s.end_time - s.start_time) / 1e6, 2),
        "status": s.status.status_code.name,
        "attributes": dict(s.attributes or {}),
    }


def setup_tracing(jsonl_path: Path | None = None, console: bool = False) -> TracerProvider:
    """Install the global tracer provider once; later calls only add exporters."""
    global _provider
    if _provider is None:
        _provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
        trace.set_tracer_provider(_provider)
    if jsonl_path is not None:
        _provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(jsonl_path)))
    if console:
        _provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    return _provider


def add_span_processor(processor: SpanProcessor) -> None:
    setup_tracing().add_span_processor(processor)


def _clean(value: Any) -> Any:
    if isinstance(value, (bool, int, float)):
        return value
    if not isinstance(value, str):
        value = json.dumps(value, default=str)
    return redact(value)[:MAX_ATTR_LEN]


def set_attrs(sp: trace.Span, **attributes: Any) -> None:
    for k, v in attributes.items():
        if v is not None:
            sp.set_attribute(k, _clean(v))


def parent_context(trace_id: str, span_id: str) -> otel_context.Context:
    """Context whose parent is a previously recorded span (used to continue a trace after HITL resume)."""
    sc = SpanContext(int(trace_id, 16), int(span_id, 16), is_remote=True, trace_flags=TraceFlags(TraceFlags.SAMPLED))
    return trace.set_span_in_context(NonRecordingSpan(sc))


@contextmanager
def span(
    name: str,
    *,
    parent: otel_context.Context | None = None,
    passthrough: tuple[type[BaseException], ...] = (),
    **attributes: Any,
) -> Iterator[trace.Span]:
    """Open a span with redacted attributes and latency_ms.

    Exceptions are recorded as span errors, except `passthrough` types (e.g. a
    LangGraph interrupt), which are control flow rather than failures.
    """
    tracer = trace.get_tracer("helpdesk")
    start = time.perf_counter()
    with tracer.start_as_current_span(
        name, context=parent, record_exception=False, set_status_on_exception=False
    ) as sp:
        set_attrs(sp, **attributes)
        try:
            yield sp
        except passthrough:
            raise
        except Exception as e:
            sp.record_exception(e)
            sp.set_status(Status(StatusCode.ERROR, f"{type(e).__name__}: {e}"[:200]))
            raise
        finally:
            sp.set_attribute("latency_ms", round((time.perf_counter() - start) * 1000, 2))


def current_ids() -> tuple[str, str]:
    sc = trace.get_current_span().get_span_context()
    return format(sc.trace_id, "032x"), format(sc.span_id, "016x")


def load_spans(path: Path, *, session_id: str | None = None, trace_id: str | None = None) -> list[dict]:
    """Read exported spans, filtered by session_id attribute and/or trace_id."""
    if not path.exists():
        return []
    spans = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if trace_id:
        spans = [s for s in spans if s["trace_id"] == trace_id]
    if session_id:
        traces = {s["trace_id"] for s in spans if s["attributes"].get("session_id") == session_id}
        spans = [s for s in spans if s["trace_id"] in traces]
    return sorted(spans, key=lambda s: s["start_ns"])
