"""Single entry point for tool calls: policy -> validate input -> run (timeout + retry) -> validate output.

Agents never call tool handlers directly; they go through ToolExecutor.execute,
so every call is policy-checked and typed. Failures are returned as a
ToolResult with a non-"ok" status instead of raised, so graph nodes can route on them.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Literal

import openai
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, ValidationError
from tenacity import RetryError, Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from helpdesk.policy import PolicyDecision, PolicyEngine
from helpdesk.registry import ToolRegistry, ToolSpec
from helpdesk.telemetry import set_attrs, span

MAX_ATTEMPTS = 3


class TransientToolError(Exception):
    """Raise from a tool handler for failures worth retrying."""


class ToolTimeoutError(TransientToolError):
    pass


TRANSIENT_ERRORS = (
    TransientToolError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)

ToolStatus = Literal["ok", "denied", "needs_approval", "invalid_input", "error"]


class ToolResult(BaseModel):
    tool: str
    status: ToolStatus
    args: dict[str, Any]
    output: dict[str, Any] | None = None
    decision: PolicyDecision | None = None
    error: str | None = None
    attempts: int = 0
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class CallContext:
    session_id: str
    agent: str
    role: str = "user"
    facts: dict[str, Any] = field(default_factory=dict)  # session facts for policy, e.g. account_found


class ToolExecutor:
    def __init__(self, tools: ToolRegistry, policy: PolicyEngine, retry_wait_s: float = 0.5) -> None:
        self.tools = tools
        self.policy = policy
        self.retry_wait_s = retry_wait_s
        self._pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="tool")

    def execute(self, tool: str, args: dict[str, Any], ctx: CallContext, approved: bool = False) -> ToolResult:
        with span(
            f"tool.{tool}", session_id=ctx.session_id, agent_name=ctx.agent, tool_name=tool,
            role=ctx.role, approved=approved, tool_args=args,
        ) as sp:
            r = self._execute(tool, args, ctx, approved)
            set_attrs(
                sp, tool_status=r.status, attempts=r.attempts, tool_output=r.output, error=r.error,
                decision=r.decision.effect if r.decision else None,
                policy_rule=r.decision.rule_id if r.decision else None,
            )
            if r.status == "error":
                sp.set_status(Status(StatusCode.ERROR, r.error or "tool error"))
            return r

    def _execute(self, tool: str, args: dict[str, Any], ctx: CallContext, approved: bool) -> ToolResult:
        start = time.perf_counter()

        def result(status: ToolStatus, **kw) -> ToolResult:
            return ToolResult(
                tool=tool, status=status, args=args, latency_ms=round((time.perf_counter() - start) * 1000, 2), **kw
            )

        decision = self.policy.evaluate(tool, ctx.agent, ctx.role, args, ctx.facts)
        if decision.effect == "deny":
            return result("denied", decision=decision, error=decision.reason)
        if decision.effect == "require_approval" and not approved:
            return result("needs_approval", decision=decision)

        spec = self.tools.get(tool)
        try:
            inp = spec.input_model.model_validate(args)
        except ValidationError as e:
            return result("invalid_input", decision=decision, error=_short(e))

        attempts = 0
        try:
            for attempt in Retrying(
                stop=stop_after_attempt(MAX_ATTEMPTS),
                wait=wait_exponential(multiplier=self.retry_wait_s, max=4 * self.retry_wait_s),
                retry=retry_if_exception_type(TRANSIENT_ERRORS),
                reraise=False,
            ):
                with attempt:
                    attempts = attempt.retry_state.attempt_number
                    out = self._run_with_timeout(spec, inp)
        except RetryError as e:
            err = e.last_attempt.exception()
            return result("error", decision=decision, attempts=attempts,
                          error=f"{type(err).__name__} after {attempts} attempts: {err}")
        except Exception as e:  # non-transient: fail fast, no retry
            return result("error", decision=decision, attempts=attempts, error=f"{type(e).__name__}: {e}")

        try:
            out = spec.output_model.model_validate(out.model_dump() if isinstance(out, BaseModel) else out)
        except ValidationError as e:
            return result("error", decision=decision, attempts=attempts, error=f"bad tool output: {_short(e)}")
        return result("ok", decision=decision, attempts=attempts, output=out.model_dump())

    def _run_with_timeout(self, spec: ToolSpec, inp: BaseModel) -> BaseModel:
        future = self._pool.submit(spec.handler, inp)
        try:
            return future.result(timeout=spec.timeout_s)
        except FutureTimeout:
            future.cancel()
            raise ToolTimeoutError(f"{spec.name} exceeded {spec.timeout_s}s") from None


def _short(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
