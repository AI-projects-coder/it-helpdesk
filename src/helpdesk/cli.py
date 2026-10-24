"""CLI: query, resume, conversations, eval, ui, list-agents, list-tools, trace."""
import json
import os

import click

from helpdesk.config import get_settings
from helpdesk.telemetry import load_spans, setup_tracing

TRACE_FILE = "traces.jsonl"


def _traces_path():
    return get_settings().traces_dir / TRACE_FILE


@click.group()
@click.option("--otel-console", is_flag=True, envvar="OTEL_CONSOLE", help="Also print raw OTel spans to stdout.")
def cli(otel_console: bool) -> None:
    """IT Helpdesk multi-agent system."""
    setup_tracing(_traces_path(), console=otel_console)


@cli.command()
@click.argument("text")
@click.option("--customer-id", help="e.g. CUST_05883 (enables account check and actions)")
@click.option("--role", default="user", type=click.Choice(["viewer", "user", "admin"]), show_default=True)
@click.option("--conversation", "conversation_id", help="Continue this conversation (agents see earlier turns).")
@click.option("--approve/--reject", "decision", default=None, help="Answer HITL non-interactively.")
@click.option("--defer", is_flag=True, help="Leave a HITL approval pending; decide later with `helpdesk resume`.")
@click.option("--trace/--no-trace", "show_trace", default=True, show_default=True)
def query(text: str, customer_id: str | None, role: str, conversation_id: str | None, decision: bool | None,
          defer: bool, show_trace: bool) -> None:
    """Run one helpdesk turn (a new conversation unless --conversation is given)."""
    app = _app()
    try:
        result = app.start(text, customer_id=customer_id, role=role, conversation_id=conversation_id)
    except ValueError as e:
        raise click.ClickException(str(e)) from None

    if result.status == "awaiting_approval":
        _print_approval(result.interrupt)
        if defer:
            click.echo(f"\nPending. Decide later: helpdesk resume {result.session_id} --approve|--reject")
            click.echo(f"Conversation {result.conversation_id}")
            return
        approved = decision if decision is not None else click.confirm("   Approve this action?", default=False)
        comment = "" if decision is not None else click.prompt("   Comment", default="", show_default=False)
        result = app.resume(result.session_id, approved=approved, approver=os.getenv("USER", "cli-user"),
                            comment=comment)

    _print_run(result.state)
    click.echo(f"\nConversation {result.conversation_id}  (continue with --conversation {result.conversation_id})")
    if show_trace:
        _print_spans(load_spans(_traces_path(), session_id=result.session_id))


@cli.command()
@click.argument("session_id")
@click.option("--approve/--reject", "decision", required=True)
@click.option("--comment", default="")
def resume(session_id: str, decision: bool, comment: str) -> None:
    """Approve or reject a pending action (works after a restart)."""
    app = _app()
    try:
        pending = app.get_run(session_id)
    except FileNotFoundError:
        raise click.ClickException(f"unknown session {session_id}") from None
    if pending.status != "awaiting_approval":
        raise click.ClickException(f"session {session_id} is not awaiting approval")
    _print_approval(pending.interrupt)
    result = app.resume(session_id, approved=decision, approver=os.getenv("USER", "cli-user"), comment=comment)
    _print_run(result.state)


@cli.command()
@click.option("--limit", default=15, show_default=True)
def conversations(limit: int) -> None:
    """List recent conversations (⏸ = an action awaits approval)."""
    from helpdesk.conversation_store import ConversationStore

    store = ConversationStore(get_settings().state_dir / "conversations.sqlite")
    rows = store.recent(limit)
    if not rows:
        click.echo("No conversations yet.")
    for c in rows:
        flag = click.style(" ⏸", fg="yellow") if c.pending else ""
        click.echo(f"{c.id}  {c.updated_at}  {c.turn_count:>2} turn(s)  {c.customer_id or '-':<11} {c.title}{flag}")


def _app():
    from helpdesk.app import HelpdeskApp
    from helpdesk.config import api_key_problem

    if problem := api_key_problem():
        raise click.ClickException(problem)
    return HelpdeskApp()


def _print_approval(p: dict) -> None:
    click.secho("\n⏸  APPROVAL REQUIRED", fg="yellow", bold=True)
    click.echo(f"   tool:   {p['tool']}  (policy rule: {p['policy_rule']})")
    click.echo(f"   args:   {json.dumps(p['args'])}")
    click.echo(f"   reason: {p['reason']}")


@cli.command("eval")
@click.option("--split", default="all", type=click.Choice(["all", "panel", "extended"]), show_default=True)
@click.option("--limit", type=int, help="Only the first N tickets.")
@click.option("--ticket", "ticket_ids", multiple=True, help="Specific ticket id(s).")
@click.option("--concurrency", default=4, show_default=True)
@click.option("--save-baseline", is_flag=True, help="Store this run as the regression baseline.")
@click.option("--tolerance", default=0.05, show_default=True, help="Allowed absolute drop per quality metric.")
def eval_cmd(split, limit, ticket_ids, concurrency, save_baseline, tolerance) -> None:
    """Batch-evaluate the golden subset; exit code 1 on regression."""
    from helpdesk.config import api_key_problem
    from helpdesk.eval import baseline as bl
    from helpdesk.eval.runner import load_golden, run_eval, save_run

    if problem := api_key_problem():
        raise click.ClickException(problem)
    tickets = load_golden(split)
    if ticket_ids:
        tickets = [t for t in tickets if t["ticket_id"] in ticket_ids]
    tickets = tickets[:limit] if limit else tickets
    if not tickets:
        raise click.ClickException("no tickets selected")
    click.secho(f"Evaluating {len(tickets)} tickets (concurrency {concurrency})\n", bold=True)

    def progress(r: dict) -> None:
        s = r["scores"]
        mark = click.style("✓", fg="green") if s["correctness"] >= bl.PASS_THRESHOLD else click.style("✗", fg="red")
        click.echo(
            f" {mark} {r['ticket_id']}  {r['issue_type']:<17} → {str(r['predicted_issue_type']):<17} "
            f"corr={s['correctness']:.1f} grnd={s['groundedness']:.1f} tool={s['tool_use']:.1f}  "
            f"{r['outcome']:<16} {r['latency_ms']:>6.0f}ms {r['tokens']:>5} tok"
        )

    run = run_eval(tickets, concurrency=concurrency, on_result=progress)
    path = save_run(run)
    _print_summary(run["summary"])
    click.echo(f"\nSaved {path}")

    base = bl.load_baseline()
    if base and not save_baseline:
        report = bl.compare(run, base, tolerance=tolerance)
        click.secho(f"\nRegression check vs baseline {base['run_id']}", bold=True)
        for c in report.config_changes:
            click.secho(f"  config changed {c} (scores are not like-for-like)", fg="yellow")
        for r in report.regressions:
            click.secho(f"  REGRESSION {r.metric}: {r.baseline} → {r.current} ({r.detail})", fg="red")
        if report.newly_failing:
            click.secho(f"  NEWLY FAILING: {', '.join(report.newly_failing)}", fg="red")
        for imp in report.improvements:
            click.secho(f"  improved {imp}", fg="green")
        if not report.has_regression:
            click.secho("  no regression", fg="green")
        else:
            raise SystemExit(1)
    elif not base and not save_baseline:
        click.echo("No baseline yet; run with --save-baseline to create one.")
    if save_baseline:
        click.secho(f"Baseline saved: {bl.save_baseline(run)}", fg="green")


def _print_summary(s: dict) -> None:
    click.secho("\nSummary", bold=True)
    keys = ("correctness", "groundedness", "tool_use", "classification", "resolution_match", "hitl_rate", "error_rate")
    for k in keys:
        click.echo(f"  {k:<18} {s[k]:.3f}")
    click.echo(f"  {'latency p50/p95':<18} {s['latency_p50_ms']:.0f} / {s['latency_p95_ms']:.0f} ms")
    click.echo(f"  {'tokens avg':<18} {s['tokens_avg']:.0f}")
    click.echo(f"  {'outcomes':<18} {s['outcomes']}")
    click.echo("  correctness by issue type:")
    for k, v in s["correctness_by_issue_type"].items():
        click.echo(f"    {k:<18} {v:.2f}")


@cli.command()
@click.option("--port", default=8501, show_default=True, envvar="PORT")
@click.option("--host", default="localhost", show_default=True, help="Use 0.0.0.0 inside containers.")
@click.option("--headless", is_flag=True, help="Don't open a browser.")
def ui(port: int, host: str, headless: bool) -> None:
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys
    from pathlib import Path

    from helpdesk.config import ROOT

    app = Path(ROOT) / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(port), "--server.address", host,
           "--server.headless", str(headless).lower()]
    raise SystemExit(subprocess.call(cmd))


@cli.command("list-agents")
def list_agents() -> None:
    """List agents in the AI registry."""
    from helpdesk.registry import AIRegistry

    for a in AIRegistry.from_yaml().all():
        click.secho(f"{a.name} v{a.version}", bold=True)
        click.echo(f"  {a.description}")
        click.echo(f"  capabilities:  {', '.join(a.capabilities)}")
        click.echo(f"  allowed_tools: {', '.join(a.allowed_tools) or '-'}")


@cli.command("list-tools")
def list_tools() -> None:
    """List tools in the tool registry."""
    from helpdesk.registry import ToolRegistry

    for t in ToolRegistry.from_yaml().all():
        flag = click.style(" [approval required]", fg="yellow") if t.approval_required else ""
        click.secho(f"{t.name} v{t.version}  risk={t.risk}{flag}", bold=True)
        click.echo(f"  {t.description}")
        click.echo(f"  params: {', '.join(t.input_model.model_fields)}")


@cli.command()
@click.argument("session_id")
def trace(session_id: str) -> None:
    """Show the OpenTelemetry span tree for a session."""
    spans = load_spans(_traces_path(), session_id=session_id)
    if not spans:
        raise click.ClickException(f"no spans found for session {session_id}")
    _print_spans(spans)


def _print_run(state: dict) -> None:
    click.secho(f"\nSession {state['session_id']}  trace {state.get('trace_id')}", bold=True)

    click.secho("\nAgent flow", bold=True)
    for e in state.get("events", []):
        click.echo(f"  → {e['agent']:<17} {e['decision']:<34} {e['latency_ms']:>8.0f} ms")

    if state.get("tool_results"):
        click.secho("\nTool calls", bold=True)
        for r in state["tool_results"]:
            d = r.get("decision") or {}
            policy = d.get("effect", "-")
            click.echo(f"  {r['tool']:<26} {r['status']:<15} policy={policy:<17} {r['latency_ms']:>7.0f} ms")

    if state.get("errors"):
        click.secho("\nErrors", bold=True, fg="red")
        for err in state["errors"]:
            click.echo(f"  {err}")

    color = {"resolved": "green", "action_completed": "green", "ticket_created": "cyan", "escalated": "yellow"}
    outcome = state.get("outcome", "unknown")
    click.secho(f"\nOutcome: {outcome}", bold=True, fg=color.get(outcome))
    if state.get("draft"):
        click.echo(f"Resolution: {state['draft']['resolution_summary']}")
    click.secho("\nResponse", bold=True)
    click.echo(state.get("final_response", "(none)"))


def _print_spans(spans: list[dict]) -> None:
    if not spans:
        return
    click.secho("\nOpenTelemetry trace", bold=True)
    children: dict[str | None, list[dict]] = {}
    ids = {s["span_id"] for s in spans}
    for s in spans:
        parent = s["parent_span_id"] if s["parent_span_id"] in ids else None
        children.setdefault(parent, []).append(s)

    def walk(parent: str | None, depth: int) -> None:
        for s in children.get(parent, []):
            a = s["attributes"]
            extra = [f"{k}={a[k]}" for k in ("decision", "tool_status", "policy_rule", "token_count") if k in a]
            status = click.style(" ERROR", fg="red") if s["status"] == "ERROR" else ""
            name = f"{'  ' * depth}{s['name']}"
            click.echo(f"  {name:<40} {s['duration_ms']:>8.0f} ms  {' '.join(extra)}{status}")
            walk(s["span_id"], depth + 1)

    walk(None, 0)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
