"""Streamlit dashboard: chat with the helpdesk, approve HITL actions, inspect traces and evals.

Run with `helpdesk ui` (or `streamlit run streamlit_app.py`). Each chat message is
a new helpdesk session; the dashboard is a thin layer over HelpdeskApp.
"""
import threading
import time

import altair as alt
import pandas as pd
import streamlit as st
import yaml

from helpdesk.app import HelpdeskApp, RunResult
from helpdesk.config import CONFIG_DIR, api_key_problem, get_settings
from helpdesk.eval import baseline as bl
from helpdesk.telemetry import load_spans, setup_tracing
from helpdesk.ui import views

TRACE_FILE = "traces.jsonl"


# --- resources -------------------------------------------------------------

@st.cache_resource
def get_app() -> HelpdeskApp:
    """One app per server process; LangGraph checkpoints live in it, so HITL can resume across reruns."""
    setup_tracing(get_settings().traces_dir / TRACE_FILE)
    return HelpdeskApp()


@st.cache_data
def golden_tickets() -> list[dict]:
    return views.load_json(get_settings().golden_path) or []


@st.cache_data
def accounts() -> dict[str, dict]:
    return views.load_json(get_settings().accounts_path) or {}


def theme() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except AttributeError:
        return "light"


def init_state() -> None:
    st.session_state.setdefault("turns", [])        # [{"query", "customer_id", "role", "result": RunResult}]
    st.session_state.setdefault("trace_session", None)


# --- sidebar ---------------------------------------------------------------

def sidebar() -> tuple[str | None, str, str | None]:
    """Returns (customer_id, role, example query to send now)."""
    s = get_settings()
    with st.sidebar:
        st.header("Requester")
        accts = accounts()
        options = ["(none)"] + sorted(accts)

        def label(cid: str) -> str:
            if cid == "(none)":
                return "No customer id"
            a = accts[cid]
            return f"{cid} · {a['sla_plan']} · {a['account_status']}"

        customer = st.selectbox("Customer", options, format_func=label, key="customer")
        role = st.radio("Role", ["user", "viewer", "admin"], horizontal=True, key="role",
                        help="viewer can only search the KB; admin is still subject to HITL")

        st.header("Try a golden ticket")
        tickets = golden_tickets()
        idx = st.selectbox(
            "Example", range(len(tickets)), key="example",
            format_func=lambda i: f"{tickets[i]['issue_type']} · {tickets[i]['initial_message'][:48]}",
        )
        example = None
        if st.button("Send example", width="stretch"):
            example = tickets[idx]["initial_message"]
            st.session_state["example_customer"] = tickets[idx]["customer_id"]

        st.divider()
        if st.button("New conversation", width="stretch"):
            st.session_state.turns = []
            st.rerun()
        st.caption(f"LLM: `{s.llm_provider}/{s.llm_model}` · judge `{s.judge_model}` · KB top-k {s.kb_top_k}")

    return (None if customer == "(none)" else customer), role, example


# --- chat ------------------------------------------------------------------

def render_details(state: dict) -> None:
    with st.expander("How this was handled", expanded=False):
        plan = state.get("plan") or {}
        if plan:
            st.markdown(f"**Plan** · `{plan.get('issue_type')}` · urgency `{plan.get('urgency')}`  \n"
                        + " → ".join(plan.get("steps", [])))
        st.markdown("**Agent flow**")
        st.dataframe(views.events_frame(state), hide_index=True, width="stretch")
        tools = views.tools_frame(state)
        if not tools.empty:
            st.markdown("**Tool calls** (every call passes the policy engine)")
            st.dataframe(tools, hide_index=True, width="stretch")
        refl = state.get("reflection")
        if refl:
            verdict = "grounded ✅" if refl["grounded"] else f"not grounded ⚠️ — {refl['feedback']}"
            st.markdown(f"**Reflection** ({state.get('reflection_attempts', 1)} review(s)): {verdict}")
        if state.get("errors"):
            st.error("\n".join(state["errors"]))


def render_turn(i: int, turn: dict) -> None:
    with st.chat_message("user"):
        who = turn["customer_id"] or "anonymous"
        st.markdown(turn["query"])
        st.caption(f"{who} · role {turn['role']}")

    result: RunResult = turn["result"]
    state = result.state
    with st.chat_message("assistant"):
        if result.status == "awaiting_approval":
            render_approval(i, turn)
            return
        icon, text = views.OUTCOME_BADGES.get(state.get("outcome"), ("•", state.get("outcome")))
        st.markdown(state.get("final_response", ""))
        cols = st.columns([2, 2, 3])
        cols[0].caption(f"{icon} {text}")
        cols[1].caption(f"issue `{(state.get('plan') or {}).get('issue_type', '-')}`")
        if cols[2].button(f"Trace {result.session_id}", key=f"trace_{i}", type="tertiary"):
            st.session_state.trace_session = result.session_id
            st.toast("Open the Traces tab to see this session")
        render_details(state)


def render_approval(i: int, turn: dict) -> None:
    result: RunResult = turn["result"]
    p = result.interrupt
    st.warning(f"**Approval required** before running `{p['tool']}` (policy rule `{p['policy_rule']}`)", icon="⏸️")
    st.markdown(f"**Why:** {p['reason']}")
    st.json(p["args"])
    comment = st.text_input("Comment for the audit log", key=f"comment_{i}")
    c1, c2, _ = st.columns([1, 1, 4])
    approve = c1.button("Approve", key=f"approve_{i}", type="primary")
    reject = c2.button("Reject", key=f"reject_{i}")
    if approve or reject:
        with st.spinner("Resuming…"):
            turn["result"] = get_app().resume(result.session_id, approved=approve, approver="dashboard-user",
                                              comment=comment)
        st.rerun()
    render_details(result.state)


def chat_tab(customer_id: str | None, role: str, example: str | None) -> None:
    turns = st.session_state.turns
    if not turns:
        st.info("Describe an IT problem, or send a golden ticket from the sidebar. Pick a customer to enable "
                "account checks and actions. Try *“My account is locked”* with **CUST_02280** to see approval.")
    for i, turn in enumerate(turns):
        render_turn(i, turn)

    pending = any(t["result"].status == "awaiting_approval" for t in turns)
    typed = st.chat_input("Describe your issue…", disabled=pending)
    query = typed or example
    if pending and example:
        st.toast("Approve or reject the pending action first")
        return
    if query:
        if example:
            customer_id = st.session_state.pop("example_customer", customer_id)
        with st.spinner("Agents working…"):
            result = get_app().start(query, customer_id=customer_id, role=role)
        turns.append({"query": query, "customer_id": customer_id, "role": role, "result": result})
        st.rerun()


# --- traces ----------------------------------------------------------------

def waterfall(df: pd.DataFrame) -> alt.Chart:
    mode = theme()
    order = df.sort_values("order")["label"].tolist()
    return (
        alt.Chart(df)
        .mark_bar(cornerRadius=4, height=14)
        .encode(
            x=alt.X("start_ms:Q", title="ms since session start"),
            x2="end_ms:Q",
            y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=320)),
            color=alt.Color("kind:N", scale=alt.Scale(domain=views.SPAN_KINDS, range=views.KIND_COLORS[mode]),
                            legend=alt.Legend(title="span kind", orient="top")),
            tooltip=["name", "kind", alt.Tooltip("duration_ms:Q", format=".0f"), "decision", "status", "tokens"],
        )
        .properties(height=max(160, 22 * len(df)))
    )


def traces_tab() -> None:
    s = get_settings()
    recent = views.recent_sessions(s.sessions_dir)
    if not recent:
        st.info("No sessions yet. Ask something in the Chat tab.")
        return
    ids = [r["session_id"] for r in recent]
    labels = {r["session_id"]: f"{r['session_id']} · {r['outcome'] or r['status']} · {r['query']}" for r in recent}
    default = st.session_state.trace_session
    session_id = st.selectbox("Session", ids, index=ids.index(default) if default in ids else 0,
                              format_func=labels.get)
    df = views.spans_frame(load_spans(s.traces_dir / TRACE_FILE, session_id=session_id))
    if df.empty:
        st.warning("No spans recorded for this session (it may predate tracing to file).")
        return

    stats = views.trace_stats(df)
    c = st.columns(5)
    c[0].metric("Duration", f"{stats['duration_ms'] / 1000:.1f} s")
    c[1].metric("Spans", stats["spans"])
    c[2].metric("LLM calls", stats["llm_calls"])
    c[3].metric("Tool calls", stats["tool_calls"])
    c[4].metric("Tokens", f"{stats['tokens']:,}")
    if stats["errors"]:
        st.error(f"{stats['errors']} span(s) ended with ERROR")

    view = st.segmented_control("View", ["Timeline", "Table"], default="Timeline", key="trace_view")
    if view == "Table":
        st.dataframe(df.drop(columns=["order"]), hide_index=True, width="stretch")
    else:
        st.altair_chart(waterfall(df), width="stretch")
    st.caption("Spans are exported by OpenTelemetry to data/traces/traces.jsonl; every span carries session_id.")


# --- evaluation ------------------------------------------------------------

METRICS = [("correctness", "Correctness"), ("groundedness", "Groundedness"), ("tool_use", "Tool use"),
           ("classification", "Classification"), ("resolution_match", "Resolution match")]


def run_eval_with_progress(limit: int, split: str) -> None:
    from helpdesk.eval.runner import load_golden, run_eval, save_run

    tickets = load_golden(split)[:limit]
    done: list[dict] = []
    box: dict = {}

    def work() -> None:
        try:
            box["run"] = run_eval(tickets, app=get_app(), concurrency=4, on_result=done.append)
        except Exception as e:  # surfaced in the main thread below
            box["error"] = e

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    bar = st.progress(0.0, text="Starting…")
    while worker.is_alive():
        bar.progress(len(done) / len(tickets), text=f"{len(done)}/{len(tickets)} tickets")
        time.sleep(0.5)
    if "error" in box:
        st.error(f"Evaluation failed: {box['error']}")
        return
    save_run(box["run"])
    bar.progress(1.0, text="Done")
    st.rerun()


def eval_tab() -> None:
    s = get_settings()
    latest = views.load_json(s.eval_dir / "latest.json")
    base = bl.load_baseline()

    with st.expander("Run an evaluation", expanded=latest is None):
        c1, c2, c3 = st.columns([1, 1, 1])
        split = c1.selectbox("Split", ["all", "panel", "extended"])
        limit = c2.number_input("Tickets", 1, 50, 10)
        c3.write("")
        if c3.button("Run", type="primary", width="stretch"):
            run_eval_with_progress(int(limit), split)

    if latest is None:
        st.info("No evaluation results yet.")
        return

    cur = latest["summary"]
    st.caption(f"Run `{latest['run_id']}` · {cur['tickets']} tickets · model `{latest['config']['llm_model']}` · "
               f"judge `{latest['config']['judge_model']}` (prompt v{latest['config']['judge_prompt_version']})"
               + (f" · baseline `{base['run_id']}`" if base else " · no baseline"))

    cols = st.columns(len(METRICS) + 2)
    for col, (key, name) in zip(cols, METRICS, strict=False):  # extra cols hold latency/tokens
        delta = f"{cur[key] - base['summary'][key]:+.2f}" if base and key in base["summary"] else None
        col.metric(name, f"{cur[key]:.2f}", delta)
    lat_delta = f"{cur['latency_p50_ms'] - base['summary']['latency_p50_ms']:+.0f} ms" if base else None
    cols[-2].metric("Latency p50", f"{cur['latency_p50_ms'] / 1000:.1f} s", lat_delta, delta_color="inverse")
    tok_delta = f"{cur['tokens_avg'] - base['summary']['tokens_avg']:+.0f}" if base else None
    cols[-1].metric("Tokens/ticket", f"{cur['tokens_avg']:,.0f}", tok_delta, delta_color="inverse")

    if base:
        report = bl.compare(latest, base)
        for c in report.config_changes:
            st.warning(f"Config changed since baseline: {c} — scores are not like-for-like")
        if report.has_regression:
            lines = [f"- **{r.metric}**: {r.baseline} → {r.current} ({r.detail})" for r in report.regressions]
            if report.newly_failing:
                lines.append(f"- newly failing: {', '.join(report.newly_failing)}")
            st.error("**Regression detected**\n" + "\n".join(lines))
        else:
            st.success("No regression against the baseline"
                       + (f" · improved: {', '.join(report.improvements)}" if report.improvements else ""))
    if st.button("Save this run as baseline"):
        bl.save_baseline(latest)
        st.rerun()

    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Correctness by issue type**")
        by_type = pd.DataFrame(sorted(cur["correctness_by_issue_type"].items()), columns=["issue_type", "correctness"])
        chart = (
            alt.Chart(by_type)
            .mark_bar(cornerRadiusEnd=4, height=14, color=views.SERIES_1[theme()])
            .encode(
                x=alt.X("correctness:Q", scale=alt.Scale(domain=[0, 1]), title=None),
                y=alt.Y("issue_type:N", title=None, sort="-x"),
                tooltip=["issue_type", alt.Tooltip("correctness:Q", format=".2f")],
            )
            .properties(height=24 * len(by_type) + 20)
        )
        st.altair_chart(chart, width="stretch")
        st.markdown("**Outcomes**")
        st.dataframe(pd.DataFrame(cur["outcomes"].items(), columns=["outcome", "tickets"]), hide_index=True)
    with right:
        st.markdown("**Tickets**")
        df = views.eval_tickets_frame(latest)
        only_failing = st.toggle("Only tickets below 1.0 on any score")
        if only_failing:
            df = df[(df[[k for k, _ in METRICS]] < 1).any(axis=1)]
        st.dataframe(df, hide_index=True, width="stretch", height=420)


# --- registry --------------------------------------------------------------

def registry_tab() -> None:
    app = get_app()
    st.markdown("**AI registry** — agents and the only tools each may call")
    st.dataframe(pd.DataFrame([{"agent": a.name, "version": a.version, "capabilities": ", ".join(a.capabilities),
                                "allowed_tools": ", ".join(a.allowed_tools) or "—", "description": a.description}
                               for a in app.agents.all()]), hide_index=True, width="stretch")
    st.markdown("**Tool registry**")
    st.dataframe(pd.DataFrame([{"tool": t.name, "version": t.version, "risk": t.risk,
                                "approval_required": t.approval_required, "timeout_s": t.timeout_s,
                                "params": ", ".join(t.input_model.model_fields), "description": t.description}
                               for t in app.tools.all()]), hide_index=True, width="stretch")
    policy = yaml.safe_load((CONFIG_DIR / "policy.yaml").read_text())
    st.markdown(f"**Policy** v{policy['version']} · default `{policy['default_effect']}` · first matching rule wins")
    st.dataframe(pd.DataFrame([{"rule": r["id"], "effect": r["effect"], "when": str(r["when"]),
                                "description": r.get("description", "")} for r in policy["rules"]]),
                 hide_index=True, width="stretch")
    st.dataframe(pd.DataFrame([{"role": k, "permitted_tools": ", ".join(v)}
                               for k, v in policy["role_permissions"].items()]),
                 hide_index=True, width="stretch")


# --- main ------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="IT Helpdesk", page_icon="🛠️", layout="wide")
    init_state()
    if problem := api_key_problem():
        st.error(problem, icon="🔑")
        st.stop()
    customer_id, role, example = sidebar()
    st.title("IT Helpdesk")
    st.caption("Multi-agent helpdesk · LangGraph · OpenTelemetry · policy-gated tools with human approval")
    chat, traces, evals, registry = st.tabs(["💬 Chat", "🔭 Traces", "📊 Evaluation", "🗂️ Registry"])
    with chat:
        chat_tab(customer_id, role, example)
    with traces:
        traces_tab()
    with evals:
        eval_tab()
    with registry:
        registry_tab()
