import copy

import pytest
from click.testing import CliRunner

from helpdesk.app import HelpdeskApp
from helpdesk.eval import baseline as bl
from helpdesk.eval import runner, scorers
from helpdesk.session_store import SessionStore
from helpdesk.tools import kb
from tests.fakes import CUSTOMER, GROUNDED, KB_HITS, FakeLLM, draft, plan

ACCESS_TICKET = {
    "ticket_id": "T1", "customer_id": CUSTOMER, "initial_message": "I am locked out of my account",
    "issue_type": "account_access", "scenario_key": "access_locked",
    "resolution_summary": "Reset account credentials and confirmed successful login.",
}
HOWTO_TICKET = {
    "ticket_id": "T2", "customer_id": CUSTOMER, "initial_message": "How do I export data?",
    "issue_type": "how_to", "scenario_key": "howto_export_data",
    "resolution_summary": "Provided step-by-step instructions and relevant documentation.",
}


def kb_result(*ids):
    hits = [h for h in KB_HITS if h["article_id"] in ids]
    return {"tool": "search_kb", "status": "ok", "output": {"hits": hits}}


ACCOUNT_RESULT = {"tool": "get_account_status", "status": "ok", "output": {}}


# --- deterministic scorers ---

def test_classification():
    assert scorers.score_classification(HOWTO_TICKET, {"plan": {"issue_type": "how_to"}}) == 1.0
    assert scorers.score_classification(HOWTO_TICKET, {"plan": {"issue_type": "bug"}}) == 0.0
    assert scorers.score_classification(HOWTO_TICKET, {}) == 0.0


def test_resolution_match_uses_first_cited_article():
    tools = [kb_result("KB_HOW_TO", "KB_BUG")]
    good = {"tool_results": tools, "draft": {"cited_articles": ["KB_HOW_TO", "KB_BUG"]}}
    bad = {"tool_results": tools, "draft": {"cited_articles": ["KB_BUG"]}}
    none = {"tool_results": tools, "draft": {"cited_articles": []}}
    assert scorers.score_resolution_match(HOWTO_TICKET, good) == 1.0
    assert scorers.score_resolution_match(HOWTO_TICKET, bad) == 0.0
    assert scorers.score_resolution_match(HOWTO_TICKET, none) == 0.0


@pytest.mark.parametrize(
    "ticket, action, looked_up, expected",
    [
        (ACCESS_TICKET, "reset_credentials", True, 1.0),
        (ACCESS_TICKET, "reset_credentials", False, 0.5),  # missing lookup
        (ACCESS_TICKET, "create_ticket", True, 0.5),       # wrong action
        (HOWTO_TICKET, "none", False, 1.0),
        (HOWTO_TICKET, "none", True, 0.5),                 # unneeded lookup
        (HOWTO_TICKET, "issue_refund", True, 0.0),
    ],
)
def test_tool_use(ticket, action, looked_up, expected):
    state = {"draft": {"action": action}, "tool_results": [ACCOUNT_RESULT] if looked_up else []}
    score, _ = scorers.score_tool_use(ticket, state)
    assert score == expected


def test_scenario_override_for_compliance_question():
    ticket = {**ACCESS_TICKET, "issue_type": "security_concern", "scenario_key": "security_compliance"}
    state = {"draft": {"action": "none"}, "tool_results": [ACCOUNT_RESULT]}
    assert scorers.score_tool_use(ticket, state)[0] == 1.0


# --- baseline / regression ---

def make_run(correctness=0.9, latency=1000.0, per_ticket=(1.0, 1.0)):
    return {
        "run_id": "r",
        "summary": {"correctness": correctness, "groundedness": 0.93, "tool_use": 0.8, "classification": 1.0,
                    "resolution_match": 1.0, "latency_p50_ms": latency, "tokens_avg": 1500},
        "tickets": [{"ticket_id": f"T{i}", "scores": {"correctness": c}} for i, c in enumerate(per_ticket)],
    }


def test_identical_runs_have_no_regression():
    assert not bl.compare(make_run(), make_run()).has_regression


def test_quality_drop_beyond_tolerance_flagged():
    report = bl.compare(make_run(correctness=0.8), make_run(correctness=0.9), tolerance=0.05)
    assert [r.metric for r in report.regressions] == ["correctness"]
    assert report.has_regression


def test_drop_equal_to_tolerance_not_flagged():
    cur, base = make_run(), make_run()
    cur["summary"]["groundedness"], base["summary"]["groundedness"] = 0.88, 0.93
    assert not bl.compare(cur, base, tolerance=0.05).regressions


def test_latency_growth_flagged():
    report = bl.compare(make_run(latency=2000), make_run(latency=1000), cost_tolerance=0.5)
    assert [r.metric for r in report.regressions] == ["latency_p50_ms"]


def test_newly_failing_ticket_flagged_even_if_average_holds():
    report = bl.compare(make_run(per_ticket=(0.0, 1.0)), make_run(per_ticket=(1.0, 1.0)))
    assert report.newly_failing == ["T0"]
    assert report.has_regression


def test_improvement_reported():
    report = bl.compare(make_run(correctness=1.0), make_run(correctness=0.8))
    assert report.improvements == ["correctness +0.200"]
    assert not report.has_regression


def test_config_changes_reported():
    cur, base = make_run(), make_run()
    cur["config"] = {"judge_prompt_version": "1.1", "kb_top_k": 3}
    base["config"] = {"judge_prompt_version": "1.0", "kb_top_k": 3}
    assert bl.compare(cur, base).config_changes == ["judge_prompt_version: 1.0 -> 1.1"]


def test_baseline_roundtrip(tmp_path):
    path = tmp_path / "baseline.json"
    assert bl.load_baseline(path) is None
    bl.save_baseline(make_run(), path)
    assert bl.load_baseline(path)["summary"]["correctness"] == 0.9


# --- runner end-to-end (fake LLMs, stub KB) ---

def scripted_agents() -> FakeLLM:
    def planner(user):
        return plan("account_access", needs_account=True) if "locked" in user else plan("how_to")

    def reasoning(user):
        if "locked" in user:
            return draft("reset_credentials", cited=["KB_ACCOUNT_ACCESS"])
        return draft(cited=["KB_HOW_TO"])

    return FakeLLM({"planner": planner, "reasoning": reasoning, "reflection": GROUNDED})


def test_run_eval_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(kb, "search", lambda query, top_k=3: copy.deepcopy(KB_HITS[:top_k]))
    app = HelpdeskApp(llm=scripted_agents(), store=SessionStore(tmp_path))
    judge = FakeLLM({
        "judge_correctness": {"score": 1.0, "reason": "same"},
        "judge_groundedness": {"score": 0.5, "reason": "minor"},
    })
    seen = []
    run = runner.run_eval([ACCESS_TICKET, HOWTO_TICKET], app=app, judge_llm=judge, concurrency=1,
                          on_result=seen.append)

    assert [r["ticket_id"] for r in seen] == ["T1", "T2"]
    access, howto = run["tickets"]
    assert access["hitl"] is True and access["outcome"] == "action_completed"  # auto-approved
    assert "reset_credentials:ok" in access["tools"]
    assert access["scores"] == {"correctness": 1.0, "groundedness": 0.5, "tool_use": 1.0,
                                "classification": 1.0, "resolution_match": 1.0}
    assert howto["outcome"] == "resolved"
    assert howto["scores"]["tool_use"] == 1.0  # no action, no account lookup: both appropriate
    assert access["tokens"] == 15 * access["llm_calls"] and access["llm_calls"] == 3  # judge calls excluded

    s = run["summary"]
    assert s["tickets"] == 2 and s["hitl_rate"] == 0.5 and s["correctness"] == 1.0
    assert run["config"]["judge_prompt_version"] == scorers.JUDGE_PROMPT_VERSION


# --- CLI exit code ---

def test_cli_eval_exits_1_on_regression(monkeypatch):
    import helpdesk.config
    from helpdesk.cli import cli

    monkeypatch.setattr(helpdesk.config, "api_key_problem", lambda: None)
    monkeypatch.setattr(runner, "load_golden", lambda split: [HOWTO_TICKET])
    monkeypatch.setattr(runner, "run_eval", lambda tickets, **kw: {**make_run(correctness=0.5), "run_id": "cur"} | {
        "summary": {**make_run(correctness=0.5)["summary"], "tickets": 1, "hitl_rate": 0, "error_rate": 0,
                    "latency_p95_ms": 1, "outcomes": {}, "correctness_by_issue_type": {}}})
    monkeypatch.setattr(runner, "save_run", lambda run: "/dev/null")
    monkeypatch.setattr(bl, "load_baseline", lambda path=None: make_run(correctness=0.9))

    result = CliRunner().invoke(cli, ["eval"])
    assert result.exit_code == 1, result.output
    assert "REGRESSION correctness" in result.output
