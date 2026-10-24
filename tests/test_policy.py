import pytest

from helpdesk.policy import PolicyEngine

CUST = {"customer_id": "CUST_05883"}
FOUND = {"account_found": True}


@pytest.mark.parametrize(
    "tool, agent",
    [("search_kb", "knowledge_search"), ("get_account_status", "access_check"), ("create_ticket", "action_ticket")],
)
def test_low_and_medium_risk_tools_allowed(policy, tool, agent):
    d = policy.evaluate(tool, agent, "user", CUST)
    assert d.effect == "allow"
    assert d.rule_id == "allow-low-medium-risk"


@pytest.mark.parametrize("tool", ["reset_credentials", "update_security_settings", "issue_refund"])
def test_high_risk_tools_require_approval(policy, tool):
    d = policy.evaluate(tool, "action_ticket", "user", {**CUST, "amount": 10}, FOUND)
    assert d.effect == "require_approval"
    assert d.rule_id == "approval-required-tools"


def test_unknown_tool_denied(policy):
    d = policy.evaluate("drop_database", "action_ticket", "admin", {})
    assert (d.effect, d.rule_id) == ("deny", "unknown-tool")


def test_least_privilege_denies_tool_outside_agent_scope(policy):
    d = policy.evaluate("reset_credentials", "knowledge_search", "admin", CUST)
    assert (d.effect, d.rule_id) == ("deny", "least-privilege")


def test_role_permission_denies_viewer(policy):
    d = policy.evaluate("get_account_status", "access_check", "viewer", CUST)
    assert (d.effect, d.rule_id) == ("deny", "role-permission")


def test_unknown_role_denied(policy):
    d = policy.evaluate("search_kb", "knowledge_search", "intruder", {})
    assert (d.effect, d.rule_id) == ("deny", "role-permission")


def test_admin_wildcard_still_needs_approval_for_high_risk(policy):
    d = policy.evaluate("reset_credentials", "action_ticket", "admin", CUST, FOUND)
    assert d.effect == "require_approval"


def test_refund_over_limit_denied_before_approval(policy):
    d = policy.evaluate("issue_refund", "action_ticket", "user", {**CUST, "amount": 900}, FOUND)
    assert (d.effect, d.rule_id) == ("deny", "refund-limit")


def test_refund_at_limit_needs_approval(policy):
    d = policy.evaluate("issue_refund", "action_ticket", "user", {**CUST, "amount": 500}, FOUND)
    assert d.effect == "require_approval"


def test_default_deny_when_no_rule_matches(tools, agents):
    engine = PolicyEngine(
        {"default_effect": "deny", "role_permissions": {"user": ["*"]}, "rules": []}, tools, agents
    )
    d = engine.evaluate("search_kb", "knowledge_search", "user", {})
    assert (d.effect, d.rule_id) == ("deny", "default")


@pytest.mark.parametrize("found", [False, None])
@pytest.mark.parametrize("tool", ["reset_credentials", "update_security_settings", "issue_refund"])
def test_account_actions_need_a_found_account(policy, tool, found):
    d = policy.evaluate(tool, "action_ticket", "admin", {**CUST, "amount": 10}, {"account_found": found})
    assert (d.effect, d.rule_id) == ("deny", "account-must-exist")


def test_ticket_creation_does_not_need_account_lookup(policy):
    d = policy.evaluate("create_ticket", "action_ticket", "user", CUST, {"account_found": None})
    assert d.effect == "allow"
