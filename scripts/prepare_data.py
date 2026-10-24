"""Freezes the golden tickets CSV into data/golden_subset.json.

Also writes:
  data/accounts.json     - mock account/SLA records keyed by customer_id
  data/kb_articles.json  - one KB article per issue_type (resolution from the CSV)
and builds the Chroma index (skip with --no-index).

Only initial_message is used as agent input; resolution_summary and
agent_first_reply are kept as evaluation references. No model is trained.
"""
import argparse
import csv
import json
from collections import Counter
from dataclasses import replace

from helpdesk.config import get_settings

GOLDEN_FIELDS = [
    "ticket_id", "customer_id", "initial_message", "resolution_summary",
    "agent_first_reply", "issue_type", "product_area", "priority",
    "platform", "channel", "eval_split", "scenario_key",
]

# Hand-written descriptions of when each resolution applies. Deliberately not
# copied from ticket text so the KB doesn't contain the eval queries verbatim.
ARTICLE_META = {
    "account_access": (
        "Account access and login problems",
        "user cannot log in, password rejected, account locked out, or two-factor (2FA) codes not working.",
        ["login", "log", "password", "locked", "2fa", "two", "factor", "code", "access", "sign"],
    ),
    "billing_problem": (
        "Billing and invoice problems",
        "user was charged incorrectly or twice, invoice is wrong, or a plan change was billed incorrectly.",
        ["charged", "charge", "twice", "invoice", "refund", "billing", "payment", "downgrade", "plan"],
    ),
    "security_concern": (
        "Security concerns",
        "suspicious login activity, unrecognized devices, a possible vulnerability, or a security compliance question.",
        ["suspicious", "security", "vulnerability", "compliance", "unauthorized", "breach", "unknown", "device"],
    ),
    "bug": (
        "Product bugs",
        "the product shows an error, crashes, or fails to save changes.",
        ["error", "crash", "crashes", "bug", "broken", "saving", "save", "fails", "failing"],
    ),
    "performance": (
        "Performance degradation",
        "the product or API is slow, requests time out, or performance is degraded.",
        ["slow", "timeout", "timing", "latency", "performance", "degraded", "lag", "loading"],
    ),
    "how_to": (
        "How-to and configuration questions",
        "user asks how to set up, configure, or export something and needs instructions.",
        ["how", "configure", "setup", "set", "export", "instructions", "steps", "guide"],
    ),
    "feature_request": (
        "Feature requests",
        "user asks for a new capability, more customization, or support for something not yet available.",
        ["feature", "request", "would", "like", "wish", "add", "support", "customize", "advanced"],
    ),
    "other": (
        "General questions",
        "general account or workspace questions that don't fit another category.",
        ["general", "question", "workspace", "unsure", "check", "account"],
    ),
}

# Synthetic account status per scenario; everything else is "active".
SCENARIO_ACCOUNT_STATUS = {"access_locked": "locked"}


def load_rows(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_golden(rows: list[dict]) -> list[dict]:
    return [{k: r[k] for k in GOLDEN_FIELDS} for r in rows]


def build_accounts(rows: list[dict]) -> dict[str, dict]:
    return {
        r["customer_id"]: {
            "customer_id": r["customer_id"],
            "segment": r["customer_segment"],
            "sla_plan": r["sla_plan"],
            "region": r["region"],
            "platform": r["platform"],
            "account_status": SCENARIO_ACCOUNT_STATUS.get(r["scenario_key"], "active"),
            "mfa_enabled": r["scenario_key"] == "access_2fa",
        }
        for r in rows
    }


def build_articles(rows: list[dict]) -> list[dict]:
    resolutions = {}
    for r in rows:
        resolutions.setdefault(r["issue_type"], r["resolution_summary"])
    articles = []
    for issue_type, resolution in sorted(resolutions.items()):
        title, applies_when, keywords = ARTICLE_META[issue_type]
        articles.append({
            "article_id": f"KB_{issue_type.upper()}",
            "issue_type": issue_type,
            "title": title,
            "applies_when": applies_when,
            "resolution": resolution,
            "keywords": keywords,
        })
    return articles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-index", action="store_true", help="skip building the Chroma index")
    args = parser.parse_args()

    s = get_settings()
    rows = load_rows(s.raw_csv)

    golden = build_golden(rows)
    s.golden_path.write_text(json.dumps(golden, indent=2))
    print(f"golden subset: {len(golden)} tickets -> {s.golden_path}")
    print("  by issue_type:", dict(Counter(t["issue_type"] for t in golden)))

    accounts = build_accounts(rows)
    s.accounts_path.write_text(json.dumps(accounts, indent=2))
    print(f"accounts: {len(accounts)} -> {s.accounts_path}")

    articles = build_articles(rows)
    s.kb_articles_path.write_text(json.dumps(articles, indent=2))
    print(f"kb articles: {len(articles)} -> {s.kb_articles_path}")

    if not args.no_index:
        from helpdesk.config import api_key_problem
        from helpdesk.tools.kb import build_index

        if problem := api_key_problem(replace(s, llm_provider="openai")):  # embeddings always use OpenAI
            raise SystemExit(problem)
        print(f"chroma index: {build_index(articles)} docs -> {s.chroma_dir}")


if __name__ == "__main__":
    main()
