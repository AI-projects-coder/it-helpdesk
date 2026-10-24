"""Mock account / SLA status derived from ticket metadata."""
import json
from functools import lru_cache

from helpdesk.config import get_settings
from helpdesk.tools.schemas import AccountStatusInput, AccountStatusOutput


@lru_cache
def _accounts() -> dict[str, dict]:
    return json.loads(get_settings().accounts_path.read_text())


def get_account_status(inp: AccountStatusInput) -> AccountStatusOutput:
    acct = _accounts().get(inp.customer_id)
    if acct is None:
        return AccountStatusOutput(customer_id=inp.customer_id, found=False)
    return AccountStatusOutput(
        customer_id=inp.customer_id,
        found=True,
        account_status=acct["account_status"],
        sla_plan=acct["sla_plan"],
        segment=acct["segment"],
        mfa_enabled=acct["mfa_enabled"],
    )
