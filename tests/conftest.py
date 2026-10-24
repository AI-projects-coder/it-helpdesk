import pytest

from helpdesk.policy import PolicyEngine
from helpdesk.registry import AIRegistry, ToolRegistry
from helpdesk.tools.executor import CallContext, ToolExecutor


@pytest.fixture(scope="session")
def tools() -> ToolRegistry:
    return ToolRegistry.from_yaml()


@pytest.fixture(scope="session")
def agents() -> AIRegistry:
    return AIRegistry.from_yaml()


@pytest.fixture(scope="session")
def policy(tools, agents) -> PolicyEngine:
    return PolicyEngine.from_yaml(tools, agents)


@pytest.fixture
def executor(tools, policy) -> ToolExecutor:
    return ToolExecutor(tools, policy, retry_wait_s=0)


@pytest.fixture
def action_ctx() -> CallContext:
    return CallContext(session_id="test", agent="action_ticket", role="user", facts={"account_found": True})
