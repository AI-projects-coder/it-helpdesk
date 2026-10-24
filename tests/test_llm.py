from dataclasses import replace

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from helpdesk.config import api_key_problem, get_settings
from helpdesk.llm import create_chat_model


@pytest.fixture
def settings():
    return replace(get_settings(), openai_api_key="sk-test", anthropic_api_key="sk-ant-test")


def test_openai_provider(settings):
    model = create_chat_model("openai", "gpt-4o-mini", settings)
    assert isinstance(model, ChatOpenAI)
    assert model.temperature == 0


def test_anthropic_provider(settings):
    assert isinstance(create_chat_model("anthropic", "claude-haiku-5-5", settings), ChatAnthropic)


def test_unknown_provider_rejected(settings):
    with pytest.raises(ValueError, match="unsupported LLM_PROVIDER"):
        create_chat_model("mystery", "x", settings)


def test_api_key_problem(settings):
    assert api_key_problem(settings) is None
    assert "OPENAI_API_KEY" in api_key_problem(replace(settings, openai_api_key="sk-..."))
    assert "OPENAI_API_KEY" in api_key_problem(replace(settings, openai_api_key=None))
    assert "ANTHROPIC_API_KEY" in api_key_problem(replace(settings, llm_provider="anthropic", anthropic_api_key=""))


# --- structured output resilience ---

class _Runnable:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return self.outputs.pop(0)


class _Chat:
    def __init__(self, runnable):
        self.runnable = runnable

    def with_structured_output(self, schema, **kw):
        return self.runnable


def _llm_with(outputs):
    from helpdesk.llm import LangChainLLM

    llm = LangChainLLM.__new__(LangChainLLM)  # skip provider client construction
    llm.provider, llm.model = "openai", "test"
    runnable = _Runnable(outputs)
    llm._chat = _Chat(runnable)
    return llm, runnable


def test_unparseable_output_is_retried_once():
    from helpdesk.agents.reflection import Reflection

    good = Reflection(grounded=True, unsupported_claims=[], feedback="")
    llm, runnable = _llm_with([{"parsed": None, "parsing_error": "bad"}, {"parsed": good, "raw": None}])
    assert llm.structured(Reflection, "s", "u", name="reflection").parsed == good
    assert runnable.calls == 2


def test_still_unparseable_raises():
    from helpdesk.agents.reflection import Reflection
    from helpdesk.llm import LLMOutputError

    llm, _ = _llm_with([{"parsed": None, "parsing_error": "bad"}] * 2)
    with pytest.raises(LLMOutputError):
        llm.structured(Reflection, "s", "u", name="reflection")


def test_string_where_list_expected_is_accepted():
    """The exact failure seen live: unsupported_claims came back as one string."""
    from helpdesk.agents.planner import Plan
    from helpdesk.agents.reasoning import Draft
    from helpdesk.agents.reflection import Reflection

    r = Reflection.model_validate({"grounded": False, "unsupported_claims": "The draft implies that t...",
                                   "feedback": "x"})
    assert r.unsupported_claims == ["The draft implies that t..."]
    assert Reflection.model_validate({"grounded": True, "unsupported_claims": None, "feedback": ""}
                                     ).unsupported_claims == []
    d = Draft.model_validate({"resolution_summary": "r", "customer_reply": "c", "cited_articles": "KB_BUG",
                              "action": "none", "action_reason": "x"})
    assert d.cited_articles == ["KB_BUG"]
    p = Plan.model_validate({"issue_type": "bug", "summary": "s", "urgency": "low", "needs_account_check": False,
                             "steps": "search the KB"})
    assert p.steps == ["search the KB"] and p.in_scope is True
