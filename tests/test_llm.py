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
