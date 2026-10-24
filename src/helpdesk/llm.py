"""LLM provider factory (openai | anthropic), switchable via config.

Agents depend on the small `LLMClient` protocol, not on a provider SDK, so
tests can swap in a scripted fake and providers can change via LLM_PROVIDER.
"""
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from helpdesk.config import Settings, get_settings
from helpdesk.telemetry import span

T = TypeVar("T", bound=BaseModel)

REQUEST_TIMEOUT_S = 30
MAX_RETRIES = 2  # provider SDK retries transient errors with exponential backoff


class LLMOutputError(RuntimeError):
    pass


@dataclass
class LLMResult(Generic[T]):
    parsed: T
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    def structured(self, schema: type[T], system: str, user: str, *, name: str) -> LLMResult[T]: ...


def create_chat_model(provider: str, model: str, settings: Settings | None = None) -> BaseChatModel:
    s = settings or get_settings()
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model, api_key=s.openai_api_key, base_url=s.openai_base_url,
            temperature=0, timeout=REQUEST_TIMEOUT_S, max_retries=MAX_RETRIES,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model, api_key=s.anthropic_api_key,
            temperature=0, timeout=REQUEST_TIMEOUT_S, max_retries=MAX_RETRIES,
        )
    raise ValueError(f"unsupported LLM_PROVIDER {provider!r} (expected 'openai' or 'anthropic')")


class LangChainLLM:
    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        self._chat = create_chat_model(provider, model)

    def structured(self, schema: type[T], system: str, user: str, *, name: str) -> LLMResult[T]:
        with span(f"llm.{name}", llm_provider=self.provider, llm_model=self.model) as sp:
            kwargs = {"method": "function_calling"} if self.provider == "openai" else {}
            runnable = self._chat.with_structured_output(schema, include_raw=True, **kwargs)
            out = runnable.invoke([SystemMessage(system), HumanMessage(user)])
            if out.get("parsing_error") or out.get("parsed") is None:
                raise LLMOutputError(f"{name}: could not parse {schema.__name__}: {out.get('parsing_error')}")
            usage = getattr(out["raw"], "usage_metadata", None) or {}
            result = LLMResult(
                parsed=out["parsed"], model=self.model,
                input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
            )
            sp.set_attribute("input_tokens", result.input_tokens)
            sp.set_attribute("output_tokens", result.output_tokens)
            sp.set_attribute("token_count", result.input_tokens + result.output_tokens)
            return result


def get_llm(settings: Settings | None = None) -> LLMClient:
    s = settings or get_settings()
    return LangChainLLM(s.llm_provider, s.llm_model)
