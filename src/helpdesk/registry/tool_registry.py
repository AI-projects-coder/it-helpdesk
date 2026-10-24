"""Loads config/tools.yaml; tools are looked up by name."""
import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from helpdesk.config import CONFIG_DIR
from helpdesk.tools import schemas


class UnknownToolError(KeyError):
    pass


class ToolSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    version: str
    description: str
    risk: Literal["low", "medium", "high"]
    approval_required: bool
    timeout_s: float = Field(default=10, gt=0)
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[BaseModel], BaseModel]

    def describe(self) -> dict:
        """Serializable view (for CLI listing / LLM tool descriptions)."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "risk": self.risk,
            "approval_required": self.approval_required,
            "parameters": self.input_model.model_json_schema(),
        }


def _resolve_handler(ref: str) -> Callable:
    module_name, _, func_name = ref.partition(":")
    if not func_name:
        raise ValueError(f"handler must be 'module:function', got {ref!r}")
    return getattr(importlib.import_module(module_name), func_name)


def _resolve_schema(name: str) -> type[BaseModel]:
    model = getattr(schemas, name, None)
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise ValueError(f"unknown schema {name!r} in helpdesk.tools.schemas")
    return model


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    @classmethod
    def from_yaml(cls, path: Path = CONFIG_DIR / "tools.yaml") -> "ToolRegistry":
        registry = cls()
        for entry in yaml.safe_load(path.read_text())["tools"]:
            registry.register(ToolSpec(
                name=entry["name"],
                version=entry["version"],
                description=entry["description"],
                risk=entry["risk"],
                approval_required=entry["approval_required"],
                timeout_s=entry.get("timeout_s", 10),
                input_model=_resolve_schema(entry["input_schema"]),
                output_model=_resolve_schema(entry["output_schema"]),
                handler=_resolve_handler(entry["handler"]),
            ))
        return registry

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} already registered")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise UnknownToolError(name) from None

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())
