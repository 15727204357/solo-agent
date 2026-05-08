from __future__ import annotations

from collections.abc import Callable
from typing import Any

SubagentFactory = Callable[..., Any]


class SubagentRegistry:
    """Registry for subagent runner factories."""

    def __init__(self):
        self._factories: dict[str, SubagentFactory] = {}

    def register(self, name: str, factory: SubagentFactory) -> None:
        self._factories[name] = factory

    def get(self, name: str) -> SubagentFactory:
        if name not in self._factories:
            raise ValueError(f"Unknown subagent type: {name}")
        return self._factories[name]

    def list_types(self) -> list[str]:
        return list(self._factories.keys())

    def has(self, name: str) -> bool:
        return name in self._factories


_builtin_registry = SubagentRegistry()


def get_builtin_registry() -> SubagentRegistry:
    return _builtin_registry
