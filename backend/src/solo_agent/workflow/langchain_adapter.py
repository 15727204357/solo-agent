from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from solo_agent.providers.base import ChatMessage, ProviderError, ProviderToolCall, ProviderToolSpec


def _lc_to_provider_message(msg: BaseMessage) -> ChatMessage:
    if isinstance(msg, HumanMessage):
        return ChatMessage(role="user", content=_string_content(msg.content))
    elif isinstance(msg, AIMessage):
        return ChatMessage(
            role="assistant",
            content=_string_content(msg.content),
            tool_calls=tuple(_lc_tool_call_to_provider(tool_call) for tool_call in getattr(msg, "tool_calls", []) or []),
        )
    elif isinstance(msg, SystemMessage):
        return ChatMessage(role="system", content=_string_content(msg.content))
    elif isinstance(msg, ToolMessage):
        tool_call_id = getattr(msg, "tool_call_id", "")
        return ChatMessage(role="tool", content=_string_content(msg.content), tool_call_id=tool_call_id)
    else:
        role = getattr(msg, "type", "user")
        return ChatMessage(
            role=role if role in ("user", "assistant", "system") else "user",
            content=_string_content(msg.content),
        )


def _string_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _lc_tool_call_to_provider(tool_call: Any) -> ProviderToolCall:
    if isinstance(tool_call, dict):
        return ProviderToolCall(
            id=str(tool_call.get("id") or ""),
            name=str(tool_call.get("name") or ""),
            arguments=dict(tool_call.get("args") or {}),
        )
    return ProviderToolCall(
        id=str(getattr(tool_call, "id", "") or ""),
        name=str(getattr(tool_call, "name", "") or ""),
        arguments=dict(getattr(tool_call, "args", {}) or {}),
    )


def _provider_tool_call_to_lc(tool_call: ProviderToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "name": tool_call.name,
        "args": dict(tool_call.arguments),
    }


def _tool_to_openai(tool: Any) -> ProviderToolSpec:
    if isinstance(tool, dict):
        if tool.get("type") == "function":
            return dict(tool)
        if "name" in tool:
            return {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                },
            }

    name = getattr(tool, "name", None)
    if not name:
        raise ValueError(f"Tool is missing a name: {tool!r}")
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is None:
        parameters: dict[str, Any] = {"type": "object", "properties": {}}
    elif hasattr(args_schema, "model_json_schema"):
        parameters = args_schema.model_json_schema()
    elif hasattr(args_schema, "schema"):
        parameters = args_schema.schema()
    else:
        parameters = {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": getattr(tool, "description", "") or "",
            "parameters": parameters,
        },
    }


class LangChainChatAdapter(BaseChatModel):
    """将现有 ChatProvider 包装为 LangChain BaseChatModel。"""

    provider: Any
    temperature: float = 0.2
    max_tokens: int = 1400
    bound_tools: tuple[ProviderToolSpec, ...] = Field(default_factory=tuple)
    tool_choice: Any = None

    def __init__(self, *, provider: Any, temperature: float = 0.2, max_tokens: int = 1400, **kwargs: Any):
        super().__init__(provider=provider, temperature=temperature, max_tokens=max_tokens, **kwargs)
        self._provider_name = getattr(provider, "name", "unknown")

    @property
    def _llm_type(self) -> str:
        return f"solo-agent-{getattr(self.provider, 'name', 'unknown')}"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "provider": getattr(self.provider, "name", "unknown"),
            "model": getattr(self.provider, "model", "unknown"),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LangChainChatAdapter:
        if not getattr(self.provider, "supports_tool_calling", False):
            name = getattr(self.provider, "name", "unknown")
            raise ProviderError(f"{name} provider does not support tool calling")
        return self.model_copy(
            update={
                "bound_tools": tuple(_tool_to_openai(tool) for tool in tools),
                "tool_choice": tool_choice,
            }
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        provider_messages = [_lc_to_provider_message(m) for m in messages]
        if self.bound_tools:
            if not hasattr(self.provider, "complete_message"):
                name = getattr(self.provider, "name", "unknown")
                raise ProviderError(f"{name} provider does not support tool calling")
            response = await self.provider.complete_message(
                provider_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=self.bound_tools,
                tool_choice=self.tool_choice,
            )
            message = AIMessage(
                content=response.content,
                tool_calls=[_provider_tool_call_to_lc(tool_call) for tool_call in response.tool_calls],
                response_metadata={
                    "finish_reason": response.finish_reason,
                    "raw": response.raw,
                },
            )
        else:
            content = await self.provider.complete(
                provider_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:  # noqa: F821
        from langchain_core.outputs import ChatGenerationChunk

        provider_messages = [_lc_to_provider_message(m) for m in messages]
        async for chunk in self.provider.stream_chat(
            provider_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=self.bound_tools or None,
            tool_choice=self.tool_choice,
        ):
            if chunk.content:
                yield ChatGenerationChunk(message=AIMessageChunk(content=chunk.content))  # noqa: F821
            for tool_call in chunk.tool_calls:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        additional_kwargs={
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.name,
                                        "arguments": tool_call.raw_arguments or json.dumps(tool_call.arguments),
                                    },
                                }
                            ]
                        },
                    )
                )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("Use async _agenerate")
