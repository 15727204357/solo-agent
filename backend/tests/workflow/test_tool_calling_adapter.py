from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from langchain_core.messages import HumanMessage
from solo_agent.providers.base import ChatMessage, ProviderChunk, ProviderError, ProviderResponse, ProviderToolCall
from solo_agent.providers.openai_compatible import OpenAICompatibleProvider
from solo_agent.workflow.langchain_adapter import LangChainChatAdapter
from solo_agent.workflow.sandbox.tool_adapter import build_langchain_tool


class ToolCallingProvider:
    name = "fake-tools"
    model = "fake-model"
    supports_tool_calling = True

    def __init__(self):
        self.calls = []

    async def complete(self, messages, *, temperature=None, max_tokens=None):
        return "plain"

    async def complete_message(
        self,
        messages,
        *,
        temperature=None,
        max_tokens=None,
        tools=None,
        tool_choice=None,
    ):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
            }
        )
        return ProviderResponse(
            content="",
            tool_calls=(
                ProviderToolCall(
                    id="call_1",
                    name="read_file",
                    arguments={"path": "README.md"},
                ),
            ),
            finish_reason="tool_calls",
        )

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature=None,
        max_tokens=None,
        tools=None,
        tool_choice=None,
    ) -> AsyncIterator[ProviderChunk]:
        yield ProviderChunk(content="stream")


class NoToolProvider(ToolCallingProvider):
    name = "fake-no-tools"
    supports_tool_calling = False


@pytest.mark.asyncio
async def test_langchain_adapter_bind_tools_returns_ai_message_tool_calls():
    provider = ToolCallingProvider()
    tool = build_langchain_tool("read_file", "read", lambda **kwargs: {}, {"path": "Path"})
    model = LangChainChatAdapter(provider=provider).bind_tools([tool], tool_choice="auto")

    result = await model._agenerate([HumanMessage(content="read README")])
    message = result.generations[0].message

    assert provider.calls[0]["tool_choice"] == "auto"
    assert provider.calls[0]["tools"][0]["function"]["name"] == "read_file"
    assert message.tool_calls == [{"name": "read_file", "args": {"path": "README.md"}, "id": "call_1", "type": "tool_call"}]


def test_langchain_adapter_bind_tools_raises_for_unsupported_provider():
    model = LangChainChatAdapter(provider=NoToolProvider())

    with pytest.raises(ProviderError, match="does not support tool calling"):
        model.bind_tools([{"name": "read_file", "parameters": {}}])


def test_openai_compatible_parses_completion_tool_calls():
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider.name = "openai"

    response = provider._parse_completion_response(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": "{\"path\":\"README.md\"}",
                                },
                            }
                        ],
                    },
                }
            ]
        }
    )

    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].name == "read_file"
    assert response.tool_calls[0].arguments == {"path": "README.md"}


def test_openai_compatible_parses_stream_tool_call_delta():
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider.name = "openai"

    chunk = provider._parse_sse_line(
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
        '"type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}]},'
        '"finish_reason":null}]}'
    )

    assert chunk is not None
    assert chunk.tool_calls[0].id == "call_1"
    assert chunk.tool_calls[0].arguments == {"path": "README.md"}
