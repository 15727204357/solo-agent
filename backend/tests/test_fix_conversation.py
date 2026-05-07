# -*- coding: utf-8 -*-
"""ConversationRepair / fix_conversation 管道单元测试。"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from solo_agent.agent.fix_conversation import (
    ConversationRepair,
    fix_conversation,
)


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


@pytest.fixture
def repair() -> ConversationRepair:
    return ConversationRepair()


def _msg(role: str, content: str, **extra: Any) -> Dict[str, Any]:
    """快捷构造测试消息。"""
    msg: Dict[str, Any] = {"role": role, "content": content}
    msg.update(extra)
    return msg


def _tool(source: str, content: str, **extra: Any) -> Dict[str, Any]:
    """快捷构造工具结果消息。"""
    msg: Dict[str, Any] = {"source": source, "content": content}
    msg.update(extra)
    return msg


# ---------------------------------------------------------------------------
# 步骤 2: trim_assistant_text_whitespace
# ---------------------------------------------------------------------------


class TestTrimWhitespace:
    def test_trim_trailing_spaces(self, repair: ConversationRepair) -> None:
        msgs = [
            _msg("assistant", "hello   "),
            _msg("user", "hi"),
        ]
        result = repair.fix_conversation(msgs)
        assert result[0]["content"] == "hello"
        assert result[1]["content"] == "hi"

    def test_trim_trailing_newlines(self, repair: ConversationRepair) -> None:
        msgs = [_msg("assistant", "line1\nline2\n\n")]
        result = repair.fix_conversation(msgs)
        assert result[0]["content"] == "line1\nline2"

    def test_preserve_user_whitespace(self, repair: ConversationRepair) -> None:
        msgs = [_msg("user", "  keep spaces  ")]
        result = repair.fix_conversation(msgs)
        # user 消息保留原始空白（不去除首尾）
        assert result[0]["content"] == "  keep spaces  "


# ---------------------------------------------------------------------------
# 步骤 3: remove_empty_messages
# ---------------------------------------------------------------------------


class TestRemoveEmpty:
    def test_remove_empty_string(self, repair: ConversationRepair) -> None:
        msgs = [
            _msg("user", "hello"),
            _msg("assistant", ""),
            _msg("user", "world"),
        ]
        result = repair.fix_conversation(msgs)
        # assistant "" 被 remove_empty 移除后，两个 user 被 merge_consecutive 合并
        assert len(result) == 1
        assert result[0]["content"] == "hello\nworld"

    def test_remove_whitespace_only(self, repair: ConversationRepair) -> None:
        msgs = [
            _msg("user", "   "),
            _msg("assistant", "\n\t"),
        ]
        result = repair.fix_conversation(msgs)
        # 全部移除后 populate 填充 "Hello"
        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    def test_keep_system_messages(self, repair: ConversationRepair) -> None:
        msgs = [
            _msg("system", "System instruction"),
            _msg("user", "hello"),
        ]
        result = repair.fix_conversation(msgs)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 步骤 4: fix_empty_tool_results
# ---------------------------------------------------------------------------


class TestFixEmptyToolResults:
    def test_empty_tool_result_gets_placeholder(self, repair: ConversationRepair) -> None:
        msgs = [
            _msg("user", "run command"),
            _tool("tool:bash", ""),
        ]
        result = repair.fix_conversation(msgs)
        assert result[1]["content"] == "(empty result)"

    def test_whitespace_tool_result_gets_placeholder(self, repair: ConversationRepair) -> None:
        msgs = [
            _msg("user", "run command"),
            _tool("tool:bash", "   "),
        ]
        result = repair.fix_conversation(msgs)
        assert result[1]["content"] == "(empty result)"

    def test_non_empty_tool_result_untouched(self, repair: ConversationRepair) -> None:
        msgs = [
            _msg("user", "run command"),
            _tool("tool:bash", "output"),
        ]
        result = repair.fix_conversation(msgs)
        assert result[1]["content"] == "output"

    def test_regular_message_untouched(self, repair: ConversationRepair) -> None:
        msgs = [_msg("user", "")]
        result = repair.fix_conversation(msgs)
        # 空 user 消息被 remove_empty 移除，populate 填充 "Hello"
        assert len(result) == 1
        assert result[0]["content"] == "Hello"


# ---------------------------------------------------------------------------
# 步骤 5: fix_tool_calling
# ---------------------------------------------------------------------------


class TestFixToolCalling:
    def test_remove_empty_tool_calls(self, repair: ConversationRepair) -> None:
        msgs = [
            _msg("user", "do something"),
            _msg("assistant", "ok", tool_calls=[]),
            _tool("tool:bash", "output"),
        ]
        result = repair.fix_conversation(msgs)
        # assistant 消息的 tool_calls 为空列表，应被移除
        assert "tool_calls" not in result[1]

    def test_keep_tool_calls_with_entries(self, repair: ConversationRepair) -> None:
        msgs = [
            _msg("user", "run test"),
            _msg(
                "assistant",
                "running",
                tool_calls=[{"id": "1", "name": "pytest"}],
            ),
        ]
        result = repair.fix_conversation(msgs)
        # 有内容的 tool_calls 保留
        assert "tool_calls" in result[1]


# ---------------------------------------------------------------------------
# 步骤 6: merge_consecutive_messages
# ---------------------------------------------------------------------------


class TestMergeConsecutive:
    def test_merge_two_user_messages(self, repair: ConversationRepair) -> None:
        msgs = [
            _msg("user", "first"),
            _msg("user", "second"),
        ]
        result = repair.fix_conversation(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "first\nsecond"

    def test_merge_two_assistant_messages(self, repair: ConversationRepair) -> None:
        msgs = [
            _msg("user", "hi"),
            _msg("assistant", "hello"),
            _msg("assistant", "world"),
        ]
        result = repair.fix_conversation(msgs)
        assert len(result) == 2
        assert result[1]["content"] == "hello\nworld"

    def test_no_merge_different_roles(self, repair: ConversationRepair) -> None:
        msgs = [
            _msg("user", "hi"),
            _msg("assistant", "hello"),
        ]
        result = repair.fix_conversation(msgs)
        assert len(result) == 2

    def test_no_merge_across_tool(self, repair: ConversationRepair) -> None:
        msgs = [
            _msg("user", "run test"),
            _tool("tool:bash", "passed"),
            _msg("user", "next"),
        ]
        result = repair.fix_conversation(msgs)
        # 工具结果前后不合并
        assert len(result) == 3

    def test_no_merge_assistant_with_tool_calls(self, repair: ConversationRepair) -> None:
        msgs = [
            _msg("assistant", "planning", tool_calls=[{"id": "1"}]),
            _msg("assistant", "more"),
        ]
        result = repair.fix_conversation(msgs)
        # 有 tool_calls 的 assistant 不参与合并
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 步骤 7: fix_lead_trail
# ---------------------------------------------------------------------------


class TestFixLeadTrail:
    def test_starts_with_user(self, repair: ConversationRepair) -> None:
        msgs = [
            _msg("system", "sys"),
            _msg("assistant", "before"),
            _msg("user", "actual start"),
            _msg("assistant", "reply"),
        ]
        result = repair.fix_conversation(msgs)
        # 新的 fix_lead_trail 保留 system 前导和 assistant 消息
        assert len(result) == 4
        assert result[2]["role"] == "user"
        assert result[2]["content"] == "actual start"

    def test_ends_with_user_or_assistant(self, repair: ConversationRepair) -> None:
        msgs = [
            _msg("user", "hi"),
            _msg("assistant", "hello"),
            _tool("tool:bash", "result"),
            _msg("system", "trailing"),
        ]
        result = repair.fix_conversation(msgs)
        # 末尾 system "trailing" 被移除，剩 3 条：user, assistant, tool
        assert len(result) == 3
        # 工具消息保留在末尾
        assert result[-1]["source"] == "tool:bash"


# ---------------------------------------------------------------------------
# 步骤 8: populate_if_empty
# ---------------------------------------------------------------------------


class TestPopulateIfEmpty:
    def test_empty_list_gets_hello(self, repair: ConversationRepair) -> None:
        result = repair.fix_conversation([])
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_non_empty_list_unchanged(self, repair: ConversationRepair) -> None:
        msgs = [_msg("user", "real question")]
        result = repair.fix_conversation(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "real question"


# ---------------------------------------------------------------------------
# 集成测试：完整管道
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_all_steps_in_order(self, repair: ConversationRepair) -> None:
        """测试 8 步管道按顺序执行后消息结构正确。"""
        msgs = [
            _msg("system", "prelude"),       # lead 保留（合法 system 前导）
            _msg("user", "  first  "),       # 保留
            _msg("assistant", "hello   \n"), # trim → "hello"
            _msg("assistant", ""),           # remove_empty 移除
            _msg("user", ""),                # remove_empty 移除
            _msg("user", "second"),          # merge 与上一条 user 合并
            _tool("tool:bash", ""),          # tool 保留 → fix_empty → "(empty result)"
            _msg("system", "end"),           # trail 移除（末尾 system）
        ]
        result = repair.fix_conversation(msgs)

        # 期望结果：
        # 1. system "prelude" → 保留
        # 2. user "  first  " → 保留（与 second 之间有 assistant 隔开，不合并）
        # 3. assistant "hello" → 去尾部空白
        # 4. user "second" → 保留
        # 5. tool:bash "(empty result)" → 空占位
        #    system "end" → trail 移除
        assert len(result) == 5
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "prelude"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "hello"
        assert result[-1]["source"] == "tool:bash"
        assert result[-1]["content"] == "(empty result)"


class TestDisabledFixes:
    def test_disable_specific_step(self) -> None:
        repair = ConversationRepair(disabled_fixes={"remove_empty_messages"})
        msgs = [
            _msg("user", "hi"),
            _msg("assistant", ""),
        ]
        result = repair.fix_conversation(msgs)
        # remove_empty_messages 被禁用，空消息保留
        assert len(result) == 2

    def test_active_fixes_property(self) -> None:
        repair = ConversationRepair(disabled_fixes={"merge_consecutive_messages"})
        assert "merge_consecutive_messages" not in repair.active_fixes
        assert "trim_assistant_text_whitespace" in repair.active_fixes


class TestInputImmutability:
    def test_original_messages_not_modified(self) -> None:
        msgs = [
            _msg("user", "hello"),
            _msg("assistant", "world   "),
        ]
        original = [dict(m) for m in msgs]
        repair = ConversationRepair()
        repair.fix_conversation(msgs)
        # 原始输入不应被修改
        for i, msg in enumerate(msgs):
            assert msg["content"] == original[i]["content"]


class TestConvenienceFunction:
    def test_fix_conversation_function(self) -> None:
        msgs = [
            _msg("user", "hi"),
            _msg("assistant", ""),
        ]
        result = fix_conversation(msgs)
        assert len(result) == 1  # 空 assistant 被移除
        assert result[0]["content"] == "hi"


# ---------------------------------------------------------------------------
# 边界情况
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_message(self, repair: ConversationRepair) -> None:
        result = repair.fix_conversation([_msg("user", "hello")])
        assert len(result) == 1

    def test_only_tool_messages(self, repair: ConversationRepair) -> None:
        msgs = [
            _tool("tool:a", "result1"),
            _tool("tool:b", "result2"),
        ]
        result = repair.fix_conversation(msgs)
        # tool 消息不含 role，不被 fix_lead_trail 移除（除非全是 system）
        # 工具结果正常保留
        assert len(result) == 2
        assert result[0]["source"] == "tool:a"

    def test_only_system_messages(self, repair: ConversationRepair) -> None:
        msgs = [
            _msg("system", "instruction"),
        ]
        result = repair.fix_conversation(msgs)
        # 只有 system 消息，lead_fix 移除，populate 填充
        assert len(result) == 0 or result[0]["content"] == "Hello"
