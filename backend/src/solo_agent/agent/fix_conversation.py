"""fix_conversation 对话修复管道。

参考 goose 的 fix_conversation 实现（crates/goose/src/conversation/mod.rs），
适配 solo-agent 的 ChatMessage / dict 消息格式，在 LLM 调用前清洗消息历史。

管道步骤（8 步按顺序执行）：
1. merge_text_content_items      — 合并不适用（solo-agent 使用简单 role/content 格式）
2. trim_assistant_text_whitespace — 去除 assistant 消息尾部空白
3. remove_empty_messages          — 移除空内容消息
4. fix_empty_tool_results         — 空工具结果添加占位文本
5. fix_tool_calling               — 移除孤立工具结果（无对应 tool_call 的 source:* 记录）
6. merge_consecutive_messages     — 合并相同 role 的连续消息
7. fix_lead_trail                 — 确保对话以 user 开始、以 user 结束
8. populate_if_empty              — 空对话填充 "Hello"

设计原则：
- 纯函数管道，无状态，无副作用
- 每步可独立开关（disabled_fixes 配置）
- 只操作 visible 消息（非 visible 消息保持原样）
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

# ---------------------------------------------------------------------------
# 消息类型辅助
# ---------------------------------------------------------------------------

def _is_assistant(msg: dict[str, Any]) -> bool:
    """判断消息 role 是否为 assistant。"""
    return msg.get("role", "") == "assistant"


def _is_user(msg: dict[str, Any]) -> bool:
    """判断消息 role 是否为 user 或 human。"""
    return msg.get("role", "") in ("user", "human")


def _is_system(msg: dict[str, Any]) -> bool:
    return msg.get("role", "") == "system"


def _is_tool(msg: dict[str, Any]) -> bool:
    """判断消息是否为工具结果（source 以 'tool:' 开头）。"""
    source = msg.get("source", "")
    return isinstance(source, str) and source.startswith("tool:")


def _is_tool_call(msg: dict[str, Any]) -> bool:
    """判断消息是否包含 tool_calls（assistant 消息中的工具调用）。"""
    return _is_assistant(msg) and "tool_calls" in msg


# ---------------------------------------------------------------------------
# 步骤实现
# ---------------------------------------------------------------------------


def _merge_text_content_items(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """步骤 1: 合并不适用。

    solo-agent 使用简单的 role/content 字符串格式（非 multi-block content），
    此步骤为保留位，直接返回原列表。
    """
    return messages


def _trim_assistant_text_whitespace(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """步骤 2: 去除 assistant 消息 content 尾部空白。"""
    result = []
    for msg in messages:
        msg = dict(msg)  # 浅拷贝
        if _is_assistant(msg) and isinstance(msg.get("content"), str):
            msg["content"] = msg["content"].rstrip()
        result.append(msg)
    return result


def _remove_empty_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """步骤 3: 移除 content 为空或仅空白字符的消息（保留工具结果消息）。"""
    return [
        msg
        for msg in messages
        if _is_tool(msg) or (
            isinstance(msg.get("content"), str) and msg["content"].strip()
        )
    ]


def _fix_empty_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """步骤 4: 空工具结果添加 "(empty result)" 占位。"""
    result = []
    for msg in messages:
        msg = dict(msg)
        if _is_tool(msg):
            content = msg.get("content", "")
            if not isinstance(content, str) or not content.strip():
                msg["content"] = "(empty result)"
        result.append(msg)
    return result


def _fix_tool_calling(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """步骤 5: 移除孤立工具调用和工具结果。

    规则：
    - 有 tool_calls 的 assistant 消息，如果后续没有对应的 tool:* source 结果，移除该 tool_call
    - 没有前置 assistant tool_calls 的独立 tool:* 结果，保留（可能是外部注入）
    """
    # solo-agent 的简单消息格式不包含显式 tool_call_id 绑定。
    # 采用保守策略：保留所有有内容的消息，只移除无对应结果的 tool_call 标记。
    result = []
    for msg in messages:
        msg = dict(msg)
        if _is_tool_call(msg) and not msg.get("tool_calls"):
            # 空 tool_calls 列表，移除该字段
            msg.pop("tool_calls", None)
        result.append(msg)
    return result


def _merge_consecutive_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """步骤 6: 合并相同 role 的连续消息（不含 tool_calls 的 assistant）。"""
    if not messages:
        return messages

    merged: list[dict[str, Any]] = []
    for msg in messages:
        # 可以合并的 role：user/human, assistant(无 tool_calls), system
        role = msg.get("role", "")
        if not merged:
            merged.append(dict(msg))
            continue

        prev = merged[-1]
        prev_role = prev.get("role", "")

        # 判断是否应合并：相同 role 且都不是工具结果，且 prev 无 tool_calls
        can_merge = (
            role == prev_role
            and not _is_tool(msg)
            and not _is_tool(prev)
            and not _is_tool_call(prev)
            and role in ("user", "human", "assistant", "system")
        )

        if can_merge:
            prev_content = prev.get("content", "")
            cur_content = msg.get("content", "")
            prev["content"] = f"{prev_content}\n{cur_content}".strip()
        else:
            merged.append(dict(msg))

    return merged


def _fix_lead_trail(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """步骤 7: 修复对话头尾结构。

    规则：
    - 仅移除单独出现在末尾的孤立 system 消息（无后续对话内容）
    - 保留 system 在开头的对话（system 是合法的对话前导）
    - 不强制要求以 user 开头或以 user 结尾
    """
    if not messages:
        return messages

    # 末尾：移除末尾的孤立 system 消息
    end_idx = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        if not _is_system(messages[i]):
            end_idx = i + 1
            break
    else:
        # 全是 system 消息
        end_idx = 0

    return messages[:end_idx]


def _populate_if_empty(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """步骤 8: 空对话填充 "Hello" 占位消息。"""
    if not messages:
        return [{"role": "user", "content": "Hello"}]
    return messages


# ---------------------------------------------------------------------------
# 管道步骤注册
# ---------------------------------------------------------------------------

# 步骤名称 → 函数映射（按顺序）
_PIPELINE_STEPS: list[tuple[str, Any]] = [
    ("merge_text_content_items", _merge_text_content_items),
    ("trim_assistant_text_whitespace", _trim_assistant_text_whitespace),
    ("remove_empty_messages", _remove_empty_messages),
    ("fix_empty_tool_results", _fix_empty_tool_results),
    ("fix_tool_calling", _fix_tool_calling),
    ("merge_consecutive_messages", _merge_consecutive_messages),
    ("fix_lead_trail", _fix_lead_trail),
    ("populate_if_empty", _populate_if_empty),
]

_PIPELINE_STEP_NAMES = [name for name, _ in _PIPELINE_STEPS]


# ---------------------------------------------------------------------------
# ConversationRepair 主类
# ---------------------------------------------------------------------------


class ConversationRepair:
    """对话修复器：在 LLM 调用前清洗消息历史。

    用法::

        repair = ConversationRepair(disabled_fixes={"merge_consecutive_messages"})
        cleaned = repair.fix_conversation(messages)
    """

    def __init__(self, disabled_fixes: set[str] | None = None) -> None:
        """初始化修复器。

        :param disabled_fixes: 禁用的步骤名称集合（如 {"merge_consecutive_messages"}）
        """
        self.disabled_fixes: set[str] = disabled_fixes or set()

    def fix_conversation(
        self, messages: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """执行完整的 8 步修复管道，返回修复后的消息列表。

        :param messages: dict 格式的消息列表（role + content）
        :returns: 修复后的消息列表（深拷贝，不修改输入）
        """
        # 深拷贝输入，不修改原始数据
        current = deepcopy(list(messages))

        for step_name, step_fn in _PIPELINE_STEPS:
            if step_name in self.disabled_fixes:
                continue
            current = step_fn(current)

        return current

    @property
    def available_fixes(self) -> list[str]:
        """返回所有可用的修复步骤名称列表。"""
        return list(_PIPELINE_STEP_NAMES)

    @property
    def active_fixes(self) -> list[str]:
        """返回当前启用的修复步骤名称列表。"""
        return [name for name in _PIPELINE_STEP_NAMES if name not in self.disabled_fixes]


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------


def fix_conversation(
    messages: Sequence[dict[str, Any]],
    disabled_fixes: set[str] | None = None,
) -> list[dict[str, Any]]:
    """便捷函数：创建默认修复器并执行 fix_conversation。

    :param messages: dict 格式的消息列表
    :param disabled_fixes: 可选，禁用的步骤名称集合
    :returns: 修复后的消息列表
    """
    repair = ConversationRepair(disabled_fixes=disabled_fixes)
    return repair.fix_conversation(messages)
