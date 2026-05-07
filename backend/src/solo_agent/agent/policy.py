from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .error_classifier import ErrorClassification, ErrorClassifier

CODE_CHANGE_MARKERS = (
    "edit",
    "modify",
    "fix",
    "refactor",
    "implement",
    "change",
    "write code",
    "修改",
    "修复",
    "实现",
    "重构",
    "改代码",
)
FAILING_TEST_MARKERS = (
    "failing test",
    "failed test",
    "test fails",
    "test failure",
    "red test",
    "pytest failed",
    "pytest failure",
    "失败测试",
    "测试失败",
    "失败的测试",
    "红灯测试",
)
IRON_LAW_WARNING_MARKERS = (
    "read-only",
    "read only",
    "readonly",
    "explore",
    "exploration",
    "inspect only",
    "只读",
    "探索",
    "仅查看",
    "不要修改",
)
QUALITY_TOOL_NAMES = {"run_pytest", "targeted_pytest", "run_ruff_check", "run_ruff_format_check"}
EDIT_PROOF_TOOL_NAMES = {"prepare_edit", "get_file_hash"}
EDIT_TOOL_NAMES = {"prepare_edit", "preview_patch", "apply_text_edit"}
CONTEXT_PROOF_TOOL_NAMES = {"workspace_snapshot", "search_text", "read_file", "inspect_python_symbols"}


class BehaviorPolicy:
    """Graph 层行为策略：skill 是输入，硬约束和恢复由这里执行。"""

    def __init__(self) -> None:
        self._default_error_classifier = ErrorClassifier()
        self._run_error_classifiers: dict[str, ErrorClassifier] = {}

    def start_error_run(self, run_id: str) -> None:
        self._run_error_classifiers[run_id] = ErrorClassifier()

    def finish_error_run(self, run_id: str) -> None:
        self._run_error_classifiers.pop(run_id, None)

    def build_snapshot(self, state: Any) -> dict[str, Any]:
        skills = [str(skill.get("name", "")).lower() for skill in getattr(state, "selected_skills", [])]
        return {
            "engine": "graph_behavior_policy",
            "enforced_principles": [
                "superpowers_tdd_iron_law",
                "karpathy_think_before_coding",
                "karpathy_simplicity_first",
                "karpathy_surgical_changes",
                "karpathy_goal_driven_execution",
                "hash_anchored_editing",
                "read_before_edit",
            ],
            "selected_behavior_skills": [
                skill
                for skill in getattr(state, "selected_skills", [])
                if str(skill.get("category", "")).lower() == "behavior"
            ],
            "hard_gates": {
                "production_edit_requires_failing_test_signal": "iron-law" in skills,
                "existing_file_edit_requires_context_read": True,
                "apply_text_edit_requires_hash_and_preview": True,
            },
        }

    def new_tool_protocol_state(self, state: Any) -> dict[str, Any]:
        protocol_state: dict[str, Any] = {
            "edit_proofs": set(),
            "previews": set(),
            "context_proofs": set(),
        }
        for record in getattr(state, "tool_calls", []):
            if not getattr(record, "blocked", False) and tool_result_ok(getattr(record, "result", None)):
                self.record_tool_success(protocol_state, record.name, record.arguments, record.result)
        return protocol_state

    def iron_law_decision(
        self,
        state: Any,
        proposed: Iterable[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        skill_names = {str(skill.get("name", "")).lower() for skill in getattr(state, "selected_skills", [])}
        proposed_calls = list(proposed or [])
        production_paths = [
            path
            for call in proposed_calls
            if str(call.get("name", "")) == "apply_text_edit"
            for path in [normalize_path(dict(call.get("arguments") or {}).get("path"))]
            if path and is_production_path(path)
        ]
        if "iron-law" not in skill_names and not production_paths:
            return {"action": "none"}

        text = f"{getattr(state, 'user_input', '')}\n{getattr(state, 'plan', '')}".lower()
        has_code_change_intent = any(marker in text for marker in CODE_CHANGE_MARKERS)
        if not has_code_change_intent and not production_paths:
            return {"action": "none"}
        if has_failing_test_signal(state):
            return {"action": "none", "reason": "failing_test_signal_present"}
        if quality_tool_runs_before_production_edit(proposed_calls):
            return {"action": "none", "reason": "awaiting_quality_tool_signal"}

        warning_only = any(marker in text for marker in IRON_LAW_WARNING_MARKERS)
        return {
            "action": "warning" if warning_only else "blocked",
            "reason": "production_edit_without_failing_test_signal",
            "current_task_priority": "user_input",
            "production_paths": production_paths,
            "warning_only": warning_only,
        }

    def tool_protocol_violation(
        self,
        state: Any,
        name: str,
        arguments: Mapping[str, Any],
        protocol_state: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if name not in EDIT_TOOL_NAMES:
            return None

        path = normalize_path(arguments.get("path"))
        if not path:
            return {"reason": f"{name}_missing_path", "recoverable": False}

        if not protocol_state.get("context_proofs"):
            return {
                "reason": "context_required_before_edit",
                "path": path,
                "missing": ["read_file_or_search_text"],
                "recoverable": True,
                "recovery_stage": "collect_context",
            }

        if name != "apply_text_edit":
            return None

        expected_hash = normalize_hash(arguments.get("expected_hash"))
        if not expected_hash:
            return {"reason": "apply_text_edit_missing_expected_hash", "path": path, "recoverable": False}

        iron_law = self.iron_law_decision(state, [{"name": name, "arguments": dict(arguments)}])
        if iron_law["action"] == "blocked":
            return {"reason": "iron_law_blocked", "path": path, "expected_hash": expected_hash, "recoverable": False}

        key = (path, expected_hash)
        has_edit_proof = key in protocol_state.get("edit_proofs", set())
        has_preview = key in protocol_state.get("previews", set())
        if not has_edit_proof or not has_preview:
            missing = []
            if not has_edit_proof:
                missing.append("prepare_edit_or_get_file_hash")
            if not has_preview:
                missing.append("preview_patch")
            return {
                "reason": "apply_text_edit_protocol_incomplete",
                "path": path,
                "expected_hash": expected_hash,
                "missing": missing,
                "recoverable": can_recover_edit_protocol(arguments, missing),
                "recovery_stage": "hash_anchored_editing",
            }

        return None

    def recovery_tool_calls(
        self,
        violation: Mapping[str, Any],
        original_name: str,
        original_arguments: Mapping[str, Any],
        available_tools: set[str],
    ) -> list[dict[str, Any]]:
        path = str(violation.get("path") or original_arguments.get("path") or "")
        reason = str(violation.get("reason", ""))
        calls: list[dict[str, Any]] = []

        if reason == "context_required_before_edit":
            if path and "read_file" in available_tools:
                calls.append({"name": "read_file", "arguments": {"path": path}, "category": "context"})
            elif "search_text" in available_tools:
                calls.append(
                    {
                        "name": "search_text",
                        "arguments": {"query": path or str(original_arguments)[:200], "max_matches": 20},
                        "category": "context",
                    }
                )
            return calls

        if reason == "apply_text_edit_protocol_incomplete":
            expected_hash = str(violation.get("expected_hash") or original_arguments.get("expected_hash") or "")
            missing = set(violation.get("missing", []))
            if "prepare_edit_or_get_file_hash" in missing and path and "prepare_edit" in available_tools:
                calls.append({"name": "prepare_edit", "arguments": {"path": path}, "category": "edit"})
            if "preview_patch" in missing and path and expected_hash and "preview_patch" in available_tools:
                preview_args = preview_arguments(original_arguments, path=path, expected_hash=expected_hash)
                if preview_args:
                    calls.append({"name": "preview_patch", "arguments": preview_args, "category": "edit"})
            return calls

        return []

    def record_tool_success(
        self,
        protocol_state: dict[str, Any],
        name: str,
        arguments: Mapping[str, Any],
        result: Any,
    ) -> None:
        if name in CONTEXT_PROOF_TOOL_NAMES:
            protocol_state.setdefault("context_proofs", set()).add(name)
            return
        if name not in EDIT_PROOF_TOOL_NAMES and name != "preview_patch":
            return

        path = normalize_path(
            first_present(
                arguments.get("path"),
                extract_protocol_field(result, ("path", "file_path", "target_path")),
            )
        )
        expected_hash = normalize_hash(
            first_present(
                arguments.get("expected_hash"),
                extract_protocol_field(
                    result,
                    ("expected_hash", "hash", "file_hash", "sha256", "digest", "current_hash"),
                ),
            )
        )
        if not path or not expected_hash:
            return

        key = (path, expected_hash)
        if name in EDIT_PROOF_TOOL_NAMES:
            protocol_state.setdefault("edit_proofs", set()).add(key)
        elif name == "preview_patch":
            protocol_state.setdefault("previews", set()).add(key)

    # ------------------------------------------------------------------
    # 错误恢复策略方法（Error Handling Layer）
    # ------------------------------------------------------------------

    def classify_error(
        self,
        exception: Exception,
        stage: str = "",
        attempt_count: int = 0,
        run_id: str | None = None,
    ) -> ErrorClassification:
        """委托给 ErrorClassifier 进行错误分类。

        :param exception: 捕获的异常
        :param stage: 失败发生的 agent 阶段
        :param attempt_count: 当前已尝试次数
        """
        if run_id:
            classifier = self._run_error_classifiers.setdefault(run_id, ErrorClassifier())
        else:
            classifier = self._default_error_classifier
        return classifier.classify(exception, stage=stage, attempt_count=attempt_count)

    def should_retry(
        self,
        classification: ErrorClassification,
        retry_count: int,
    ) -> tuple[bool, str]:
        """判断是否应重试。

        :param classification: ErrorClassifier 返回的分类结果
        :param retry_count: 当前已重试次数
        :returns: (是否应重试, 决策理由)
        """
        if classification.category == "retryable" and retry_count < 3:
            return True, "可重试的临时错误"
        if (
            classification.category == "fixable"
            and classification.recovery_stage
            and retry_count < 2
        ):
            return True, f"可修复错误，恢复阶段: {classification.recovery_stage}"
        if classification.category == "architectural":
            return False, "三次相同失败，判定为架构问题"
        if classification.category == "fatal":
            return False, "致命错误，不可恢复"
        return False, f"重试次数已达上限 (retry_count={retry_count})"

    def build_fix_prompt(self, classification: ErrorClassification) -> str:
        """根据错误分类生成修复提示文本，注入到对话上下文中。

        :param classification: ErrorClassifier 返回的分类结果
        :returns: 修复提示文本（可能为空字符串）
        """
        if classification.recovery_stage == "collect_context":
            return (
                "[修复提示] 工具调用失败：缺少文件上下文。"
                "请先使用 read_file 或 search_text 收集相关代码内容，再尝试编辑。"
            )
        if classification.recovery_stage == "fix_conversation":
            return (
                "[修复提示] LLM 消息格式错误。"
                "请确保工具调用参数格式正确，避免空内容或无效 role 组合。"
            )
        if classification.recovery_stage == "hash_anchored_editing":
            return (
                "[修复提示] 编辑协议失败：缺少文件哈希或预览。"
                "请先使用 prepare_edit 获取文件哈希，再使用 preview_patch 确认变更。"
            )
        return ""


def preview_arguments(
    original_arguments: Mapping[str, Any],
    *,
    path: str,
    expected_hash: str,
) -> dict[str, Any] | None:
    new_text = first_present(original_arguments.get("new_text"), original_arguments.get("new"))
    if new_text is None:
        return None
    args: dict[str, Any] = {"path": path, "expected_hash": expected_hash, "new_text": new_text}
    for source, target in (
        ("old_text", "old_text"),
        ("old", "old_text"),
        ("line_start", "line_start"),
        ("line_end", "line_end"),
    ):
        value = original_arguments.get(source)
        if value is not None and target not in args:
            args[target] = value
    return args


def can_recover_edit_protocol(arguments: Mapping[str, Any], missing: Iterable[str]) -> bool:
    missing_set = set(missing)
    if "preview_patch" in missing_set and first_present(arguments.get("new_text"), arguments.get("new")) is None:
        return False
    return True


def quality_tool_runs_before_production_edit(proposed_calls: Iterable[Mapping[str, Any]]) -> bool:
    for call in proposed_calls:
        name = str(call.get("name", ""))
        if name in QUALITY_TOOL_NAMES:
            return True
        if name == "apply_text_edit":
            path = normalize_path(dict(call.get("arguments") or {}).get("path"))
            return not path or not is_production_path(path)
    return False


def has_failing_test_signal(state: Any) -> bool:
    text = f"{getattr(state, 'user_input', '')}\n{getattr(state, 'plan', '')}".lower()
    if any(marker in text for marker in FAILING_TEST_MARKERS):
        return True
    return any(
        record.name in QUALITY_TOOL_NAMES and tool_result_failed(record.result)
        for record in getattr(state, "tool_calls", [])
        if not getattr(record, "blocked", False)
    )


def normalize_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").strip()


def normalize_hash(value: Any) -> str:
    return str(value or "").strip()


def is_production_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return False
    return not any(part in {"tests", "test", "__tests__", "docs", "documentation"} for part in parts)


def tool_result_ok(result: Any) -> bool:
    if isinstance(result, Mapping):
        if "ok" in result:
            return bool(result["ok"])
        if "exit_code" in result:
            return int(result.get("exit_code") or 0) == 0
    return result is not None


def tool_result_failed(result: Any) -> bool:
    if isinstance(result, Mapping):
        if result.get("failed") is True:
            return True
        if "exit_code" in result:
            try:
                return int(result.get("exit_code") or 0) != 0
            except (TypeError, ValueError):
                return False
        if result.get("ok") is False:
            return True
    return False


def first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def extract_protocol_field(result: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(result, Mapping):
        for key in keys:
            if key in result:
                return result[key]
        nested = result.get("result")
        if isinstance(nested, Mapping):
            for key in keys:
                if key in nested:
                    return nested[key]
    return None
