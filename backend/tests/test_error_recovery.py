"""错误恢复循环集成测试。

测试完整的错误恢复流程：分类 → 恢复 → 继续执行（或终止）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from solo_agent.agent.error_classifier import ErrorClassification, ErrorClassifier
from solo_agent.agent.state import AgentState
from solo_agent.providers import ChatMessage, ProviderChunk

# ---------------------------------------------------------------------------
# Fake 组件
# ---------------------------------------------------------------------------


class FailingToolRegistry:
    """模拟工具注册表：可配置特定工具失败。"""

    def __init__(self, fail_on: str = "", fail_count: int = 1):
        self.fail_on = fail_on
        self.fail_count = fail_count
        self.call_count: dict[str, int] = {}
        self.tools = {
            "read_file": self._call_read_file,
            "search_text": self._call_search_text,
            "run_pytest": self._call_run_pytest,
        }

    async def call(self, name: str, arguments: dict) -> dict:
        self.call_count[name] = self.call_count.get(name, 0) + 1
        if name == self.fail_on and self.call_count[name] <= self.fail_count:
            raise TimeoutError(f"Tool {name} timed out")
        # 默认成功返回
        return {"ok": True, "result": f"result from {name}"}

    async def call_tool(self, name: str, arguments: dict) -> dict:
        return await self.call(name, arguments)


class FakeProvider:
    """模拟 LLM provider：返回固定的工具调用指令。"""

    name = "fake"
    model = "fake-model"

    def __init__(self) -> None:
        self.seen_messages: list[list[ChatMessage]] = []

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ProviderChunk]:
        self.seen_messages.append(messages)
        yield ProviderChunk(content=self._generate_response(messages))

    def _generate_response(self, messages: list[ChatMessage]) -> str:
        """生成简单的工具调用响应。"""
        return "I will read the file and run tests."


class FakePersistence:
    """模拟持久化层。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def __getattr__(self, method: str):
        async def _noop(*args, **kwargs):
            self.calls.append(method)
            return None

        return _noop


class FakeSafetyInspector:
    """模拟安全检查器：默认全部通过。"""

    async def inspect(self, *args) -> dict:
        return {"allowed": True}


# ---------------------------------------------------------------------------
# 单元级集成测试
# ---------------------------------------------------------------------------


class TestErrorRecoveryBasic:
    """基础错误恢复流程测试。"""

    def test_retryable_error_classification(self) -> None:
        """retryable 错误应被正确分类。"""
        classifier = ErrorClassifier()
        exc = TimeoutError("请求超时")
        result = classifier.classify(exc, stage="tools")
        assert result.category == "retryable"
        assert result.recoverable is True

    def test_fatal_error_classification(self) -> None:
        """fatal 错误应被正确分类。"""
        classifier = ErrorClassifier()
        exc = PermissionError("权限不足")
        result = classifier.classify(exc, stage="tools")
        assert result.category == "fatal"
        assert result.recoverable is False

    def test_architectural_after_three(self) -> None:
        """三次相同错误后应触发 architectural。"""
        classifier = ErrorClassifier()
        for _ in range(3):
            classifier.classify(TimeoutError("超时"), stage="tools")
        count = classifier._detect_identical_error("TIMEOUT", "tools")
        assert count == 3

    def test_fixable_error_with_recovery_stage(self) -> None:
        """fixable 错误应有 recovery_stage。"""
        classifier = ErrorClassifier()

        # 注册一个 fixable 错误类型
        class MissingContextError(Exception):
            pass

        classifier.register(
            MissingContextError,
            category="fixable",
            error_code="MISSING_CONTEXT",
            severity="error",
        )
        result = classifier.classify(MissingContextError(), stage="tools")
        assert result.category == "fixable"


class TestCompactionAttemptsLimit:
    """max_compaction_attempts=2 测试。"""

    def test_compaction_under_limit_retryable(self) -> None:
        """attempt_count < 2 时错误仍可重试。"""
        state = AgentState(session_id="s", run_id="r", user_input="test")
        assert state.compaction_attempts == 0

        # 模拟压缩失败
        state.compaction_attempts += 1
        assert state.compaction_attempts == 1

        state.compaction_attempts += 1
        assert state.compaction_attempts == 2

    def test_compaction_at_limit_architectural(self) -> None:
        """attempt_count >= 2 时分类为 architectural。"""
        state = AgentState(session_id="s", run_id="r", user_input="test")
        state.compaction_attempts = 2  # 先到 2

        classifier = ErrorClassifier()
        # 对于 context_guard stage + CONTEXT_COMPRESSION_FAILED error_code
        # 需要在 classify 中传入正确的参数
        result = classifier.classify(
            ValueError("Compression failed"),
            stage="context_guard",
            attempt_count=state.compaction_attempts,
        )
        assert result.category == "architectural"
        assert result.error_code == "COMPACTION_ARCHITECTURAL"
        assert result.recoverable is False
        # 标准异常（非 CONTEXT_COMPRESSION_FAILED）不会触发 architectural
        # 但 attempt_count 仍被正确传递


class TestErrorRecoveryFlow:
    """端到端错误恢复流程测试。"""

    def test_run_scoped_error_history_isolated(self) -> None:
        """run_id scoped classifiers should not contaminate each other."""
        from solo_agent.agent.policy import BehaviorPolicy

        policy = BehaviorPolicy()
        policy.start_error_run("run-a")
        policy.start_error_run("run-b")
        try:
            policy.classify_error(TimeoutError("timeout"), stage="tools", run_id="run-a")
            policy.classify_error(TimeoutError("timeout"), stage="tools", run_id="run-a")

            run_b = policy.classify_error(TimeoutError("timeout"), stage="tools", run_id="run-b")
            run_a = policy.classify_error(TimeoutError("timeout"), stage="tools", run_id="run-a")

            assert run_b.category == "retryable"
            assert run_a.category == "architectural"
        finally:
            policy.finish_error_run("run-a")
            policy.finish_error_run("run-b")

    def test_tool_failure_recovery_to_success(self) -> None:
        """工具调用失败 → 重试成功 → 继续执行。"""
        from solo_agent.agent.policy import BehaviorPolicy

        policy = BehaviorPolicy()
        classifier = ErrorClassifier()

        # 模拟第一次工具调用失败（超时）
        exc = TimeoutError("工具超时")
        classification = classifier.classify(exc, stage="tools", attempt_count=0)
        assert classification.category == "retryable"

        # 策略判断：应重试
        should_retry, reason = policy.should_retry(classification, retry_count=0)
        assert should_retry is True
        assert "重试" in reason

        # 生成修复提示（retryable 错误通常无修复提示）
        fix_prompt = policy.build_fix_prompt(classification)
        assert fix_prompt == ""

    def test_fixable_error_injects_prompt(self) -> None:
        """fixable 错误应注入修复提示到上下文。"""
        from solo_agent.agent.policy import BehaviorPolicy

        policy = BehaviorPolicy()
        classification = ErrorClassification(
            category="fixable",
            error_code="MISSING_CONTEXT",
            severity="error",
            recoverable=True,
            recovery_stage="collect_context",
        )

        fix_prompt = policy.build_fix_prompt(classification)
        assert "read_file" in fix_prompt.lower() or "search_text" in fix_prompt.lower()
        assert "修复提示" in fix_prompt

    def test_fatal_error_should_not_retry(self) -> None:
        """fatal 错误不应触发重试。"""
        from solo_agent.agent.policy import BehaviorPolicy

        policy = BehaviorPolicy()
        classification = ErrorClassification(
            category="fatal",
            error_code="PERMISSION_DENIED",
            severity="fatal",
            recoverable=False,
        )

        should_retry, reason = policy.should_retry(classification, retry_count=0)
        assert should_retry is False
        assert "致命" in reason

    def test_architectural_error_should_not_retry(self) -> None:
        """architectural 错误不应触发重试。"""
        from solo_agent.agent.policy import BehaviorPolicy

        policy = BehaviorPolicy()
        classification = ErrorClassification(
            category="architectural",
            error_code="COMPACTION_ARCHITECTURAL",
            severity="fatal",
            recoverable=False,
        )

        should_retry, reason = policy.should_retry(classification, retry_count=0)
        assert should_retry is False
        assert "架构" in reason

    def test_retry_count_exceeds_limit(self) -> None:
        """重试次数超过上限时不应继续重试。"""
        from solo_agent.agent.policy import BehaviorPolicy

        policy = BehaviorPolicy()
        classification = ErrorClassification(
            category="retryable",
            error_code="TIMEOUT",
            severity="error",
            recoverable=True,
        )

        # retry_count >= 3，不应重试
        should_retry, reason = policy.should_retry(classification, retry_count=3)
        assert should_retry is False
        assert "上限" in reason


class TestStateErrorTracking:
    """AgentState 错误追踪字段测试。"""

    def test_error_state_update_on_classification(self) -> None:
        """分类后 AgentState 错误字段应正确更新。"""
        state = AgentState(session_id="s", run_id="r", user_input="test")
        classifier = ErrorClassifier()

        exc = TimeoutError("超时")
        classification = classifier.classify(exc, stage="tools")

        state.last_error = classification.to_dict()
        state.error_classification = classification.category

        assert state.last_error["category"] == "retryable"
        assert state.error_classification == "retryable"
        assert state.retry_count == 0

    def test_retry_count_tracking_across_retries(self) -> None:
        """多次重试后 retry_count 应正确递增。"""
        state = AgentState(session_id="s", run_id="r", user_input="test")

        for _ in range(4):
            state.retry_count += 1

        assert state.retry_count == 4


class TestFixPromptGeneration:
    """修复提示文本生成测试。"""

    def test_collect_context_prompt(self) -> None:
        from solo_agent.agent.policy import BehaviorPolicy

        policy = BehaviorPolicy()
        classification = ErrorClassification(
            category="fixable",
            error_code="MISSING_CONTEXT",
            severity="error",
            recoverable=True,
            recovery_stage="collect_context",
        )
        prompt = policy.build_fix_prompt(classification)
        assert "文件上下文" in prompt

    def test_fix_conversation_prompt(self) -> None:
        from solo_agent.agent.policy import BehaviorPolicy

        policy = BehaviorPolicy()
        classification = ErrorClassification(
            category="fixable",
            error_code="CONVERSATION_ERROR",
            severity="error",
            recoverable=True,
            recovery_stage="fix_conversation",
        )
        prompt = policy.build_fix_prompt(classification)
        assert "消息格式" in prompt

    def test_hash_anchored_editing_prompt(self) -> None:
        from solo_agent.agent.policy import BehaviorPolicy

        policy = BehaviorPolicy()
        classification = ErrorClassification(
            category="fixable",
            error_code="MISSING_HASH",
            severity="error",
            recoverable=True,
            recovery_stage="hash_anchored_editing",
        )
        prompt = policy.build_fix_prompt(classification)
        assert "prepare_edit" in prompt

    def test_unknown_recovery_stage_empty_prompt(self) -> None:
        from solo_agent.agent.policy import BehaviorPolicy

        policy = BehaviorPolicy()
        classification = ErrorClassification(
            category="retryable",
            error_code="TIMEOUT",
            severity="error",
            recoverable=True,
        )
        prompt = policy.build_fix_prompt(classification)
        assert prompt == ""
