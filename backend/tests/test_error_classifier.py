"""ErrorClassifier 模块单元测试。"""

from __future__ import annotations

import pytest
from solo_agent.agent.error_classifier import (
    ErrorClassification,
    ErrorClassifier,
)

# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


@pytest.fixture
def classifier() -> ErrorClassifier:
    return ErrorClassifier()


# ---------------------------------------------------------------------------
# ErrorClassification 序列化测试
# ---------------------------------------------------------------------------


class TestErrorClassification:
    """ErrorClassification dataclass 基础测试。"""

    def test_create_retryable(self) -> None:
        c = ErrorClassification(
            category="retryable",
            error_code="TIMEOUT",
            severity="error",
            recoverable=True,
        )
        assert c.category == "retryable"
        assert c.error_code == "TIMEOUT"
        assert c.severity == "error"
        assert c.recoverable is True
        assert c.recovery_stage is None

    def test_to_dict(self) -> None:
        c = ErrorClassification(
            category="fixable",
            error_code="FIX_001",
            severity="error",
            recoverable=True,
            recovery_stage="collect_context",
            reason="测试原因",
        )
        d = c.to_dict()
        assert d["category"] == "fixable"
        assert d["error_code"] == "FIX_001"
        assert d["severity"] == "error"
        assert d["recoverable"] is True
        assert d["recovery_stage"] == "collect_context"
        assert d["reason"] == "测试原因"


# ---------------------------------------------------------------------------
# 内置异常类型映射测试
# ---------------------------------------------------------------------------


class TestBuiltinExceptionMapping:
    """内置异常类型 → 分类映射测试。"""

    @pytest.mark.parametrize(
        "exc,expected_category",
        [
            (TimeoutError(), "retryable"),
            (ConnectionError(), "retryable"),
            (PermissionError(), "fatal"),
            (ValueError(), "fatal"),
            (KeyError(), "fatal"),
            (TypeError(), "fatal"),
            (RuntimeError(), "fatal"),
        ],
    )
    def test_builtin_mapping(
        self, classifier: ErrorClassifier, exc: Exception, expected_category: str
    ) -> None:
        result = classifier.classify(exc)
        assert result.category == expected_category


class TestSubclassInheritance:
    """通过 issubclass 语义匹配父类。"""

    def test_timeout_subclass(self, classifier: ErrorClassifier) -> None:
        # TimeoutError 的子类应匹配 retryable
        # ConnectionError 不是 TimeoutError 的子类，但 OSError 的子类...
        # 这里测试的是直接继承关系
        class CustomTimeout(TimeoutError):
            pass

        result = classifier.classify(CustomTimeout())
        assert result.category == "retryable"


class TestCauseChain:
    """通过 __cause__ 链遍历匹配。"""

    def test_cause_chain_match(self, classifier: ErrorClassifier) -> None:
        inner = TimeoutError("内部超时")
        outer = ValueError("包装错误")
        outer.__cause__ = inner
        result = classifier.classify(outer)
        # 应通过 __cause__ 链找到 TimeoutError
        assert result.category == "retryable"


# ---------------------------------------------------------------------------
# LLM HTTP 状态码分类测试
# ---------------------------------------------------------------------------


class TestHTTPStatusClassification:
    """HTTP 状态码通过错误消息模式匹配分类。"""

    @pytest.mark.parametrize(
        "message,expected_category",
        [
            ("HTTP 408 Request Timeout", "retryable"),
            ("status_code: 429 too many requests", "retryable"),
            ("HTTPStatus.502 Bad Gateway", "retryable"),
            ("HTTP 503 Service Unavailable", "retryable"),
            ("HTTP 504 Gateway Timeout", "retryable"),
            ("HTTP 401 Unauthorized", "fatal"),
            ("HTTP status code=403 Forbidden", "fatal"),
        ],
    )
    def test_http_status_from_message(
        self, classifier: ErrorClassifier, message: str, expected_category: str
    ) -> None:
        exc = RuntimeError(message)
        result = classifier.classify(exc)
        assert result.category == expected_category


# ---------------------------------------------------------------------------
# 默认分类与未知异常
# ---------------------------------------------------------------------------


class TestDefaultClassification:
    """未匹配异常默认分类为 fatal。"""

    def test_unknown_exception_defaults_to_fatal(
        self, classifier: ErrorClassifier
    ) -> None:
        exc = Exception("未知错误")
        result = classifier.classify(exc)
        assert result.category == "fatal"
        assert result.error_code == "UNKNOWN_ERROR"


# ---------------------------------------------------------------------------
# 可扩展注册
# ---------------------------------------------------------------------------


class TestRegisterCustomException:
    """register() 方法注册自定义异常映射。"""

    def test_register_custom_error(self, classifier: ErrorClassifier) -> None:
        class CustomError(Exception):
            pass

        classifier.register(CustomError, category="fixable", error_code="CUSTOM_001")
        result = classifier.classify(CustomError("自定义错误"))
        assert result.category == "fixable"
        assert result.error_code == "CUSTOM_001"

    def test_register_auto_error_code(self, classifier: ErrorClassifier) -> None:
        class AnotherError(Exception):
            pass

        classifier.register(AnotherError, category="retryable")
        result = classifier.classify(AnotherError())
        assert result.error_code == "ANOTHERERROR"


# ---------------------------------------------------------------------------
# 上下文压缩特殊逻辑
# ---------------------------------------------------------------------------


class TestCompactionClassification:
    """上下文压缩 attempt_count 判断。"""

    def test_compaction_less_than_2_retryable(self, classifier: ErrorClassifier) -> None:
        exc = RuntimeError("上下文压缩失败")
        result = classifier.classify(
            exc,
            stage="context_guard",
            attempt_count=0,
        )
        # 即使 RuntimeError 是 fatal，但 stage=context_guard + attempt<2 的特殊情况
        # 注意：这里没有 error_code 的覆盖，仅测试 attempt_count 传递
        # 真正的 context_guard 分类在 BehaviorPolicy 层结合 error_code
        assert result.category == "retryable"
        assert result.error_code == "CONTEXT_COMPRESSION_FAILED"
        assert result.recoverable is True
        assert result.recovery_stage == "context_guard"

    def test_compaction_at_2_triggers_architectural(
        self, classifier: ErrorClassifier
    ) -> None:
        exc = RuntimeError("上下文压缩失败")
        result = classifier.classify(
            exc,
            stage="context_guard",
            attempt_count=2,
        )
        assert result.category == "architectural"
        assert result.error_code == "COMPACTION_ARCHITECTURAL"
        assert result.severity == "fatal"
        assert result.recoverable is False
        # 由于 error_code 是 "RUNTIME_ERROR" 而非 "CONTEXT_COMPRESSION_FAILED"，
        # 特殊逻辑不触发。实际集成中由 BehaviorPolicy 先设置 error_code。
        # 这里测试 attempt_count 参数传递正确。
        assert result.reason  # 有原因描述


# ---------------------------------------------------------------------------
# 三次相同错误检测
# ---------------------------------------------------------------------------


class TestIdenticalErrorDetection:
    """_detect_identical_error 和 _track_identical_error 测试。"""

    def test_first_error_no_detect(self, classifier: ErrorClassifier) -> None:
        count = classifier._detect_identical_error("TIMEOUT", "tools")
        assert count == 0

    def test_three_same_errors_triggers_architectural(
        self, classifier: ErrorClassifier
    ) -> None:
        # 模拟两次 error 出现
        classifier.classify(TimeoutError(), stage="tools")
        classifier.classify(TimeoutError(), stage="tools")
        assert classifier._detect_identical_error("TIMEOUT", "tools") == 2

        # 第三次应触发 architectural
        result = classifier.classify(TimeoutError(), stage="tools")
        assert classifier._detect_identical_error("TIMEOUT", "tools") == 3
        assert result.category == "architectural"
        assert result.severity == "fatal"
        assert result.recoverable is False

    def test_different_error_resets_count(self, classifier: ErrorClassifier) -> None:
        classifier.classify(TimeoutError(), stage="tools")
        classifier.classify(ValueError(), stage="tools")  # 不同的 error_code
        assert classifier._detect_identical_error("TIMEOUT", "tools") == 1
        assert classifier._detect_identical_error("VALUE_ERROR", "tools") == 1

    def test_reset_history(self, classifier: ErrorClassifier) -> None:
        classifier.classify(TimeoutError(), stage="tools")
        classifier.reset_history()
        assert classifier._detect_identical_error("TIMEOUT", "tools") == 0
