"""错误分类器模块：将异常映射为 retryable / fixable / fatal / architectural 四类。

设计参考 Hermes 的 ErrorClassifier 模式，配合 goose 的 fix_conversation 反馈循环。
原则：三次相同失败等于架构问题（max_compaction_attempts=2）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

# ---------------------------------------------------------------------------
# 错误分类数据类
# ---------------------------------------------------------------------------

ErrorCategory = Literal["retryable", "fixable", "fatal", "architectural"]
ErrorSeverity = Literal["warn", "error", "fatal"]


@dataclass(frozen=True)
class ErrorClassification:
    """错误分类结果，由 ErrorClassifier.classify() 返回。"""

    category: ErrorCategory
    error_code: str
    severity: ErrorSeverity
    recoverable: bool
    recovery_stage: str | None = None  # 如 "collect_context", "fix_conversation"
    reason: str = ""  # 分类理由，方便调试

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "error_code": self.error_code,
            "severity": self.severity,
            "recoverable": self.recoverable,
            "recovery_stage": self.recovery_stage,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# 可重试的 HTTP 状态码（LLM 调用临时错误）
# ---------------------------------------------------------------------------

_RETRYABLE_HTTP_STATUSES: tuple[int, ...] = (408, 429, 502, 503, 504)

# HTTP 状态码关键词模式，用于从错误消息中提取状态码
_HTTP_STATUS_PATTERNS = (
    "status_code",
    "status code",
    "HTTPStatus",
    "http_status",
    "HTTP ",
    "error code",
)

# fatal HTTP 状态码（认证/权限问题）
_FATAL_HTTP_STATUSES: tuple[int, ...] = (401, 403)


# ---------------------------------------------------------------------------
# 异常类型 → 错误分类 映射表
# ---------------------------------------------------------------------------

# 默认分类：未匹配的异常归类为 fatal
_DEFAULT_CLASSIFICATION: dict[str, Any] = {
    "category": "fatal",
    "error_code": "UNKNOWN_ERROR",
    "severity": "fatal",
    "recoverable": False,
}


def _build_default_mapping() -> dict[type[Exception], dict[str, Any]]:
    """构建内置异常类型 → 分类配置映射。"""
    return {
        # --- retryable: 可重试的临时错误 ---
        TimeoutError: {
            "category": "retryable",
            "error_code": "TIMEOUT",
            "severity": "error",
            "recoverable": True,
        },
        ConnectionError: {
            "category": "retryable",
            "error_code": "CONNECTION_ERROR",
            "severity": "error",
            "recoverable": True,
        },
        # --- fatal: 不可恢复的致命错误 ---
        PermissionError: {
            "category": "fatal",
            "error_code": "PERMISSION_DENIED",
            "severity": "fatal",
            "recoverable": False,
        },
        ValueError: {
            "category": "fatal",
            "error_code": "VALUE_ERROR",
            "severity": "error",
            "recoverable": False,
        },
        KeyError: {
            "category": "fatal",
            "error_code": "KEY_ERROR",
            "severity": "error",
            "recoverable": False,
        },
        TypeError: {
            "category": "fatal",
            "error_code": "TYPE_ERROR",
            "severity": "error",
            "recoverable": False,
        },
        RuntimeError: {
            "category": "fatal",
            "error_code": "RUNTIME_ERROR",
            "severity": "error",
            "recoverable": False,
        },
    }


# ---------------------------------------------------------------------------
# ErrorClassifier 主类
# ---------------------------------------------------------------------------


class ErrorClassifier:
    """错误分类器。

    根据异常类型、HTTP 状态码、错误消息和调用阶段将异常映射为 ErrorClassification。
    支持通过 register() 注册自定义异常类型映射。
    通过 _history 追踪 (error_code, stage) 组合，实现"三次相同失败 = 架构问题"。

    用法::

        classifier = ErrorClassifier()
        classification = classifier.classify(exc, stage="tools", attempt_count=0)
    """

    def __init__(self) -> None:
        # 异常类型 → 分类配置
        self._mapping: dict[type[Exception], dict[str, Any]] = _build_default_mapping()
        # 错误历史：(error_code, stage) → 出现次数
        self._history: dict[tuple[str, str], int] = {}

    # ---- 公共方法 ----

    def classify(
        self,
        exception: Exception,
        stage: str = "",
        attempt_count: int = 0,
    ) -> ErrorClassification:
        """对异常进行分类，返回 ErrorClassification。

        :param exception: Python 异常对象
        :param stage: 失败发生时的 agent 阶段（如 "tools", "context_guard"）
        :param attempt_count: 当前阶段/错误码的已尝试次数（用于 architectural 检测）
        """
        # 1) 先尝试从错误消息中提取 HTTP 状态码（比类型匹配更精确）
        http_result = self._try_http_status_classification(exception)
        if http_result:
            return self._finalize(http_result, exception, stage, attempt_count)

        # 2) 通过异常类型链匹配（优先 __cause__ 链的内层异常）
        result = self._match_by_exception_type(exception)
        if result:
            return self._finalize(result, exception, stage, attempt_count)

        # 3) 默认分类为 fatal
        return self._finalize(
            dict(_DEFAULT_CLASSIFICATION),
            exception,
            stage,
            attempt_count,
        )

    def register(
        self,
        exception_type: type[Exception],
        category: ErrorCategory = "fatal",
        error_code: str = "",
        severity: ErrorSeverity = "error",
    ) -> None:
        """注册自定义异常类型映射。

        :param exception_type: 要注册的异常类型
        :param category: 错误类别 (retryable/fixable/fatal/architectural)
        :param error_code: 机器可读错误码（留空则自动生成）
        :param severity: 严重级别
        """
        code = error_code or exception_type.__name__.upper()
        self._mapping[exception_type] = {
            "category": category,
            "error_code": code,
            "severity": severity,
            "recoverable": category in ("retryable", "fixable"),
        }

    def reset_history(self) -> None:
        """重置错误历史追踪（通常在新会话或新 run 开始时调用）。"""
        self._history.clear()

    # ---- 内部方法 ----

    def _match_by_exception_type(
        self, exception: Exception
    ) -> dict[str, Any] | None:
        """通过异常类型链（优先 __cause__ 链内层）匹配内置映射表。"""
        # 先遍历 __cause__ 链（内层异常通常包含更精确的根因）
        cause = exception.__cause__
        while cause is not None:
            cause_type = type(cause)
            if cause_type in self._mapping:
                return dict(self._mapping[cause_type])
            for mapped_type, config in self._mapping.items():
                if issubclass(cause_type, mapped_type):
                    return dict(config)
            cause = cause.__cause__

        # 再匹配外层异常类型
        exc_type = type(exception)
        if exc_type in self._mapping:
            return dict(self._mapping[exc_type])
        for mapped_type, config in self._mapping.items():
            if issubclass(exc_type, mapped_type):
                return dict(config)
        return None

    @staticmethod
    def _extract_http_status_code(message: str) -> int | None:
        """从错误消息中尝试提取 HTTP 状态码。"""
        import re

        # 模式: "status_code: 429", "HTTP 503", "status code=408", "HTTPStatus.429"
        patterns = [
            r"(?:status[_\s]?code|HTTP|error[_\s]?code)[\s:=]+(\d{3})",
            r"HTTP(?:Status)?[.\s]+(\d{3})",
            r"\b(\d{3})\b.*?(?:status|http|error)",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                code = int(match.group(1))
                if 100 <= code <= 599:
                    return code
        return None

    @classmethod
    def _try_http_status_classification(
        cls, exception: Exception
    ) -> dict[str, Any] | None:
        """尝试基于 HTTP 状态码进行分类。"""
        message = str(exception)
        status_code = cls._extract_http_status_code(message)
        if status_code is None:
            # 也检查异常链中的消息
            cause = exception.__cause__
            while cause is not None:
                status_code = cls._extract_http_status_code(str(cause))
                if status_code is not None:
                    break
                cause = cause.__cause__

        if status_code is not None:
            if status_code in _RETRYABLE_HTTP_STATUSES:
                return {
                    "category": "retryable",
                    "error_code": f"HTTP_{status_code}",
                    "severity": "error",
                    "recoverable": True,
                }
            if status_code in _FATAL_HTTP_STATUSES:
                return {
                    "category": "fatal",
                    "error_code": f"HTTP_{status_code}",
                    "severity": "fatal",
                    "recoverable": False,
                }
        return None

    def _finalize(
        self,
        base: dict[str, Any],
        exception: Exception,
        stage: str,
        attempt_count: int,
    ) -> ErrorClassification:
        """最终化分类结果：应用 architectural 检测和上下文压缩特殊逻辑。"""
        error_code = base.get("error_code", "UNKNOWN_ERROR")
        category = base["category"]
        severity = base.get("severity", "error")
        recoverable = bool(base.get("recoverable", False))
        recovery_stage = base.get("recovery_stage")

        if stage == "context_guard":
            if attempt_count >= 2:
                category = "architectural"
                severity = "fatal"
                error_code = "COMPACTION_ARCHITECTURAL"
                recoverable = False
                recovery_stage = None
            else:
                category = "retryable"
                severity = "error"
                error_code = "CONTEXT_COMPRESSION_FAILED"
                recoverable = True
                recovery_stage = "context_guard"

        # 三次相同错误检测（通用 architectural）
        same_count = self._track_identical_error(error_code, stage)
        if same_count >= 3:
            category = "architectural"
            severity = "fatal"
            recoverable = False
            recovery_stage = None

        return ErrorClassification(
            category=category,
            error_code=error_code,
            severity=severity,
            recoverable=recoverable,
            recovery_stage=recovery_stage,
            reason=f"{type(exception).__name__}: {str(exception)[:200]}",
        )

    def _track_identical_error(self, error_code: str, stage: str) -> int:
        """追踪相同错误出现次数。返回当前出现次数（含本次）。"""
        key = (error_code, stage)
        self._history[key] = self._history.get(key, 0) + 1
        return self._history[key]

    def _detect_identical_error(self, error_code: str, stage: str) -> int:
        """查询 (error_code, stage) 组合的历史出现次数（不含本次）。"""
        key = (error_code, stage)
        return self._history.get(key, 0)
