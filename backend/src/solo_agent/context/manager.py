from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from solo_agent.providers import ChatMessage, ChatProvider

from .auxiliary import AuxiliaryClient

MAIN_COMPRESSION_RATIO = 0.80
AUXILIARY_COMPRESSION_RATIO = 0.50
AUXILIARY_AFTER_COUNT = 2
DEFAULT_CONTEXT_TOKEN_BUDGET = 128_000
DEFAULT_TOOL_OUTPUT_CUTOFF = 3
DEFAULT_SUMMARY_MAX_TOKENS = 700
TOOL_RESULT_FRAGMENT_CHARS = 600


@dataclass(frozen=True)
class ContextBudgetReport:
    current_tokens: int
    token_budget: int
    threshold_ratio: float
    threshold_tokens: int
    compression_count: int
    provider_role: str
    provider_model: str | None
    should_compress: bool
    reason: str


@dataclass(frozen=True)
class CompressionResult:
    compressed: bool
    report: ContextBudgetReport
    compression_count: int
    provider_role: str
    provider_model: str | None
    summary: str = ""
    state: Any | None = None
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)


class ContextManager:
    def __init__(
        self,
        *,
        settings: Any | None = None,
        main_provider: ChatProvider | None = None,
        auxiliary_provider: ChatProvider | None = None,
        estimator: Any | None = None,
    ) -> None:
        self.settings = settings
        self.main_provider = main_provider
        self.auxiliary_provider = auxiliary_provider
        self.estimator = estimator

    def evaluate(
        self,
        state: Any,
        *,
        settings: Any | None = None,
        estimator: Any | None = None,
        provider: ChatProvider | None = None,
        auxiliary_provider: ChatProvider | None = None,
    ) -> ContextBudgetReport:
        active_settings = settings or self.settings
        compression_count = _compression_count(state)
        provider_role = _provider_role_for_count(compression_count, active_settings)
        selected_provider = self._select_provider(
            provider_role,
            settings=active_settings,
            main_provider=provider,
            auxiliary_provider=auxiliary_provider,
            build_auxiliary=False,
        )
        token_budget = _token_budget(active_settings, estimator or self.estimator)
        threshold_ratio = _threshold_ratio(compression_count, active_settings)
        threshold_tokens = int(token_budget * threshold_ratio)
        current_tokens = _estimate_tokens(state, estimator or self.estimator)
        should_compress = current_tokens >= threshold_tokens
        reason = "threshold_exceeded" if should_compress else "within_budget"

        return ContextBudgetReport(
            current_tokens=current_tokens,
            token_budget=token_budget,
            threshold_ratio=threshold_ratio,
            threshold_tokens=threshold_tokens,
            compression_count=compression_count,
            provider_role=provider_role,
            provider_model=getattr(selected_provider, "model", None),
            should_compress=should_compress,
            reason=reason,
        )

    async def maybe_compress(
        self,
        state: Any,
        *,
        settings: Any | None = None,
        provider: ChatProvider | None = None,
        auxiliary_provider: ChatProvider | None = None,
        estimator: Any | None = None,
        force: bool = False,
    ) -> CompressionResult:
        active_settings = settings or self.settings
        report = self.evaluate(
            state,
            settings=active_settings,
            estimator=estimator,
            provider=provider,
            auxiliary_provider=auxiliary_provider,
        )
        if not report.should_compress and not force:
            return CompressionResult(
                compressed=False,
                report=report,
                compression_count=report.compression_count,
                provider_role=report.provider_role,
                provider_model=report.provider_model,
                state=state,
                tool_outputs=summarize_tool_outputs(state, active_settings),
            )

        provider_role = report.provider_role
        selected_provider = self._select_provider(
            report.provider_role,
            settings=active_settings,
            main_provider=provider,
            auxiliary_provider=auxiliary_provider,
            build_auxiliary=True,
        )
        if selected_provider is None:
            provider_role = "main"
            selected_provider = self._select_provider(
                "main",
                settings=active_settings,
                main_provider=provider,
                auxiliary_provider=auxiliary_provider,
                build_auxiliary=False,
            )
        if selected_provider is None:
            raise RuntimeError("No provider available for context compression")
        prompt = _compression_prompt(state, active_settings)
        summary = await selected_provider.complete(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "You compress coding-agent context. Do not call tools, do not request tool access, "
                        "and do not invent facts. Write the final summary in Chinese."
                    ),
                ),
                ChatMessage(role="user", content=prompt),
            ],
            temperature=0.1,
            max_tokens=int(_setting(active_settings, "summary_max_tokens", DEFAULT_SUMMARY_MAX_TOKENS)),
        )
        next_count = report.compression_count + 1
        compressed_state = _with_compression_summary(state, summary, next_count)

        return CompressionResult(
            compressed=True,
            report=replace(
                report,
                provider_model=getattr(selected_provider, "model", None),
            ),
            compression_count=next_count,
            provider_role=provider_role,
            provider_model=getattr(selected_provider, "model", None),
            summary=summary,
            state=compressed_state,
            tool_outputs=summarize_tool_outputs(state, active_settings),
        )

    def _select_provider(
        self,
        provider_role: str,
        *,
        settings: Any | None,
        main_provider: ChatProvider | None,
        auxiliary_provider: ChatProvider | None,
        build_auxiliary: bool,
    ) -> ChatProvider | None:
        if provider_role == "main":
            return main_provider or self.main_provider
        provider = auxiliary_provider or self.auxiliary_provider
        if provider is None and build_auxiliary:
            provider = AuxiliaryClient.for_task("compression", settings)
            self.auxiliary_provider = provider
        return provider


def summarize_tool_outputs(state: Any, settings: Any | None = None) -> list[dict[str, Any]]:
    records = list(_tool_records(state))
    cutoff = int(_setting(settings, "context_tool_output_cutoff", DEFAULT_TOOL_OUTPUT_CUTOFF))
    if cutoff <= 0:
        recent: list[Any] = []
        old = records
    else:
        recent = records[-cutoff:]
        old = records[:-cutoff]

    summarized = [_summarize_old_tool_record(record) for record in old]
    summarized.extend(_full_tool_record(record) for record in recent)
    return summarized


def _compression_prompt(state: Any, settings: Any | None) -> str:
    payload = {
        "user_input": _get_state_value(state, "user_input", ""),
        "plan": _get_state_value(state, "plan", ""),
        "memory_summary": _get_state_value(state, "conversation_context", {}).get("summary", "")
        if isinstance(_get_state_value(state, "conversation_context", {}), Mapping)
        else "",
        "context": _get_state_value(state, "context", []),
        "task_state": _task_state_payload(state),
        "tool_outputs": summarize_tool_outputs(state, settings),
        "response_so_far": _get_state_value(state, "response", ""),
    }
    return (
        "Compress the following coding-agent context into a concise Chinese summary.\n"
        "Preserve user intent, decisions, file paths, commands, errors, blocked tool calls, and unresolved work.\n"
        "Older tool outputs may already be shortened; keep important evidence and omit noise.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, default=str)}"
    )


def _task_state_payload(state: Any) -> Any:
    snapshots = _get_state_value(state, "snapshots", {})
    if isinstance(snapshots, Mapping):
        return snapshots.get("task_state", {})
    if isinstance(state, Mapping):
        return state.get("task_state", {})
    return {}


def _threshold_ratio(compression_count: int, settings: Any | None = None) -> float:
    if compression_count <= int(_setting(settings, "context_long_task_after_compressions", AUXILIARY_AFTER_COUNT)):
        return float(_setting(settings, "context_regular_threshold", MAIN_COMPRESSION_RATIO))
    return float(_setting(settings, "context_long_task_threshold", AUXILIARY_COMPRESSION_RATIO))


def _provider_role_for_count(compression_count: int, settings: Any | None = None) -> str:
    after_count = int(_setting(settings, "context_long_task_after_compressions", AUXILIARY_AFTER_COUNT))
    return "main" if compression_count <= after_count else "auxiliary"


def _compression_count(state: Any) -> int:
    for key in ("compression_count", "context_compression_count"):
        value = _get_state_value(state, key, None)
        if value is not None:
            return int(value)
    snapshots = _get_state_value(state, "snapshots", {})
    if isinstance(snapshots, Mapping):
        return int(snapshots.get("compression_count", 0) or 0)
    return 0


def _token_budget(settings: Any | None, estimator: Any | None) -> int:
    for key in ("context_window_tokens", "context_token_budget", "max_context_tokens", "token_budget"):
        value = _setting(settings, key, None)
        if value is not None:
            return int(value)
    for key in ("token_budget", "max_tokens", "budget"):
        value = getattr(estimator, key, None)
        if value is not None and not callable(value):
            return int(value)
    return DEFAULT_CONTEXT_TOKEN_BUDGET


def _estimate_tokens(state: Any, estimator: Any | None) -> int:
    if estimator is not None:
        for method_name in ("estimate_state", "estimate", "count_state_tokens", "count_tokens"):
            method = getattr(estimator, method_name, None)
            if method is None:
                continue
            try:
                return int(method(state))
            except TypeError:
                continue
    text = json.dumps(_state_payload(state), ensure_ascii=False, default=str)
    return max(1, len(text) // 4)


def _with_compression_summary(state: Any, summary: str, compression_count: int) -> Any:
    if isinstance(state, dict):
        updated = dict(state)
        updated["compression_count"] = compression_count
        updated["context_summary"] = summary
        return updated

    for key, value in (("compression_count", compression_count), ("context_summary", summary)):
        try:
            setattr(state, key, value)
        except (AttributeError, TypeError):
            pass

    snapshots = _get_state_value(state, "snapshots", None)
    if isinstance(snapshots, dict):
        snapshots["compression_count"] = compression_count
        snapshots["context_summary"] = summary
    return state


def _tool_records(state: Any) -> Sequence[Any]:
    records = _get_state_value(state, "tool_calls", [])
    if isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)):
        return records
    return []


def _summarize_old_tool_record(record: Any) -> dict[str, Any]:
    result = _record_value(record, "result", None)
    return {
        "name": _record_value(record, "name", ""),
        "blocked": bool(_record_value(record, "blocked", False)),
        "reason": _record_value(record, "reason", None),
        "result": _fragment(result),
        "compressed": True,
    }


def _full_tool_record(record: Any) -> dict[str, Any]:
    return {
        "name": _record_value(record, "name", ""),
        "arguments": _record_value(record, "arguments", {}),
        "result": _record_value(record, "result", None),
        "blocked": bool(_record_value(record, "blocked", False)),
        "reason": _record_value(record, "reason", None),
        "compressed": False,
    }


def _fragment(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= TOOL_RESULT_FRAGMENT_CHARS:
        return text
    return f"{text[:TOOL_RESULT_FRAGMENT_CHARS]}...[truncated]"


def _state_payload(state: Any) -> Any:
    if isinstance(state, Mapping):
        return dict(state)
    snapshot = getattr(state, "snapshot", None)
    if callable(snapshot):
        return snapshot()
    return getattr(state, "__dict__", str(state))


def _get_state_value(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, Mapping):
        return state.get(key, default)
    return getattr(state, key, default)


def _record_value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _setting(settings: Any | None, key: str, default: Any = None) -> Any:
    if settings is None:
        return default
    if isinstance(settings, Mapping):
        return settings.get(key, default)
    return getattr(settings, key, default)
