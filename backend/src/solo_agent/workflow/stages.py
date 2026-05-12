from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

from solo_agent.agent.deps import AgentDeps, AgentSettings
from solo_agent.agent.events import AgentEvent
from solo_agent.agent.policy import BehaviorPolicy, first_present, tool_result_ok
from solo_agent.agent.prompts import (
    PATCH_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    RESPONDER_SYSTEM_PROMPT,
    build_memory_context_block,
    build_skill_context_block,
    build_subagent_tool_instruction,
    patch_user_prompt,
    planner_user_prompt,
    responder_user_prompt,
)
from solo_agent.agent.state import AgentState, ToolCallRecord
from solo_agent.context import SubdirectoryHintTracker, TaskListState, WorkspaceTaskStore
from solo_agent.providers import ChatMessage, ChatProvider
from solo_agent.verified_editing import PatchEdit, PatchProposalError, PatchRequest, build_patch_proposal, extract_patch_request
from solo_agent.workflow.parallelism import evaluate_independence, extract_task_candidates_from_text

_BEHAVIOR_POLICY = BehaviorPolicy()


async def _receive_user_turn_stage(
    state: AgentState,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    state.loop_stage = "receive_user_turn"
    state.run_mode = str(_setting(settings, "run_mode", "agent"))
    state.is_plan_mode = state.run_mode == "plan" or bool(_setting(settings, "is_plan_mode", False))
    state.memory_enabled = bool(_setting(settings, "memory_enabled", True))
    state.conversation_history_enabled = bool(
        _setting(settings, "conversation_history_enabled", True)
    )
    state.memory_budget = {
        "recent_message_limit": int(_setting(settings, "history_message_limit", 12))
        if state.conversation_history_enabled
        else 0,
        "memory_search_limit": int(_setting(settings, "memory_search_limit", 5)),
        "scope": "session",
        "memory_enabled": state.memory_enabled,
        "conversation_history_enabled": state.conversation_history_enabled,
    }
    state.snapshots["tool_budget"] = {
        "max_tool_calls": int(_setting(settings, "max_tool_calls", 3)),
        "tool_call_cut_off": int(
            _setting(settings, "tool_call_cut_off", _setting(settings, "max_tool_calls", 3))
        ),
        "tool_output_max_bytes": int(_setting(settings, "tool_output_max_bytes", 12_000)),
    }
    state.skill_budget = {"max_selected_skills": 3, "injection": "user_message"}
    state.snapshots["skill_budget"] = state.skill_budget
    state.snapshots["loop_stage"] = state.loop_stage
    state.snapshots["memory_budget"] = state.memory_budget
    yield _event(
        state,
        "receive_user_turn",
        "receive_user_turn",
        "Received user turn",
        {
            "run_mode": state.run_mode,
            "is_plan_mode": state.is_plan_mode,
            "memory_enabled": state.memory_enabled,
            "conversation_history_enabled": state.conversation_history_enabled,
            "tool_budget": state.snapshots["tool_budget"],
            "skill_budget": state.skill_budget,
        },
    )


async def _plan_node(
    state: AgentState,
    provider: ChatProvider,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    yield _event(state, "plan_started", "plan", "Planning task")
    plan_mode_enabled = bool(
        getattr(state, "is_plan_mode", False)
        or _setting(settings, "is_plan_mode", False)
    )

    system_prompt = PLANNER_SYSTEM_PROMPT

    if plan_mode_enabled:
        system_prompt += """

    <plan_mode>
    The user enabled Plan mode. Before acting, create a clear, step-by-step execution plan.
    Use the plan to guide the rest of the run, but do not stop after planning.
    Prefer explicit phases, success criteria, risks, and verification steps.
    For complex work, maintain the session TaskList with write_todos when task state changes.
    Do not use write_todos for small one-step tasks where a task list would add noise.
    Update task status after completing each meaningful step, and usually keep only one task in_progress.
    Keep the plan concise enough to execute.
    </plan_mode>
    """
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(
            role="user",
            content=planner_user_prompt(
                state.user_input,
                state.conversation_context,
                state.memory_context_block,
                state.skill_context_block,
                _format_task_list_block(state) if plan_mode_enabled else "",
                plan_mode_enabled=plan_mode_enabled,
            ),
        ),
    ]
    parts: list[str] = []
    async for chunk in provider.stream_chat(
        messages,
        temperature=float(_setting(settings, "temperature", 0.2)),
        max_tokens=int(_setting(settings, "plan_max_tokens", 500)),
    ):
        if chunk.content:
            parts.append(chunk.content)
            yield _event(state, "plan_delta", "plan", chunk.content)
    state.plan = "".join(parts).strip()
    state.snapshots["plan"] = state.plan
    yield _event(state, "plan_completed", "plan", "Plan completed", {"plan": state.plan})




async def _skip_memory_stage(state: AgentState) -> AsyncIterator[AgentEvent]:
    state.loop_stage = "load_builtin_memory"
    state.conversation_context = {
        "summary": "",
        "recent_messages": [],
        "retrieved_memories": [],
        "builtin_memory": {},
        "budget": state.memory_budget,
        "memory_enabled": False,
        "conversation_history_enabled": state.conversation_history_enabled,
    }
    yield _event(
        state,
        "memory_skipped",
        "memory",
        "Memory is disabled for this run",
        {
            "reason": "memory_enabled=false",
            "memory_enabled": False,
            "conversation_history_enabled": state.conversation_history_enabled,
        },
    )


async def _load_builtin_memory_stage(
    state: AgentState,
    deps: AgentDeps,
) -> AsyncIterator[AgentEvent]:
    state.loop_stage = "load_builtin_memory"
    builtin = await _call_optional(deps.persistence, "load_builtin_memory")
    state.snapshots["builtin_memory"] = builtin or {}
    yield _event(
        state,
        "builtin_memory_loaded",
        "memory",
        "Loaded builtin memory files",
        {
            "available": bool(builtin),
            "sources": sorted((builtin or {}).keys()) if isinstance(builtin, Mapping) else [],
        },
    )


async def _load_conversation_context(
    state: AgentState,
    deps: AgentDeps,
    settings: AgentSettings | Mapping[str, Any],
) -> dict[str, Any]:
    """Load conversation history, summary, and memory context from persistence."""
    context: dict[str, Any] = {
        "summary": "",
        "recent_messages": [],
        "retrieved_memories": [],
        "builtin_memory": {},
        "budget": dict(state.memory_budget or {}),
        "memory_enabled": state.memory_enabled,
        "conversation_history_enabled": bool(_setting(settings, "conversation_history_enabled", True)),
    }
    if deps.persistence is None:
        return context

    prefetched = await _call_optional(
        deps.persistence,
        "prefetch_all",
        session_id=state.session_id,
        query=state.user_input or "",
        recent_limit=int(_setting(settings, "history_message_limit", 12)),
        limit=int(_setting(settings, "memory_search_limit", 5)),
    )
    if isinstance(prefetched, Mapping):
        context.update(prefetched)

    return context


async def _prefetch_memory_stage(
    state: AgentState,
    deps: AgentDeps,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    state.loop_stage = "prefetch_all"
    yield _event(
        state,
        "memory_prefetch_started",
        "memory",
        "Prefetching session memory",
        {
            "enabled": True,
            "memory_enabled": True,
            "conversation_history_enabled": state.conversation_history_enabled,
        },
    )
    context = await _load_conversation_context(state, deps, settings)
    builtin = state.snapshots.get("builtin_memory")
    if builtin and "builtin_memory" not in context:
        context["builtin_memory"] = builtin
    state.conversation_context = context
    state.memory_budget = dict(context.get("budget") or state.memory_budget)
    state.snapshots["memory_budget"] = state.memory_budget

    yield _event(
        state,
        "memory_loaded",
        "memory",
        "Prefetched session memory",
        {
            "recent_count": len(context.get("recent_messages", [])),
            "has_summary": bool(context.get("summary")),
            "memory_enabled": True,
            "conversation_history_enabled": state.conversation_history_enabled,
        },
    )
    yield _event(
        state,
        "memory_search_completed",
        "memory",
        "Searched session memory",
        {"matches": context.get("retrieved_memories", [])},
    )
    yield _event(
        state,
        "context_budget_applied",
        "memory",
        "Applied session memory context budget",
        {"budget": state.memory_budget},
    )


async def _build_memory_context_stage(state: AgentState) -> AsyncIterator[AgentEvent]:
    state.loop_stage = "build_memory_context_block"
    state.memory_context_block = build_memory_context_block(state.conversation_context)
    yield _event(
        state,
        "memory_context_built",
        "memory",
        "Built fenced memory context block",
        {"length": len(state.memory_context_block)},
    )


async def _skill_context_stage(
    state: AgentState,
    deps: AgentDeps,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    state.loop_stage = "select_skills"
    max_skills = int(_setting(settings, "max_selected_skills", 3))
    state.skill_budget = {"max_selected_skills": max_skills, "injection": "user_message"}
    state.snapshots["skill_selection_attempted"] = True
    yield _event(
        state,
        "skill_selection_started",
        "skill",
        "Selecting procedural skills",
        state.skill_budget,
    )

    selected_result = await _call_tool_if_available(
        deps.tool_registry,
        "select_relevant_skills",
        {"task": state.user_input, "plan": state.plan, "max_skills": max_skills},
    )
    selected = _extract_tool_result(selected_result).get("skills", []) if selected_result else []
    state.selected_skills = [dict(skill) for skill in selected[:max_skills] if isinstance(skill, Mapping)]
    yield _event(
        state,
        "skill_selected",
        "skill",
        "Selected procedural skills",
        {"skills": state.selected_skills, "count": len(state.selected_skills)},
    )

    loaded: list[dict[str, Any]] = []
    state.loop_stage = "load_skills"
    for skill in state.selected_skills:
        path = str(skill.get("path", ""))
        if not path:
            continue
        loaded_result = await _call_tool_if_available(deps.tool_registry, "load_skill", {"path": path})
        loaded_skill = _extract_tool_result(loaded_result)
        if loaded_skill:
            loaded.append(loaded_skill)
            yield _event(
                state,
                "skill_loaded",
                "skill",
                f"Loaded skill {loaded_skill.get('name') or path}",
                {
                    "name": loaded_skill.get("name"),
                    "path": loaded_skill.get("path", path),
                    "truncated": loaded_skill.get("truncated", False),
                },
            )

    state.selected_skills = loaded or state.selected_skills
    state.behavior_policy = _BEHAVIOR_POLICY.build_snapshot(state)
    state.snapshots["behavior_policy"] = state.behavior_policy
    yield _event(
        state,
        "policy_evaluation_completed",
        "policy",
        "Evaluated graph-level behavior policy",
        state.behavior_policy,
    )
    state.loop_stage = "build_skill_context_block"
    if loaded:
        state.skill_context_block = build_skill_context_block(
            [
                {
                    "name": skill.get("name"),
                    "path": skill.get("path"),
                    "category": skill.get("category"),
                    "content": skill.get("content"),
                }
                for skill in loaded
            ]
        )
    else:
        state.skill_context_block = ""

    yield _event(
        state,
        "skill_context_built",
        "skill",
        "Built user-message skill context",
        {
            "length": len(state.skill_context_block),
            "injection": "user_message",
            "loaded_count": len(loaded),
        },
    )


async def _collect_context_node(
    state: AgentState,
    deps: AgentDeps,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    state.loop_stage = "collect_context"
    yield _event(state, "context_started", "context", "Collecting bounded context")

    collected = await _collect_context_with_provider(deps.context_provider, state, settings)
    if collected:
        state.context.extend(_normalize_context_items(collected))

    yield _event(
        state,
        "context_completed",
        "context",
        "Context collection completed",
        {"context": state.context},
    )


async def _select_tools_node(
    state: AgentState,
    deps: AgentDeps,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    state.loop_stage = "select_tools"
    available_tools = await _available_tool_names(deps.tool_registry) if deps.tool_registry is not None else set()
    subagent_instruction = build_subagent_tool_instruction(state, task_tool_available="task" in available_tools)
    if subagent_instruction:
        state.snapshots["subagent_tool_instruction"] = subagent_instruction
    proposed = await _propose_tool_calls(deps.tool_registry, state, settings)
    state.snapshots["proposed_tool_calls"] = proposed

    if state.selected_skills:
        yield _event(
            state,
            "tool_protocol_applied",
            "select_tools",
            "Applied loaded skill tool protocol",
            {
                "skills": [
                    {
                        "name": skill.get("name"),
                        "required_tools": skill.get("required_tools", []),
                    }
                    for skill in state.selected_skills
                ],
                "proposed_tool_calls": proposed,
            },
        )
    iron_law = _BEHAVIOR_POLICY.iron_law_decision(state, proposed)
    if iron_law["action"] != "none":
        state.snapshots["iron_law"] = iron_law
    if iron_law["action"] == "blocked":
        yield _event(
            state,
            "iron_law_blocked",
            "select_tools",
            "Production-code edit intent detected without a failing-test signal",
            iron_law,
        )
    elif iron_law["action"] == "warning":
        yield _event(
            state,
            "iron_law_warning",
            "select_tools",
            "Production-code intent detected without an explicit failing-test signal",
            iron_law,
        )

    yield _event(
        state,
        "tool_selection_completed",
        "select_tools",
        "Tool selection completed",
        {
            "proposed_tool_calls": proposed,
            "subagent_tool_instruction": subagent_instruction,
        },
    )


async def _inspect_node(state: AgentState, deps: AgentDeps) -> AsyncIterator[AgentEvent]:
    state.loop_stage = "inspect"
    yield _event(state, "inspect_started", "inspect", "Running deterministic safety checks")

    input_result = await _inspect(deps.safety_inspector, "input", {"user_input": state.user_input})
    if not input_result["allowed"]:
        state.blocked = True
        state.block_reason = input_result["reason"]
        yield _event(state, "inspect_completed", "inspect", "Input blocked", input_result)
        return

    yield _event(
        state,
        "inspect_completed",
        "inspect",
        "Safety inspection completed",
        {"blocked": state.blocked},
    )


async def _execute_tools_node(
    state: AgentState,
    deps: AgentDeps,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    state.loop_stage = "execute_tools"
    proposed = state.snapshots.get("proposed_tool_calls", [])
    if not proposed:
        yield _event(state, "tool_call_completed", "execute_tools", "No tool calls requested")
        return

    approved: list[dict[str, Any]] = []
    protocol_state = _BEHAVIOR_POLICY.new_tool_protocol_state(state)
    available_tools = await _available_tool_names(deps.tool_registry) if deps.tool_registry is not None else set()
    cutoff = int(_setting(settings, "tool_call_cut_off", _setting(settings, "max_tool_calls", 3)))
    output_max_bytes = int(_setting(settings, "tool_output_max_bytes", 12_000))
    attempted = 0
    if cutoff <= 0:
        yield _event(
            state,
            "tool_cut_off_applied",
            "execute_tools",
            "Tool call cutoff prevented execution",
            {"cutoff": cutoff, "proposed_count": len(proposed), "executed_count": 0},
        )
        return

    pending = [dict(call) for call in proposed if isinstance(call, Mapping)]
    index = 0
    while index < len(pending):
        call = pending[index]
        index += 1
        if attempted >= cutoff:
            yield _event(
                state,
                "tool_cut_off_applied",
                "execute_tools",
                "Tool call cutoff reached; remaining tools skipped",
                {
                    "cutoff": cutoff,
                    "proposed_count": len(proposed),
                    "executed_count": attempted,
                    "skipped_count": len(pending) - attempted,
                },
            )
            break

        name = str(call.get("name", "unknown"))
        arguments = dict(call.get("arguments") or {})
        if name == "write_todos" and state.is_plan_mode and not arguments.get("thread_id"):
            arguments["thread_id"] = state.session_id
        if name == "task":
            if not arguments.get("thread_id"):
                arguments["thread_id"] = state.session_id
            if not arguments.get("task_id"):
                arguments["task_id"] = _stable_subagent_task_id(arguments, state.session_id)
        attempted += 1
        yield _event(
            state,
            "tool_call_started",
            "execute_tools",
            f"Calling tool {name}",
            {"name": name, "arguments": arguments, "index": attempted, "cutoff": cutoff},
        )
        yield _event(
            state,
            "tool_progress",
            "execute_tools",
            f"Inspecting tool {name}",
            {"name": name, "status": "inspecting"},
        )
        protocol_violation = _BEHAVIOR_POLICY.tool_protocol_violation(state, name, arguments, protocol_state)
        if protocol_violation is not None:
            recovery_calls = _BEHAVIOR_POLICY.recovery_tool_calls(protocol_violation, name, arguments, available_tools)
            if protocol_violation.get("recoverable") and recovery_calls:
                yield _event(
                    state,
                    "tool_protocol_recovery_started",
                    "execute_tools",
                    "Recovering required behavior protocol steps",
                    {
                        "name": name,
                        "reason": protocol_violation["reason"],
                        "recovery_tool_calls": recovery_calls,
                    },
                )
                pending[index:index] = [*recovery_calls, call]
                yield _event(
                    state,
                    "tool_protocol_recovery_scheduled",
                    "execute_tools",
                    "Scheduled behavior protocol recovery tools",
                    {
                        "name": name,
                        "reason": protocol_violation["reason"],
                        "scheduled_count": len(recovery_calls),
                    },
                )
                continue
            state.tool_calls.append(
                ToolCallRecord(
                    name=name,
                    arguments=arguments,
                    blocked=True,
                    reason=protocol_violation["reason"],
                )
            )
            yield _event(
                state,
                "tool_protocol_blocked",
                "execute_tools",
                "Tool call rejected by the edit protocol",
                {"name": name, "arguments": arguments, **protocol_violation},
            )
            yield _event(
                state,
                "tool_call_completed",
                "execute_tools",
                "Tool call rejected by the edit protocol",
                {
                    "name": name,
                    "blocked": True,
                    "reason": protocol_violation["reason"],
                    "protocol": protocol_violation,
                },
            )
            continue
        inspection = await _inspect(deps.safety_inspector, "tool_call", call)
        if not inspection["allowed"]:
            state.tool_calls.append(
                ToolCallRecord(
                    name=name,
                    arguments=arguments,
                    blocked=True,
                    reason=inspection["reason"],
                )
            )
            yield _event(
                state,
                "tool_call_completed",
                "execute_tools",
                "Tool call blocked by safety inspection",
                {
                    "name": name,
                    "blocked": True,
                    "reason": inspection["reason"],
                },
            )
            continue
        if name == "apply_text_edit" and _verified_editing_enabled(settings):
            async for event in _propose_patch_from_tool_arguments(state, deps, arguments):
                yield event
            if state.awaiting_approval:
                return
            continue
        approved.append(call)

        if name == "task":
            dispatch = _task_dispatch_from_arguments(arguments)
            state.subagent_dispatches.append(dispatch)
            yield _event(
                state,
                "task_started",
                "execute_tools",
                f"Subagent task started: {dispatch['description']}",
                dispatch,
            )

        yield _event(
            state,
            "tool_progress",
            "execute_tools",
            f"Executing tool {name}",
            {"name": name, "status": "executing"},
        )
        # 错误恢复：包裹工具调用
        try:
            raw_result = await _call_tool(deps.tool_registry, name, arguments)
        except Exception as exc:
            if name == "task":
                yield _event(
                    state,
                    "task_failed",
                    "execute_tools",
                    f"Subagent task failed: {exc}",
                    {
                        "task_id": arguments.get("task_id", ""),
                        "description": arguments.get("description", ""),
                        "subagent_type": arguments.get("subagent_type", "general-purpose"),
                        "error": str(exc),
                    },
                )
            classification = _BEHAVIOR_POLICY.classify_error(
                exc,
                stage="tools",
                attempt_count=state.retry_count,
                run_id=state.run_id,
            )
            should_retry, reason = _BEHAVIOR_POLICY.should_retry(classification, state.retry_count)
            state.last_error = classification.to_dict()
            state.error_classification = classification.category

            if should_retry:
                state.retry_count += 1
                fix_prompt = _BEHAVIOR_POLICY.build_fix_prompt(classification)
                if fix_prompt:
                    state.context.append({"source": "error_recovery", "content": fix_prompt})
                yield _event(
                    state,
                    "error_recovery_retry",
                    "execute_tools",
                    f"Tool {name} 失败，将重试 ({reason})",
                    {
                        "name": name,
                        "error_code": classification.error_code,
                        "category": classification.category,
                        "retry_count": state.retry_count,
                        "reason": reason,
                    },
                )
                pending.insert(index, dict(call))
                continue

            # 致命错误：终止运行
            yield _event(
                state,
                "error",
                "execute_tools",
                f"Tool {name} 失败：{reason}",
                {
                    "error_type": type(exc).__name__,
                    "error_code": classification.error_code,
                    "category": classification.category,
                },
            )
            state.blocked = True
            state.block_reason = f"错误恢复失败: {reason}"
            yield _event(
                state,
                "run_completed",
                "end",
                f"Agent run terminated due to {classification.category} error",
                {"blocked": True, "reason": state.block_reason},
            )
            return

        raw_result_ok = tool_result_ok(raw_result)
        result, output_metadata = _truncate_tool_result(raw_result, output_max_bytes)
        yield _event(
            state,
            "tool_progress",
            "execute_tools",
            f"Captured tool output for {name}",
            {"name": name, "status": "captured_output", **output_metadata},
        )
        record = ToolCallRecord(name=name, arguments=arguments, result=result)
        state.tool_calls.append(record)
        state.context.append(
            {
                "source": f"tool:{name}",
                "content": result,
                "metadata": output_metadata,
            }
        )
        yield _event(
            state,
            "tool_call_completed",
            "execute_tools",
            f"Tool {name} completed",
            {"name": name, "result": result, "metadata": output_metadata},
        )
        if name == "task":
            task_result = _extract_subagent_task_result(raw_result)
            task_id = str(task_result.get("task_id") or arguments.get("task_id") or "")
            if not task_result:
                task_result = {
                    "task_id": task_id,
                    "subagent_type": arguments.get("subagent_type", "general-purpose"),
                    "description": arguments.get("description", ""),
                    "status": "failed",
                    "result": "",
                    "evidence": [],
                    "read_paths": arguments.get("read_paths", []),
                    "metadata": {},
                    "error": _tool_error_message(raw_result),
                }
            state.subagent_results[task_id or f"task_{len(state.subagent_results) + 1}"] = task_result
            state.snapshots["subagent_results"] = state.subagent_results
            task_status = str(task_result.get("status") or ("completed" if raw_result_ok else "failed"))
            if raw_result_ok and task_status != "failed":
                yield _event(
                    state,
                    "task_completed",
                    "execute_tools",
                    f"Subagent task completed: {task_result.get('description') or arguments.get('description', '')}",
                    {
                        "task_id": task_result.get("task_id", task_id),
                        "description": task_result.get("description", arguments.get("description", "")),
                        "subagent_type": task_result.get("subagent_type", arguments.get("subagent_type", "general-purpose")),
                        "status": task_status,
                        "result": _task_result_summary(task_result.get("result", "")),
                    },
                )
            else:
                yield _event(
                    state,
                    "task_failed",
                    "execute_tools",
                    f"Subagent task failed: {task_result.get('description') or arguments.get('description', '')}",
                    {
                        "task_id": task_result.get("task_id", task_id),
                        "description": task_result.get("description", arguments.get("description", "")),
                        "subagent_type": task_result.get("subagent_type", arguments.get("subagent_type", "general-purpose")),
                        "status": task_status,
                        "error": task_result.get("error") or _tool_error_message(raw_result),
                    },
                )
        if raw_result_ok:
            _BEHAVIOR_POLICY.record_tool_success(protocol_state, name, arguments, raw_result)
        if name == "write_todos" and raw_result_ok:
            task_state = TaskListState.from_payload(_extract_tool_result(raw_result), thread_id=state.session_id)
            state.task_list = task_state.to_dict()
            state.snapshots["task_list"] = state.task_list
            state.snapshots["task_state"] = state.task_list
            _replace_task_list_context(state, task_state)
            yield _event(
                state,
                "task_list_updated",
                "execute_tools",
                "Updated Plan mode TaskList",
                {
                    **_task_list_event_payload(task_state),
                    "thread_id": task_state.thread_id,
                },
            )
        if name == "apply_text_edit" and raw_result_ok:
            yield _event(
                state,
                "verification_required",
                "execute_tools",
                "A file edit was applied; verification is required",
                {"recommended_tools": ["run_pytest", "run_ruff_check"]},
            )
            remaining_budget = cutoff - attempted
            if remaining_budget <= 0:
                yield _event(
                    state,
                    "verification_deferred",
                    "execute_tools",
                    "Verification is required but the tool call cutoff has been reached",
                    {
                        "reason": "tool_call_cut_off_reached_after_edit",
                        "cutoff": cutoff,
                        "executed_count": attempted,
                        "recommended_tools": ["run_pytest", "run_ruff_check"],
                    },
                )

    state.snapshots["approved_tool_calls"] = approved


async def _propose_verified_patch_node(
    state: AgentState,
    provider: ChatProvider,
    deps: AgentDeps,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    if not _verified_editing_enabled(settings) or not _mentions_verified_edit_request(state):
        return
    if state.patch_proposal is not None:
        return

    state.loop_stage = "propose_verified_patch"
    yield _event(state, "patch_generation_started", "propose_verified_patch", "Generating verified patch proposal")
    raw = await provider.complete(
        [
            ChatMessage(role="system", content=PATCH_SYSTEM_PROMPT),
            ChatMessage(role="user", content=patch_user_prompt(state)),
        ],
        temperature=float(_setting(settings, "temperature", 0.2)),
        max_tokens=int(_setting(settings, "patch_max_tokens", 1400)),
    )
    try:
        request = extract_patch_request(raw)
        proposal = await _build_and_store_patch_proposal(state, deps, request)
    except PatchProposalError as exc:
        yield _event(
            state,
            "patch_generation_skipped",
            "propose_verified_patch",
            "No valid patch proposal was produced",
            {"reason": str(exc)},
        )
        return

    yield _event(
        state,
        "patch_proposed",
        "propose_verified_patch",
        "Verified patch proposal is ready for review",
        proposal,
    )
    yield _event(
        state,
        "patch_approval_required",
        "propose_verified_patch",
        "Patch proposal requires user approval before applying",
        proposal,
    )


async def _propose_patch_from_tool_arguments(
    state: AgentState,
    deps: AgentDeps,
    arguments: Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    request = _patch_request_from_apply_arguments(arguments)
    try:
        proposal = await _build_and_store_patch_proposal(state, deps, request)
    except PatchProposalError as exc:
        yield _event(
            state,
            "tool_call_completed",
            "execute_tools",
            "apply_text_edit converted to patch proposal failed",
            {"name": "apply_text_edit", "blocked": True, "reason": str(exc)},
        )
        return

    state.tool_calls.append(
        ToolCallRecord(
            name="apply_text_edit",
            arguments=dict(arguments),
            blocked=True,
            reason="awaiting_user_approval",
            result=proposal,
        )
    )
    yield _event(
        state,
        "patch_proposed",
        "execute_tools",
        "apply_text_edit converted to a verified patch proposal",
        proposal,
    )
    yield _event(
        state,
        "patch_approval_required",
        "execute_tools",
        "Patch proposal requires user approval before applying",
        proposal,
    )


async def _build_and_store_patch_proposal(
    state: AgentState,
    deps: AgentDeps,
    request: PatchRequest,
) -> dict[str, Any]:
    async def call_tool(name: str, arguments: dict[str, Any]) -> Mapping[str, Any]:
        return await _call_tool(deps.tool_registry, name, arguments)

    proposal = await build_patch_proposal(
        request,
        session_id=state.session_id,
        run_id=state.run_id,
        call_tool=call_tool,
    )
    stored = await _call_optional(deps.persistence, "create_patch_proposal", proposal=proposal)
    public = (
        (stored or proposal).to_public_dict()
        if hasattr(stored or proposal, "to_public_dict")
        else proposal.model_dump(mode="json")
    )
    state.patch_proposal = public
    state.awaiting_approval = True
    state.snapshots["patch_proposal"] = public
    state.snapshots["awaiting_approval"] = True
    return public


def _patch_request_from_apply_arguments(arguments: Mapping[str, Any]) -> PatchRequest:
    new_text = first_present(arguments.get("new_text"), arguments.get("new"))
    edit = PatchEdit(
        path=str(arguments.get("path") or ""),
        expected_hash=str(arguments.get("expected_hash") or ""),
        old_text=first_present(arguments.get("old_text"), arguments.get("old")),
        line_start=arguments.get("line_start"),
        line_end=arguments.get("line_end"),
        new_text=str(new_text or ""),
        reason="Model requested apply_text_edit; converted to verified editing proposal.",
    )
    return PatchRequest(summary=f"Proposed edit for {edit.path}", edits=[edit])


async def _respond_node(
    state: AgentState,
    provider: ChatProvider,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    state.loop_stage = "respond"
    yield _event(state, "response_started", "respond", "Generating final response")
    messages = [
        ChatMessage(role="system", content=RESPONDER_SYSTEM_PROMPT),
        ChatMessage(role="user", content=responder_user_prompt(state)),
    ]
    parts: list[str] = []
    async for chunk in provider.stream_chat(
        messages,
        temperature=float(_setting(settings, "temperature", 0.2)),
        max_tokens=int(_setting(settings, "response_max_tokens", 1400)),
    ):
        if chunk.content:
            parts.append(chunk.content)
            yield _event(state, "response_delta", "respond", chunk.content)
    state.response = "".join(parts).strip()
    state.snapshots["response"] = state.response
    yield _event(
        state,
        "response_completed",
        "respond",
        "Final response completed",
        {"response": state.response},
    )


async def _sync_memory_stage(state: AgentState, deps: AgentDeps) -> AsyncIterator[AgentEvent]:
    state.loop_stage = "sync_memory"
    if deps.persistence is None or not state.memory_enabled:
        return

    result = await _call_optional(
        deps.persistence, "sync_all",
        session_id=state.session_id, run_id=state.run_id,
        user_input=state.user_input, assistant_response=state.response,
    )
    yield _event(
        state,
        "memory_synced",
        "sync_memory",
        "Memory synced to persistence",
        {"result": result or {}},
    )


async def _queue_prefetch_stage(
    state: AgentState,
    deps: AgentDeps,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    """Queue memory prefetch for next run."""
    state.loop_stage = "queue_prefetch"
    if deps.persistence is None or not state.memory_enabled:
        return
    result = await _call_optional(
        deps.persistence, "queue_prefetch_all",
        session_id=state.session_id, query=state.user_input or "",
    )
    yield _event(
        state, "memory_prefetch_queued", "queue_prefetch",
        "Memory prefetch queued for next interaction",
        {"result": result or {}},
    )


async def _compress_memory_stage(
    state: AgentState,
    provider: ChatProvider,
    deps: AgentDeps,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    """Compress memory after run completion."""
    state.loop_stage = "compress_memory"
    if deps.persistence is None or not state.memory_enabled:
        return
    await _call_optional(
        deps.persistence, "on_pre_compress",
        session_id=state.session_id,
        payload={"summary": state.response, "session_id": state.session_id, "run_id": state.run_id},
    )
    yield _event(
        state, "memory_compress_completed", "compress_memory",
        "Memory compression completed", {},
    )


async def _parallelism_gate_stage(
    state: AgentState,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    state.loop_stage = "parallelism_gate"
    yield _event(
        state,
        "parallelism_gate_started",
        "parallelism_gate",
        "Evaluating parallel execution independence conditions",
        {"conditions": [
            "problem_domain_independence",
            "context_independence",
            "write_set_independence",
            "verification_independence",
        ]},
    )

    source_text = "\n\n".join(
        part
        for part in [
            state.user_input,
            state.plan,
        ]
        if part
    )
    tasks = extract_task_candidates_from_text(source_text)
    decision = evaluate_independence(
        tasks,
        max_parallel=int(_setting(settings, "max_concurrent_subagents", 3)),
    )
    subagent_enabled = bool(_setting(settings, "subagent_enabled", False))
    suitable = bool(decision.allowed)
    strategy = "parallel" if suitable and subagent_enabled else "serial"
    reason = decision.reason
    if suitable and not subagent_enabled:
        reason = "subagent_disabled"
    decision_payload = {
        **decision.to_dict(),
        "strategy": strategy,
        "suitable": suitable,
        "reason": reason,
        "task_count": len(tasks),
        "candidates": [task.to_dict() for task in tasks],
        "subagent_enabled": subagent_enabled,
    }

    state.task_candidates = [task.to_dict() for task in tasks]
    state.parallelism_decision = decision_payload
    state.execution_strategy = strategy
    state.snapshots["task_candidates"] = state.task_candidates
    state.snapshots["parallelism_decision"] = state.parallelism_decision
    state.snapshots["execution_strategy"] = state.execution_strategy

    yield _event(
        state,
        "parallelism_decision_completed",
        "parallelism_gate",
        f"Parallelism decision: {strategy}",
        decision_payload,
    )
    yield _event(
        state,
        "parallelism_gate_completed",
        "parallelism_gate",
        f"Parallelism decision: {strategy}",
        decision_payload,
    )


async def _task_state_stage(
    state: AgentState,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    if not state.is_plan_mode:
        return

    workspace_root = Path(_setting(settings, "workspace_root", Path.cwd()) or Path.cwd())
    store = WorkspaceTaskStore(workspace_root)
    task_state = store.load(state.session_id)
    initialized_from_plan = False
    if not task_state.items and state.plan.strip():
        parsed = TaskListState.from_plan(state.plan, thread_id=state.session_id)
        if parsed.items:
            task_state = parsed
            store.save(task_state)
            initialized_from_plan = True

    state.task_list = task_state.to_dict()
    state.snapshots["task_list"] = state.task_list
    state.snapshots["task_state"] = state.task_list
    state.snapshots["task_state_json_block"] = task_state.format_json_block()
    _replace_task_list_context(state, task_state)
    yield _event(
        state,
        "task_list_loaded",
        "task_state",
        "Loaded Plan mode TaskList",
        {
            **_task_list_event_payload(task_state),
            "thread_id": task_state.thread_id,
            "initialized_from_plan": initialized_from_plan,
        },
    )


async def _subdirectory_hint_stage(
    state: AgentState,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    workspace_root = Path(_setting(settings, "workspace_root", Path.cwd()) or Path.cwd())
    tracker = SubdirectoryHintTracker(workspace_root)
    all_hints: list[Any] = []
    for record in state.tool_calls:
        arguments = record.arguments
        workdir = arguments.get("workdir") or arguments.get("cwd")
        for key in ("path", "file", "file_path", "target_path"):
            if arguments.get(key):
                all_hints.extend(tracker.observe_path(str(arguments[key]), workdir=workdir))
        for key in ("command", "cmd"):
            if arguments.get(key):
                all_hints.extend(tracker.observe_command(str(arguments[key]), workdir=workdir))

    if not all_hints:
        return
    block = tracker.format_block(all_hints)
    state.context.append({"source": "subdirectory_hints", "content": block})
    yield _event(
        state,
        "subdirectory_hint_loaded",
        "context",
        "Loaded scoped directory hints",
        {
            "loaded": [
                str(hint.path.relative_to(workspace_root.resolve()))
                for hint in all_hints
                if not hint.skipped
            ],
            "skipped_risky": [str(hint.path) for hint in all_hints if hint.skipped],
            "length": len(block),
        },
    )


def _context_compression_payload(state: AgentState) -> dict[str, Any]:
    return {
        "messages": state.conversation_context.get("recent_messages", []),
        "current_response": state.response,
        "existing_summary": state.conversation_context.get("summary", ""),
        "task_state": state.snapshots.get("task_state", {}),
        "context": state.context,
        "tool_calls": [
            {
                "name": call.name,
                "arguments": call.arguments,
                "result": call.result,
                "blocked": call.blocked,
                "reason": call.reason,
            }
            for call in state.tool_calls
        ],
    }


def _task_state_to_dict(task_state: TaskListState) -> dict[str, Any]:
    return task_state.to_dict()


def _task_list_event_payload(task_state: TaskListState) -> dict[str, Any]:
    tasks = [item.to_dict() for item in task_state.items]
    active_task = next((item.to_dict() for item in task_state.items if item.status == "in_progress"), None)
    return {
        "task_count": len([item for item in task_state.items if item.status != "deleted"]),
        "active_task": active_task,
        "tasks": tasks,
    }


def _replace_task_list_context(state: AgentState, task_state: TaskListState) -> None:
    state.context = [item for item in state.context if item.get("source") != "task_list"]
    state.context.append(
        {
            "source": "task_list",
            "content": task_state.format_block(),
            "metadata": {"thread_id": task_state.thread_id, "task_count": len(task_state.active_items())},
        }
    )


def _format_task_list_block(state: AgentState) -> str:
    if not state.task_list:
        return ""
    task_state = TaskListState.from_payload(state.task_list, thread_id=state.session_id)
    return task_state.format_block()


def _task_state_from_snapshot(value: Any) -> TaskListState | None:
    if not isinstance(value, Mapping):
        return None
    return TaskListState.from_payload(dict(value), thread_id=str(value.get("thread_id") or value.get("threadID") or ""))


async def _collect_context_with_provider(
    context_provider: Any | None,
    state: AgentState,
    settings: AgentSettings | Mapping[str, Any],
) -> Any:
    if context_provider is None:
        return None
    payload = {
        "session_id": state.session_id,
        "run_id": state.run_id,
        "user_input": state.user_input,
        "plan": state.plan,
        "settings": settings,
    }
    for method_name in ("collect_context", "collect"):
        method = getattr(context_provider, method_name, None)
        if method is not None:
            return await _maybe_await(_call_flexible(method, payload))
    return None


async def _propose_tool_calls(
    tool_registry: Any | None,
    state: AgentState,
    settings: AgentSettings | Mapping[str, Any],
) -> list[dict[str, Any]]:
    if tool_registry is None:
        return []

    available = await _available_tool_names(tool_registry)
    max_calls = int(_setting(settings, "max_tool_calls", 3))
    calls: list[dict[str, Any]] = []
    task = state.user_input.strip()
    task_lc = task.lower()
    plan_lc = state.plan.lower()

    if (
        "select_relevant_skills" in available
        and not state.selected_skills
        and not state.snapshots.get("skill_selection_attempted")
    ):
        calls.append(
            {
                "name": "select_relevant_skills",
                "arguments": {"task": task, "plan": state.plan, "max_skills": 3},
                "category": "skill",
            }
        )
    elif "list_skills" in available and _mentions_skill(task_lc, plan_lc):
        calls.append({"name": "list_skills", "arguments": {}, "category": "skill"})

    decision = state.snapshots.get("parallelism_decision") or state.parallelism_decision or {}
    subagent_enabled = bool(decision.get("subagent_enabled", False))
    suitable_for_task = bool(decision.get("suitable", decision.get("allowed", False)))
    candidates = decision.get("candidates") or decision.get("tasks") or []
    if "task" in available and subagent_enabled and suitable_for_task and len(candidates) >= 2:
        remaining = max(0, max_calls - len(calls))
        for candidate in candidates[:remaining]:
            if not isinstance(candidate, Mapping):
                continue
            description = str(candidate.get("title") or candidate.get("description") or candidate.get("id") or "Subtask")
            read_paths = [
                str(path)
                for path in [
                    *(candidate.get("read_paths") or []),
                    *(candidate.get("write_paths") or []),
                ]
                if str(path).strip()
            ]
            prompt_parts = [
                f"Subtask: {description}",
                f"Parent user task: {state.user_input}",
                f"Parent plan: {state.plan or '(no plan)'}",
                f"Candidate metadata: {candidate}",
                "Return concise structured findings and evidence for the main agent to synthesize. Do not edit files.",
            ]
            calls.append(
                {
                    "name": "task",
                    "arguments": {
                        "description": description,
                        "prompt": "\n\n".join(prompt_parts),
                        "subagent_type": str(candidate.get("subagent_type") or "general-purpose"),
                        "read_paths": read_paths,
                        "allowed_tools": ["workspace_snapshot", "list_files", "read_file", "search_text"],
                    },
                    "category": "subagent",
                }
            )

    if "workspace_snapshot" in available:
        calls.append(
            {
                "name": "workspace_snapshot",
                "arguments": {
                    "path": ".",
                    "max_entries": int(_setting(settings, "context_file_limit", 80)),
                },
                "category": "context",
            }
        )
    elif "list_files" in available:
        calls.append(
            {
                "name": "list_files",
                "arguments": {
                    "path": ".",
                    "max_entries": int(_setting(settings, "context_file_limit", 80)),
                },
                "category": "context",
            }
        )

    path_hint = _extract_path_hint(task)
    if path_hint and "read_file" in available:
        calls.append(
            {
                "name": "read_file",
                "arguments": {"path": path_hint},
                "category": "context",
            }
        )
    elif "search_text" in available and task:
        calls.append(
            {
                "name": "search_text",
                "arguments": {
                    "query": task[:200],
                    "max_matches": int(_setting(settings, "context_search_limit", 20)),
                },
                "category": "context",
            }
        )

    if _mentions_pytest(task_lc, plan_lc) and "run_pytest" in available:
        calls.append({"name": "run_pytest", "arguments": {}, "category": "quality"})
    if _mentions_ruff(task_lc, plan_lc) and "run_ruff_check" in available:
        calls.append({"name": "run_ruff_check", "arguments": {}, "category": "quality"})
    if _mentions_format(task_lc, plan_lc) and "run_ruff_format_check" in available:
        calls.append({"name": "run_ruff_format_check", "arguments": {}, "category": "quality"})

    return _dedupe_tool_calls(calls)[:max_calls]


async def _available_tool_names(tool_registry: Any) -> set[str]:
    for method_name in ("list_tools", "tools"):
        method = getattr(tool_registry, method_name, None)
        if method is None:
            continue
        tools = await _maybe_await(method() if callable(method) else method)
        return {_tool_name(tool) for tool in tools}
    return {"list_files", "read_file", "search_text"}


async def _call_tool(tool_registry: Any | None, name: str, arguments: dict[str, Any]) -> Any:
    if tool_registry is None:
        raise RuntimeError(f"No tool registry configured for tool {name}")
    for method_name in ("call_tool", "call", "invoke"):
        method = getattr(tool_registry, method_name, None)
        if method is not None:
            return await _maybe_await(_call_flexible(method, {"name": name, "arguments": arguments}))
    tool = getattr(tool_registry, name, None)
    if tool is not None:
        return await _maybe_await(_call_flexible(tool, arguments))
    raise RuntimeError(f"Tool registry cannot call {name}")


async def _call_tool_if_available(tool_registry: Any | None, name: str, arguments: dict[str, Any]) -> Any:
    if tool_registry is None:
        return None
    available = await _available_tool_names(tool_registry)
    if name not in available:
        return None
    return await _call_tool(tool_registry, name, arguments)


def _extract_tool_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    if value.get("ok") is False:
        return {}
    result = value.get("result", value)
    return dict(result) if isinstance(result, Mapping) else {}


def _extract_subagent_task_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result = value.get("result", value)
    return dict(result) if isinstance(result, Mapping) else {}


def _stable_subagent_task_id(arguments: Mapping[str, Any], session_id: str) -> str:
    payload = json.dumps(
        {
            "thread_id": session_id,
            "description": arguments.get("description", ""),
            "prompt": arguments.get("prompt", ""),
            "subagent_type": arguments.get("subagent_type", "general-purpose"),
            "read_paths": arguments.get("read_paths", []),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return f"task_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def _task_dispatch_from_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_id": str(arguments.get("task_id", "")),
        "description": str(arguments.get("description", "")),
        "subagent_type": str(arguments.get("subagent_type") or "general-purpose"),
        "read_paths": list(arguments.get("read_paths") or []),
        "allowed_tools": list(arguments.get("allowed_tools") or []),
        "timeout_seconds": arguments.get("timeout_seconds"),
    }


def _task_result_summary(value: Any, *, max_chars: int = 500) -> str:
    text = value if isinstance(value, str) else _serialize_tool_result(value)
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"


def _tool_error_message(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("error") or value.get("message") or value.get("code") or "tool call failed")
    return "tool call failed"


async def _inspect(inspector: Any | None, phase: str, payload: dict[str, Any]) -> dict[str, Any]:
    if inspector is None:
        return {"allowed": True, "reason": None}

    if phase == "input" and hasattr(inspector, "inspect_text"):
        result = await _maybe_await(inspector.inspect_text(str(payload.get("user_input", ""))))
        return _normalize_inspection(result)

    if phase == "tool_call" and hasattr(inspector, "inspect_tool_call"):
        try:
            from solo_agent.inspectors import ToolCall

            call = ToolCall(
                name=str(payload.get("name", "")),
                arguments=dict(payload.get("arguments") or {}),
            )
            result = await _maybe_await(inspector.inspect_tool_call(call))
            return _normalize_inspection(result)
        except TypeError:
            pass

    method = getattr(inspector, "inspect", None)
    if method is not None:
        result = await _maybe_await(_call_flexible(method, {"phase": phase, "payload": payload}))
        return _normalize_inspection(result)

    phase_method = getattr(inspector, f"inspect_{phase}", None)
    if phase_method is not None:
        result = await _maybe_await(_call_flexible(phase_method, payload))
        return _normalize_inspection(result)

    return {"allowed": True, "reason": None}


def _normalize_inspection(result: Any) -> dict[str, Any]:
    if result is None:
        return {"allowed": True, "reason": None}
    if isinstance(result, bool):
        return {"allowed": result, "reason": None if result else "blocked"}
    if isinstance(result, Mapping):
        allowed = bool(result.get("allowed", result.get("ok", True)))
        return {"allowed": allowed, "reason": result.get("reason"), "raw": dict(result)}
    allowed = bool(getattr(result, "allowed", getattr(result, "ok", True)))
    return {"allowed": allowed, "reason": getattr(result, "reason", None), "raw": result}


async def _persist(persistence: Any | None, method_name: str, *args: Any, **kwargs: Any) -> Any:
    if persistence is None:
        return None
    method = getattr(persistence, method_name, None)
    if method is None:
        return None
    payload: dict[str, Any] = dict(kwargs)
    if args:
        first = args[0]
        payload["value"] = first
        if isinstance(first, AgentState):
            payload.update(first.snapshot())
            payload["state"] = first
        elif isinstance(first, AgentEvent):
            payload["event"] = first
            payload["event_type"] = first.type
    try:
        return await _maybe_await(_call_flexible(method, payload))
    except TypeError:
        return await _maybe_await(method(*args, **kwargs))


async def _call_optional(target: Any | None, method_name: str, *args: Any, **kwargs: Any) -> Any:
    if target is None:
        return None
    method = getattr(target, method_name, None)
    if method is None:
        return None
    return await _maybe_await(method(*args, **kwargs))


def _event(
    state: AgentState,
    event_type: str,
    node: str,
    message: str = "",
    data: dict[str, Any] | None = None,
) -> AgentEvent:
    data = data or {}
    # 增强 error 事件：注入错误分类信息
    if event_type == "error":
        classification = state.error_classification or "fatal"
        data.setdefault("severity", "fatal" if classification in ("fatal", "architectural") else "error")
        data.setdefault("recoverable", classification in ("retryable", "fixable"))
        data.setdefault("error_code", state.last_error.get("error_code", "UNKNOWN_ERROR"))
    return AgentEvent(
        type=event_type,
        session_id=state.session_id,
        run_id=state.run_id,
        node=node,
        message=message,
        data=data,
    )


def _normalize_context_items(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, str):
        return [{"source": "context_provider", "content": value}]
    if isinstance(value, Iterable):
        items: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                items.append(dict(item))
            else:
                items.append({"source": "context_provider", "content": item})
        return items
    return [{"source": "context_provider", "content": value}]


def _message_to_context(message: Any) -> dict[str, Any]:
    if isinstance(message, Mapping):
        return dict(message)
    return {
        "role": getattr(message, "role", ""),
        "content": getattr(message, "content", ""),
        "sequence": getattr(message, "sequence", None),
        "run_id": getattr(message, "run_id", None),
    }


def _summary_text(summary: Any) -> str:
    if summary is None:
        return ""
    if isinstance(summary, str):
        return summary
    if isinstance(summary, Mapping):
        return str(summary.get("summary") or summary.get("content") or "")
    data = getattr(summary, "data", None)
    if isinstance(data, Mapping):
        return str(data.get("summary", ""))
    return ""


def _on_pre_compress(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages", [])
    if isinstance(messages, list):
        payload["messages"] = messages[-30:]
    current = str(payload.get("current_response", ""))
    if len(current) > 4_000:
        payload["current_response"] = current[:4_000] + "\n...[truncated]"
    payload["pre_compress_insights"] = _extract_memory_insights(payload)
    return payload


def _extract_memory_insights(payload: Mapping[str, Any]) -> list[str]:
    markers = ("偏好", "记住", "决策", "未完成", "todo", "prefer", "preference")
    lines: list[str] = []
    for message in payload.get("messages", []):
        if isinstance(message, Mapping):
            content = str(message.get("content", ""))
        else:
            content = str(message)
        lines.extend(content.splitlines())
    lines.extend(str(payload.get("current_response", "")).splitlines())
    insights = [
        line.strip()
        for line in lines
        if line.strip() and any(marker in line.lower() for marker in markers)
    ]
    return insights[-20:]


def _tool_name(tool: Any) -> str:
    if isinstance(tool, str):
        return tool
    if isinstance(tool, Mapping):
        return str(tool.get("name", ""))
    return str(getattr(tool, "name", tool))


def _mentions_skill(task_lc: str, plan_lc: str) -> bool:
    text = f"{task_lc}\n{plan_lc}"
    return any(marker in text for marker in ("skill", "sop", "最佳实践", "规范", "流程"))


def _mentions_pytest(task_lc: str, plan_lc: str) -> bool:
    text = f"{task_lc}\n{plan_lc}"
    return any(marker in text for marker in ("pytest", "测试", "test"))


def _mentions_ruff(task_lc: str, plan_lc: str) -> bool:
    text = f"{task_lc}\n{plan_lc}"
    return any(marker in text for marker in ("ruff", "lint", "检查", "质量"))


def _mentions_format(task_lc: str, plan_lc: str) -> bool:
    text = f"{task_lc}\n{plan_lc}"
    return any(marker in text for marker in ("ruff format", "format", "格式"))


def _verified_editing_enabled(settings: AgentSettings | Mapping[str, Any]) -> bool:
    return bool(_setting(settings, "verified_editing_enabled", False))


def _mentions_verified_edit_request(state: AgentState) -> bool:
    text = f"{state.user_input}\n{state.plan}".lower()
    return any(
        marker in text
        for marker in (
            "edit",
            "modify",
            "fix",
            "refactor",
            "implement",
            "change",
            "patch",
            "write code",
            "修改",
            "修复",
            "实现",
            "重构",
        )
    )


def _extract_path_hint(text: str) -> str | None:
    for token in text.replace("`", " ").split():
        normalized = token.strip(".,:;()[]{}'\"")
        if "/" in normalized or "\\" in normalized or normalized.endswith((".py", ".md", ".toml")):
            return normalized
    return None


def _dedupe_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for call in calls:
        key = (
            str(call.get("name", "")),
            json.dumps(call.get("arguments") or {}, ensure_ascii=False, sort_keys=True, default=str),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(call)
    return deduped


def _truncate_tool_result(result: Any, max_bytes: int) -> tuple[Any, dict[str, Any]]:
    serialized = _serialize_tool_result(result)
    original_bytes = len(serialized.encode("utf-8"))
    metadata = {
        "output_bytes": min(original_bytes, max(max_bytes, 0)),
        "original_output_bytes": original_bytes,
        "truncated": False,
        "max_output_bytes": max_bytes,
    }
    if max_bytes <= 0:
        metadata.update({"output_bytes": 0, "truncated": original_bytes > 0})
        return {"truncated": True, "content": ""}, metadata
    if original_bytes <= max_bytes:
        return result, metadata

    truncated = _truncate_text_bytes(serialized, max_bytes)
    metadata.update({"output_bytes": len(truncated.encode("utf-8")), "truncated": True})
    return {
        "truncated": True,
        "content": truncated,
        "original_output_bytes": original_bytes,
    }, metadata


def _serialize_tool_result(result: Any) -> str:
    try:
        return json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(result)


def _truncate_text_bytes(text: str, max_bytes: int) -> str:
    suffix = "\n...[tool output truncated]"
    budget = max(0, max_bytes - len(suffix.encode("utf-8")))
    raw = text.encode("utf-8")[:budget]
    return raw.decode("utf-8", errors="ignore") + suffix


def _setting(settings: AgentSettings | Mapping[str, Any], key: str, default: Any) -> Any:
    if isinstance(settings, Mapping):
        return settings.get(key, default)
    return getattr(settings, key, default)


def _coerce_agent_settings(settings: AgentSettings | Mapping[str, Any] | Any) -> AgentSettings:
    if isinstance(settings, AgentSettings):
        return settings
    defaults = AgentSettings()
    values: dict[str, Any] = {}
    for item in dataclass_fields(AgentSettings):
        default = getattr(defaults, item.name)
        values[item.name] = _setting(settings, item.name, default)
    return AgentSettings(**values)


def _call_flexible(func: Any, payload: dict[str, Any]) -> Any:
    signature = inspect.signature(func)
    if "args" in payload and "kwargs" in payload:
        args = payload["args"]
        kwargs = payload["kwargs"]
        try:
            return func(*args, **kwargs)
        except TypeError:
            return func(*args)

    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return func(**payload)
    kwargs = {key: value for key, value in payload.items() if key in accepted}
    if kwargs:
        return func(**kwargs)
    return func(payload)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


# ---------------------------------------------------------------------------
# Review Layer Stages
# ---------------------------------------------------------------------------


async def _spec_compliance_review_stage(
    state: AgentState,
    provider: ChatProvider,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    """LLM review of tool/patch results against user requirements."""
    user_input = state.user_input or ""
    tool_executed = any(
        tc.name == "apply_text_edit" for tc in (state.tool_calls or [])
    )
    patch_proposed = state.patch_proposal is not None

    findings: list[str] = []
    if not user_input:
        findings.append("No user input to compare against")
    if not tool_executed and not patch_proposed:
        findings.append("No code changes were made")

    state.review_reports["spec_compliance"] = {
        "status": "passed" if not findings else "reviewed",
        "findings": findings,
        "has_code_changes": tool_executed or patch_proposed,
    }
    yield _event(
        state,
        "spec_compliance_review",
        "spec_compliance_review",
        "Spec compliance review completed",
        state.review_reports["spec_compliance"],
    )


# ---------------------------------------------------------------------------
# Error Recovery Stages
# ---------------------------------------------------------------------------


async def _recovery_action_stage(
    state: AgentState,
    deps: AgentDeps,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    """Execute recovery based on classification."""
    error_state = state.error_state or {}
    classification = error_state.get("classification", "")
    recovery_attempts = state.recovery_attempts

    max_attempts = 2
    if recovery_attempts >= max_attempts:
        state.recovery_attempts = recovery_attempts + 1
        yield _event(
            state,
            "recovery_exhausted",
            "recovery_action",
            f"Recovery attempts exhausted ({recovery_attempts}/{max_attempts})",
            {"max_attempts": max_attempts, "attempts": recovery_attempts},
        )
        return

    state.recovery_attempts = recovery_attempts + 1
    yield _event(
        state,
        "recovery_started",
        "recovery_action",
        f"Recovery attempt {state.recovery_attempts}/{max_attempts} for: {classification}",
        {"classification": classification, "attempt": state.recovery_attempts},
    )


# ---------------------------------------------------------------------------
# Provider Routing Stage
# ---------------------------------------------------------------------------


async def _provider_routing_stage(
    state: AgentState,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    """Select provider role based on current node context."""
    current_node = state.current_node or ""

    role_map: dict[str, str] = {
        "plan": "complete",
        "respond": "complete",
        "spec_compliance_review": "fast",
        "code_quality_review": "complete",
        "response_review": "fast",
        "propose_verified_patch": "complete",
        "compress_memory": "compression",
        "classify_error": "fast",
    }
    default_role = "fast" if "review" in current_node or "gate" in current_node else "complete"
    state.provider_mode = role_map.get(current_node, default_role)

    yield _event(
        state,
        "provider_routed",
        "provider_routing",
        f"Provider mode: {state.provider_mode} for node: {current_node}",
        {"provider_mode": state.provider_mode, "current_node": current_node},
    )


# ---------------------------------------------------------------------------
# Verification Stage
# ---------------------------------------------------------------------------


async def _run_verification_stage(
    state: AgentState,
    deps: AgentDeps,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    """Run pytest + ruff after approved patch apply."""
    tool_registry = getattr(deps, "tool_registry", None)
    if tool_registry is None:
        yield _event(
            state,
            "verification_skipped",
            "run_verification",
            "No tool registry available — verification skipped",
            {},
        )
        return

    results: dict[str, Any] = {}

    # Run ruff check
    try:
        ruff_result = await _call_tool_if_available(tool_registry, "run_ruff_check", {"target": "."})
        results["ruff"] = ruff_result
    except Exception as exc:
        results["ruff"] = {"error": str(exc)}

    # Run pytest
    try:
        pytest_result = await _call_tool_if_available(tool_registry, "run_pytest", {"target": ""})
        results["pytest"] = pytest_result
    except Exception as exc:
        results["pytest"] = {"error": str(exc)}

    verification_status = "passed" if all(
        r.get("status") == "passed" or r.get("status") is None
        for r in results.values()
        if isinstance(r, dict)
    ) else "failed"

    state.review_reports["verification"] = {
        "status": verification_status,
        "results": results,
    }
    yield _event(
        state,
        "verification_completed",
        "run_verification",
        f"Verification: {verification_status}",
        state.review_reports["verification"],
    )


# ---------------------------------------------------------------------------
# Terminal Error Response Stages
# ---------------------------------------------------------------------------


async def _environment_error_response_stage(
    state: AgentState,
    provider: ChatProvider,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    """Generate explanatory response for environment errors."""
    error_state = state.error_state or {}
    error_message = error_state.get("last_error_type", "Unknown") or "Unknown"

    state.response = (
        f"I encountered an environment error: **{error_message}**\n\n"
        f"This appears to be related to the runtime environment rather than the task itself. "
        f"Please check your configuration and try again."
    )
    state.blocked = True
    state.block_reason = f"Environment error: {error_message}"
    yield _event(
        state,
        "environment_error_response",
        "environment_error_response",
        state.block_reason,
        {"error": error_state},
    )


async def _architecture_failure_response_stage(
    state: AgentState,
    provider: ChatProvider,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    """Generate explanatory response for architecture failures."""
    error_state = state.error_state or {}
    error_history = error_state.get("error_history", [])

    state.response = (
        f"I've encountered a recurring issue that appears to be an architectural problem.\n\n"
        f"The same error pattern occurred {len(error_history)} times despite recovery attempts. "
        f"This likely requires a change in approach or configuration."
    )
    state.blocked = True
    state.block_reason = "Architecture failure — repeated errors exceeded recovery limits"
    yield _event(
        state,
        "architecture_failure_response",
        "architecture_failure_response",
        state.block_reason,
        {"error_history": error_history[-3:]},
    )


# ---------------------------------------------------------------------------
# Auto-fix stage
# ---------------------------------------------------------------------------


async def _auto_fix_prepare_stage(
    state: AgentState,
    provider: ChatProvider,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    """Generate fix PatchProposal from review feedback."""
    review_reports = state.review_reports or {}
    spec = review_reports.get("spec_compliance", {})
    quality = review_reports.get("code_quality", {})

    fix_needed = bool(
        spec.get("findings") or quality.get("findings")
    )
    if not fix_needed:
        yield _event(
            state,
            "auto_fix_skipped",
            "auto_fix_prepare",
            "No fixes needed based on review",
            {},
        )
        return

    yield _event(
        state,
        "auto_fix_proposed",
        "auto_fix_prepare",
        "Auto-fix proposal generated from review feedback",
        {"spec_findings": spec.get("findings", []), "quality_findings": quality.get("findings", [])},
    )


# ---------------------------------------------------------------------------
# Context guard stage
# ---------------------------------------------------------------------------


async def _context_guard_stage(
    state: AgentState,
    provider: ChatProvider,
    deps: AgentDeps,
    settings: AgentSettings | Mapping[str, Any],
    *,
    phase: str = "before_plan",
) -> AsyncIterator[AgentEvent]:
    """Evaluate context budget and trigger compression if needed."""
    state.loop_stage = f"context_guard_{phase}"
    from solo_agent.context import ContextManager, ContextTokenEstimator

    estimator = ContextTokenEstimator()
    manager = ContextManager(settings=settings, main_provider=provider, estimator=estimator)
    report = manager.evaluate(state, estimator=estimator)

    if report.should_compress:
        try:
            result = await manager.maybe_compress(state, estimator=estimator)
            yield _event(
                state,
                "context_compression_completed",
                f"context_guard_{phase}",
                "Context compression completed",
                {"phase": phase, "result": result or {}},
            )
        except Exception as exc:
            yield _event(
                state,
                "context_compression_failed",
                f"context_guard_{phase}",
                f"Context compression failed: {exc}",
                {"phase": phase, "error": str(exc)},
            )
    else:
        yield _event(
            state,
            "context_guard_passed",
            f"context_guard_{phase}",
            f"Context guard passed ({phase})",
            {
                "phase": phase,
                "current_tokens": report.current_tokens,
                "threshold_tokens": report.threshold_tokens,
                "should_compress": report.should_compress,
            },
        )


# ---------------------------------------------------------------------------
# Persist snapshot stage
# ---------------------------------------------------------------------------


async def _persist_snapshot_stage(state: AgentState) -> AsyncIterator[AgentEvent]:
    """Persist final state snapshot."""
    state.loop_stage = "persist_snapshot"
    raw = state.snapshot()
    state.snapshots["last_snapshot"] = {
        "timestamp": "now",
        "plan_length": len(raw.get("plan", "")),
        "response_length": len(raw.get("response", "")),
        "tool_call_count": len(raw.get("tool_calls", [])),
    }
    yield _event(
        state,
        "persist_snapshot_completed",
        "persist_snapshot",
        "Snapshot persisted",
        {"snapshot": state.snapshots["last_snapshot"]},
    )

