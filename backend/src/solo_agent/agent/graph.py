from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Iterable, Mapping
from pathlib import Path
from typing import Any, TypedDict

from solo_agent.context import ContextManager, ContextTokenEstimator, SubdirectoryHintTracker, TaskListState
from solo_agent.providers import ChatMessage, ChatProvider, create_provider_from_settings

from .deps import AgentDeps, AgentSettings
from .events import AgentEvent
from .prompts import (
    PLANNER_SYSTEM_PROMPT,
    RESPONDER_SYSTEM_PROMPT,
    build_memory_context_block,
    build_skill_context_block,
    planner_user_prompt,
    responder_user_prompt,
)
from .state import AgentState, ToolCallRecord


class LangGraphTopologyState(TypedDict, total=False):
    session_id: str
    run_id: str
    user_input: str
    plan: str
    context: list[dict[str, Any]]
    approved_tool_calls: list[dict[str, Any]]
    response: str
    persisted: bool


def build_langgraph_topology() -> Any:
    """Return the milestone topology as a compiled LangGraph graph.

    The Web runner uses the streaming event loop below so it can surface token
    deltas and tool progress directly. This topology keeps the LangGraph seam
    explicit for future checkpointing and richer graph execution.
    """

    from langgraph.graph import END, START, StateGraph

    async def pass_through(state: LangGraphTopologyState) -> LangGraphTopologyState:
        return state

    graph = StateGraph(LangGraphTopologyState)
    nodes = (
        "receive_user_turn",
        "load_builtin_memory",
        "prefetch_all",
        "build_memory_context_block",
        "select_skills",
        "load_skills",
        "build_skill_context_block",
        "context_guard_before_plan",
        "plan",
        "task_state_update",
        "collect_context",
        "inspect",
        "select_tools",
        "execute_tools",
        "subdirectory_hint_track",
        "context_guard_before_respond",
        "respond",
        "sync_all",
        "queue_prefetch_all",
        "context_guard_after_run",
        "persist_snapshot",
        "finish",
    )
    for node in nodes:
        graph.add_node(node, pass_through)

    graph.add_edge(START, nodes[0])
    for index in range(len(nodes) - 1):
        graph.add_edge(nodes[index], nodes[index + 1])
    graph.add_edge(nodes[-1], END)
    return graph.compile()


async def run_agent_events(
    session_id: str,
    run_id: str,
    user_input: str,
    deps: AgentDeps | None = None,
    settings: AgentSettings | Mapping[str, Any] | None = None,
) -> AsyncIterator[AgentEvent]:
    """Run the Milestone 1 agent graph and stream visible progress events."""

    deps = deps or AgentDeps()
    settings = settings or deps.settings or AgentSettings()
    provider = deps.provider or create_provider_from_settings(settings)
    state = AgentState(session_id=session_id, run_id=run_id, user_input=user_input)

    try:
        await _persist(deps.persistence, "start_run", state)
        async for event in _run_graph(state, provider, deps, settings):
            await _persist(deps.persistence, "save_event", event, state)
            yield event
    except Exception as exc:
        event = _event(state, "error", "error", str(exc), {"error_type": type(exc).__name__})
        await _persist(deps.persistence, "save_event", event, state)
        await _persist(deps.persistence, "finish_run", state, status="error", error=str(exc))
        yield event
    else:
        await _persist(deps.persistence, "finish_run", state, status="completed")


async def _run_graph(
    state: AgentState,
    provider: ChatProvider,
    deps: AgentDeps,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    async for event in _receive_user_turn_stage(state, settings):
        yield event

    if state.memory_enabled:
        async for event in _load_builtin_memory_stage(state, deps):
            yield event
        async for event in _prefetch_memory_stage(state, deps, settings):
            yield event
        async for event in _build_memory_context_stage(state):
            yield event
    else:
        async for event in _skip_memory_stage(state):
            yield event

    async for event in _skill_context_stage(state, deps, settings):
        yield event

    async for event in _context_guard_stage(state, provider, deps, settings, phase="before_plan"):
        yield event

    async for event in _plan_node(state, provider, settings):
        yield event
    async for event in _task_state_stage(state):
        yield event

    async for event in _collect_context_node(state, deps, settings):
        yield event

    async for event in _inspect_node(state, deps):
        yield event
    if state.blocked:
        state.loop_stage = "finish"
        yield _event(
            state,
            "run_completed",
            "end",
            "Agent run blocked by safety inspection",
            {"blocked": True, "reason": state.block_reason},
        )
        return

    async for event in _select_tools_node(state, deps, settings):
        yield event

    async for event in _execute_tools_node(state, deps, settings):
        yield event
    async for event in _subdirectory_hint_stage(state, settings):
        yield event

    async for event in _context_guard_stage(state, provider, deps, settings, phase="before_respond"):
        yield event

    async for event in _respond_node(state, provider, settings):
        yield event

    if state.memory_enabled:
        async for event in _sync_memory_stage(state, deps):
            yield event
        async for event in _queue_prefetch_stage(state, deps, settings):
            yield event
        async for event in _compress_memory_stage(state, provider, deps, settings):
            yield event

    async for event in _persist_snapshot_stage(state):
        yield event

    state.loop_stage = "finish"
    yield _event(state, "run_completed", "end", "Agent run completed", state.snapshot())


async def _receive_user_turn_stage(
    state: AgentState,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    state.loop_stage = "receive_user_turn"
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
    messages = [
        ChatMessage(role="system", content=PLANNER_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=planner_user_prompt(
                state.user_input,
                state.conversation_context,
                state.memory_context_block,
                state.skill_context_block,
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
    iron_law = _iron_law_decision(state, proposed)
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
        {"proposed_tool_calls": proposed},
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
    protocol_state = _new_tool_protocol_state(state)
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

    for call in proposed:
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
                    "skipped_count": len(proposed) - attempted,
                },
            )
            break

        name = str(call.get("name", "unknown"))
        arguments = dict(call.get("arguments") or {})
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
        protocol_violation = _tool_protocol_violation(state, name, arguments, protocol_state)
        if protocol_violation is not None:
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
        approved.append(call)

        yield _event(
            state,
            "tool_progress",
            "execute_tools",
            f"Executing tool {name}",
            {"name": name, "status": "executing"},
        )
        raw_result = await _call_tool(deps.tool_registry, name, arguments)
        raw_result_ok = _tool_result_ok(raw_result)
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
        if raw_result_ok:
            _record_tool_protocol_success(protocol_state, name, arguments, raw_result)
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
    state.loop_stage = "sync_all"
    sync_result = await _call_optional(
        deps.persistence,
        "sync_all",
        session_id=state.session_id,
        run_id=state.run_id,
        user_input=state.user_input,
        assistant_response=state.response,
    )
    if sync_result is not None:
        yield _event(state, "memory_synced", "memory", "Synced current turn into memory", sync_result)


async def _queue_prefetch_stage(
    state: AgentState,
    deps: AgentDeps,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    state.loop_stage = "queue_prefetch_all"
    queue_result = await _call_optional(
        deps.persistence,
        "queue_prefetch_all",
        session_id=state.session_id,
        query=state.response or state.user_input,
        limit=int(_setting(settings, "memory_search_limit", 5)),
    )
    if queue_result is not None:
        yield _event(state, "memory_prefetch_queued", "memory", "Queued memory prefetch for next turn", queue_result)


async def _compress_memory_stage(
    state: AgentState,
    provider: ChatProvider,
    deps: AgentDeps,
    settings: AgentSettings | Mapping[str, Any],
) -> AsyncIterator[AgentEvent]:
    async for event in _context_guard_stage(state, provider, deps, settings, phase="after_run"):
        yield event


async def _persist_snapshot_stage(state: AgentState) -> AsyncIterator[AgentEvent]:
    state.loop_stage = "persist_snapshot"
    state.snapshots["loop_stage"] = state.loop_stage
    yield _event(
        state,
        "persist_snapshot_completed",
        "persist_snapshot",
        "Persisted run snapshot",
        {"snapshot": state.snapshot()},
    )


async def _load_conversation_context(
    state: AgentState,
    deps: AgentDeps,
    settings: AgentSettings | Mapping[str, Any],
) -> dict[str, Any]:
    persistence = deps.persistence
    if persistence is None:
        return {"summary": "", "recent_messages": [], "retrieved_memories": [], "budget": {}}

    history_enabled = bool(_setting(settings, "conversation_history_enabled", True))
    history_limit = int(_setting(settings, "history_message_limit", 12)) if history_enabled else 0
    memory_limit = int(_setting(settings, "memory_search_limit", 5))

    prefetched = await _call_prefetch_all(
        persistence,
        session_id=state.session_id,
        query=state.user_input,
        recent_limit=history_limit,
        limit=memory_limit,
        include_history=history_enabled,
    )
    if prefetched:
        if not history_enabled:
            prefetched["recent_messages"] = []
        return {
            **prefetched,
            "budget": {
                "recent_message_limit": history_limit,
                "memory_search_limit": memory_limit,
                "scope": "session",
                "conversation_history_enabled": history_enabled,
            },
        }

    recent_messages = (
        await _call_optional(persistence, "list_messages", state.session_id, limit=history_limit)
        if history_enabled
        else []
    )
    summary = await _call_optional(persistence, "get_latest_summary", state.session_id)
    retrieved = await _call_optional(
        persistence,
        "search_memory",
        session_id=state.session_id,
        query=state.user_input,
        limit=memory_limit,
    )

    return {
        "summary": _summary_text(summary),
        "recent_messages": [_message_to_context(message) for message in recent_messages or []],
        "retrieved_memories": retrieved or [],
        "budget": {
            "recent_message_limit": history_limit,
            "memory_search_limit": memory_limit,
            "scope": "session",
            "memory_enabled": True,
            "conversation_history_enabled": history_enabled,
        },
    }


async def _context_guard_stage(
    state: AgentState,
    provider: ChatProvider,
    deps: AgentDeps,
    settings: AgentSettings | Mapping[str, Any],
    *,
    phase: str,
) -> AsyncIterator[AgentEvent]:
    state.loop_stage = f"context_guard_{phase}"
    if not bool(_setting(settings, "memory_enabled", True)):
        return

    stats = await _call_optional(deps.persistence, "get_context_stats", state.session_id)
    if isinstance(stats, Mapping):
        context_stats = dict(stats.get("context_stats") or stats)
        if "compression_count" in context_stats:
            state.snapshots["compression_count"] = int(context_stats.get("compression_count") or 0)

    estimator = ContextTokenEstimator()
    manager = ContextManager(settings=settings, main_provider=provider, estimator=estimator)
    report = manager.evaluate(state, estimator=estimator)
    yield _event(
        state,
        "context_budget_checked",
        "context",
        "Checked context token budget",
        {
            "phase": phase,
            "current_tokens": report.current_tokens,
            "threshold_tokens": report.threshold_tokens,
            "threshold_ratio": report.threshold_ratio,
            "compression_count": report.compression_count,
            "provider_role": report.provider_role,
            "should_compress": report.should_compress,
        },
    )
    force_compress = False
    if not report.should_compress and phase == "after_run":
        force_compress = await _legacy_summary_trigger_met(state, deps, settings)
    if not report.should_compress and not force_compress:
        yield _event(
            state,
            "context_compression_skipped",
            "context",
            "Context compression skipped; budget is healthy",
            {"phase": phase, "reason": report.reason},
        )
        return

    yield _event(
        state,
        "context_compression_started",
        "context",
        "Compressing context before continuing",
        {
            "phase": phase,
            "strategy": report.provider_role,
            "compression_count": report.compression_count,
            "forced": force_compress,
        },
    )
    try:
        await _call_optional(
            deps.persistence,
            "on_pre_compress",
            session_id=state.session_id,
            payload=_context_compression_payload(state),
        )
        result = await manager.maybe_compress(state, estimator=estimator, force=force_compress)
    except Exception as exc:
        try:
            result = await _fallback_main_compression(state, provider, settings, estimator, exc)
        except Exception as fallback_error:
            state.summary_status = "failed"
            yield _event(
                state,
                "memory_summary_failed",
                "context",
                "Context compression failed; continuing without blocking this run",
                {"phase": phase, "error": str(fallback_error)},
            )
            return

    await _persist_context_summary(state, deps, result, phase=phase)
    _inject_task_state_block(state)
    state.summary_status = "updated" if result.compressed else state.summary_status
    yield _event(
        state,
        "context_compression_completed",
        "context",
        "Context compression completed",
        {
            "phase": phase,
            "compression_count": result.compression_count,
            "strategy": result.provider_role,
            "model": result.provider_model,
            "summary_chars": len(result.summary),
        },
    )
    if state.snapshots.get("task_state_block"):
        yield _event(
            state,
            "task_state_injected",
            "context",
            "Injected TaskList state after compression",
            {"phase": phase, "length": len(str(state.snapshots.get("task_state_block", "")))},
        )


async def _fallback_main_compression(
    state: AgentState,
    provider: ChatProvider,
    settings: AgentSettings | Mapping[str, Any],
    estimator: ContextTokenEstimator,
    original_error: Exception,
) -> Any:
    manager = ContextManager(
        settings=settings,
        main_provider=provider,
        auxiliary_provider=provider,
        estimator=estimator,
    )
    try:
        return await manager.maybe_compress(state, estimator=estimator, force=True)
    except Exception as fallback_error:
        raise RuntimeError(
            f"context compression failed: {original_error}; fallback failed: {fallback_error}"
        ) from fallback_error


async def _legacy_summary_trigger_met(
    state: AgentState,
    deps: AgentDeps,
    settings: AgentSettings | Mapping[str, Any],
) -> bool:
    if deps.persistence is None:
        return False
    trigger = int(_setting(settings, "summary_trigger_messages", 0))
    if trigger <= 0:
        return False
    count = await _call_optional(deps.persistence, "count_messages", state.session_id)
    expected_count = int(count or 0)
    if state.response and expected_count == 0:
        expected_count = 1
    return expected_count >= trigger


async def _persist_context_summary(
    state: AgentState,
    deps: AgentDeps,
    result: Any,
    *,
    phase: str,
) -> None:
    if deps.persistence is None or not result.compressed:
        return
    context_stats = {
        "compression_count": result.compression_count,
        "last_estimated_tokens": result.report.current_tokens,
        "last_threshold": result.report.threshold_ratio,
        "last_strategy": result.provider_role,
        "last_model": result.provider_model,
    }
    state.snapshots["compression_count"] = result.compression_count
    state.snapshots["context_stats"] = context_stats
    state.conversation_context["summary"] = result.summary.strip()
    await _call_optional(
        deps.persistence,
        "append_or_update_summary_snapshot",
        session_id=state.session_id,
        run_id=state.run_id,
        summary=result.summary.strip(),
        metadata={
            "source": "context_manager",
            "phase": phase,
            "context_stats": context_stats,
        },
    )


async def _task_state_stage(state: AgentState) -> AsyncIterator[AgentEvent]:
    task_state = TaskListState.from_text(state.plan, thread_id=state.session_id)
    if not task_state.items:
        return
    state.snapshots["task_state"] = _task_state_to_dict(task_state)
    state.snapshots["task_state_json_block"] = task_state.format_json_block()
    yield _event(
        state,
        "task_state_injected",
        "context",
        "Captured TaskList state from plan",
        {
            "continue_from": task_state.continue_from,
            "items": state.snapshots["task_state"]["items"],
            "thread_id": task_state.thread_id,
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


def _inject_task_state_block(state: AgentState) -> None:
    task_state = _task_state_from_snapshot(state.snapshots.get("task_state"))
    if task_state is None:
        task_state = TaskListState.from_text(state.plan, thread_id=state.session_id)
    if not task_state.items:
        return
    block = task_state.format_block()
    state.snapshots["task_state"] = _task_state_to_dict(task_state)
    state.snapshots["task_state_block"] = block
    state.memory_context_block = "\n\n".join(part for part in (state.memory_context_block, block) if part)


def _task_state_to_dict(task_state: TaskListState) -> dict[str, Any]:
    return task_state.to_dict()


def _task_state_from_snapshot(value: Any) -> TaskListState | None:
    if not isinstance(value, Mapping):
        return None
    return TaskListState.from_payload(dict(value), thread_id=str(value.get("thread_id") or value.get("threadID") or ""))


async def _maybe_update_summary(
    state: AgentState,
    provider: ChatProvider,
    deps: AgentDeps,
    settings: AgentSettings | Mapping[str, Any],
    *,
    include_response: bool = False,
) -> AsyncIterator[AgentEvent]:
    persistence = deps.persistence
    if persistence is None or not bool(_setting(settings, "memory_enabled", True)):
        state.summary_status = "skipped"
        return

    count = await _call_optional(persistence, "count_messages", state.session_id)
    trigger = int(_setting(settings, "summary_trigger_messages", 8))
    expected_count = int(count or 0)
    if include_response and state.response and expected_count == 0:
        expected_count = 1
    if expected_count < trigger:
        state.summary_status = "skipped_threshold"
        return

    messages = await _call_optional(
        persistence,
        "list_messages",
        state.session_id,
        limit=min(expected_count, 40),
    )
    payload = _on_pre_compress(
        {
            "messages": [_message_to_context(message) for message in messages or []],
            "current_response": state.response if include_response else "",
            "existing_summary": state.conversation_context.get("summary", ""),
        }
    )

    state.summary_status = "compressing"
    yield _event(
        state,
        "memory_compress_started",
        "memory",
        "Preparing memory compression",
        {"message_count": expected_count},
    )
    try:
        await _call_optional(
            persistence,
            "on_pre_compress",
            session_id=state.session_id,
            payload=payload,
        )
        summary = await provider.complete(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "Summarize this coding session memory for future turns. "
                        "Keep user preferences, decisions, unresolved tasks, and important context. "
                        "Use concise Chinese when possible."
                    ),
                ),
                ChatMessage(role="user", content=str(payload)),
            ],
            temperature=float(_setting(settings, "temperature", 0.2)),
            max_tokens=int(_setting(settings, "summary_max_tokens", 700)),
        )
        await _call_optional(
            persistence,
            "append_or_update_summary_snapshot",
            session_id=state.session_id,
            run_id=state.run_id,
            summary=summary.strip(),
            metadata={"source": "model", "message_count": expected_count},
        )
        state.summary_status = "updated"
        yield _event(
            state,
            "memory_summary_updated",
            "memory",
            "Updated session summary memory",
            {"message_count": expected_count},
        )
    except Exception as exc:
        state.summary_status = "failed"
        yield _event(
            state,
            "memory_summary_failed",
            "memory",
            "Summary update failed; continuing without blocking this run",
            {"error": str(exc), "message_count": expected_count},
        )


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


async def _call_prefetch_all(
    target: Any | None,
    *,
    session_id: str,
    query: str,
    recent_limit: int,
    limit: int,
    include_history: bool,
) -> Any:
    if target is None:
        return None
    method = getattr(target, "prefetch_all", None)
    if method is None:
        return None

    kwargs = {
        "session_id": session_id,
        "query": query,
        "recent_limit": recent_limit,
        "limit": limit,
    }
    signature = inspect.signature(method)
    accepts_include_history = "include_history" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_include_history:
        kwargs["include_history"] = include_history
    return await _maybe_await(method(**kwargs))


def _event(
    state: AgentState,
    event_type: str,
    node: str,
    message: str = "",
    data: dict[str, Any] | None = None,
) -> AgentEvent:
    return AgentEvent(
        type=event_type,
        session_id=state.session_id,
        run_id=state.run_id,
        node=node,
        message=message,
        data=data or {},
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


_CODE_CHANGE_MARKERS = (
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
_FAILING_TEST_MARKERS = (
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
_IRON_LAW_WARNING_MARKERS = (
    "read-only",
    "read only",
    "readonly",
    "explore",
    "exploration",
    "inspect only",
    "skip tests",
    "skip testing",
    "without tests",
    "只读",
    "探索",
    "仅查看",
    "不要修改",
    "跳过测试",
    "不用测试",
)
_QUALITY_TOOL_NAMES = {"run_pytest", "run_ruff_check", "run_ruff_format_check"}
_EDIT_PROOF_TOOL_NAMES = {"prepare_edit", "get_file_hash"}


def _iron_law_decision(
    state: AgentState,
    proposed: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    skill_names = {str(skill.get("name", "")).lower() for skill in state.selected_skills}
    proposed_calls = list(proposed or [])
    production_paths = [
        path
        for call in proposed_calls
        if str(call.get("name", "")) == "apply_text_edit"
        for path in [_normalize_protocol_path(dict(call.get("arguments") or {}).get("path"))]
        if path and _is_production_path(path)
    ]
    if "iron-law" not in skill_names and not production_paths:
        return {"action": "none"}

    text = f"{state.user_input}\n{state.plan}".lower()
    has_code_change_intent = any(marker in text for marker in _CODE_CHANGE_MARKERS)
    if not has_code_change_intent and not production_paths:
        return {"action": "none"}
    if _has_failing_test_signal(state):
        return {"action": "none", "reason": "failing_test_signal_present"}

    warning_only = any(marker in text for marker in _IRON_LAW_WARNING_MARKERS)
    action = "warning" if warning_only else "blocked"
    return {
        "action": action,
        "reason": "production_edit_without_failing_test_signal",
        "current_task_priority": "user_input",
        "production_paths": production_paths,
        "warning_only": warning_only,
    }


def _new_tool_protocol_state(state: AgentState) -> dict[str, Any]:
    protocol_state: dict[str, Any] = {
        "edit_proofs": set(),
        "previews": set(),
    }
    for record in state.tool_calls:
        if not record.blocked and _tool_result_ok(record.result):
            _record_tool_protocol_success(
                protocol_state,
                record.name,
                record.arguments,
                record.result,
            )
    return protocol_state


def _tool_protocol_violation(
    state: AgentState,
    name: str,
    arguments: Mapping[str, Any],
    protocol_state: Mapping[str, Any],
) -> dict[str, Any] | None:
    if name != "apply_text_edit":
        return None

    path = _normalize_protocol_path(arguments.get("path"))
    expected_hash = _normalize_protocol_hash(arguments.get("expected_hash"))
    if not path:
        return {"reason": "apply_text_edit_missing_path"}
    if not expected_hash:
        return {"reason": "apply_text_edit_missing_expected_hash", "path": path}

    iron_law = _iron_law_decision(state, [{"name": name, "arguments": dict(arguments)}])
    if iron_law["action"] == "blocked":
        return {"reason": "iron_law_blocked", "path": path, "expected_hash": expected_hash}

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
        }

    return None


def _record_tool_protocol_success(
    protocol_state: dict[str, Any],
    name: str,
    arguments: Mapping[str, Any],
    result: Any,
) -> None:
    if name not in _EDIT_PROOF_TOOL_NAMES and name != "preview_patch":
        return

    path = _normalize_protocol_path(
        _first_present(
            arguments.get("path"),
            _extract_protocol_field(result, ("path", "file_path", "target_path")),
        )
    )
    expected_hash = _normalize_protocol_hash(
        _first_present(
            arguments.get("expected_hash"),
            _extract_protocol_field(
                result,
                ("expected_hash", "hash", "file_hash", "sha256", "digest", "current_hash"),
            ),
        )
    )
    if not path or not expected_hash:
        return

    key = (path, expected_hash)
    if name in _EDIT_PROOF_TOOL_NAMES:
        protocol_state.setdefault("edit_proofs", set()).add(key)
    elif name == "preview_patch":
        protocol_state.setdefault("previews", set()).add(key)


def _has_failing_test_signal(state: AgentState) -> bool:
    text = f"{state.user_input}\n{state.plan}".lower()
    if any(marker in text for marker in _FAILING_TEST_MARKERS):
        return True
    return any(
        record.name in _QUALITY_TOOL_NAMES and _tool_result_failed(record.result)
        for record in state.tool_calls
        if not record.blocked
    )


def _tool_result_failed(result: Any) -> bool:
    if isinstance(result, Mapping):
        for key in ("ok", "success", "passed"):
            if key in result:
                return not bool(result[key])
        for key in ("exit_code", "returncode", "status_code"):
            if key in result:
                try:
                    return int(result[key]) != 0
                except (TypeError, ValueError):
                    return False
        if bool(result.get("failed", False)):
            return True
        nested = result.get("result")
        if isinstance(nested, Mapping):
            return _tool_result_failed(nested)
    return False


def _extract_protocol_field(value: Any, names: Iterable[str]) -> Any:
    if not isinstance(value, Mapping):
        return None
    names_set = set(names)
    for name in names_set:
        if value.get(name) is not None:
            return value[name]
    for nested_name in ("result", "data", "metadata"):
        nested = value.get(nested_name)
        if isinstance(nested, Mapping):
            found = _extract_protocol_field(nested, names_set)
            if found is not None:
                return found
    return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _normalize_protocol_path(value: Any) -> str | None:
    if value is None:
        return None
    path = str(value).strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path or None


def _normalize_protocol_hash(value: Any) -> str | None:
    if value is None:
        return None
    expected_hash = str(value).strip()
    return expected_hash or None


def _is_production_path(path: str) -> bool:
    normalized = _normalize_protocol_path(path) or ""
    parts = normalized.lower().split("/")
    filename = parts[-1] if parts else ""
    return not (
        "tests" in parts
        or filename.startswith("test_")
        or filename.endswith("_test.py")
        or normalized.lower().endswith(".md")
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


def _tool_result_ok(result: Any) -> bool:
    if isinstance(result, Mapping):
        return bool(result.get("ok", True))
    return True


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
