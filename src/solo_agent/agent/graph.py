from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any, TypedDict

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
        "plan",
        "collect_context",
        "inspect",
        "select_tools",
        "execute_tools",
        "respond",
        "sync_all",
        "queue_prefetch_all",
        "on_pre_compress",
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

    async for event in _plan_node(state, provider, settings):
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
    if _needs_iron_law_warning(state):
        yield _event(
            state,
            "iron_law_warning",
            "select_tools",
            "Production-code intent detected without an explicit failing-test signal",
            {
                "reason": "iron-law skill is active and no failing test was referenced",
                "current_task_priority": "user_input",
            },
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
        if name == "apply_text_edit" and _tool_result_ok(result):
            yield _event(
                state,
                "verification_required",
                "execute_tools",
                "A file edit was applied; verification is required",
                {"recommended_tools": ["run_pytest", "run_ruff_check"]},
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
    state.loop_stage = "on_pre_compress"
    async for event in _maybe_update_summary(state, provider, deps, settings, include_response=True):
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


def _needs_iron_law_warning(state: AgentState) -> bool:
    skill_names = {str(skill.get("name", "")).lower() for skill in state.selected_skills}
    if "iron-law" not in skill_names:
        return False
    task = state.user_input.lower()
    has_code_change_intent = any(
        marker in task
        for marker in (
            "edit",
            "modify",
            "fix",
            "refactor",
            "implement",
            "change",
            "写",
            "修改",
            "修复",
            "实现",
            "重构",
        )
    )
    has_test_signal = any(marker in task for marker in ("failing test", "失败测试", "pytest", "test", "测试"))
    return has_code_change_intent and not has_test_signal


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
