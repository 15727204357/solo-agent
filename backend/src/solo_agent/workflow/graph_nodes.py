from __future__ import annotations

from typing import Any

from solo_agent.agent.state import AgentState
from solo_agent.workflow.graph_state import (
    SoloGraphState,
    agent_state_from_graph_data,
    agent_state_to_graph_data,
)
from solo_agent.workflow.stages import (
    _build_memory_context_stage,
    _collect_context_node,
    _compress_memory_stage,
    _context_guard_stage,
    _execute_tools_node,
    _inspect_node,
    _load_builtin_memory_stage,
    _parallelism_gate_stage,
    _persist_snapshot_stage,
    _plan_node,
    _prefetch_memory_stage,
    _propose_verified_patch_node,
    _queue_prefetch_stage,
    _receive_user_turn_stage,
    _respond_node,
    _select_tools_node,
    _skill_context_stage,
    _skip_memory_stage,
    _subdirectory_hint_stage,
    _sync_memory_stage,
    _task_state_stage,
)

StageKwArgs = dict[str, Any]


def _run_stage(
    state: AgentState,
    stage_fn: Any,
    *args: Any,
    **kwargs: Any,
) -> tuple[AgentState, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    collected_error: dict[str, Any] | None = None
    try:
        import asyncio
        agen = stage_fn(state, *args, **kwargs)
        while True:
            try:
                event = asyncio.get_event_loop().run_until_complete(agen.__anext__())
                events.append(event.to_dict())
            except StopAsyncIteration:
                break
    except Exception as exc:
        collected_error = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    return state, events, collected_error


async def _run_stage_async(
    state: AgentState,
    stage_fn: Any,
    *args: Any,
    **kwargs: Any,
) -> tuple[AgentState, list[dict[str, Any]], dict[str, Any] | None]:
    events: list[dict[str, Any]] = []
    collected_error: dict[str, Any] | None = None
    try:
        async for event in stage_fn(state, *args, **kwargs):
            events.append(event.to_dict())
    except Exception as exc:
        collected_error = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    return state, events, collected_error


def _make_node(stage_fn: Any, *extra_args: Any, **extra_kwargs: Any):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, stage_fn, *extra_args, **extra_kwargs)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            existing = graph_state.get("events") or []
            graph_state["events"] = existing + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_receive_user_turn_node(settings: Any):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _receive_user_turn_stage, settings)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_skip_memory_node():
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _skip_memory_stage)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_load_builtin_memory_node(deps: Any):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _load_builtin_memory_stage, deps)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_prefetch_memory_node(deps: Any, settings: Any):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _prefetch_memory_stage, deps, settings)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_build_memory_context_node():
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _build_memory_context_stage)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_skill_context_node(deps: Any, settings: Any):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _skill_context_stage, deps, settings)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_context_guard_node(provider: Any, deps: Any, settings: Any, *, phase: str):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        async for _event in _context_guard_stage(agent_state, provider, deps, settings, phase=phase):
            pass  # events collected via state mutations
        graph_state["agent_state"] = agent_state_to_graph_data(agent_state)
        return graph_state
    return node


def make_plan_node(provider: Any, settings: Any):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _plan_node, provider, settings)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_task_state_node():
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _task_state_stage)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_parallelism_gate_node(settings: Any):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _parallelism_gate_stage, settings)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_collect_context_node(deps: Any, settings: Any):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _collect_context_node, deps, settings)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_inspect_node(deps: Any):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _inspect_node, deps)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_select_tools_node(deps: Any, settings: Any):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _select_tools_node, deps, settings)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_execute_tools_node(deps: Any, settings: Any):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _execute_tools_node, deps, settings)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_propose_verified_patch_node(provider: Any, deps: Any, settings: Any):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(
            agent_state, _propose_verified_patch_node, provider, deps, settings
        )
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_subdirectory_hint_node(settings: Any):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _subdirectory_hint_stage, settings)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_respond_node(provider: Any, settings: Any):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _respond_node, provider, settings)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_sync_memory_node(deps: Any):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _sync_memory_stage, deps)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_queue_prefetch_node(deps: Any, settings: Any):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _queue_prefetch_stage, deps, settings)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_compress_memory_node(provider: Any, deps: Any, settings: Any):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(
            agent_state, _compress_memory_stage, provider, deps, settings
        )
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_persist_snapshot_node():
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _persist_snapshot_stage)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


async def parallel_dispatch_placeholder_node(graph_state: SoloGraphState) -> SoloGraphState:
    events = graph_state.get("events") or []
    events.append({
        "type": "parallel_dispatch_placeholder",
        "session_id": graph_state.get("agent_state", {}).get("session_id", ""),
        "run_id": graph_state.get("agent_state", {}).get("run_id", ""),
        "node": "parallel_dispatch",
        "message": "Parallel dispatch is a placeholder — real parallel scheduler coming in a follow-up",
        "data": {},
    })
    agent_state = agent_state_from_graph_data(graph_state["agent_state"])
    agent_state.execution_strategy = "serial"
    graph_state["agent_state"] = agent_state_to_graph_data(agent_state)
    graph_state["events"] = events
    return graph_state
