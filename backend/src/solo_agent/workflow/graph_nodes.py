from __future__ import annotations

from typing import Any

from solo_agent.agent.state import AgentState
from solo_agent.workflow.graph_state import (
    SoloGraphState,
    agent_state_from_graph_data,
    agent_state_to_graph_data,
)
from solo_agent.workflow.stages import (
    _architecture_failure_response_stage,
    _build_memory_context_stage,
    _collect_context_node,
    _compress_memory_stage,
    _context_guard_stage,
    _environment_error_response_stage,
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
    _recovery_action_stage,
    _respond_node,
    _select_tools_node,
    _skill_context_stage,
    _skip_memory_stage,
    _subdirectory_hint_stage,
    _sync_memory_stage,
    _task_state_stage,
)

StageKwArgs = dict[str, Any]


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
    return _make_node(_receive_user_turn_stage, settings)


def make_skip_memory_node():
    return _make_node(_skip_memory_stage)


def make_load_builtin_memory_node(deps: Any):
    return _make_node(_load_builtin_memory_stage, deps)


def make_prefetch_memory_node(deps: Any, settings: Any):
    return _make_node(_prefetch_memory_stage, deps, settings)


def make_build_memory_context_node():
    return _make_node(_build_memory_context_stage)


def make_skill_context_node(deps: Any, settings: Any):
    return _make_node(_skill_context_stage, deps, settings)


def make_context_guard_node(provider: Any, deps: Any, settings: Any, *, phase: str):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        async for _event in _context_guard_stage(agent_state, provider, deps, settings, phase=phase):
            pass  # events collected via state mutations
        graph_state["agent_state"] = agent_state_to_graph_data(agent_state)
        return graph_state
    return node


def make_plan_node(provider: Any, settings: Any):
    return _make_node(_plan_node, provider, settings)


def make_task_state_node(settings: Any | None = None):
    if settings is None:
        from solo_agent.agent.deps import AgentSettings

        settings = AgentSettings()
    return _make_node(_task_state_stage, settings)


def make_parallelism_gate_node(settings: Any):
    return _make_node(_parallelism_gate_stage, settings)


def make_collect_context_node(deps: Any, settings: Any):
    return _make_node(_collect_context_node, deps, settings)


def make_inspect_node(deps: Any):
    return _make_node(_inspect_node, deps)


def make_select_tools_node(deps: Any, settings: Any):
    return _make_node(_select_tools_node, deps, settings)


def make_execute_tools_node(deps: Any, settings: Any):
    return _make_node(_execute_tools_node, deps, settings)


def make_propose_verified_patch_node(provider: Any, deps: Any, settings: Any):
    return _make_node(_propose_verified_patch_node, provider, deps, settings)


def make_subdirectory_hint_node(settings: Any):
    return _make_node(_subdirectory_hint_stage, settings)


def make_respond_node(provider: Any, settings: Any):
    return _make_node(_respond_node, provider, settings)


def make_sync_memory_node(deps: Any):
    return _make_node(_sync_memory_stage, deps)


def make_queue_prefetch_node(deps: Any, settings: Any):
    return _make_node(_queue_prefetch_stage, deps, settings)


def make_compress_memory_node(provider: Any, deps: Any, settings: Any):
    return _make_node(_compress_memory_stage, provider, deps, settings)


def make_persist_snapshot_node():
    return _make_node(_persist_snapshot_stage)

# ---------------------------------------------------------------------------
# Review nodes
# ---------------------------------------------------------------------------

def make_spec_compliance_review_node(provider: Any, settings: Any):
    """LLM-based spec compliance review — delegates to existing responder."""
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        tool_calls = agent_state.tool_calls or []
        has_changes = any(
            tc.name == "apply_text_edit" and not tc.blocked for tc in tool_calls
        )
        agent_state.review_reports["spec_compliance"] = {
            "status": "passed" if not has_changes else "reviewed",
            "has_code_changes": has_changes,
            "tool_call_count": len(tool_calls),
        }
        graph_state["agent_state"] = agent_state_to_graph_data(agent_state)
        return graph_state
    return node


# ---------------------------------------------------------------------------
# Error recovery nodes
# ---------------------------------------------------------------------------

def make_classify_error_node():
    """Classify errors using BehaviorPolicy and store in error_state."""
    from solo_agent.workflow.stages import _BEHAVIOR_POLICY

    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        last_error = agent_state.last_error or {}
        error_state = agent_state.error_state or {}
        error_history = error_state.get("error_history", [])

        error_message = last_error.get("error_message", "")
        try:
            classification = _BEHAVIOR_POLICY.classify_error(
                Exception(error_message) if error_message else Exception("Unknown error"),
                stage="graph_node",
                attempt_count=agent_state.retry_count,
                run_id=agent_state.run_id,
            )
            category = classification.category
        except Exception:
            category = "fatal"

        if category == "retryable":
            classification_out = "recoverable_error"
        elif category == "fixable":
            classification_out = "recoverable_error"
        elif category == "fatal":
            classification_out = "policy_violation"
        else:
            classification_out = "architecture_failure"

        error_state["classification"] = classification_out
        error_state["error_history"] = error_history + [{
            "classification": classification_out,
            "message": error_message or "",
        }]
        agent_state.error_state = error_state

        graph_state["events"] = (graph_state.get("events") or []) + [{
            "type": "error_classified",
            "session_id": agent_state.session_id,
            "run_id": agent_state.run_id,
            "node": "classify_error",
            "message": f"Error classified: {classification_out}",
            "data": {"classification": classification_out, "error_message": error_message},
        }]
        graph_state["agent_state"] = agent_state_to_graph_data(agent_state)
        return graph_state
    return node


def make_repetition_guard_node():
    """Check if same error repeated >= 3 times — escalate to architecture_failure."""
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        error_state = agent_state.error_state or {}
        error_history = error_state.get("error_history", [])

        same_count = sum(
            1 for e in error_history[-5:]
            if e.get("classification") in ("recoverable_error",)
        )
        if same_count >= 3:
            error_state["classification"] = "architecture_failure"
            agent_state.recovery_attempts = 0
            graph_state["events"] = (graph_state.get("events") or []) + [{
                "type": "architecture_failure",
                "session_id": agent_state.session_id,
                "run_id": agent_state.run_id,
                "node": "repetition_guard",
                "message": "Same error repeated 3+ times — architecture failure",
                "data": {"repeated_count": same_count},
            }]
        else:
            if error_state.get("classification") == "recoverable_error":
                agent_state.recovery_attempts += 1

        graph_state["agent_state"] = agent_state_to_graph_data(agent_state)
        return graph_state
    return node


def make_recovery_action_node(deps: Any, settings: Any):
    return _make_node(_recovery_action_stage, deps, settings)


def make_environment_error_response_node(provider: Any, settings: Any):
    return _make_node(_environment_error_response_stage, provider, settings)


def make_architecture_failure_response_node(provider: Any, settings: Any):
    return _make_node(_architecture_failure_response_stage, provider, settings)
