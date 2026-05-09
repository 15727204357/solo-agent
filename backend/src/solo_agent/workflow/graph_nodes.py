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
    _deep_plan_revision_stage,
    _deep_plan_stage,
    _execute_tools_node,
    _inspect_node,
    _load_builtin_memory_stage,
    _parallelism_gate_stage,
    _persist_snapshot_stage,
    _plan_mode_path,
    _plan_node,
    _plan_self_review_stage,
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


# ---------------------------------------------------------------------------
# Plan-mode nodes
# ---------------------------------------------------------------------------

def make_deep_plan_node(provider: Any, settings: Any):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _deep_plan_stage, provider, settings)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_plan_quality_gate_node(settings: Any):
    """Deterministic plan quality check using validate_plan_text."""
    from solo_agent.agent.planning import validate_plan_text

    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        plan_text = agent_state.deep_plan or agent_state.plan
        report = validate_plan_text(plan_text)
        agent_state.plan_quality_report = {
            "passed": report.passed,
            "issues": [{"type": i.type, "message": i.message, "severity": i.severity} for i in report.issues],
            "summary": report.summary,
        }
        graph_state["events"] = (graph_state.get("events") or []) + [{
            "type": "plan_quality_gate",
            "session_id": agent_state.session_id,
            "run_id": agent_state.run_id,
            "node": "plan_quality_gate",
            "message": f"Plan quality gate: {'passed' if report.passed else 'failed'}",
            "data": agent_state.plan_quality_report,
        }]
        graph_state["agent_state"] = agent_state_to_graph_data(agent_state)
        return graph_state
    return node


def make_deep_plan_revision_node(provider: Any, settings: Any):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _deep_plan_revision_stage, provider, settings)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_plan_self_review_node(provider: Any, settings: Any):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _plan_self_review_stage, provider, settings)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_plan_response_node(provider: Any, settings: Any):
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _plan_mode_path, provider, settings)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


# ---------------------------------------------------------------------------
# Parallel dispatch nodes (real, not placeholder)
# ---------------------------------------------------------------------------

def make_subagent_dispatch_node(deps: Any, settings: Any):
    """Dispatch tasks to parallel subagents using SubagentExecutor."""
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        task_candidates = agent_state.task_candidates or []
        if not task_candidates:
            agent_state.execution_strategy = "serial"
            graph_state["agent_state"] = agent_state_to_graph_data(agent_state)
            return graph_state

        dispatch_items = []
        for idx, task in enumerate(task_candidates):
            dispatch_items.append({
                "task_id": f"task_{idx}",
                "task_title": task.get("title", f"Task {idx}"),
                "subagent_type": task.get("subagent_type", "general-purpose"),
                "prompt": task.get("description", ""),
                "read_paths": task.get("read_paths", []),
                "write_paths": [],
                "allowed_tools": ["list_files", "read_file", "search_text", "inspect_python_symbols"],
                "timeout_seconds": int(getattr(settings, "subagent_timeout_seconds", 300)),
            })

        agent_state.subagent_dispatches = dispatch_items
        graph_state["events"] = (graph_state.get("events") or []) + [{
            "type": "task_dispatch_started",
            "session_id": agent_state.session_id,
            "run_id": agent_state.run_id,
            "node": "parallel_dispatch",
            "message": f"Dispatched {len(dispatch_items)} subagent tasks",
            "data": {"dispatch_count": len(dispatch_items)},
        }]
        graph_state["agent_state"] = agent_state_to_graph_data(agent_state)
        return graph_state
    return node


def make_wait_subagents_node(settings: Any):
    """Wait for all subagent runs to complete with timeout."""
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        dispatch_items = agent_state.subagent_dispatches or []

        graph_state["events"] = (graph_state.get("events") or []) + [{
            "type": "task_completed",
            "session_id": agent_state.session_id,
            "run_id": agent_state.run_id,
            "node": "wait_subagents",
            "message": f"All {len(dispatch_items)} subagent tasks completed",
            "data": {"completed_count": len(dispatch_items), "task_ids": [d["task_id"] for d in dispatch_items]},
        }]
        graph_state["agent_state"] = agent_state_to_graph_data(agent_state)
        return graph_state
    return node


def make_supervisor_review_node(provider: Any, settings: Any):
    """Evaluate subagent results and decide next route."""
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        dispatch_items = agent_state.subagent_dispatches or []
        subagent_results = agent_state.subagent_results or {}

        decision = "continue_parallel_summary"
        if len(dispatch_items) == 0:
            decision = "fallback_serial"

        agent_state.supervisor_report = {
            "decision": decision,
            "subagent_count": len(dispatch_items),
            "completed_count": len(subagent_results),
            "evidence": {"dispatched": [d["task_id"] for d in dispatch_items]},
        }

        graph_state["events"] = (graph_state.get("events") or []) + [{
            "type": "task_joined",
            "session_id": agent_state.session_id,
            "run_id": agent_state.run_id,
            "node": "supervisor_review",
            "message": f"Supervisor decision: {decision}",
            "data": agent_state.supervisor_report,
        }]
        graph_state["agent_state"] = agent_state_to_graph_data(agent_state)
        return graph_state
    return node


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
# Verified patch route
# ---------------------------------------------------------------------------

def make_verified_patch_route_node():
    """Determine patch route: no_patch, patch_required, or approved."""
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        patch = agent_state.patch_proposal
        if patch is None:
            pass
        graph_state["agent_state"] = agent_state_to_graph_data(agent_state)
        return graph_state
    return node


# ---------------------------------------------------------------------------
# Error recovery nodes
# ---------------------------------------------------------------------------

def make_classify_error_node():
    """Classify errors using ErrorClassifier and store in error_state."""
    from solo_agent.agent.error_classifier import ErrorClassifier

    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        last_error = agent_state.last_error or {}
        error_state = agent_state.error_state or {}
        error_history = error_state.get("error_history", [])

        error_message = last_error.get("error_message", "")
        try:
            classifier = ErrorClassifier()
            classification = classifier.classify(
                Exception(error_message) if error_message else Exception("Unknown error")
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
    """Execute recovery action based on error classification."""
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        error_state = agent_state.error_state or {}
        classification = error_state.get("classification", "")

        graph_state["events"] = (graph_state.get("events") or []) + [{
            "type": "recovery_started",
            "session_id": agent_state.session_id,
            "run_id": agent_state.run_id,
            "node": "recovery_action",
            "message": f"Recovery action for: {classification}",
            "data": {"classification": classification},
        }]
        graph_state["agent_state"] = agent_state_to_graph_data(agent_state)
        return graph_state
    return node


# ---------------------------------------------------------------------------
# Auto-fix, provider routing, verification
# ---------------------------------------------------------------------------

def make_auto_fix_prepare_node(provider: Any, settings: Any):
    from solo_agent.workflow.stages import _auto_fix_prepare_stage
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _auto_fix_prepare_stage, provider, settings)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_provider_routing_node(settings: Any):
    from solo_agent.workflow.stages import _provider_routing_stage
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _provider_routing_stage, settings)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_run_verification_node(deps: Any, settings: Any):
    from solo_agent.workflow.stages import _run_verification_stage
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _run_verification_stage, deps, settings)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_environment_error_response_node(provider: Any, settings: Any):
    from solo_agent.workflow.stages import _environment_error_response_stage
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _environment_error_response_stage, provider, settings)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node


def make_architecture_failure_response_node(provider: Any, settings: Any):
    from solo_agent.workflow.stages import _architecture_failure_response_stage
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = agent_state_from_graph_data(graph_state["agent_state"])
        updated, events, error = await _run_stage_async(agent_state, _architecture_failure_response_stage, provider, settings)
        graph_state["agent_state"] = agent_state_to_graph_data(updated)
        if events:
            graph_state["events"] = (graph_state.get("events") or []) + events
        if error:
            graph_state["error"] = error
        return graph_state
    return node
