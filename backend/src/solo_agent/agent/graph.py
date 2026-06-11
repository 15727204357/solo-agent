from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

from solo_agent.providers import create_provider_from_settings

from .deps import AgentDeps, AgentSettings
from .events import AgentEvent
from .state import AgentState


async def run_agent_events(
    session_id: str,
    run_id: str,
    user_input: str,
    deps: AgentDeps | None = None,
    settings: AgentSettings | Mapping[str, Any] | None = None,
    initial_state: AgentState | Mapping[str, Any] | None = None,
    resume_from_node: str | None = None,
) -> AsyncIterator[AgentEvent]:
    """Run the primary workflow runtime and stream visible progress events."""

    deps = deps or AgentDeps()
    from solo_agent.workflow.stages import _coerce_agent_settings, _setting

    settings = _coerce_agent_settings(settings or deps.settings or AgentSettings())
    deps.settings = settings
    provider = deps.provider or create_provider_from_settings(settings)
    if isinstance(initial_state, AgentState):
        state = initial_state
        state.session_id = session_id
        state.run_id = run_id
        state.user_input = user_input or state.user_input
    elif isinstance(initial_state, Mapping):
        from solo_agent.workflow.graph_state import agent_state_from_graph_data

        state = agent_state_from_graph_data(dict(initial_state))
        state.session_id = session_id
        state.run_id = run_id
        state.user_input = user_input or state.user_input
    else:
        state = AgentState(session_id=session_id, run_id=run_id, user_input=user_input)
    state.run_mode = str(_setting(settings, "run_mode", "agent"))
    state.tool_loop_mode = str(_setting(settings, "tool_loop_mode", "heuristic"))
    state.approval_mode = str(_setting(settings, "approval_mode", "confirm"))
    state.workspace_backend = str(_setting(settings, "workspace_backend", "copy"))
    state.eval_suite_id = _setting(settings, "eval_suite_id", None)
    state.is_plan_mode = state.run_mode == "plan" or bool(_setting(settings, "is_plan_mode", False))
    state.memory_enabled = bool(_setting(settings, "memory_enabled", True))
    state.conversation_history_enabled = bool(
        _setting(settings, "conversation_history_enabled", True)
    )
    state.resume_target = {
        "from_node": resume_from_node or _setting(settings, "resume_from_node", None),
        "recovery_hints": dict(_setting(settings, "recovery_hints", {}) or {}),
    }
    state.human_feedback = dict(_setting(settings, "human_feedback", {}) or {})

    from solo_agent.workflow.stages import _BEHAVIOR_POLICY, _event, _persist

    _BEHAVIOR_POLICY.start_error_run(run_id)

    from solo_agent.workflow.runtime import WorkflowRuntime

    runtime = WorkflowRuntime(
        deps=deps,
        state=state,
        provider=provider,
    )

    try:
        await _persist(deps.persistence, "start_run", state)
        if state.resume_target.get("from_node") in {"team_develop", "team_test", "team_supervisor"}:
            async for event in _run_team_resume_events(state, provider, deps, settings):
                await _persist(deps.persistence, "save_event", event, state)
                yield event
        else:
            async for event in runtime.run():
                await _persist(deps.persistence, "save_event", event, state)
                yield event
    except Exception as exc:
        event = _event(state, "error", "error", str(exc), {"error_type": type(exc).__name__})
        await _persist(deps.persistence, "save_event", event, state)
        await _persist(deps.persistence, "finish_run", state, status="error", error=str(exc))
        yield event
    else:
        await _persist(
            deps.persistence,
            "finish_run",
            state,
            status="awaiting_approval" if state.awaiting_approval else "completed",
        )
    finally:
        _BEHAVIOR_POLICY.finish_error_run(run_id)


async def _run_team_resume_events(
    state: AgentState,
    provider: Any,
    deps: AgentDeps,
    settings: AgentSettings,
) -> AsyncIterator[AgentEvent]:
    from solo_agent.workflow.graph_state import agent_state_to_graph_data
    from solo_agent.workflow.graphs import route_after_team_test
    from solo_agent.workflow.stages import (
        _event,
        _persist_snapshot_stage,
        _team_develop_stage,
        _team_supervisor_stage,
        _team_test_stage,
    )

    from_node = str(state.resume_target.get("from_node") or "")
    yield _event(
        state,
        "run_resumed",
        "resume",
        f"Resuming team workflow from {from_node}",
        {"resume_target": state.resume_target, "human_feedback": state.human_feedback},
    )
    if state.human_feedback:
        state.context.append({"source": "human_feedback", "content": state.human_feedback})
        state.snapshots["human_feedback"] = state.human_feedback

    if from_node == "team_develop":
        async for event in _team_develop_stage(state, provider, deps, settings):
            yield event
        from_node = "team_test"

    if from_node == "team_test":
        async for event in _team_test_stage(state, deps, settings):
            yield event
        graph_state = {"agent_state": agent_state_to_graph_data(state), "events": [], "error": None}
        if route_after_team_test(graph_state) == "team_develop":
            async for event in _team_develop_stage(state, provider, deps, settings):
                yield event
            async for event in _team_test_stage(state, deps, settings):
                yield event
        from_node = "team_supervisor"

    if from_node == "team_supervisor":
        async for event in _team_supervisor_stage(state, deps, settings):
            yield event

    async for event in _persist_snapshot_stage(state):
        yield event
    if not state.awaiting_approval:
        yield AgentEvent(
            type="run_completed",
            session_id=state.session_id,
            run_id=state.run_id,
            node="workflow",
            message="Workflow run completed",
            data={"blocked": state.blocked, "block_reason": state.block_reason},
        )
