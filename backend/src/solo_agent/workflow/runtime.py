"""Workflow runtime lifecycle orchestration — multi-agent LangGraph path."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from solo_agent.agent.events import AgentEvent
from solo_agent.agent.state import AgentState
from solo_agent.workflow.checkpoints import create_checkpointer
from solo_agent.workflow.graph_state import agent_state_from_graph_data, initial_graph_state


def _use_multi_agent(settings: Any) -> bool:
    return bool(getattr(settings, "use_multi_agent", False)) or getattr(settings, "run_mode", "") == "coordinator"


class WorkflowRuntime:
    """Run the workflow graph and stream AgentEvent values.

    Uses the monolithic single-agent graph by default (backward compatible).
    When settings.use_multi_agent is True, uses the coordinator multi-agent graph.
    """

    def __init__(
        self,
        *,
        deps: Any,
        state: AgentState,
        provider: Any,
    ):
        self._deps = deps
        self._agent_state = state
        self._provider = provider

    async def run(self) -> AsyncIterator[AgentEvent]:
        """Orchestration path: build graph, compile, stream, update state."""
        settings = self._deps.settings
        gs = initial_graph_state(self._agent_state)
        is_multi = _use_multi_agent(settings)
        default_agent_source = "coordinator" if is_multi else "agent"

        if is_multi:
            from solo_agent.workflow.coordinator_graph import build_coordinator_graph
            graph = build_coordinator_graph(
                provider=self._provider, deps=self._deps, settings=settings,
            )
        else:
            from solo_agent.workflow.graphs import build_main_workflow_graph
            graph = build_main_workflow_graph(
                provider=self._provider, deps=self._deps, settings=settings,
            )
        checkpointer = await create_checkpointer(settings)
        compiled = graph.compile(checkpointer=checkpointer if checkpointer else False)

        thread_id = f"{self._agent_state.run_id}/coordinator" if is_multi else self._agent_state.run_id
        config = {"configurable": {"thread_id": thread_id}}

        seen_event_keys: set[tuple[str, str, str]] = set()

        async for update in compiled.astream(gs, config=config, stream_mode="values"):
            if not isinstance(update, dict):
                continue

            for event_dict in update.get("events", []):
                if isinstance(event_dict, dict):
                    key = (
                        event_dict.get("type", ""),
                        event_dict.get("session_id", ""),
                        event_dict.get("created_at", ""),
                    )
                    if key not in seen_event_keys:
                        seen_event_keys.add(key)
                        yield AgentEvent(
                            type=event_dict.get("type", ""),
                            session_id=event_dict.get("session_id", ""),
                            run_id=event_dict.get("run_id", ""),
                            node=event_dict.get("node", "workflow"),
                            message=event_dict.get("message", ""),
                            data=event_dict.get("data", {}),
                            agent_source=event_dict.get("agent_source", default_agent_source),
                        )

            if "agent_state" in update:
                self._agent_state = agent_state_from_graph_data(update["agent_state"])

        if self._agent_state.awaiting_approval:
            return

        yield AgentEvent(
            type="run_completed",
            session_id=self._agent_state.session_id,
            run_id=self._agent_state.run_id,
            node="workflow",
            message="Workflow run completed",
            data={"blocked": self._agent_state.blocked, "block_reason": self._agent_state.block_reason},
        )
