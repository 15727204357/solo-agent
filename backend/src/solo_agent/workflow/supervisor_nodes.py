"""Supervisor node factories — sub-task dispatch, evaluation, and loop control."""

from __future__ import annotations

from typing import Any

from solo_agent.agent.state import ResearcherState
from solo_agent.workflow.graph_state import (
    SoloGraphState,
    coordinator_state_from_graph_data,
    coordinator_state_to_graph_data,
)


def make_dispatch_researchers_node(deps: Any, settings: Any):
    """Dispatch sub-tasks to parallel researchers via SubagentExecutor."""
    from solo_agent.workflow.state import SubagentRunRecord, WorkflowState

    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = coordinator_state_from_graph_data(graph_state["agent_state"])
        agent_state.loop_stage = "dispatch_researchers"
        task_specs = agent_state.supervisor_task_specs

        if not task_specs:
            task_specs = agent_state.supervisor_task_specs or [{
                "task_id": "research_main",
                "title": agent_state.deep_plan or agent_state.plan,
                "description": agent_state.user_input,
                "subagent_type": "general-purpose",
            }]

        workflow_state = WorkflowState.from_agent_state(agent_state)
        from solo_agent.workflow.subagent.executor import SubagentExecutor
        executor = SubagentExecutor(
            max_concurrent=int(getattr(settings, "max_concurrent_subagents", 3)),
            timeout_seconds=float(getattr(settings, "subagent_timeout_seconds", 900)),
        )

        researcher_states = []
        for spec in task_specs:
            task_id = spec.get("task_id", f"task_{len(workflow_state.subagent_runs)}")
            record = SubagentRunRecord(
                run_id=task_id,
                subagent_type=spec.get("subagent_type", "general-purpose"),
                description=spec.get("description", "")[:200],
                status="pending",
                started_at=str(__import__("time").time()),
            )
            workflow_state.add_subagent_run(record)

            researcher = ResearcherState(
                session_id=f"{agent_state.session_id}/researcher/{task_id}",
                run_id=f"{agent_state.run_id}/researcher/{task_id}",
                user_input=agent_state.user_input,
                research_prompt=spec.get("description", agent_state.user_input),
                subagent_type=spec.get("subagent_type", "general-purpose"),
                memory_enabled=agent_state.memory_enabled,
                conversation_history_enabled=False,
            )
            researcher.plan = agent_state.deep_plan or agent_state.plan
            researcher_states.append(researcher.snapshot())

        agent_state.subagent_dispatches = [dict(spec) for spec in task_specs]
        graph_state["agent_state"] = coordinator_state_to_graph_data(agent_state)
        graph_state["supervisor_executor"] = executor
        graph_state["supervisor_workflow_state"] = workflow_state
        graph_state["researcher_states"] = researcher_states
        return graph_state

    return node


def make_wait_researchers_node(settings: Any):
    """Wait for all dispatched researchers to complete."""
    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = coordinator_state_from_graph_data(graph_state["agent_state"])
        agent_state.loop_stage = "wait_researchers"

        executor = graph_state.get("supervisor_executor")
        if executor is not None:
            await executor.wait_for_all(
                timeout=float(getattr(settings, "subagent_timeout_seconds", 900))
            )

        workflow_state = graph_state.get("supervisor_workflow_state")
        if workflow_state is not None:
            completed = {}
            events = graph_state.get("events") or []
            for rid, record in workflow_state.subagent_runs.items():
                event_type = {
                    "completed": "task_completed",
                    "failed": "task_failed",
                    "running": "task_started",
                    "pending": "task_started",
                }.get(record.status, "task_started")
                events.append({
                    "type": event_type,
                    "session_id": agent_state.session_id,
                    "run_id": agent_state.run_id,
                    "node": "supervisor",
                    "message": f"Subagent {rid}: {record.status}",
                    "data": {
                        "task_id": rid,
                        "subagent_type": record.subagent_type,
                        "status": record.status,
                        "error": record.error[:200] if record.error else "",
                    },
                    "agent_source": "supervisor",
                })
                result_text = record.result if record.status == "completed" else f"error: {record.error}"
                completed[rid] = {
                    "title": record.description[:100],
                    "status": record.status,
                    "findings": result_text[:2000],
                }
            agent_state.subagent_results = completed
            graph_state["events"] = events

        graph_state["agent_state"] = coordinator_state_to_graph_data(agent_state)
        return graph_state
    return node


def make_evaluate_results_node(provider: Any, settings: Any):
    """LLM-driven routing: evaluate researcher results and decide next step."""
    from solo_agent.providers import ChatMessage

    async def node(graph_state: SoloGraphState) -> SoloGraphState:
        agent_state = coordinator_state_from_graph_data(graph_state["agent_state"])
        agent_state.loop_stage = "evaluate_results"

        loop_count = agent_state.snapshots.get("exploration_loop_count", 0)
        if loop_count >= 2:
            agent_state.snapshots["exploration_decision"] = "summarize_return"
            agent_state.snapshots["exploration_loop_count"] = loop_count + 1
            graph_state["agent_state"] = coordinator_state_to_graph_data(agent_state)
            return graph_state

        results_summary = str(agent_state.subagent_results)
        messages = [
            ChatMessage(role="system", content=(
                "You are a research supervisor. Based on the research results below, "
                "decide whether to: continue_explore (need more investigation), "
                "summarize_return (enough findings to answer), or fallback_serial "
                "(research approach failed, try single-agent). "
                "Respond with ONLY one word: continue_explore / summarize_return / fallback_serial"
            )),
            ChatMessage(
                role="user",
                content=f"Research question: {agent_state.user_input}\n\nResults so far:\n{results_summary}",
            ),
        ]
        try:
            raw = await provider.complete(messages, temperature=0.1, max_tokens=50)
            decision = _normalize_routing_key(str(raw).strip())
        except Exception:
            decision = "summarize_return"

        agent_state.snapshots["exploration_loop_count"] = loop_count + 1
        agent_state.snapshots["exploration_decision"] = decision

        graph_state["agent_state"] = coordinator_state_to_graph_data(agent_state)
        return graph_state
    return node


def _normalize_routing_key(raw: str) -> str:
    allowed = {"continue_explore", "summarize_return", "fallback_serial"}
    key = raw.strip().lower().replace(".", "").replace('"', "").replace("'", "")
    if key in allowed:
        return key
    return "summarize_return"
