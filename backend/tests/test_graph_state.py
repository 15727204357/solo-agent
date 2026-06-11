from __future__ import annotations

import pytest
from solo_agent.agent.state import AgentState, ToolCallRecord
from solo_agent.workflow.graph_state import (
    agent_state_from_graph_data,
    agent_state_to_graph_data,
    initial_graph_state,
)


@pytest.mark.asyncio
async def test_agent_state_round_trips_tool_calls() -> None:
    state = AgentState(
        session_id="s1",
        run_id="r1",
        user_input="test input",
        plan="test plan",
        execution_strategy="parallel",
        tool_calls=[
            ToolCallRecord(name="read_file", arguments={"path": "test.py"}, result="content"),
        ],
    )
    data = agent_state_to_graph_data(state)
    restored = agent_state_from_graph_data(data)
    assert restored.session_id == "s1"
    assert restored.run_id == "r1"
    assert restored.user_input == "test input"
    assert restored.plan == "test plan"
    assert restored.execution_strategy == "parallel"
    assert len(restored.tool_calls) == 1
    assert restored.tool_calls[0].name == "read_file"
    assert restored.tool_calls[0].arguments == {"path": "test.py"}
    assert restored.tool_calls[0].result == "content"


@pytest.mark.asyncio
async def test_execution_strategy_task_candidates_parallelism_decision_survive() -> None:
    state = AgentState(
        session_id="s1",
        run_id="r1",
        user_input="do it",
        task_list={"thread_id": "s1", "tasks": [{"id": "T-1", "subject": "Plan", "status": "in_progress"}]},
        task_candidates=[{"id": "T1", "title": "Task one"}],
        parallelism_decision={"allowed": True, "mode": "parallel", "reason": "All pass"},
        execution_strategy="parallel",
    )
    data = agent_state_to_graph_data(state)
    restored = agent_state_from_graph_data(data)
    assert isinstance(data["task_list"], dict)
    assert restored.task_list["tasks"][0]["subject"] == "Plan"
    assert restored.task_candidates == [{"id": "T1", "title": "Task one"}]
    assert restored.parallelism_decision == {"allowed": True, "mode": "parallel", "reason": "All pass"}
    assert restored.execution_strategy == "parallel"


@pytest.mark.asyncio
async def test_initial_graph_state_returns_dict_shaped() -> None:
    state = AgentState(session_id="s1", run_id="r1", user_input="hello")
    gs = initial_graph_state(state)
    assert isinstance(gs, dict)
    assert "agent_state" in gs
    assert "events" in gs
    assert "error" in gs
    assert gs["agent_state"]["session_id"] == "s1"
    assert gs["agent_state"]["run_id"] == "r1"
    assert gs["agent_state"]["user_input"] == "hello"


@pytest.mark.asyncio
async def test_agent_state_round_trips_error_fields() -> None:
    state = AgentState(
        session_id="s1",
        run_id="r1",
        user_input="test",
        last_error={"error_code": "E001", "category": "retryable"},
        retry_count=2,
        error_classification="retryable",
        compaction_attempts=1,
    )
    data = agent_state_to_graph_data(state)
    restored = agent_state_from_graph_data(data)
    assert restored.last_error == {"error_code": "E001", "category": "retryable"}
    assert restored.retry_count == 2
    assert restored.error_classification == "retryable"
    assert restored.compaction_attempts == 1


@pytest.mark.asyncio
async def test_agent_state_round_trips_skill_evolution_fields() -> None:
    state = AgentState(
        session_id="s1",
        run_id="r1",
        user_input="test",
        skill_evolution_candidates=[{"target_skill": "python-backend-change"}],
        skill_evolution_proposal={"proposal": {"id": "skillchg_1"}},
    )

    data = agent_state_to_graph_data(state)
    restored = agent_state_from_graph_data(data)

    assert restored.skill_evolution_candidates == [{"target_skill": "python-backend-change"}]
    assert restored.skill_evolution_proposal == {"proposal": {"id": "skillchg_1"}}
