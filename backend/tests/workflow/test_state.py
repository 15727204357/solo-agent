from __future__ import annotations

from solo_agent.agent.state import AgentState
from solo_agent.workflow.state import SubagentRunRecord, WorkflowState


def test_workflow_state_creation():
    agent_state = AgentState(session_id="s1", run_id="r1", user_input="hello")
    wf = WorkflowState.from_agent_state(agent_state)
    assert wf.session_id == "s1"
    assert wf.run_id == "r1"
    assert wf.messages == []
    assert wf.artifacts == {}
    assert wf.subagent_runs == {}
    assert wf.active_subagent_count == 0


def test_snapshot_backward_compatibility():
    agent_state = AgentState(session_id="s1", run_id="r1", user_input="hello")
    wf = WorkflowState.from_agent_state(agent_state)
    snap = wf.snapshot()

    assert "session_id" in snap
    assert snap["session_id"] == "s1"
    assert snap["run_id"] == "r1"
    assert snap["user_input"] == "hello"
    assert "workflow_artifacts" in snap
    assert "subagent_runs" in snap


def test_workflow_fields_isolation():
    agent_state = AgentState(session_id="s1", run_id="r1", user_input="hello")
    wf = WorkflowState.from_agent_state(agent_state)
    wf.thread_data["key"] = "value"
    wf.artifacts["result"] = "done"

    snap = wf.snapshot()
    assert snap["workflow_artifacts"]["result"] == "done"
    assert agent_state.response == ""
    assert "thread_data" not in snap


def test_subagent_run_tracking():
    agent_state = AgentState(session_id="s1", run_id="r1", user_input="test")
    wf = WorkflowState.from_agent_state(agent_state)

    record = SubagentRunRecord(
        run_id="task_1",
        subagent_type="code-review",
        description="Review app.py",
    )
    wf.add_subagent_run(record)
    assert "task_1" in wf.subagent_runs
    assert wf.subagent_runs["task_1"].subagent_type == "code-review"


def test_active_subagent_count():
    agent_state = AgentState(session_id="s1", run_id="r1", user_input="test")
    wf = WorkflowState.from_agent_state(agent_state)

    r1 = SubagentRunRecord(run_id="t1", subagent_type="general-purpose", description="a", status="running")
    r2 = SubagentRunRecord(run_id="t2", subagent_type="general-purpose", description="b", status="completed")
    r3 = SubagentRunRecord(run_id="t3", subagent_type="general-purpose", description="c", status="pending")
    wf.add_subagent_run(r1)
    wf.add_subagent_run(r2)
    wf.add_subagent_run(r3)

    assert wf.get_active_subagent_count() == 2  # running + pending
