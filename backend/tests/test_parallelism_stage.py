from __future__ import annotations

import pytest
from solo_agent.agent.deps import AgentSettings
from solo_agent.agent.state import AgentState
from solo_agent.workflow.stages import _parallelism_gate_stage


@pytest.mark.asyncio
async def test_parallelism_gate_sets_parallel_strategy_when_all_conditions_pass() -> None:
    state = AgentState(session_id="s1", run_id="r1", user_input="implement independent tasks")
    state.plan = '''
```json
{
  "parallel_tasks": [
    {
      "id": "T1",
      "title": "Provider tests",
      "domain": "providers",
      "read_paths": ["backend/src/solo_agent/providers/"],
      "write_paths": ["backend/tests/test_provider_config.py"],
      "verify_commands": ["pytest backend/tests/test_provider_config.py -q"]
    },
    {
      "id": "T2",
      "title": "Memory tests",
      "domain": "memory",
      "read_paths": ["backend/src/solo_agent/memory/"],
      "write_paths": ["backend/tests/test_memory_inbox.py"],
      "verify_commands": ["pytest backend/tests/test_memory_inbox.py -q"]
    }
  ]
}
```
'''

    events = [event async for event in _parallelism_gate_stage(state, AgentSettings())]

    assert [event.type for event in events] == [
        "parallelism_gate_started",
        "parallelism_gate_completed",
    ]
    assert state.execution_strategy == "parallel"
    assert state.parallelism_decision["allowed"] is True
    assert state.snapshots["execution_strategy"] == "parallel"


@pytest.mark.asyncio
async def test_parallelism_gate_falls_back_to_serial_without_metadata() -> None:
    state = AgentState(session_id="s1", run_id="r1", user_input="fix everything")
    state.plan = "1. Inspect the codebase. 2. Fix the issue. 3. Run pytest."

    events = [event async for event in _parallelism_gate_stage(state, AgentSettings())]

    assert events[-1].type == "parallelism_gate_completed"
    assert state.execution_strategy == "serial"
    assert state.parallelism_decision["allowed"] is False
    assert state.task_candidates[0]["risk_flags"] == ["unstructured_plan"]
