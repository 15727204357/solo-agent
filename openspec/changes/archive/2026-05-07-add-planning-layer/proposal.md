## Why

Solo Agent currently operates exclusively in "agent" mode: it plans, collects context, executes tools, and produces results in a single run. Users often want to first generate a thorough implementation plan that can be handed to a zero-context engineer (or another agent instance) for execution. Superpowers' `writing-plans` skill defines a proven standard for such plans: no placeholders, zero-context-reader coverage, inline self-review, and explicit execution choices. Solo Agent should support a `plan` mode that generates these deep plans and stops, keeping `agent` mode as the backward-compatible default.

## What Changes

- New `run_mode` field (`"agent"` | `"plan"`) on `CreateRunRequest` and `AgentSettings`, defaulting to `"agent"`
- New `planning.py` module with `PlanningMode`, `PlanQualityReport`, deep plan message builder, and plan text validator
- `plan` mode graph path: memory/skills → context guard → deep plan → self-review → response/persist (skips tool execution and patch proposals)
- `agent` mode graph path unchanged (backward compatible)
- New deep plan system prompt enforcing no-placeholders, zero-context-reader, inline self-reflection, execution options
- New events: `deep_plan_started`, `deep_plan_delta`, `plan_self_review_completed` alongside existing `plan_completed`
- Frontend composer: `Agent` / `Plan` segmented toggle submitted with run payload
- Unit tests for plan validation, graph mode branching, API mode input, and frontend smoke

## Capabilities

### New Capabilities

- `planning-mode`: Core planning mode infrastructure — `run_mode` on request/settings/state, graph branching between `agent` and `plan` paths, and mode-aware event generation
- `deep-plan-prompt`: Superpowers-style deep planning system/user prompts that produce file maps, TDD steps, expected outputs, and execution options without placeholders
- `plan-validation`: Deterministic quality validation that rejects placeholder expressions (TODO, TBD, "implement later", "类似上一步", etc.), missing file maps, and missing self-review sections
- `run-mode-api`: Web API extension — `CreateRunRequest.run_mode`, run metadata persistence, 422 on invalid modes, runner passes mode to `AgentSettings`
- `planning-ui`: Frontend mode selector (Agent / Plan segmented control), mode submission in payload, plan summary display in monitoring panel

### Modified Capabilities

None. All existing `agent` mode behavior is preserved unchanged. New capabilities are additive.

## Impact

- `backend/src/solo_agent/agent/planning.py` (new): planning module
- `backend/src/solo_agent/agent/deps.py`: add `run_mode` field to `AgentSettings`
- `backend/src/solo_agent/agent/state.py`: add `run_mode` and deep plan fields
- `backend/src/solo_agent/agent/prompts.py`: add deep plan prompts (preserve existing)
- `backend/src/solo_agent/agent/graph.py`: add `plan` mode path (preserve `agent` mode)
- `backend/src/solo_agent/agent/events.py`: no changes needed (generic event structure)
- `backend/src/solo_agent/web/routes.py`: add `run_mode` to `CreateRunRequest`, validate
- `backend/src/solo_agent/web/runner.py`: pass `run_mode` to `AgentSettings`
- `backend/src/solo_agent/web/models.py`: `RunRecord.metadata` already supports arbitrary fields
- `backend/src/solo_agent/settings.py`: no changes needed
- `frontend/templates/index.html`: add mode selector UI
- `frontend/static/app.js`: wire mode toggle to run payload
- `backend/tests/test_planning_layer.py` (new): planning module tests
- `backend/tests/test_agent_graph.py`: add plan-mode graph tests
- `backend/tests/test_web_api.py`: add run_mode API tests
