## 1. Foundation: Settings, State, and Interface

- [x] 1.1 Add `run_mode: str = "agent"` field to `AgentSettings` in `backend/src/solo_agent/agent/deps.py`
- [x] 1.2 Add `run_mode`, `deep_plan`, `plan_quality_report` fields to `AgentState` in `backend/src/solo_agent/agent/state.py`, include in `snapshot()`
- [x] 1.3 Add `run_mode: str | None = None` to `CreateRunRequest` Pydantic model in `backend/src/solo_agent/web/routes.py` with Literal validation
- [x] 1.4 Store `run_mode` in `RunRecord.metadata` on run creation in `create_run` route
- [x] 1.5 Read `run_mode` from run metadata in `AgentRunner.run()` in `backend/src/solo_agent/web/runner.py` and pass to `AgentSettings` and `AgentState`

## 2. Worker A: Planning Module and Prompts

- [x] 2.1 Create `backend/src/solo_agent/agent/planning.py` with dataclasses: `PlanningMode` (Literal["agent", "plan"]), `PlanQualityIssue`, `PlanQualityReport`
- [x] 2.2 Implement `validate_plan_text(plan_text: str) -> PlanQualityReport` with placeholder detection (TBD/TODO/implement later/类似上一步/适当处理/待定/略/同理/同上)
- [x] 2.3 Implement structural completeness checks in `validate_plan_text`: file map section, self-review section, numbered steps
- [x] 2.4 Add `DEEP_PLAN_SYSTEM_PROMPT` to `backend/src/solo_agent/agent/prompts.py` — Superpowers-style prompt enforcing no placeholders, file map, TDD steps, commands, expected results, execution options, self-review
- [x] 2.5 Add `build_deep_plan_messages()` function to `prompts.py` constructing system + user messages with memory context, skill context, task, and Superpowers conventions
- [x] 2.6 Export new prompt functions and constants from `backend/src/solo_agent/agent/__init__.py`

## 3. Worker B: Graph Integration and Events

- [x] 3.1 Set `state.run_mode` from settings in `_receive_user_turn_stage` in `backend/src/solo_agent/agent/graph.py`
- [x] 3.2 Implement `_deep_plan_stage()` coroutine — streaming LLM call with deep plan prompts, emitting `deep_plan_started`, `deep_plan_delta`, `plan_completed` events
- [x] 3.3 Implement `_plan_self_review_stage()` coroutine — second LLM call reviewing the plan, emitting `plan_self_review_completed` with `PlanQualityReport` data
- [x] 3.4 Implement `_plan_mode_path()` coroutine that chains: context_guard → deep_plan → self_review → response → persist
- [x] 3.5 Branch `_run_graph()`: after context guard, if `run_mode == "plan"`, yield from `_plan_mode_path()` and return; else continue existing `agent` path
- [x] 3.6 In `plan` mode, emit `plan_completed` after deep plan generation for backward compatibility with existing event consumers
- [x] 3.7 Set `state.response` to include deep plan text in `plan` mode so the Web UI renders it

## 4. Worker C: Web API and Frontend

- [x] 4.1 Validate `run_mode` in `CreateRunRequest`: reject values other than `"agent"` or `"plan"` with 422
- [x] 4.2 Include `run_mode` in `RunRecord.metadata` in `create_run` route handler
- [x] 4.3 Consume `run_mode` in `AgentRunner.run()` and propagate to `AgentSettings` and `AgentState`
- [x] 4.4 Add Agent/Plan segmented control to `frontend/templates/index.html` composer area
- [x] 4.5 Add CSS styles for segmented mode control to match existing UI palette
- [x] 4.6 Wire mode toggle in `frontend/static/app.js`: default "agent", append `run_mode` to POST body
- [x] 4.7 Handle `deep_plan_started`, `deep_plan_delta`, `plan_self_review_completed` events in SSE stream consumer (app.js)
- [x] 4.8 Display deep plan summary and quality badges in monitoring panel after plan completion

## 5. Worker D: Testing

- [x] 5.1 Create `backend/tests/test_planning_layer.py` with unit tests for `validate_plan_text` (placeholder detection, structural checks, clean plan passes)
- [x] 5.2 Add `test_agent_graph.py` tests: `agent` mode event order unchanged; `plan` mode produces `deep_plan_started`/`deep_plan_delta`/`plan_self_review_completed` and skips tool execution
- [x] 5.3 Add `test_web_api.py` tests: POST run with `run_mode` default/agent/plan invalid; GET run returns metadata; 422 on invalid mode
- [x] 5.4 Add smoke test for `build_deep_plan_messages()` verifying prompt structure
- [x] 5.5 Add smoke test for `PlanQualityReport` serialization

## 6. Verification and Regression

- [x] 6.1 Run full test suite: `uv run --extra dev python -m pytest backend/tests/test_planning_layer.py backend/tests/test_agent_graph.py backend/tests/test_web_api.py -q`
- [x] 6.2 Run linter: `uv run --extra dev ruff check .`
- [x] 6.3 Manual smoke: start server, create session, send run with `"run_mode":"plan"`, verify events and response
- [x] 6.4 Verify `agent` mode still works identically to before (regression check)
