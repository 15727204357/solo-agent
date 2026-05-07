## Context

Solo Agent currently uses a linear agent graph (Milestone 1): plan → context → inspect → tools → response. The `_plan_node` generates a short lightweight plan (3-6 steps, max 500 tokens). Users have requested a dedicated planning mode that produces Superpowers-style deep implementation plans for handoff to zero-context engineers or parallel agents. The `plan` mode must coexist with the existing `agent` mode without altering `agent` mode behavior.

Key constraints:
- `plan` mode is read-only: generates a plan and stops, no tool execution, no code modification
- Deep plans must follow Superpowers `writing-plans` conventions: no placeholders (TBD, TODO, "implement later"), zero-context-reader coverage, inline self-review, execution options
- Chinese comments/docs, English system/user prompts (per project convention)
- Backward compatible: `agent` mode unchanged; existing events and UI still work

Existing architecture:
- `AgentState` (dataclass): carries all run state
- `AgentSettings` (frozen dataclass): typed settings
- `AgentDeps`: dependency injection container
- `run_agent_events()` in `graph.py`: streaming async generator that yields `AgentEvent`
- `_run_graph()`: orchestrates stages as coroutines
- `prompts.py`: `PLANNER_SYSTEM_PROMPT`, `planner_user_prompt()` — lightweight plan
- `routes.py`: `CreateRunRequest` with `prompt`, `memory_enabled`, `conversation_history_enabled`
- `runner.py`: `AgentRunner.run()` bridges Web API to agent graph
- Frontend: Jinja2 templates with `app.js` for interactivity
- Events are stored via `RunEvent` in SQLite and streamed as SSE

## Goals / Non-Goals

**Goals:**
- Add `run_mode: Literal["agent", "plan"]` to `AgentSettings` (default `"agent"`) and `CreateRunRequest`
- Create `planning.py` module with `build_deep_plan_messages()`, `validate_plan_text()`, `PlanQualityReport`
- Branch `_run_graph()`: `agent` mode follows existing path; `plan` mode follows memory/skills → context guard → deep plan → self-review → response/persist
- Deep plan prompt enforces: file map, TDD steps, commands, expected results, execution options, self-review pass
- Plan validation rejects placeholder patterns (TBD, TODO, "类似上一步", "适当处理", "implement later", etc.)
- New streaming events: `deep_plan_started`, `deep_plan_delta`, `plan_self_review_completed`
- Frontend: Agent/Plan segmented toggle in composer, mode included in POST payload
- Unit tests for planning module, graph branching, API validation

**Non-Goals:**
- Changing `agent` mode behavior (planning, tool execution, patch proposals, etc.)
- Real-time web search integration at runtime (web search is development-phase only, for capturing Superpowers conventions)
- Auto-execution of plans in `plan` mode
- Plan template storage or plan versioning
- Multi-turn plan iteration within a single `plan` mode run
- Modifying `BehaviorPolicy`, verified editing, context compression logic

## Decisions

### D1: `run_mode` on `AgentSettings` stringly-typed vs. Literal

Using `str` with validation (default `"agent"`) rather than an enum. The existing codebase uses plain strings for settings fields (`provider: str`, `model: str | None`). A `Literal` type is used at the API boundary (`CreateRunRequest.run_mode`) where Pydantic provides validation; the internal `AgentSettings` uses `str = "agent"` for simplicity. Invalid modes are caught at the API layer (422 response), not in `AgentSettings`.

### D2: Graph branching: conditional path vs. separate graph

A single graph with conditional path (`if state.run_mode == "plan": yield from _plan_mode_path(...)`) rather than two separate compiled graphs. This keeps session/run lifecycle identical (persistence, events, error handling) and avoids duplicating the outer `run_agent_events()` wrapper. The `plan` mode path simply skips the tool-execution and patch-proposal stages.

### D3: Deep plan implementation: streaming LLM call + post-hoc validation

The deep plan is generated via a single streaming LLM call (like the existing `_plan_node`) with a specialized system prompt, followed by `validate_plan_text()` for deterministic quality checks. This avoids the complexity of multi-step plan refinement within a single run. Self-review is a second LLM call asking the model to check its own plan against the quality rules, emitted as a separate event.

### D4: Plan validation: regex-based rules vs. LLM-based

Regex-based deterministic rules for placeholder detection (TBD, TODO, "implement later", "类似上一步", "适当处理", etc.) and structural checks (file map present, self-review section present). This is fast, reliable, and does not consume LLM tokens. Specific Chinese and English placeholder patterns are defined in a configurable list.

### D5: Event structure: new events vs. reusing existing

New event types (`deep_plan_started`, `deep_plan_delta`, `plan_self_review_completed`) for the `plan` mode path. The existing `plan_completed` continues to fire for `agent` mode. This allows the frontend to distinguish between lightweight and deep plans without breaking existing event consumers. The `plan` mode also emits `plan_completed` as a final snapshot with the full validated plan text, for backward compatibility with any existing UI that listens for it.

### D6: Frontend: segmented control vs. dropdown

A segmented button control (Agent | Plan) in the composer area, matching the existing minimal UI style. The selection is included in the POST `/api/sessions/{id}/runs` body as `run_mode`. Default is `"agent"` to preserve current behavior.

## Risks / Trade-offs

- **[R1] LLM quality variance for deep plans**: Some models may produce placeholder-laden plans despite the prompt. → `validate_plan_text()` provides a deterministic safety net; `plan_self_review_completed` event surfaces quality flags to the UI.
- **[R2] `plan` mode token cost**: Deep plans use higher `plan_max_tokens` (increased from 500 to 2000 for plan mode), consuming more API budget. → Configurable via `plan_deep_max_tokens` setting; users aware that `plan` mode is a deliberate deep-planning step.
- **[R3] Frontend complexity creep**: Adding a mode selector adds surface area. → Minimal segmented control (2 options, no sub-options), styled to match existing UI.
- **[R4] Graph complexity**: One graph with two paths increases cyclomatic complexity in `_run_graph()`. → The `plan` mode path is a single `async for` block delegating to helper coroutines (`_deep_plan_stage`, `_plan_self_review_stage`); the `agent` mode path is the existing code unchanged.

## Open Questions

- Should `plan` mode also support memory search injection? (Current design: yes, memory/skills load identically for both modes before branching.)
- Should plan validation errors block the run or just warn? (Current design: validation failures are reported as warnings in `plan_self_review_completed`; the plan is still surfaced. This matches Superpowers' "self-review" philosophy.)
