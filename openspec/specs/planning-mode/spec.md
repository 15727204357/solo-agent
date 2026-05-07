## ADDED Requirements

### Requirement: Forward planning mode

The system SHALL support a `run_mode` field on each agent run with values `"agent"` (default) and `"plan"`. In `"agent"` mode, the system SHALL execute the existing plan → context → inspect → tools → patch → response path. In `"plan"` mode, the system SHALL generate a deep implementation plan and stop without executing tools or proposing patches.

#### Scenario: Default mode is agent

- **WHEN** a run is created without specifying `run_mode`
- **THEN** the system SHALL execute in `"agent"` mode with full tool execution and patch proposal

#### Scenario: Plan mode skips tool execution

- **WHEN** a run is created with `run_mode: "plan"`
- **THEN** the system SHALL NOT select tools, execute tools, or propose verified patches

#### Scenario: Plan mode preserves memory and skills

- **WHEN** a run is created with `run_mode: "plan"` and memory is enabled
- **THEN** the system SHALL load builtin memory, prefetch session memory, and load relevant skills before generating the plan

#### Scenario: Run metadata persists run_mode

- **WHEN** a run is created with a specific `run_mode`
- **THEN** the `run_mode` SHALL be persisted in the run's metadata and retrievable via the API

### Requirement: Plan mode event stream

The system SHALL emit distinct events for `plan` mode runs that allow the frontend to render the deep plan progress. The event stream SHALL include `deep_plan_started`, `deep_plan_delta`, `plan_self_review_completed`, and `plan_completed`.

#### Scenario: Plan mode events in sequence

- **WHEN** a `plan` mode run executes
- **THEN** events SHALL be emitted in order: `deep_plan_started` → one or more `deep_plan_delta` → `plan_self_review_completed` → `plan_completed` → `run_completed`

#### Scenario: Agent mode events unchanged

- **WHEN** an `agent` mode run executes
- **THEN** the event stream SHALL match the existing event sequence including `plan_started`, `plan_delta`, `plan_completed`, and all tool/patch events

### Requirement: Plan mode response content

In `plan` mode, the final response SHALL contain the generated deep plan text and self-review results. The `plan` field in state SHALL hold the validated deep plan.

#### Scenario: Deep plan in response

- **WHEN** a `plan` mode run completes
- **THEN** `state.plan` SHALL contain the deep plan text and `state.response` SHALL include the plan with self-review notes
