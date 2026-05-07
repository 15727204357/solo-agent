## ADDED Requirements

### Requirement: Deep plan system prompt

The system SHALL include a deep plan system prompt when `run_mode` is `"plan"`. The prompt SHALL explicitly instruct the model to produce a plan with no placeholders, zero-context-reader coverage, inline self-reflection, and explicit execution options.

#### Scenario: Prompt forbids placeholders

- **WHEN** the deep plan system prompt is constructed
- **THEN** it SHALL include instructions that prohibit "TBD", "TODO", "implement later", and equivalent placeholder expressions in both English and Chinese

#### Scenario: Prompt requires file map

- **WHEN** the deep plan system prompt is constructed
- **THEN** it SHALL require the plan to include a file map listing every file that will be created or modified with its intended purpose

#### Scenario: Prompt requires TDD steps

- **WHEN** the deep plan system prompt is constructed
- **THEN** it SHALL require numbered steps with precise commands to run, exact expected outputs, and pass/fail criteria for each step

#### Scenario: Prompt requires execution options

- **WHEN** the deep plan system prompt is constructed
- **THEN** it SHALL instruct the model to list execution options (e.g., single agent, parallel agents, subagent-driven) with a recommended choice

#### Scenario: Prompt requires self-review

- **WHEN** the deep plan system prompt is constructed
- **THEN** it SHALL instruct the model to include a self-review section before presenting the final plan

### Requirement: Deep plan user prompt

The system SHALL construct a user prompt for `plan` mode that includes memory context, skill context, the user's task, and Superpowers-style planning constraints.

#### Scenario: User prompt includes task context

- **WHEN** `build_deep_plan_messages()` is called
- **THEN** the user message SHALL include the user input, conversation history summary, memory context block, and loaded skill context

#### Scenario: User prompt references Superpowers conventions

- **WHEN** `build_deep_plan_messages()` is called
- **THEN** the user message SHALL include specific instructions to follow `writing-plans` conventions: file map, TDD steps, exact commands, expected results, self-review

### Requirement: Lightweight planner prompt preserved

The existing `PLANNER_SYSTEM_PROMPT` and `planner_user_prompt()` SHALL remain unchanged and SHALL be used for `agent` mode runs.

#### Scenario: Agent mode uses lightweight planner

- **WHEN** a run executes in `agent` mode
- **THEN** the `_plan_node` SHALL use `PLANNER_SYSTEM_PROMPT` and `planner_user_prompt()` with `plan_max_tokens` budget
