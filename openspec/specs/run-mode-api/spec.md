## ADDED Requirements

### Requirement: CreateRunRequest accepts run_mode

The `CreateRunRequest` Pydantic model SHALL accept an optional `run_mode` field with values `"agent"` (default) or `"plan"`.

#### Scenario: Default run_mode

- **WHEN** a POST to `/api/sessions/{id}/runs` is made without `run_mode` in the body
- **THEN** the run SHALL execute in `"agent"` mode

#### Scenario: Explicit plan mode

- **WHEN** a POST to `/api/sessions/{id}/runs` includes `"run_mode": "plan"` in the body
- **THEN** the run SHALL execute in `"plan"` mode

#### Scenario: Invalid run_mode rejected

- **WHEN** a POST to `/api/sessions/{id}/runs` includes `"run_mode": "invalid"` in the body
- **THEN** the API SHALL respond with HTTP 422 Unprocessable Entity

### Requirement: Run metadata includes run_mode

The `run_mode` SHALL be persisted in the run's metadata and returned in API responses for run details.

#### Scenario: run_mode in run metadata

- **WHEN** a run is created with `run_mode: "plan"`
- **THEN** the `RunRecord.metadata` SHALL include `{"run_mode": "plan"}`

#### Scenario: run_mode visible in GET run

- **WHEN** a GET request retrieves a run that was created with `run_mode: "plan"`
- **THEN** the response's `metadata` SHALL include `"run_mode": "plan"`

### Requirement: Runner passes run_mode to AgentSettings

The `AgentRunner` SHALL read `run_mode` from the run metadata and pass it to `AgentSettings`.

#### Scenario: run_mode propagated to agent settings

- **WHEN** `AgentRunner.run()` processes a run with metadata `{"run_mode": "plan"}`
- **THEN** the `AgentSettings` SHALL be constructed with `run_mode="plan"`

### Requirement: AgentSettings supports run_mode

The `AgentSettings` dataclass SHALL include a `run_mode` field of type `str` with default `"agent"`.

#### Scenario: Default AgentSettings run_mode

- **WHEN** `AgentSettings()` is instantiated without arguments
- **THEN** `run_mode` SHALL be `"agent"`
