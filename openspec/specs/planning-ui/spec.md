## ADDED Requirements

### Requirement: Mode selector in composer

The frontend composer area SHALL include a segmented control with two options: "Agent" (default) and "Plan". The selected mode SHALL be submitted with the run payload.

#### Scenario: Default selection is Agent

- **WHEN** the page loads with no prior selection
- **THEN** the "Agent" segment SHALL be visually active and the hidden `run_mode` value SHALL be `"agent"`

#### Scenario: User selects Plan mode

- **WHEN** the user clicks the "Plan" segment
- **THEN** the "Plan" segment SHALL become visually active, "Agent" SHALL become inactive, and the hidden `run_mode` value SHALL be `"plan"`

#### Scenario: Mode included in POST payload

- **WHEN** the user submits a prompt with "Plan" mode selected
- **THEN** the POST body to `/api/sessions/{id}/runs` SHALL include `"run_mode": "plan"`

### Requirement: Mode toggle visual style

The segmented control SHALL match the existing minimal CSS style (light borders, smooth transitions, consistent with the `.primary-button` and `.secondary-button` classes).

#### Scenario: Visual consistency

- **WHEN** the mode selector is rendered
- **THEN** it SHALL use the same color palette, border radius, and font as the existing UI controls

### Requirement: Plan summary display in monitoring panel

When a `plan` mode run completes, the monitoring panel SHALL display a summary of the deep plan including the plan text and any self-review quality issues.

#### Scenario: Deep plan events update monitoring panel

- **WHEN** `deep_plan_delta` events are received via SSE
- **THEN** the monitoring panel SHALL accumulate and display the streaming plan text

#### Scenario: Self-review badge displayed

- **WHEN** `plan_self_review_completed` event is received with quality issues
- **THEN** the monitoring panel SHALL display a warning badge indicating that the plan has quality concerns
