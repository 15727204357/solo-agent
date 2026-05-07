## ADDED Requirements

### Requirement: Placeholder detection

The system SHALL validate plan text against a list of forbidden placeholder patterns. Validation SHALL flag any occurrence as a quality issue.

#### Scenario: English placeholders detected

- **WHEN** `validate_plan_text()` is called with text containing "TBD", "TODO", or "implement later"
- **THEN** the result SHALL include a quality issue for each occurrence with the matching pattern

#### Scenario: Chinese placeholders detected

- **WHEN** `validate_plan_text()` is called with text containing "类似上一步", "适当处理", "待定", "略", "同理", or "同上"
- **THEN** the result SHALL include a quality issue for each occurrence

#### Scenario: Clean plan passes

- **WHEN** `validate_plan_text()` is called with a plan that contains no placeholder patterns
- **THEN** the result SHALL have `issues` length zero and `passed` SHALL be True

### Requirement: Structural completeness check

The system SHALL verify that a deep plan includes a file map section, a self-review section, and at least one numbered execution step.

#### Scenario: Missing file map

- **WHEN** `validate_plan_text()` is called with a plan that has no file map section (no `## File Map` or similar heading)
- **THEN** the result SHALL include a quality issue with type `missing_file_map`

#### Scenario: Missing self-review

- **WHEN** `validate_plan_text()` is called with a plan that has no self-review section (no `## Self-Review` or similar heading)
- **THEN** the result SHALL include a quality issue with type `missing_self_review`

#### Scenario: No execution steps

- **WHEN** `validate_plan_text()` is called with a plan that has no numbered steps (no lines matching `## Steps` followed by numbered items)
- **THEN** the result SHALL include a quality issue with type `missing_steps`

#### Scenario: Complete plan passes structure check

- **WHEN** `validate_plan_text()` is called with a plan containing file map, steps, and self-review sections
- **THEN** the result SHALL have no structural issues

### Requirement: Plan quality report

The system SHALL produce a `PlanQualityReport` with a `passed` boolean, a list of `issues` (each with `type`, `pattern`, `location`), and a `summary` string suitable for display.

#### Scenario: Quality report for clean plan

- **WHEN** a clean deep plan is validated
- **THEN** `PlanQualityReport.passed` SHALL be `True` and `PlanQualityReport.summary` SHALL indicate "All checks passed"

#### Scenario: Quality report for flawed plan

- **WHEN** a plan with placeholders and missing sections is validated
- **THEN** `PlanQualityReport.passed` SHALL be `False` and each issue SHALL have a `type` and descriptive `message`
