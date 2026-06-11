# Coding Agent Tool Ecosystem

Solo Agent's tool ecosystem is intentionally coding-agent-focused. It is not a
general DeerFlow-style marketplace, browser automation suite, data-analysis
platform, or SaaS connector hub. The supported surface is the minimum needed to
inspect, edit, verify, and review real codebases with auditable safety
boundaries.

## Tool Tiers

- Context: workspace snapshots, file discovery, bounded reads, code search, and
  Python symbol inspection.
- Editing: hash-anchored prepare/preview/apply operations plus guarded
  create/move/delete filesystem tools.
- Quality: structured no-shell programming commands for tests, lint, build,
  typecheck, and formatting checks.
- Git read: status, diff, show, and recent log inspection only.
- Skill orchestration: compact skill index, full `skill_view`, declarative
  recipe preview/run, and declared skill scripts.
- Subagents: optional scoped read-only subtask execution when the graph's
  parallelism gate proves independence.

High-risk actions such as edits, deletion, moving files, skill changes,
install/publish/deploy commands, and write-like scripts remain proposal or
approval gated.

## Skill Contract

`SKILL.md` files may expose this contract through frontmatter, Hermes metadata,
or conventional Markdown sections:

- `required_tools`: tools the skill expects the agent to use.
- `tool_strategy`: how tools should be selected and ordered.
- `acceptance_criteria`: the verification bar for successful use.
- `failure_recovery`: when to stop, recover, or ask for help.
- `metadata.hermes.recipes`: declarative recipes for common workflows.
- `metadata.hermes.scripts`: declared scripts for repeatable mechanical checks.

`skills_list` returns only compact routing metadata. `skill_view` returns the
full contract, sanitized content, and available support files.

## Hybrid Orchestration

The workflow injects a compact skill index by default. Explicit
`/skill <name-or-slug>` requests are resolved during the skill context stage,
before planning, so the planner can see the full skill SOP and recipe index.
Non-explicit matches stay compact until the model or workflow has a reason to
load more.

Recipes and scripts never replace the model's feedback loop. They produce
bounded tool results, which are fed back into the run for the model to interpret
against the current codebase.

## Local Environment Check

Repository-level command wrappers such as `rtk` are treated as local developer
environment requirements, not runtime tool capabilities. Before relying on such
wrappers, verify that they are available on `PATH`; if they are missing, use the
same bounded no-shell tool policy and report the limitation in verification
notes.

## Recipe And Script Boundary

Declarative recipes may automatically run read/search/git-read/test/build/lint
and check steps. Manual or high-risk recipe steps are blocked and reported as
manual work.

Skill scripts are stricter:

- They must be declared by the skill metadata.
- The script file must stay inside the skill's `scripts/` directory.
- Only Python scripts are supported.
- Invocation uses structured argv without a shell.
- Only `auto` scripts with `low` or `medium-safe` risk and read/quality/check
  kind are executable.
- Secret-like, shell-like, network-like, and write-like arguments are rejected.

This keeps mechanical workflow helpers useful without creating a hidden general
execution channel.

## Skill Evolution

Skill evolution is a postlude analysis loop, not an automatic self-modifying
system. After the model has produced its final response, the workflow can scan
the completed run for deterministic evidence that a skill should be improved:
blocked recipe steps, successful safe tool sequences, or clear quality-check
recovery patterns.

When confidence passes the configured threshold, the postlude creates one
pending `SkillChangeProposal` at most. The proposal reuses the existing
approval API and passes a promotion gate before it is stored. A promoted recipe
must parse through the existing recipe schema and policy, and the proposal must
include both the new `references/recipes/` file and a safe metadata patch that
declares it in the target `SKILL.md`. If metadata cannot be patched safely, the
candidate remains a snapshot instead of becoming a pending proposal.

Skill evolution does not add executable scripts, read secrets, summarize `.env`
content, or apply changes without user approval.

Low-confidence findings stay as run snapshots. Approved proposals are the only
path from observed execution experience back into `skills/`.

## Coverage Governance

Skill coverage is checked through a development-only audit loop. It scans
workspace skills, validates contract fields, recipe declarations, recipe schema,
recipe policy, required tools, orphan recipes, and undeclared scripts. The first
coverage matrix is intentionally coding-agent-focused: Python backend changes,
test-failure debugging, code review, hash-anchored editing, and bounded tool use.

The coverage report is not a product UI and does not add a new runtime approval
path. It is a regression harness for answering whether common coding-agent
workflows are actually represented by skills and recipes, or whether the model
would still need to rediscover the same procedure from scratch.
