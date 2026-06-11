from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from solo_agent.skill_recipes import RecipePolicy, parse_structured_recipe_text, recipe_from_payload
from solo_agent.tools import create_default_registry
from solo_agent.tools.readonly import _metadata_hermes, _parse_frontmatter, _skill_contract, _skill_summary

IssueSeverity = Literal["info", "warning", "error"]


class SkillCoverageIssue(BaseModel):
    kind: str
    severity: IssueSeverity
    message: str
    skill_name: str | None = None
    path: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class SkillCoverageItem(BaseModel):
    name: str
    path: str
    category: str = "general"
    required_tools: list[str] = Field(default_factory=list)
    declared_recipes: list[str] = Field(default_factory=list)
    declared_scripts: list[str] = Field(default_factory=list)
    contract_complete: bool = False
    issues: list[SkillCoverageIssue] = Field(default_factory=list)


class SkillScenarioCoverage(BaseModel):
    id: str
    expected_skill: str
    expected_recipes: list[str] = Field(default_factory=list)
    required_tool_categories: list[str] = Field(default_factory=list)
    verification: str = ""
    status: Literal["covered", "partial", "missing"] = "missing"
    missing_recipes: list[str] = Field(default_factory=list)
    missing_tool_categories: list[str] = Field(default_factory=list)


class SkillCoverageReport(BaseModel):
    skills: list[SkillCoverageItem] = Field(default_factory=list)
    scenarios: list[SkillScenarioCoverage] = Field(default_factory=list)
    issues: list[SkillCoverageIssue] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class SkillQualityError(AssertionError):
    def __init__(self, blockers: Sequence[SkillCoverageIssue], report: SkillCoverageReport) -> None:
        self.blockers = list(blockers)
        self.report = report
        preview = "; ".join(issue.message for issue in self.blockers[:5])
        if len(self.blockers) > 5:
            preview = f"{preview}; ... ({len(self.blockers)} blockers total)"
        super().__init__(f"Skill quality gate failed: {preview}")


QUALITY_BLOCKING_ISSUE_KINDS = {
    "orphan_recipe_file",
    "unknown_required_tool",
    "unknown_recipe_tool",
}


DEFAULT_SKILL_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "id": "python-backend-change",
        "expected_skill": "python-backend-change",
        "expected_recipes": ["inspect", "focused-test", "verify"],
        "required_tool_categories": ["context", "edit", "quality"],
        "verification": "focused pytest plus ruff when backend behavior changes",
    },
    {
        "id": "debug-test-failure",
        "expected_skill": "debug-test-failure",
        "expected_recipes": ["failure-triage"],
        "required_tool_categories": ["context", "edit", "quality"],
        "verification": "original failing check no longer fails",
    },
    {
        "id": "code-review",
        "expected_skill": "code-review",
        "expected_recipes": ["review-context"],
        "required_tool_categories": ["context", "quality"],
        "verification": "findings first with residual risk stated",
    },
    {
        "id": "hash-anchored-editing",
        "expected_skill": "hash-anchored-editing",
        "expected_recipes": ["manual-hash-edit"],
        "required_tool_categories": ["edit"],
        "verification": "hash prepare, preview, and apply boundary exists",
    },
    {
        "id": "tool-use-discipline",
        "expected_skill": "tool-use-discipline",
        "expected_recipes": ["bounded-context-gathering"],
        "required_tool_categories": ["context", "quality"],
        "verification": "bounded context and quality checks are represented as reusable recipe steps",
    },
)


def audit_skill_coverage(
    workspace_root: str | Path,
    *,
    registry: Any | None = None,
    scenarios: Sequence[Mapping[str, Any]] = DEFAULT_SKILL_SCENARIOS,
) -> SkillCoverageReport:
    root = Path(workspace_root).resolve()
    skills_root = root / "skills"
    registry = registry or create_default_registry(root)
    tool_index = _tool_index(registry)

    items: list[SkillCoverageItem] = []
    issues: list[SkillCoverageIssue] = []
    for skill_file in sorted(skills_root.rglob("SKILL.md")) if skills_root.exists() else []:
        item = _audit_skill(skill_file, root, tool_index)
        items.append(item)
        issues.extend(item.issues)

    scenario_results = [_audit_scenario(scenario, items, tool_index) for scenario in scenarios]
    for scenario in scenario_results:
        if scenario.status != "covered":
            issues.append(
                SkillCoverageIssue(
                    kind="scenario_not_covered",
                    severity="warning",
                    message=f"Scenario {scenario.id} is {scenario.status}",
                    skill_name=scenario.expected_skill,
                    details=scenario.model_dump(mode="json"),
                )
            )

    summary = {
        "skill_count": len(items),
        "scenario_count": len(scenario_results),
        "covered_scenario_count": sum(1 for scenario in scenario_results if scenario.status == "covered"),
        "issue_count": len(issues),
        "error_count": sum(1 for issue in issues if issue.severity == "error"),
        "warning_count": sum(1 for issue in issues if issue.severity == "warning"),
    }
    return SkillCoverageReport(skills=items, scenarios=scenario_results, issues=issues, summary=summary)


def assert_skill_quality(
    workspace_root: str | Path,
    *,
    registry: Any | None = None,
    scenarios: Sequence[Mapping[str, Any]] = DEFAULT_SKILL_SCENARIOS,
) -> SkillCoverageReport:
    report = audit_skill_coverage(workspace_root, registry=registry, scenarios=scenarios)
    blockers = _quality_blockers(report)
    if blockers:
        raise SkillQualityError(blockers, report)
    return report


def _audit_skill(skill_file: Path, root: Path, tool_index: Mapping[str, Mapping[str, Any]]) -> SkillCoverageItem:
    text = skill_file.read_text(encoding="utf-8", errors="replace")
    summary = _skill_summary(skill_file, text, root)
    frontmatter = _parse_frontmatter(text)
    hermes = _metadata_hermes(frontmatter)
    contract = _normalized_contract(text)
    issues: list[SkillCoverageIssue] = []
    name = str(summary.get("name") or skill_file.parent.name)
    relative_path = skill_file.relative_to(root).as_posix()

    required_tools = _string_list(summary.get("required_tools"))
    for field_name in ("required_tools", "tool_strategy", "acceptance_criteria", "failure_recovery"):
        value = required_tools if field_name == "required_tools" else contract.get(field_name) or []
        if not value:
            issues.append(
                _issue(
                    "missing_contract_field",
                    "warning",
                    f"Skill is missing {field_name}",
                    name,
                    relative_path,
                    field=field_name,
                )
            )

    for tool_name in required_tools:
        if tool_name not in tool_index:
            issues.append(_issue("unknown_required_tool", "error", f"Unknown required tool: {tool_name}", name, relative_path))

    recipe_ids, recipe_issues = _audit_recipes(skill_file, summary, frontmatter, hermes, tool_index, root)
    issues.extend(recipe_issues)
    if not recipe_ids:
        issues.append(_issue("missing_recipes", "warning", "Skill has no declared recipes", name, relative_path))

    script_ids = [str(script.get("id")) for script in contract.get("scripts") or [] if script.get("id")]
    issues.extend(_script_issues(skill_file, name, relative_path, contract))

    contract_complete = not any(
        issue.kind in {"missing_contract_field", "missing_recipes"} and issue.severity in {"warning", "error"}
        for issue in issues
    )
    return SkillCoverageItem(
        name=name,
        path=relative_path,
        category=str(summary.get("category") or "general"),
        required_tools=required_tools,
        declared_recipes=recipe_ids,
        declared_scripts=script_ids,
        contract_complete=contract_complete,
        issues=issues,
    )


def _quality_blockers(report: SkillCoverageReport) -> list[SkillCoverageIssue]:
    blockers: list[SkillCoverageIssue] = []
    seen: set[tuple[str, str | None, str | None, str]] = set()

    def append(issue: SkillCoverageIssue) -> None:
        key = (issue.kind, issue.skill_name, issue.path, issue.message)
        if key in seen:
            return
        seen.add(key)
        blockers.append(issue)

    for issue in report.issues:
        if issue.severity == "error" or issue.kind in QUALITY_BLOCKING_ISSUE_KINDS:
            append(issue)

    for scenario in report.scenarios:
        if scenario.status == "covered":
            continue
        append(
            SkillCoverageIssue(
                kind="scenario_not_covered",
                severity="error",
                message=f"Core scenario {scenario.id} is {scenario.status}",
                skill_name=scenario.expected_skill,
                details=scenario.model_dump(mode="json"),
            )
        )
    return blockers


def _audit_recipes(
    skill_file: Path,
    skill_summary: Mapping[str, Any],
    frontmatter: Mapping[str, Any],
    hermes: Mapping[str, Any],
    tool_index: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> tuple[list[str], list[SkillCoverageIssue]]:
    name = str(skill_summary.get("name") or skill_file.parent.name)
    relative_path = skill_file.relative_to(root).as_posix()
    raw_recipes = hermes.get("recipes") or frontmatter.get("recipes") or []
    if isinstance(raw_recipes, Mapping):
        raw_recipes = [raw_recipes]
    if not isinstance(raw_recipes, list):
        return [], [_issue("invalid_recipe_metadata", "error", "Recipe metadata must be a list", name, relative_path)]

    declared_files: set[str] = set()
    recipe_ids: list[str] = []
    issues: list[SkillCoverageIssue] = []
    for raw_recipe in raw_recipes:
        if not isinstance(raw_recipe, Mapping):
            issues.append(
                _issue(
                    "invalid_recipe_metadata",
                    "error",
                    "Recipe metadata entry must be an object",
                    name,
                    relative_path,
                )
            )
            continue
        source_file = str(raw_recipe.get("file") or "SKILL.md")
        recipe_id = str(raw_recipe.get("id") or "")
        if source_file != "SKILL.md":
            declared_files.add(source_file.replace("\\", "/"))
        payload: Mapping[str, Any] = raw_recipe
        if source_file != "SKILL.md":
            recipe_path = skill_file.parent / source_file
            if not recipe_path.is_file():
                issues.append(
                    _issue(
                        "missing_recipe_file",
                        "error",
                        f"Declared recipe file is missing: {source_file}",
                        name,
                        relative_path,
                        recipe_id=recipe_id,
                    )
                )
                continue
            try:
                parsed = parse_structured_recipe_text(recipe_path.read_text(encoding="utf-8", errors="replace"))
                if not isinstance(parsed, Mapping):
                    raise ValueError("Recipe file must contain an object")
                payload = {**dict(raw_recipe), **dict(parsed), "file": source_file}
            except Exception as exc:
                issues.append(
                    _issue(
                        "recipe_schema_invalid",
                        "error",
                        f"Recipe file is invalid: {source_file}",
                        name,
                        relative_path,
                        recipe_id=recipe_id,
                        error=str(exc),
                    )
                )
                continue
        try:
            recipe = recipe_from_payload(payload, skill=skill_summary, source_file=source_file)
        except Exception as exc:
            issues.append(
                _issue(
                    "recipe_schema_invalid",
                    "error",
                    f"Recipe schema is invalid: {recipe_id or source_file}",
                    name,
                    relative_path,
                    recipe_id=recipe_id,
                    error=str(exc),
                )
            )
            continue
        recipe_ids.append(recipe.id)
        for step in recipe.steps:
            if step.tool not in tool_index:
                issues.append(
                    _issue(
                        "unknown_recipe_tool",
                        "error",
                        f"Recipe {recipe.id} references unknown tool: {step.tool}",
                        name,
                        relative_path,
                        recipe_id=recipe.id,
                        step_id=step.id,
                    )
                )
            policy = RecipePolicy.step_auto_executable(step)
            if step.run_policy == "auto" and not policy["auto_executable"]:
                issues.append(
                    _issue(
                        "auto_recipe_step_blocked",
                        "error",
                        f"Recipe {recipe.id} has an auto step that policy blocks",
                        name,
                        relative_path,
                        recipe_id=recipe.id,
                        step_id=step.id,
                        reason=policy["reason"],
                    )
                )

    recipes_dir = skill_file.parent / "references" / "recipes"
    if recipes_dir.is_dir():
        for recipe_file in sorted(recipes_dir.glob("*")):
            if recipe_file.suffix.casefold() not in {".yaml", ".yml", ".json"}:
                continue
            relative = recipe_file.relative_to(skill_file.parent).as_posix()
            if relative not in declared_files:
                issues.append(
                    _issue("orphan_recipe_file", "warning", f"Recipe file is not declared: {relative}", name, relative_path)
                )
    return recipe_ids, issues


def _audit_scenario(
    scenario: Mapping[str, Any],
    skills: Sequence[SkillCoverageItem],
    tool_index: Mapping[str, Mapping[str, Any]],
) -> SkillScenarioCoverage:
    expected_skill = str(scenario.get("expected_skill") or "")
    expected_recipes = _string_list(scenario.get("expected_recipes"))
    required_categories = _string_list(scenario.get("required_tool_categories"))
    skill = next((item for item in skills if item.name == expected_skill), None)
    if skill is None:
        return SkillScenarioCoverage(
            id=str(scenario.get("id") or expected_skill),
            expected_skill=expected_skill,
            expected_recipes=expected_recipes,
            required_tool_categories=required_categories,
            verification=str(scenario.get("verification") or ""),
            status="missing",
            missing_recipes=expected_recipes,
            missing_tool_categories=required_categories,
        )

    missing_recipes = [recipe for recipe in expected_recipes if recipe not in skill.declared_recipes]
    available_categories = {
        str(tool_index.get(tool_name, {}).get("category") or "")
        for tool_name in skill.required_tools
        if tool_name in tool_index
    }
    missing_categories = [category for category in required_categories if category not in available_categories]
    status: Literal["covered", "partial", "missing"] = "covered"
    if missing_recipes or missing_categories or not skill.contract_complete:
        status = "partial"
    return SkillScenarioCoverage(
        id=str(scenario.get("id") or expected_skill),
        expected_skill=expected_skill,
        expected_recipes=expected_recipes,
        required_tool_categories=required_categories,
        verification=str(scenario.get("verification") or ""),
        status=status,
        missing_recipes=missing_recipes,
        missing_tool_categories=missing_categories,
    )


def _normalized_contract(text: str) -> dict[str, Any]:
    contract = dict(_skill_contract(text))
    if not contract.get("tool_strategy"):
        contract["tool_strategy"] = _extract_markdown_list_section(text, "Tool Protocol")
    return contract


def _script_issues(
    skill_file: Path,
    skill_name: str,
    relative_path: str,
    contract: Mapping[str, Any],
) -> list[SkillCoverageIssue]:
    scripts_dir = skill_file.parent / "scripts"
    if not scripts_dir.is_dir():
        return []
    declared = {str(script.get("file") or "").replace("\\", "/") for script in contract.get("scripts") or []}
    issues: list[SkillCoverageIssue] = []
    for script_file in sorted(scripts_dir.glob("*.py")):
        relative = script_file.relative_to(skill_file.parent).as_posix()
        if relative not in declared:
            issues.append(
                _issue(
                    "undeclared_script_file",
                    "warning",
                    f"Script file is not declared: {relative}",
                    skill_name,
                    relative_path,
                )
            )
    return issues


def _tool_index(registry: Any) -> dict[str, dict[str, Any]]:
    try:
        tools = registry.list_tools(visibility="all")
    except TypeError:
        tools = registry.list_tools()
    return {str(tool.get("name") or ""): dict(tool) for tool in tools if isinstance(tool, Mapping)}


def _extract_markdown_list_section(text: str, heading: str) -> list[str]:
    import re

    heading_re = re.compile(rf"^##+\s+{re.escape(heading)}\s*$", re.IGNORECASE)
    items: list[str] = []
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if heading_re.match(stripped):
            in_section = True
            continue
        if in_section and stripped.startswith("#"):
            break
        if not in_section:
            continue
        if stripped.startswith(("- ", "* ")):
            items.append(stripped[2:].strip())
        elif re.match(r"^\d+\.\s+", stripped):
            items.append(re.sub(r"^\d+\.\s+", "", stripped).strip())
    return items


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value)]


def _issue(
    kind: str,
    severity: IssueSeverity,
    message: str,
    skill_name: str,
    path: str,
    **details: Any,
) -> SkillCoverageIssue:
    return SkillCoverageIssue(
        kind=kind,
        severity=severity,
        message=message,
        skill_name=skill_name,
        path=path,
        details={key: value for key, value in details.items() if value not in (None, "")},
    )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit coding-agent skill coverage and quality.")
    parser.add_argument("workspace_root", nargs="?", default=".", help="Workspace root containing skills/.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when the quality gate has blockers.")
    args = parser.parse_args(argv)

    report = audit_skill_coverage(args.workspace_root)
    blockers = _quality_blockers(report)
    payload = report.to_public_dict()
    payload["quality_gate"] = {
        "passed": not blockers,
        "blocker_count": len(blockers),
        "blockers": [issue.model_dump(mode="json") for issue in blockers],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        summary = payload["summary"]
        status = "passed" if not blockers else "failed"
        print(
            f"Skill quality gate {status}: "
            f"{summary['covered_scenario_count']}/{summary['scenario_count']} scenarios covered, "
            f"{summary['error_count']} errors, {summary['warning_count']} warnings."
        )
        for issue in blockers[:10]:
            location = f" ({issue.skill_name})" if issue.skill_name else ""
            print(f"- [{issue.kind}]{location} {issue.message}")
    return 1 if args.strict and blockers else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
