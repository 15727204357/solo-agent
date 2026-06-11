"""Schema validation for workspace SKILL.md contracts."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from solo_agent.skill_recipes import RecipePolicy, parse_structured_recipe_text, recipe_from_payload

SKILL_SCHEMA_VERSION = "v1"
SKILL_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9 _-]{0,120}$")


@dataclass(frozen=True)
class SkillSchemaIssue:
    severity: str
    kind: str
    message: str
    path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkillSchemaV1:
    name: str
    description: str = ""
    category: str = "workflow"
    version: str = "0.1.0"
    enabled: bool = True
    required_tools: list[str] = field(default_factory=list)
    recipes: list[dict[str, Any]] = field(default_factory=list)
    scripts: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_skill_file(cls, skill_file: str | Path) -> SkillSchemaV1:
        path = Path(skill_file)
        frontmatter = _parse_frontmatter(path.read_text(encoding="utf-8", errors="replace"))
        metadata = _as_mapping(frontmatter.get("metadata"))
        hermes = _as_mapping(metadata.get("hermes"))
        enabled_raw = frontmatter.get("enabled", hermes.get("enabled", True))
        status = str(frontmatter.get("status", hermes.get("status", "enabled"))).casefold()
        return cls(
            name=str(frontmatter.get("name") or path.parent.name).strip(),
            description=str(frontmatter.get("description") or "").strip(),
            category=str(frontmatter.get("category") or path.parent.parent.name or "workflow").strip(),
            version=str(frontmatter.get("version") or "0.1.0").strip(),
            enabled=bool(enabled_raw) and status not in {"disabled", "off", "inactive"},
            required_tools=_string_list(frontmatter.get("required_tools")),
            recipes=_mapping_list(hermes.get("recipes") or frontmatter.get("recipes")),
            scripts=_mapping_list(hermes.get("scripts") or frontmatter.get("scripts")),
        )

    def validate(self, *, skill_file: str | Path) -> list[SkillSchemaIssue]:
        path = Path(skill_file)
        rel = path.as_posix()
        issues: list[SkillSchemaIssue] = []
        if not SKILL_NAME_RE.fullmatch(self.name):
            issues.append(SkillSchemaIssue("error", "invalid_name", "Skill name must be a stable human slug.", rel))
        if not self.description:
            issues.append(SkillSchemaIssue("warning", "missing_description", "Skill should describe when it applies.", rel))
        if not self.enabled:
            issues.append(SkillSchemaIssue("warning", "skill_disabled", "Skill is present but disabled.", rel))
        for recipe in self.recipes:
            recipe_file = str(recipe.get("file") or "SKILL.md")
            if recipe_file == "SKILL.md":
                continue
            recipe_path = path.parent / recipe_file
            if not recipe_path.is_file():
                issues.append(SkillSchemaIssue("error", "missing_recipe_file", f"Missing recipe file {recipe_file}.", rel))
                continue
            try:
                payload = parse_structured_recipe_text(recipe_path.read_text(encoding="utf-8"))
                compiled = recipe_from_payload(
                    {**dict(recipe), **dict(payload), "file": recipe_file},
                    skill={"name": self.name, "path": rel},
                    source_file=recipe_file,
                )
            except Exception as exc:
                issues.append(SkillSchemaIssue("error", "invalid_recipe", str(exc), rel))
                continue
            for step in compiled.steps:
                policy = RecipePolicy.step_auto_executable(step)
                if not policy["auto_executable"] and step.run_policy == "auto":
                    issues.append(
                        SkillSchemaIssue(
                            "error",
                            "auto_recipe_step_blocked",
                            f"Recipe {compiled.id} step {step.id}: {policy['reason']}",
                            rel,
                        )
                    )
        for script in self.scripts:
            script_file = str(script.get("file") or "")
            if not script_file.startswith("scripts/"):
                issues.append(
                    SkillSchemaIssue("error", "script_outside_scripts", "Skill scripts must live under scripts/.", rel)
                )
        return issues


def validate_skill_file(skill_file: str | Path) -> dict[str, Any]:
    schema = SkillSchemaV1.from_skill_file(skill_file)
    issues = schema.validate(skill_file=skill_file)
    return {
        "schema_version": SKILL_SCHEMA_VERSION,
        "skill": asdict(schema),
        "issues": [issue.to_dict() for issue in issues],
        "ok": not any(issue.severity == "error" for issue in issues),
    }


def audit_skill_schemas(workspace_root: str | Path) -> dict[str, Any]:
    root = Path(workspace_root)
    skill_files = sorted((root / "skills").rglob("SKILL.md"))
    reports = [validate_skill_file(path) for path in skill_files]
    issues = [issue for report in reports for issue in report["issues"]]
    return {
        "schema_version": SKILL_SCHEMA_VERSION,
        "skill_count": len(reports),
        "issues": issues,
        "reports": reports,
        "summary": {
            "error_count": sum(1 for issue in issues if issue["severity"] == "error"),
            "warning_count": sum(1 for issue in issues if issue["severity"] == "warning"),
            "disabled_count": sum(1 for report in reports if not report["skill"]["enabled"]),
        },
    }


def _parse_frontmatter(text: str) -> dict[str, Any]:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    raw = text[3:end].strip()
    try:
        import yaml

        parsed = yaml.safe_load(raw) or {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    except Exception:
        data: dict[str, Any] = {}
        for line in raw.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            try:
                data[key.strip()] = json.loads(value)
            except Exception:
                data[key.strip()] = value
        return data


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]
