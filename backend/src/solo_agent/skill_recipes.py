from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

AUTO_TOOL_NAMES = {
    "workspace_snapshot",
    "find_files",
    "search_code",
    "read_file",
    "git_status",
    "git_diff",
    "git_show",
    "run_command",
    "skill_script_run",
}
MANUAL_TOOL_NAMES = {
    "prepare_edit",
    "preview_patch",
    "apply_text_edit",
    "create_file",
    "mkdir",
    "move_path",
    "delete_path",
    "skill_manage",
}
KNOWN_RECIPE_TOOL_NAMES = AUTO_TOOL_NAMES | MANUAL_TOOL_NAMES
RECIPE_ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,80}$")
TEMPLATE_RE = re.compile(r"{{\s*([^{}]+?)\s*}}")
UNSAFE_TEMPLATE_TEXT_RE = re.compile(r"(?:\$\(|`|;|\|\||&&|\n|\r)")
SECRET_TEMPLATE_RE = re.compile(r"(?:TOKEN|SECRET|PASSWORD|PASSWD|API[_-]?KEY|PRIVATE[_-]?KEY)", re.IGNORECASE)


@dataclass(frozen=True)
class SkillRecipeStep:
    id: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    run_policy: str = "auto"
    risk_level: str = "low"
    continue_on_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool": self.tool,
            "arguments": self.arguments,
            "description": self.description,
            "run_policy": self.run_policy,
            "risk_level": self.risk_level,
            "continue_on_error": self.continue_on_error,
        }


@dataclass(frozen=True)
class SkillRecipe:
    id: str
    name: str
    skill_name: str
    skill_path: str
    description: str = ""
    when: list[str] = field(default_factory=list)
    mode: str = "assist"
    priority: int = 0
    steps: list[SkillRecipeStep] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    failure_policy: str = "stop"
    run_policy: str = "auto"
    source_file: str = "SKILL.md"

    def compact(self, *, query: str = "") -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "skill_name": self.skill_name,
            "skill_path": self.skill_path,
            "description": self.description,
            "when": self.when,
            "mode": self.mode,
            "priority": self.priority,
            "run_policy": self.run_policy,
            "source_file": self.source_file,
            "matched": recipe_matches(self, query),
            "auto_step_count": sum(1 for step in self.steps if RecipePolicy.step_auto_executable(step)["auto_executable"]),
            "manual_step_count": sum(
                1 for step in self.steps if not RecipePolicy.step_auto_executable(step)["auto_executable"]
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.compact(),
            "steps": [step.to_dict() for step in self.steps],
            "success_criteria": self.success_criteria,
            "failure_policy": self.failure_policy,
        }


class RecipePolicy:
    @staticmethod
    def step_auto_executable(step: SkillRecipeStep) -> dict[str, Any]:
        if step.run_policy != "auto":
            return {"auto_executable": False, "reason": "step_run_policy_manual"}
        if step.risk_level not in {"low", "medium-safe"}:
            return {"auto_executable": False, "reason": f"risk_level_{step.risk_level}"}
        if step.tool not in AUTO_TOOL_NAMES:
            return {"auto_executable": False, "reason": f"tool_{step.tool}_manual_or_unsupported"}
        if step.tool == "run_command" and _run_command_is_write_like(step.arguments):
            return {"auto_executable": False, "reason": "run_command_write_like_or_install"}
        if step.tool == "skill_script_run" and _skill_script_is_write_like(step.arguments):
            return {"auto_executable": False, "reason": "skill_script_write_like"}
        return {"auto_executable": True, "reason": "allowed_auto_step"}

    @staticmethod
    def snapshot() -> dict[str, Any]:
        return {
            "engine": "skill_recipe_policy",
            "auto_tools": sorted(AUTO_TOOL_NAMES),
            "manual_tools": sorted(MANUAL_TOOL_NAMES),
            "auto_boundary": "read/search/git-read/test/build/lint/check only",
            "scripts_enabled": False,
        }


def recipe_from_payload(payload: Mapping[str, Any], *, skill: Mapping[str, Any], source_file: str = "SKILL.md") -> SkillRecipe:
    recipe_id = str(payload.get("id") or "").strip()
    if not RECIPE_ID_RE.fullmatch(recipe_id):
        raise ValueError(f"Invalid recipe id: {recipe_id}")
    raw_steps = payload.get("steps") or []
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(f"Recipe {recipe_id} requires a non-empty steps list")

    steps = [
        _step_from_payload(step, index=index + 1)
        for index, step in enumerate(raw_steps)
        if isinstance(step, Mapping)
    ]
    if len(steps) != len(raw_steps):
        raise ValueError(f"Recipe {recipe_id} contains invalid step entries")

    return SkillRecipe(
        id=recipe_id,
        name=str(payload.get("name") or recipe_id).strip(),
        skill_name=str(skill.get("name") or "").strip(),
        skill_path=str(skill.get("path") or "").strip(),
        description=str(payload.get("description") or "").strip()[:400],
        when=_string_list(payload.get("when")),
        mode=str(payload.get("mode") or "assist").strip(),
        priority=int(payload.get("priority") or 0),
        steps=steps,
        success_criteria=_string_list(payload.get("success_criteria")),
        failure_policy=str(payload.get("failure_policy") or "stop").strip(),
        run_policy=str(payload.get("run_policy") or "auto").strip(),
        source_file=source_file,
    )


def compile_recipe(recipe: SkillRecipe, context: Mapping[str, Any]) -> dict[str, Any]:
    step_context: dict[str, Any] = {}
    compiled_steps: list[dict[str, Any]] = []
    for step in recipe.steps:
        rendered_arguments = render_templates(step.arguments, {**context, "steps": step_context})
        compiled_step = SkillRecipeStep(
            id=step.id,
            tool=step.tool,
            arguments=rendered_arguments if isinstance(rendered_arguments, dict) else {},
            description=step.description,
            run_policy=step.run_policy,
            risk_level=step.risk_level,
            continue_on_error=step.continue_on_error,
        )
        policy = RecipePolicy.step_auto_executable(compiled_step)
        compiled_steps.append({**compiled_step.to_dict(), **policy})
        step_context[step.id] = {"result": {}, "compiled": compiled_steps[-1]}
    runnable = [step for step in compiled_steps if step.get("auto_executable")]
    blocked = [step for step in compiled_steps if not step.get("auto_executable")]
    return {
        "recipe": recipe.compact(query=str(context.get("user_input", ""))),
        "steps": compiled_steps,
        "runnable_steps": len(runnable),
        "manual_steps": len(blocked),
        "run_policy": recipe.run_policy,
        "policy": RecipePolicy.snapshot(),
    }


def execute_recipe(
    recipe: SkillRecipe,
    context: Mapping[str, Any],
    tool_runner: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    run_id = f"recipe_{recipe.skill_name}_{recipe.id}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
    started_at = datetime.now(UTC).isoformat()
    step_context: dict[str, Any] = {}
    results: list[dict[str, Any]] = []
    status = "completed"
    blocked = 0

    for step in recipe.steps:
        rendered = render_templates(step.arguments, {**context, "steps": step_context})
        compiled_step = SkillRecipeStep(
            id=step.id,
            tool=step.tool,
            arguments=rendered if isinstance(rendered, dict) else {},
            description=step.description,
            run_policy=step.run_policy,
            risk_level=step.risk_level,
            continue_on_error=step.continue_on_error,
        )
        policy = RecipePolicy.step_auto_executable(compiled_step)
        step_result = {
            "id": compiled_step.id,
            "tool": compiled_step.tool,
            "arguments": compiled_step.arguments,
            "description": compiled_step.description,
            **policy,
            "status": "pending",
        }
        if not policy["auto_executable"]:
            blocked += 1
            step_result["status"] = "blocked"
            results.append(step_result)
            step_context[compiled_step.id] = {"result": step_result}
            continue
        try:
            output = tool_runner(compiled_step.tool, compiled_step.arguments)
        except Exception as exc:
            output = {"ok": False, "error": str(exc), "code": "recipe_step_error"}
        ok = bool(output.get("ok", True)) if isinstance(output, Mapping) else True
        step_result.update({"status": "completed" if ok else "failed", "result": output})
        results.append(step_result)
        step_context[compiled_step.id] = {"result": output}
        if not ok and recipe.failure_policy != "continue" and not compiled_step.continue_on_error:
            status = "failed"
            break

    return {
        "run_id": run_id,
        "recipe": recipe.compact(query=str(context.get("user_input", ""))),
        "status": status,
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "steps": results,
        "blocked_steps": blocked,
        "executed_steps": sum(1 for step in results if step.get("status") in {"completed", "failed"}),
        "policy": RecipePolicy.snapshot(),
    }


def render_templates(value: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return TEMPLATE_RE.sub(lambda match: _render_token(match.group(1), context), value)
    if isinstance(value, Mapping):
        return {str(key): render_templates(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [render_templates(item, context) for item in value]
    return value


def recipe_matches(recipe: SkillRecipe, query: str) -> bool:
    terms = {term.casefold() for term in re.findall(r"[\w\u4e00-\u9fff]+", query or "") if len(term) >= 2}
    if not terms:
        return True
    haystack = " ".join([recipe.id, recipe.name, recipe.description, *recipe.when]).casefold()
    return any(term in haystack for term in terms)


def parse_structured_recipe_text(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        import yaml

        return yaml.safe_load(stripped) or {}
    except Exception:
        pass
    return json.loads(stripped)


def _step_from_payload(payload: Mapping[str, Any], *, index: int) -> SkillRecipeStep:
    step_id = str(payload.get("id") or f"step-{index}").strip()
    if not RECIPE_ID_RE.fullmatch(step_id):
        raise ValueError(f"Invalid recipe step id: {step_id}")
    tool = str(payload.get("tool") or "").strip()
    if tool not in KNOWN_RECIPE_TOOL_NAMES:
        raise ValueError(f"Unsupported recipe step tool: {tool}")
    args = payload.get("arguments") or {}
    if not isinstance(args, Mapping):
        raise ValueError(f"Recipe step {step_id} arguments must be an object")
    _validate_template_payload(args)
    return SkillRecipeStep(
        id=step_id,
        tool=tool,
        arguments=dict(args),
        description=str(payload.get("description") or "").strip(),
        run_policy=str(payload.get("run_policy") or "auto").strip(),
        risk_level=str(payload.get("risk_level") or "low").strip(),
        continue_on_error=bool(payload.get("continue_on_error", False)),
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _render_token(token: str, context: Mapping[str, Any]) -> str:
    parts = [part.strip() for part in token.split(".") if part.strip()]
    if not parts:
        raise ValueError("Empty recipe template token")
    cursor: Any = context
    for part in parts:
        if isinstance(cursor, Mapping) and part in cursor:
            cursor = cursor[part]
            continue
        raise ValueError(f"Unknown recipe template token: {token}")
    rendered = str(cursor)
    if SECRET_TEMPLATE_RE.search(token) or SECRET_TEMPLATE_RE.search(rendered):
        raise PermissionError(f"Recipe template token may expose secrets: {token}")
    if UNSAFE_TEMPLATE_TEXT_RE.search(rendered):
        raise PermissionError(f"Recipe template rendered unsafe shell-like text: {token}")
    return rendered


def _validate_template_payload(value: Any) -> None:
    if isinstance(value, str):
        if SECRET_TEMPLATE_RE.search(value):
            raise PermissionError("Recipe arguments contain secret-like text")
        if TEMPLATE_RE.search(value):
            return
        if UNSAFE_TEMPLATE_TEXT_RE.search(value):
            raise PermissionError("Recipe arguments contain shell-like metacharacters")
    elif isinstance(value, Mapping):
        for item in value.values():
            _validate_template_payload(item)
    elif isinstance(value, list):
        for item in value:
            _validate_template_payload(item)


def _run_command_is_write_like(arguments: Mapping[str, Any]) -> bool:
    command = str(arguments.get("command") or "").casefold()
    args = [str(arg).casefold() for arg in arguments.get("args") or []]
    joined = " ".join([command, *args])
    if any(marker in joined for marker in (" install", " add ", " publish", " deploy", "--fix", " format ")):
        return True
    if command == "git" and args and args[0] not in {"status", "diff", "show", "log"}:
        return True
    return False


def _skill_script_is_write_like(arguments: Mapping[str, Any]) -> bool:
    args = [str(arg).casefold() for arg in arguments.get("args") or []]
    joined = " ".join(args)
    return any(marker in joined for marker in (" install", " add ", " publish", " deploy", "--fix", " format "))
