"""计划层模块单元测试 — validate_plan_text, PlanQualityReport。"""

from __future__ import annotations

from solo_agent.agent.planning import validate_plan_text


class TestValidatePlaceholders:
    def test_detects_tbd(self) -> None:
        report = validate_plan_text("## Plan\n1. TBD - figure this out")
        assert not report.passed
        assert any(i.type == "placeholder" and "TBD" in i.message for i in report.issues)

    def test_detects_todo(self) -> None:
        report = validate_plan_text("1. TODO: add tests")
        assert not report.passed
        assert any(i.type == "placeholder" and "TODO" in i.message for i in report.issues)

    def test_detects_implement_later(self) -> None:
        report = validate_plan_text("We will implement later the rest.")
        assert not report.passed
        assert any("implement later" in i.message for i in report.issues)

    def test_detects_chinese_placeholder_lei_si(self) -> None:
        report = validate_plan_text("步骤2：类似上一步操作")
        assert not report.passed
        assert any("类似上一步" in i.message for i in report.issues)

    def test_detects_chinese_placeholder_shi_dang(self) -> None:
        report = validate_plan_text("适当处理异常情况")
        assert not report.passed
        assert any("适当处理" in i.message for i in report.issues)

    def test_detects_chinese_placeholder_dai_ding(self) -> None:
        report = validate_plan_text("待定：需要进一步确认")
        assert not report.passed
        assert any("待定" in i.message for i in report.issues)

    def test_detects_chinese_placeholder_lue(self) -> None:
        report = validate_plan_text("实现细节略")
        assert not report.passed
        assert any("略" in i.message for i in report.issues)

    def test_detects_chinese_placeholder_tong_li(self) -> None:
        report = validate_plan_text("同理操作")
        assert not report.passed
        assert any("同理" in i.message for i in report.issues)

    def test_detects_chinese_placeholder_tong_shang(self) -> None:
        report = validate_plan_text("同上处理")
        assert not report.passed
        assert any("同上" in i.message for i in report.issues)

    def test_clean_plan_passes_placeholder_check(self) -> None:
        report = validate_plan_text(
            "## Plan\n1. Create the file at src/foo.py with the Foo class.\n"
            "2. Write unit tests for Foo in tests/test_foo.py.\n"
            "3. Run pytest to verify all tests pass."
        )
        placeholder_issues = [i for i in report.issues if i.type == "placeholder"]
        assert len(placeholder_issues) == 0


class TestValidateStructure:
    def test_missing_file_map(self) -> None:
        report = validate_plan_text(
            "## Steps\n1. Write some code.\n## Self-Review\nLooks good."
        )
        assert any(i.type == "missing_file_map" for i in report.issues)

    def test_missing_self_review(self) -> None:
        report = validate_plan_text(
            "## File Map\n| file | action |\n|------|--------|\n| a.py | create |\n"
            "## Steps\n1. Write code."
        )
        assert any(i.type == "missing_self_review" for i in report.issues)

    def test_missing_steps(self) -> None:
        report = validate_plan_text(
            "## File Map\n| file | action |\n|------|--------|\n| a.py | create |\n"
            "## Self-Review\nNo issues found."
        )
        assert any(i.type == "missing_steps" for i in report.issues)

    def test_complete_plan_passes_structure(self) -> None:
        report = validate_plan_text(
            "## File Map\n"
            "| file | action | purpose |\n"
            "|------|--------|--------|\n"
            "| src/foo.py | CREATE | Foo class |\n\n"
            "## Steps\n"
            "1. Command: `New-Item -Path src/foo.py -ItemType File`\n"
            "   Expected Output: creates src/foo.py.\n"
            "   Success Criteria: src/foo.py exists.\n"
            "   Files Affected: src/foo.py.\n\n"
            "## Verification\n"
            "`python -m pytest tests/test_foo.py -q`\n\n"
            "## Execution Options\n"
            "Single Agent is recommended because this touches one file.\n\n"
            "## Self-Review\n"
            "No placeholders. All file paths concrete."
        )
        structural_issues = [
            i for i in report.issues if i.type.startswith("missing_")
        ]
        assert len(structural_issues) == 0

    def test_file_map_with_chinese_heading(self) -> None:
        report = validate_plan_text(
            "## 文件地图\n| file | action |\n|------|--------|\n| a.py | CREATE |\n"
            "## Steps\n"
            "1. Command: `New-Item -Path a.py -ItemType File`\n"
            "   Expected Output: creates a.py.\n"
            "   Success Criteria: a.py exists.\n"
            "   Files Affected: a.py.\n\n"
            "## Verification\n`python -m pytest tests/test_a.py -q`\n\n"
            "## Execution Options\n推荐 Single Agent，因为只创建一个文件。\n\n"
            "## 自查\nAll good."
        )
        structural_issues = [
            i for i in report.issues if i.type.startswith("missing_")
        ]
        assert len(structural_issues) == 0

    def test_missing_command_details(self) -> None:
        report = validate_plan_text(
            "## File Map\n| file | action | purpose |\n|---|---|---|\n| a.py | CREATE | demo |\n\n"
            "## Steps\n1. Create a.py.\n\n"
            "## Verification\nRun tests.\n\n"
            "## Execution Options\nSingle Agent is recommended.\n\n"
            "## Self-Review\nClean."
        )
        issue_types = {i.type for i in report.issues}
        assert {"missing_step_command", "missing_expected_output", "missing_success_criteria"} <= issue_types

    def test_missing_execution_options(self) -> None:
        report = validate_plan_text(
            "## File Map\n| file | action | purpose |\n|---|---|---|\n| a.py | CREATE | demo |\n\n"
            "## Steps\n"
            "1. Command: `New-Item -Path a.py -ItemType File`\n"
            "   Expected Output: creates a.py.\n"
            "   Success Criteria: a.py exists.\n"
            "   Files Affected: a.py.\n\n"
            "## Verification\n`python -m pytest tests/test_a.py -q`\n\n"
            "## Self-Review\nClean."
        )
        assert any(i.type == "missing_execution_options" for i in report.issues)

    def test_missing_recommended_execution_option(self) -> None:
        report = validate_plan_text(
            "## File Map\n| file | action | purpose |\n|---|---|---|\n| a.py | CREATE | demo |\n\n"
            "## Steps\n"
            "1. Command: `New-Item -Path a.py -ItemType File`\n"
            "   Expected Output: creates a.py.\n"
            "   Success Criteria: a.py exists.\n"
            "   Files Affected: a.py.\n\n"
            "## Verification\n`python -m pytest tests/test_a.py -q`\n\n"
            "## Execution Options\nSingle Agent and Parallel Agents are both possible.\n\n"
            "## Self-Review\nClean."
        )
        assert any(i.type == "missing_recommended_option" for i in report.issues)


class TestPlanQualityReport:
    def test_passed_true_for_clean_plan(self) -> None:
        report = validate_plan_text(
            "## File Map\n| f | act | purpose |\n|---|---|---|\n| a.py | CREATE | demo |\n"
            "## Steps\n"
            "1. Command: `New-Item -Path a.py -ItemType File`\n"
            "   Expected Output: creates a.py.\n"
            "   Success Criteria: a.py exists.\n"
            "   Files Affected: a.py.\n"
            "## Verification\n`python -m pytest tests/test_a.py -q`\n"
            "## Execution Options\nSingle Agent is recommended because this is small.\n"
            "## Self-Review\nClean."
        )
        assert report.passed
        assert "全部通过" in report.summary

    def test_passed_false_for_issues(self) -> None:
        report = validate_plan_text("TODO: nothing")
        assert not report.passed
        assert report.issues

    def test_report_to_dict(self) -> None:
        report = validate_plan_text(
            "## File Map\n| f | ac | purpose |\n|---|---|---|\n| a.py | CREATE | demo |\n"
            "## Steps\n"
            "1. Command: `New-Item -Path a.py -ItemType File`\n"
            "   Expected Output: creates a.py.\n"
            "   Success Criteria: a.py exists.\n"
            "   Files Affected: a.py.\n"
            "## Verification\n`python -m pytest tests/test_a.py -q`\n"
            "## Execution Options\nSingle Agent is recommended because this is small.\n"
            "## Self-Review\nClean."
        )
        d = report.to_dict()
        assert "passed" in d
        assert "issues" in d
        assert "summary" in d
        assert isinstance(d["issues"], list)


class TestBuildDeepPlanMessages:
    def test_build_deep_plan_messages_structure(self) -> None:
        from solo_agent.agent.prompts import build_deep_plan_messages

        messages = build_deep_plan_messages(
            user_input="Build a REST API for users",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "REST API" in messages[1]["content"]
        assert "writing-plans" in messages[1]["content"]

    def test_build_deep_plan_messages_with_context(self) -> None:
        from solo_agent.agent.prompts import build_deep_plan_messages

        messages = build_deep_plan_messages(
            user_input="Refactor auth module",
            memory_context_block="Previous session: user likes factory pattern",
            skill_context_block="<skill-context>pytest skill loaded</skill-context>",
            conversation_context={"summary": "Working on auth module refactoring"},
        )
        assert len(messages) == 2
        user_content = messages[1]["content"]
        assert "Previous session" in user_content
        assert "pytest skill loaded" in user_content
        assert "auth module" in user_content


class TestDeepPlanPromptContent:
    def test_deep_plan_system_prompt_no_placeholders(self) -> None:
        from solo_agent.agent.prompts import DEEP_PLAN_SYSTEM_PROMPT

        assert "TBD" in DEEP_PLAN_SYSTEM_PROMPT
        assert "TODO" in DEEP_PLAN_SYSTEM_PROMPT
        assert "类似上一步" in DEEP_PLAN_SYSTEM_PROMPT
        assert "no placeholders" in DEEP_PLAN_SYSTEM_PROMPT.lower()

    def test_deep_plan_system_prompt_has_sections(self) -> None:
        from solo_agent.agent.prompts import DEEP_PLAN_SYSTEM_PROMPT

        assert "File Map" in DEEP_PLAN_SYSTEM_PROMPT
        assert "Steps" in DEEP_PLAN_SYSTEM_PROMPT
        assert "Self-Review" in DEEP_PLAN_SYSTEM_PROMPT
        assert "Execution Options" in DEEP_PLAN_SYSTEM_PROMPT

    def test_deep_plan_system_prompt_allows_plan_code_blocks(self) -> None:
        from solo_agent.agent.prompts import DEEP_PLAN_SYSTEM_PROMPT

        assert "Do NOT write any code" not in DEEP_PLAN_SYSTEM_PROMPT
        assert "complete code" in DEEP_PLAN_SYSTEM_PROMPT
        assert "Do NOT modify files" in DEEP_PLAN_SYSTEM_PROMPT
