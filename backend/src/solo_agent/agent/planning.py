"""计划层模块：深度计划生成与质量验证。

plan 模式生成 Superpowers writing-plans 风格的实施计划并停止；
agent 模式保持现有正常执行链路。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

PlanningMode = Literal["agent", "plan"]

# 占位符禁止列表（中英文）
_PLACEHOLDER_PATTERNS: list[tuple[str, str]] = [
    # 英文占位符
    (r"\bTBD\b", "英文占位符 TBD"),
    (r"\bTODO\b", "英文占位符 TODO"),
    (r"\bFIXME\b", "英文占位符 FIXME"),
    (r"\bHACK\b", "英文占位符 HACK"),
    (r"\bimplement later\b", "英文占位符 implement later"),
    (r"\bto be (determined|decided|implemented|defined)\b", "英文占位符 to be determined 等"),
    # 中文占位符
    (r"类似上一步", "中文占位符 类似上一步"),
    (r"适当处理", "中文占位符 适当处理"),
    (r"似上一步", "中文占位符 似上一步"),
    (r"待定", "中文占位符 待定"),
    (r"略[（(]?[^)）\n]{0,10}[)）]?", "中文占位符 略（省略）"),
    (r"同理[，,。\s]*(?!可证|可推|可解)", "中文占位符 同理（缺少展开）"),
    (r"同上", "中文占位符 同上"),
    (r"基本同上", "中文占位符 基本同上"),
    (r"大致相同", "中文占位符 大致相同"),
    (r"参见[上上下前].{0,20}步", "中文占位符 参见上一步"),
]


@dataclass
class PlanQualityIssue:
    """计划质量检查发现的问题。"""

    type: str
    message: str
    pattern: str | None = None
    location: str | None = None
    severity: str = "warning"


@dataclass
class PlanQualityReport:
    """计划质量报告。"""

    passed: bool = True
    issues: list[PlanQualityIssue] = field(default_factory=list)
    summary: str = ""

    def add_issue(self, issue: PlanQualityIssue) -> None:
        self.issues.append(issue)
        if issue.severity != "info":
            self.passed = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [
                {
                    "type": i.type,
                    "message": i.message,
                    "pattern": i.pattern,
                    "location": i.location,
                    "severity": i.severity,
                }
                for i in self.issues
            ],
            "summary": self.summary,
        }


def validate_plan_text(plan_text: str) -> PlanQualityReport:
    """验证计划文本：禁止占位符、检查结构和执行细节完整性。

    返回 PlanQualityReport，其中 passed=True 表示全部检查通过。
    """
    report = PlanQualityReport()

    _check_placeholders(plan_text, report)
    _check_structure(plan_text, report)

    if report.passed:
        report.summary = "计划质量检查全部通过：无占位符，文件地图/步骤/验证/执行选项/自查均完整。"
    else:
        issue_count = len(report.issues)
        placeholder_count = sum(1 for i in report.issues if i.type == "placeholder")
        struct_count = sum(1 for i in report.issues if i.type.startswith("missing_"))
        report.summary = (
            f"计划质量检查发现 {issue_count} 个问题"
            f"（占位符 {placeholder_count}，结构缺失 {struct_count}）。"
        )

    return report


def _check_placeholders(text: str, report: PlanQualityReport) -> None:
    """扫描占位符。"""
    for pattern, description in _PLACEHOLDER_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            line_no = text[: match.start()].count("\n") + 1
            report.add_issue(
                PlanQualityIssue(
                    type="placeholder",
                    message=f"检测到占位符: {description}",
                    pattern=pattern,
                    location=f"第 {line_no} 行",
                )
            )


def _check_structure(text: str, report: PlanQualityReport) -> None:
    """检查计划结构完整性：文件地图、步骤、验证、执行选项、自查。"""

    # 文件地图检查
    if not _has_section(text, r"#+\s*(File Map|文件地图|文件清单|Files Modified|Modified Files|文件列表)"):
        report.add_issue(
            PlanQualityIssue(
                type="missing_file_map",
                message="计划缺少文件地图 (## File Map / 文件地图) 部分",
            )
        )

    # 验证检查
    if not _has_section(text, r"#+\s*(Verification|验证|校验|测试计划|Test Plan|Checks)"):
        report.add_issue(
            PlanQualityIssue(
                type="missing_verification",
                message="计划缺少验证命令 (## Verification / 验证) 部分",
            )
        )

    # 执行选项检查
    if not _has_section(text, r"#+\s*(Execution Options|执行选项|实施选项|执行模式|实现选项)"):
        report.add_issue(
            PlanQualityIssue(
                type="missing_execution_options",
                message="计划缺少执行选项 (## Execution Options / 执行选项) 部分",
            )
        )
    elif not re.search(r"\b(recommend(?:ed|ation)?|建议|推荐)\b|推荐|建议", text, re.IGNORECASE):
        report.add_issue(
            PlanQualityIssue(
                type="missing_recommended_option",
                message="执行选项缺少明确推荐方案 (recommended / 推荐 / 建议)",
            )
        )

    # 自省检查
    if not _has_section(
        text,
        r"#+\s*(Self.?Review|自检|自查|自省|Plan Self.?Review|Quality Check|计划自检)",
    ):
        report.add_issue(
            PlanQualityIssue(
                type="missing_self_review",
                message="计划缺少自省 (## Self-Review / 自查) 部分",
            )
        )

    # 步骤检查
    if not _has_numbered_steps(text):
        report.add_issue(
            PlanQualityIssue(
                type="missing_steps",
                message="计划缺少编号执行步骤 (## Steps 下须有 1. 2. 3. 格式)",
            )
        )
        return

    _check_step_details(text, report)


def _check_step_details(text: str, report: PlanQualityReport) -> None:
    """检查每个深度计划至少包含可执行命令、预期输出、成功标准和影响文件。"""

    field_prefix = r"(?:^|\n)\s*(?:\d+[\.\)、]\s*)?(?:[-*]\s*)?(?:\*\*)?"
    if not re.search(field_prefix + r"(Command|命令)(?:\*\*)?\s*[:：]", text, re.IGNORECASE):
        report.add_issue(
            PlanQualityIssue(
                type="missing_step_command",
                message="执行步骤缺少精确命令字段 (Command / 命令)",
            )
        )
    if not re.search(
        field_prefix + r"(Expected Output|预期输出|期望输出)(?:\*\*)?\s*[:：]",
        text,
        re.IGNORECASE,
    ):
        report.add_issue(
            PlanQualityIssue(
                type="missing_expected_output",
                message="执行步骤缺少预期输出字段 (Expected Output / 预期输出)",
            )
        )
    if not re.search(
        field_prefix + r"(Success Criteria|成功标准|验收标准|通过标准)(?:\*\*)?\s*[:：]",
        text,
        re.IGNORECASE,
    ):
        report.add_issue(
            PlanQualityIssue(
                type="missing_success_criteria",
                message="执行步骤缺少成功标准字段 (Success Criteria / 成功标准)",
            )
        )
    if not re.search(
        field_prefix + r"(Files Affected|影响文件|涉及文件)(?:\*\*)?\s*[:：]",
        text,
        re.IGNORECASE,
    ):
        report.add_issue(
            PlanQualityIssue(
                type="missing_files_affected",
                message="执行步骤缺少影响文件字段 (Files Affected / 影响文件)",
            )
        )


def _has_section(text: str, heading_pattern: str) -> bool:
    """检查文本中是否存在匹配的标题。"""
    return bool(re.search(heading_pattern, text, re.IGNORECASE | re.MULTILINE))


def _has_numbered_steps(text: str) -> bool:
    """检查文本中是否存在编号步骤 (1. 或 Step 1 格式)。"""
    # 宽松匹配：至少一个编号条目
    return bool(re.search(r"(?:^|\n)\s*\d+[\.\)、]\s+\S", text, re.MULTILINE))
