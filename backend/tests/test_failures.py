from __future__ import annotations

from solo_agent.failures import classify_command_failure, classify_failures, remediation_for_failures


def test_pytest_failure_classification_extracts_test_and_location() -> None:
    output = """
    ______________________________ test_greet ______________________________
    tests/test_service.py:12: AssertionError
    E assert 'hello' == 'hi'
    FAILED tests/test_service.py::test_greet - AssertionError
    """

    report = classify_command_failure(
        {"command": "pytest -q tests/test_service.py", "result": {"returncode": 1, "output": output}}
    )

    assert report is not None
    assert report.kind == "test_failure"
    assert report.failing_test == "tests/test_service.py::test_greet"
    assert report.file == "tests/test_service.py"
    assert report.line == 12
    assert "hello" in report.snippet


def test_ruff_failure_classification_extracts_rule() -> None:
    output = "pkg/app.py:4:1: F401 `os` imported but unused\n"

    report = classify_command_failure({"command": "ruff check .", "result": {"returncode": 1, "output": output}})

    assert report is not None
    assert report.kind == "lint_failure"
    assert report.file == "pkg/app.py"
    assert report.line == 4
    assert report.rule == "F401"


def test_dependency_missing_is_blocked_not_install() -> None:
    output = "ModuleNotFoundError: No module named 'requests'\n"

    report = classify_command_failure({"command": "pytest -q", "result": {"returncode": 1, "output": output}})
    remediation = remediation_for_failures(
        classify_failures([{"command": "pytest -q", "result": {"returncode": 1, "output": output}}])
    )

    assert report is not None
    assert report.kind == "dependency_missing"
    assert report.retryable is False
    assert remediation["status"] == "blocked"
    assert "Do not install dependencies automatically" in remediation["next_actions"][0]
