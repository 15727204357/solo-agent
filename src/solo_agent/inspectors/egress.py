"""Egress checks for data leaving the local workspace."""

from __future__ import annotations

import re

from .base import InspectionResult, ToolCall


class EgressInspector:
    """Block network egress and likely secret egress attempts.

    Read-only tools should inspect the local workspace only. Network addresses,
    webhook endpoints, and explicit upload/send instructions are blocked before
    tool execution.
    """

    _url_pattern = re.compile(r"\b(?:https?|ftp|s3)://[^\s'\"]+", re.I)
    _network_command_pattern = re.compile(
        r"\b(?:curl|wget|Invoke-WebRequest|iwr|Invoke-RestMethod|irm|scp|rsync|ssh|nc|netcat)\b",
        re.I,
    )
    _send_pattern = re.compile(
        r"\b(?:send|upload|post|exfiltrate|forward|paste)\b.*\b(?:token|secret|key|password|file|env)\b",
        re.I,
    )

    def inspect_text(self, text: str) -> InspectionResult:
        normalized = text or ""

        match = self._url_pattern.search(normalized)
        if match:
            return InspectionResult.block(
                "Network egress is not allowed for milestone 1 tools.",
                code="network_egress",
                metadata={"match": match.group(0)},
            )

        match = self._network_command_pattern.search(normalized)
        if match:
            return InspectionResult.block(
                "Network transfer commands are not allowed.",
                code="network_command",
                metadata={"match": match.group(0)},
            )

        match = self._send_pattern.search(normalized)
        if match:
            return InspectionResult.block(
                "Potential data exfiltration request was blocked.",
                code="data_exfiltration",
                metadata={"match": match.group(0)},
            )

        return InspectionResult.allow()

    def inspect_tool_call(self, call: ToolCall) -> InspectionResult:
        combined = " ".join(str(value) for value in call.arguments.values())
        return self.inspect_text(combined)
