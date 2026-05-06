"""Server-sent event helpers."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from solo_agent.web.models import RunEvent


def encode_sse(event: RunEvent | dict[str, object], event_name: str | None = None) -> str:
    if isinstance(event, RunEvent):
        event_name = event_name or event.type
        event_id = event.sequence
        data = event.to_public_dict()
    else:
        event_id = None
        data = event

    lines: list[str] = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    if event_name:
        lines.append(f"event: {event_name}")

    payload = json.dumps(data, ensure_ascii=False, default=str)
    for line in payload.splitlines() or [""]:
        lines.append(f"data: {line}")
    lines.append("")
    return "\n".join(lines) + "\n"


async def heartbeat_stream(seconds: int) -> AsyncIterator[str]:
    while True:
        yield ": heartbeat\n\n"
        import asyncio

        await asyncio.sleep(seconds)

