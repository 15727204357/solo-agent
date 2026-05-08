from __future__ import annotations

from typing import Any


async def create_checkpointer(settings: Any) -> Any:
    mode = getattr(settings, "workflow_checkpointer", "memory")
    if mode == "none":
        return False

    if mode == "memory":
        from langgraph.checkpoint.memory import InMemorySaver
        return InMemorySaver()

    if mode == "sqlite":
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "SQLite checkpointer requires langgraph-checkpoint-sqlite. "
                "Install it with: uv add langgraph-checkpoint-sqlite"
            ) from exc
        import aiosqlite

        path = getattr(settings, "workflow_checkpoint_path", ".solo-agent/checkpoints/solo_agent_graph.sqlite3")
        conn = await aiosqlite.connect(str(path))
        return AsyncSqliteSaver(conn)

    raise ValueError(f"Unknown workflow_checkpointer mode: {mode}")
