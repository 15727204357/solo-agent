from __future__ import annotations

import os
import tempfile

import pytest
from solo_agent.settings import Settings
from solo_agent.workflow.checkpoints import create_checkpointer


@pytest.mark.asyncio
async def test_memory_checkpointer_returns_saver() -> None:
    settings = Settings(workflow_checkpointer="memory")
    checkpointer = await create_checkpointer(settings)
    from langgraph.checkpoint.memory import InMemorySaver
    assert isinstance(checkpointer, InMemorySaver)


@pytest.mark.asyncio
async def test_none_mode_returns_false() -> None:
    settings = Settings(workflow_checkpointer="none")
    checkpointer = await create_checkpointer(settings)
    assert checkpointer is False


@pytest.mark.asyncio
async def test_sqlite_mode_returns_saver() -> None:
    with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as f:
        db_path = f.name
    try:
        settings = Settings(
            workflow_checkpointer="sqlite",
            workflow_checkpoint_path=db_path,
        )
        checkpointer = await create_checkpointer(settings)
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        assert isinstance(checkpointer, AsyncSqliteSaver)
        if hasattr(checkpointer, 'conn') and checkpointer.conn:
            await checkpointer.conn.close()
    finally:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except PermissionError:
                pass
