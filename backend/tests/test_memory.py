from __future__ import annotations

import pytest
from solo_agent.memory import MessageRole, init_sqlite_memory


@pytest.mark.asyncio
async def test_sqlite_memory_roundtrip(tmp_path) -> None:
    repo = await init_sqlite_memory(tmp_path / "memory.sqlite3")

    session = await repo.create_session(title="Demo", workspace_path=str(tmp_path))
    run = await repo.create_run(session_id=session.id, provider="ollama", model="llama3.1")
    message = await repo.append_message(
        session_id=session.id,
        run_id=run.id,
        role=MessageRole.USER,
        content="Summarize the project",
    )
    tool_call = await repo.append_tool_call(
        run_id=run.id,
        name="list_files",
        arguments={"path": "."},
        result="[]",
        status="completed",
    )
    timing = await repo.append_timing_point(run_id=run.id, label="plan_started")
    snapshot = await repo.append_snapshot(
        session_id=session.id,
        run_id=run.id,
        label="checkpoint",
        data={"ok": True},
    )
    completed = await repo.complete_run(run_id=run.id)

    assert session.id
    assert message.sequence == 1
    assert tool_call.name == "list_files"
    assert timing.label == "plan_started"
    assert snapshot.data == {"ok": True}
    assert completed is not None
    assert completed.status == "completed"


@pytest.mark.asyncio
async def test_sqlite_memory_url_creates_parent_directory(tmp_path) -> None:
    database_path = tmp_path / "data" / "solo_agent.sqlite3"
    repo = await init_sqlite_memory(f"sqlite+aiosqlite:///{database_path.as_posix()}")

    session = await repo.create_session(title="URL database")

    assert database_path.exists()
    assert session.id


@pytest.mark.asyncio
async def test_sqlite_memory_lists_searches_and_summarizes_by_session(tmp_path) -> None:
    repo = await init_sqlite_memory(tmp_path / "memory.sqlite3")
    session = await repo.create_session(title="Demo")
    other = await repo.create_session(title="Other")
    run = await repo.create_run(session_id=session.id)
    other_run = await repo.create_run(session_id=other.id)

    first = await repo.append_message(
        session_id=session.id,
        run_id=run.id,
        role=MessageRole.USER,
        content="我偏好中文回答",
    )
    second = await repo.append_message(
        session_id=session.id,
        run_id=run.id,
        role=MessageRole.ASSISTANT,
        content="我会记住你的中文偏好",
    )
    await repo.append_message(
        session_id=other.id,
        run_id=other_run.id,
        role=MessageRole.USER,
        content="中文偏好不应该跨 session 泄露",
    )

    messages = await repo.list_messages(session.id)
    matches = await repo.search_memory(session_id=session.id, query="我偏好中文回答", limit=5)
    other_matches = await repo.search_memory(session_id=other.id, query="我偏好中文回答", limit=5)
    summary = await repo.append_or_update_summary_snapshot(
        session_id=session.id,
        run_id=run.id,
        summary="用户偏好中文回答。",
    )
    latest = await repo.get_latest_summary(session.id)

    assert [message.id for message in messages] == [first.id, second.id]
    assert all(match["id"] in {first.id, second.id} for match in matches)
    assert other_matches == []
    assert latest is not None
    assert latest.id == summary.id
    assert latest.data["summary"] == "用户偏好中文回答。"


@pytest.mark.asyncio
async def test_sqlite_memory_prefetch_sync_queue_and_pre_compress(tmp_path) -> None:
    repo = await init_sqlite_memory(tmp_path / "memory.sqlite3", memory_root=tmp_path)
    session = await repo.create_session(title="Demo")
    run = await repo.create_run(session_id=session.id)
    (tmp_path / "MEMORY.md").write_text("项目长期记忆：使用 FTS5。", encoding="utf-8")
    (tmp_path / "USER.md").write_text("用户偏好中文。", encoding="utf-8")

    await repo.sync_all(
        session_id=session.id,
        run_id=run.id,
        user_input="请记住我偏好中文",
        assistant_response="已记住你的中文偏好",
    )
    prefetched = await repo.prefetch_all(session_id=session.id, query="偏好中文")
    memory_only = await repo.prefetch_all(
        session_id=session.id,
        query="偏好中文",
        include_history=False,
    )
    queued = await repo.queue_prefetch_all(session_id=session.id, query="FTS5")
    cached = await repo.prefetch_all(session_id=session.id, query="FTS5")
    compressed = await repo.on_pre_compress(
        session_id=session.id,
        payload={"messages": [{"content": "用户偏好中文回答"}]},
    )
    builtin_matches = await repo.search_memory(session_id=session.id, query="FTS5")

    assert prefetched["builtin_memory"]["user"] == "用户偏好中文。"
    assert prefetched["recent_messages"]
    assert memory_only["recent_messages"] == []
    assert prefetched["retrieved_memories"]
    assert any(match["source"] == "builtin" for match in builtin_matches)
    assert queued["queued"] is True
    assert cached["cache_hit"] is True
    assert compressed["session_id"] == session.id
    assert compressed["insights"]
    assert "用户偏好中文回答" in (tmp_path / "USER.md").read_text(encoding="utf-8")
