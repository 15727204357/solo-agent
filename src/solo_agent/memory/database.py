from __future__ import annotations

from pathlib import Path

from sqlalchemy import event, make_url, text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from .models import Base

DEFAULT_SQLITE_URL = "sqlite+aiosqlite:///solo_agent.db"


def sqlite_url(database_path: str | Path | None = None) -> str:
    if database_path is None:
        return DEFAULT_SQLITE_URL

    path_text = str(database_path)
    if path_text == ":memory:":
        return "sqlite+aiosqlite:///:memory:"
    if "://" in path_text:
        return path_text

    return f"sqlite+aiosqlite:///{Path(path_text).expanduser().resolve().as_posix()}"


def create_memory_engine(
    database_path: str | Path | None = None,
    *,
    echo: bool = False,
    future: bool = True,
) -> AsyncEngine:
    _ensure_sqlite_parent_directory(database_path)

    engine = create_async_engine(sqlite_url(database_path), echo=echo, future=future)

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_database(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.execute(text("DROP TRIGGER IF EXISTS messages_ai"))
        await connection.execute(text("DROP TRIGGER IF EXISTS messages_ad"))
        await connection.execute(text("DROP TRIGGER IF EXISTS messages_au"))
        await connection.execute(text("DROP TABLE IF EXISTS messages_fts"))
        await connection.execute(
            text(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    message_id UNINDEXED,
                    session_id UNINDEXED,
                    role UNINDEXED,
                    source UNINDEXED,
                    content,
                    tokenize = 'unicode61'
                )
                """
            )
        )
        await connection.execute(text("DELETE FROM messages_fts"))
        await connection.execute(
            text(
                """
                INSERT INTO messages_fts(message_id, session_id, role, source, content)
                SELECT id, session_id, role, 'message', content FROM messages
                """
            )
        )
        await connection.execute(
            text(
                """
                CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                    INSERT INTO messages_fts(message_id, session_id, role, source, content)
                    VALUES (new.id, new.session_id, new.role, 'message', new.content);
                END
                """
            )
        )
        await connection.execute(
            text(
                """
                CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                    DELETE FROM messages_fts WHERE message_id = old.id;
                END
                """
            )
        )
        await connection.execute(
            text(
                """
                CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE OF content, role ON messages BEGIN
                    DELETE FROM messages_fts WHERE message_id = old.id;
                    INSERT INTO messages_fts(message_id, session_id, role, source, content)
                    VALUES (new.id, new.session_id, new.role, 'message', new.content);
                END
                """
            )
        )


def _ensure_sqlite_parent_directory(database_path: str | Path | None) -> None:
    if database_path is None:
        return

    path_text = str(database_path)
    if path_text == ":memory:":
        return

    if "://" not in path_text:
        Path(path_text).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        return

    url = make_url(path_text)
    if not url.drivername.startswith("sqlite"):
        return

    database = url.database
    if not database or database == ":memory:" or database.startswith("file:"):
        return

    Path(database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
