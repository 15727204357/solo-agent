from __future__ import annotations

import shutil
from pathlib import Path


class LocalSandboxProvider:
    """Local filesystem sandbox for a workflow run."""

    def __init__(self, runtime_root: str | Path = ".solo-agent/runs"):
        self._root = Path(runtime_root).resolve()

    async def setup(self, session_id: str, run_id: str) -> str:
        run_dir = self._root / session_id / run_id
        for sub in ("workspace", "uploads", "outputs"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        return str(run_dir)

    async def teardown(self, session_id: str, run_id: str) -> None:
        run_dir = self._root / session_id / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

    def map_path(self, session_id: str, run_id: str, relative_path: str) -> str:
        requested = Path(relative_path)
        if requested.is_absolute():
            raise ValueError("Sandbox paths must be relative.")

        workspace = (self._root / session_id / run_id / "workspace").resolve()
        mapped = (workspace / requested).resolve()
        try:
            mapped.relative_to(workspace)
        except ValueError as exc:
            raise ValueError("Sandbox path escapes the workspace.") from exc
        return str(mapped)

    def get_workspace(self, session_id: str, run_id: str) -> str:
        return str(self._root / session_id / run_id / "workspace")
