from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SandboxProvider(Protocol):
    """沙箱抽象接口。第一版实现 LocalSandboxProvider，Docker 预留。"""

    async def setup(self, session_id: str, run_id: str) -> str:
        """初始化沙箱环境，返回 workspace 路径。"""
        ...

    async def teardown(self, session_id: str, run_id: str) -> None:
        """清理沙箱环境。"""
        ...

    def map_path(self, session_id: str, run_id: str, relative_path: str) -> str:
        """将相对路径映射到沙箱内的绝对路径。"""
        ...

    def get_workspace(self, session_id: str, run_id: str) -> str:
        """获取沙箱工作区根路径。"""
        ...
