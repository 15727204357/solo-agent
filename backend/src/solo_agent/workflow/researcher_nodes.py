"""Researcher node factories — sub-task execution with readonly tool access."""

from __future__ import annotations

from typing import Any


def make_collect_context_node(deps: Any, settings: Any):
    """Collect workspace context using readonly tools."""
    from solo_agent.workflow.graph_nodes import _make_node
    from solo_agent.workflow.stages import _collect_context_node as _collect_stage
    return _make_node(_collect_stage, deps, settings)


def make_inspect_node(deps: Any):
    """Run safety inspection on the research request."""
    from solo_agent.workflow.graph_nodes import _make_node
    from solo_agent.workflow.stages import _inspect_node as _inspect_stage
    return _make_node(_inspect_stage, deps)


def make_execute_research_tools_node(deps: Any, settings: Any):
    """Execute tools autonomously (LLM-driven tool selection)."""
    from solo_agent.workflow.graph_nodes import _make_node
    from solo_agent.workflow.stages import _execute_tools_node as _execute_stage
    return _make_node(_execute_stage, deps, settings)
