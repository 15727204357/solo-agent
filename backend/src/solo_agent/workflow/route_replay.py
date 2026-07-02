"""Replay helpers for intent-route events."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def replay_route_events(events: Iterable[Mapping[str, Any] | Any]) -> dict[str, Any]:
    """Rebuild route epochs from a run event stream."""

    epochs: dict[int, dict[str, Any]] = {}
    reroute_requests: list[dict[str, Any]] = []
    for event in events:
        event_type = str(_event_get(event, "type", ""))
        data = _event_data(event)
        if event_type in {"intent_route_completed", "intent_route_reroute_completed"}:
            epoch = int(data.get("route_epoch") or len(epochs))
            epochs[epoch] = {
                "route_id": data.get("route_id"),
                "route_epoch": epoch,
                "event_type": event_type,
                "intent": data.get("intent"),
                "confidence": data.get("confidence"),
                "searched_scopes": list(data.get("searched_scopes") or []),
                "tool_candidates": list(data.get("tool_candidates") or []),
                "selected_calls": list(data.get("proposed_tool_calls") or []),
                "risk_summary": dict(data.get("risk_summary") or {}),
                "reroute_triggers": list(data.get("reroute_triggers") or []),
                "decision_trace": list(data.get("decision_trace") or []),
            }
        elif event_type == "intent_route_reroute_requested":
            request = {
                "route_epoch": int(data.get("route_epoch") or len(epochs)),
                "triggers": list(data.get("triggers") or []),
            }
            reroute_requests.append(request)
            epoch = int(request["route_epoch"])
            if epoch in epochs:
                epochs[epoch]["reroute_request"] = request

    ordered_epochs = [epochs[key] for key in sorted(epochs)]
    return {
        "epochs": ordered_epochs,
        "reroute_requests": reroute_requests,
        "latest": ordered_epochs[-1] if ordered_epochs else None,
    }


def _event_get(event: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(event, Mapping):
        return event.get(key, default)
    return getattr(event, key, default)


def _event_data(event: Mapping[str, Any] | Any) -> dict[str, Any]:
    payload = _event_get(event, "payload", None)
    if isinstance(payload, Mapping):
        nested = payload.get("data")
        if isinstance(nested, Mapping):
            return dict(nested)
        return dict(payload)
    data = _event_get(event, "data", None)
    return dict(data) if isinstance(data, Mapping) else {}
