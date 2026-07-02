from __future__ import annotations

from solo_agent.workflow.route_replay import replay_route_events


def test_replay_route_events_rebuilds_epochs_and_reroute_requests() -> None:
    replay = replay_route_events(
        [
            {
                "type": "intent_route_completed",
                "payload": {
                    "data": {
                        "route_id": "session:run:route:0",
                        "route_epoch": 0,
                        "intent": "inspect_code",
                        "confidence": 0.8,
                        "searched_scopes": ["workspace"],
                        "proposed_tool_calls": [{"name": "search_text"}],
                    }
                },
            },
            {
                "type": "intent_route_reroute_requested",
                "payload": {"data": {"route_epoch": 1, "triggers": [{"kind": "tool_no_results"}]}},
            },
            {
                "type": "intent_route_reroute_completed",
                "payload": {
                    "data": {
                        "route_id": "session:run:route:1",
                        "route_epoch": 1,
                        "intent": "inspect_code",
                        "confidence": 0.7,
                        "searched_scopes": ["workspace", "code_index"],
                        "proposed_tool_calls": [{"name": "code_map"}],
                    }
                },
            },
        ]
    )

    assert len(replay["epochs"]) == 2
    assert replay["reroute_requests"][0]["triggers"][0]["kind"] == "tool_no_results"
    assert replay["latest"]["route_epoch"] == 1
    assert replay["latest"]["selected_calls"][0]["name"] == "code_map"
