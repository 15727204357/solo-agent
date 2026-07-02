import { describe, expect, it } from "vitest";
import { initialRunViewState, replayRunEvents, runEventReducer } from "../runReducer";
import type { AgentEvent, RunViewState } from "../types";

function apply(events: AgentEvent[], seed: RunViewState = initialRunViewState) {
  return events.reduce((state, event) => runEventReducer(state, { type: "event", event }), seed);
}

describe("runEventReducer", () => {
  it("clears and appends streaming response text", () => {
    const state = apply(
      [
        { type: "response_started" },
        { type: "response_delta", message: "Hello" },
        { type: "response_delta", message: " world" },
      ],
      { ...initialRunViewState, responseText: "old" },
    );

    expect(state.status).toBe("running");
    expect(state.responseText).toBe("Hello world");
  });

  it("handles task_list_loaded", () => {
    const state = apply([
      {
        type: "task_list_loaded",
        payload: {
          data: {
            task_count: 1,
            active_task: { id: "T1", subject: "Inspect API", status: "in_progress" },
            tasks: [{ id: "T1", subject: "Inspect API", status: "in_progress" }],
          },
        },
      },
    ]);

    expect(state.taskCount).toBe(1);
    expect(state.taskList[0].status).toBe("in_progress");
  });

  it("handles task_list_updated", () => {
    const state = apply([
      {
        type: "task_list_updated",
        payload: {
          data: {
            task_count: 2,
            active_task: { id: "T2", subject: "Wire UI", status: "in_progress" },
            tasks: [
              { id: "T1", subject: "Inspect API", status: "completed" },
              { id: "T2", subject: "Wire UI", status: "in_progress" },
            ],
          },
        },
      },
    ]);

    expect(state.taskCount).toBe(2);
    expect(state.taskList).toHaveLength(2);
    expect(state.activeTask?.id).toBe("T2");
  });

  it("handles parallelism_decision_completed", () => {
    const state = apply([
      {
        type: "parallelism_decision_completed",
        payload: {
          data: {
            strategy: "serial",
            suitable: true,
            reason: "subagent_disabled",
            task_count: 2,
            subagent_policy: "auto",
            subagent_enabled: false,
          },
        },
      },
    ]);

    expect(state.parallelismDecision?.suitable).toBe(true);
    expect(state.parallelismDecision?.subagent_policy).toBe("auto");
    expect(state.parallelismDecision?.subagent_enabled).toBe(false);
  });

  it("handles intent_route_completed and keeps it across later events", () => {
    const state = apply([
      {
        type: "intent_route_completed",
        payload: {
          data: {
            intent: "modify_code",
            confidence: 0.82,
            searched_scopes: ["workspace", "code_index"],
            tool_candidates: [{ name: "code_map", reason: "Code task needs a repository map." }],
            risk_summary: { max_risk_level: "low", requires_approval: false },
          },
        },
      },
      {
        type: "tool_call_started",
        run_id: "run_1",
        payload: {
          data: {
            name: "code_map",
            arguments: { path: "." },
            index: 1,
          },
        },
      },
    ]);

    expect(state.intentRoute?.intent).toBe("modify_code");
    expect(state.intentRoute?.searched_scopes).toEqual(["workspace", "code_index"]);
    expect(state.routeHistory).toHaveLength(1);
    expect(state.toolCalls[0].name).toBe("code_map");
  });

  it("stores reroute history without later tool events overwriting the route", () => {
    const state = apply([
      {
        type: "intent_route_completed",
        payload: {
          data: {
            route_id: "session:run:route:0",
            route_epoch: 0,
            intent: "inspect_code",
            searched_scopes: ["workspace"],
          },
        },
      },
      {
        type: "intent_route_reroute_requested",
        payload: {
          data: {
            route_epoch: 1,
            triggers: [{ kind: "tool_no_results" }],
          },
        },
      },
      {
        type: "intent_route_reroute_completed",
        payload: {
          data: {
            route_id: "session:run:route:1",
            route_epoch: 1,
            intent: "inspect_code",
            searched_scopes: ["workspace", "code_index"],
          },
        },
      },
      {
        type: "tool_call_completed",
        run_id: "run_1",
        payload: {
          data: {
            name: "code_map",
            result: { files: ["app.py"] },
          },
        },
      },
    ]);

    expect(state.intentRoute?.route_epoch).toBe(1);
    expect(state.intentRoute?.searched_scopes).toEqual(["workspace", "code_index"]);
    expect(state.routeHistory).toHaveLength(2);
    expect(state.routeRerouteRequests).toHaveLength(1);
  });

  it("handles task_started task_completed and task_failed", () => {
    const completed = apply([
      {
        type: "task_started",
        payload: {
          data: {
            task_id: "task_1",
            description: "Inspect app",
            subagent_type: "general-purpose",
          },
        },
      },
      {
        type: "task_completed",
        payload: {
          data: {
            task_id: "task_1",
            description: "Inspect app",
            subagent_type: "general-purpose",
            result: "Looks good",
          },
        },
      },
    ]);

    expect(completed.subagentTasks[0].status).toBe("completed");
    expect(completed.subagentTasks[0].result).toBe("Looks good");

    const failed = apply([
      {
        type: "task_failed",
        payload: {
          data: {
            task_id: "task_2",
            description: "Inspect missing",
            subagent_type: "general-purpose",
            error: "missing.py does not exist",
          },
        },
      },
    ]);

    expect(failed.subagentTasks[0].status).toBe("failed");
    expect(failed.subagentTasks[0].error).toContain("missing.py");
  });

  it("handles tool_call_started and tool_call_completed", () => {
    const state = apply([
      {
        type: "tool_call_started",
        run_id: "run_1",
        payload: {
          data: {
            name: "read_file",
            arguments: { path: "app.py" },
            index: 1,
          },
        },
      },
      {
        type: "tool_call_completed",
        run_id: "run_1",
        payload: {
          data: {
            name: "read_file",
            result: { path: "app.py" },
            metadata: { truncated: false },
          },
        },
      },
    ]);

    expect(state.toolCalls).toHaveLength(1);
    expect(state.toolCalls[0].status).toBe("completed");
    expect(state.toolCalls[0].result).toEqual({ path: "app.py" });
  });

  it("handles task_blocked and replay restores it", () => {
    const events: AgentEvent[] = [
      {
        type: "task_blocked",
        payload: {
          data: {
            task_id: "task_blocked",
            description: "Inspect app",
            subagent_type: "general-purpose",
            reason: "parallelism_gate_not_suitable",
            error: "Task tool execution blocked: parallelism_gate_not_suitable",
          },
        },
      },
    ];

    const state = replayRunEvents(events, true);

    expect(state.subagentTasks[0].status).toBe("blocked");
    expect(state.subagentTasks[0].reason).toBe("parallelism_gate_not_suitable");
  });

  it("tracks patch verification plan and stop gate", () => {
    const state = apply([
      {
        type: "patch_approval_required",
        payload: {
          data: {
            id: "patch_1",
            status: "pending",
            summary: "Fix bug",
            verification_plan: {
              required: true,
              commands: [{ command: "pytest -q tests/test_app.py", tool: "targeted_pytest", target: "tests/test_app.py" }],
            },
            stop_gate: {
              status: "missing",
              approval_ready: false,
              missing_evidence: ["Passing result for: pytest -q tests/test_app.py"],
            },
          },
        },
      },
      {
        type: "verification_completed",
        payload: {
          data: {
            patch_id: "patch_1",
            verification: { ok: true },
            stop_gate: { status: "passed", approval_ready: true, reason: "All planned verification commands passed." },
          },
        },
      },
    ]);

    expect(state.status).toBe("awaiting_approval");
    expect(state.patchProposal?.verification_plan?.commands?.[0].tool).toBe("targeted_pytest");
    expect(state.patchProposal?.stop_gate?.status).toBe("passed");
    expect(state.patchProposal?.stop_gate?.approval_ready).toBe(true);
  });

  it("caps raw events at 300", () => {
    const events = Array.from({ length: 305 }, (_, index) => ({ type: `event_${index}` }));
    const state = apply(events);

    expect(state.rawEvents).toHaveLength(300);
    expect(state.rawEvents[0].type).toBe("event_5");
  });

  it("replays history into the expected final state", () => {
    const state = replayRunEvents(
      [
        { type: "response_started" },
        { type: "response_delta", message: "Draft" },
        {
          type: "parallelism_decision_completed",
          payload: { data: { strategy: "parallel", suitable: true, subagent_policy: "auto" } },
        },
        {
          type: "task_completed",
          payload: { data: { task_id: "task_1", description: "Inspect", result: "Done" } },
        },
        { type: "run_completed" },
      ],
      true,
    );

    expect(state.planMode).toBe(true);
    expect(state.status).toBe("completed");
    expect(state.responseText).toBe("Draft");
    expect(state.parallelismDecision?.strategy).toBe("parallel");
    expect(state.subagentTasks[0].status).toBe("completed");
  });
});
