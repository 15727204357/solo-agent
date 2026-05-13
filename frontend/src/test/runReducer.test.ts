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
