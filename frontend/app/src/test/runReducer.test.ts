import { describe, expect, it } from "vitest";
import { initialRunViewState, runEventReducer } from "../runReducer";
import type { AgentEvent, RunViewState } from "../types";

function apply(events: AgentEvent[], seed: RunViewState = initialRunViewState) {
  return events.reduce((state, event) => runEventReducer(state, { type: "event", event }), seed);
}

describe("runEventReducer", () => {
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
            subagent_enabled: false,
          },
        },
      },
    ]);

    expect(state.parallelismDecision?.suitable).toBe(true);
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

  it("caps raw events at 300", () => {
    const events = Array.from({ length: 305 }, (_, index) => ({ type: `event_${index}` }));
    const state = apply(events);

    expect(state.rawEvents).toHaveLength(300);
    expect(state.rawEvents[0].type).toBe("event_5");
  });
});
