import type {
  AgentEvent,
  ParallelismDecision,
  RunViewState,
  SubagentTaskView,
  TaskListItem,
  ToolCallView,
} from "./types";

const MAX_RAW_EVENTS = 300;

export const initialRunViewState: RunViewState = {
  status: "idle",
  responseText: "",
  taskList: [],
  taskCount: 0,
  activeTask: null,
  planMode: false,
  parallelismDecision: null,
  toolCalls: [],
  subagentTasks: [],
  rawEvents: [],
};

export type RunReducerAction =
  | { type: "reset"; planMode?: boolean; status?: RunViewState["status"] }
  | { type: "event"; event: AgentEvent };

export function runEventReducer(state: RunViewState, action: RunReducerAction): RunViewState {
  if (action.type === "reset") {
    return {
      ...initialRunViewState,
      planMode: Boolean(action.planMode),
      status: action.status || "idle",
    };
  }

  const event = action.event;
  const data = agentData(event);
  let next: RunViewState = {
    ...state,
    rawEvents: [...state.rawEvents, event].slice(-MAX_RAW_EVENTS),
  };

  if (event.type === "started" || event.type === "run_started" || event.type === "receive_user_turn") {
    next = { ...next, status: "running" };
  }

  if (event.type === "response_started") {
    next = { ...next, status: "running", responseText: "" };
  }

  if (event.type === "response_delta") {
    next = { ...next, status: "running", responseText: `${next.responseText}${event.message || ""}` };
  }

  if (event.type === "response_completed") {
    const response = typeof data.response === "string" ? data.response : next.responseText;
    next = { ...next, responseText: response };
  }

  if (event.type === "task_list_loaded" || event.type === "task_list_updated") {
    const tasks = Array.isArray(data.tasks) ? (data.tasks as TaskListItem[]) : [];
    next = {
      ...next,
      taskList: tasks,
      taskCount: numeric(data.task_count, tasks.filter((item) => item.status !== "deleted").length),
      activeTask: isTask(data.active_task) ? data.active_task : tasks.find((item) => item.status === "in_progress") || null,
    };
  }

  if (event.type === "task_list_skipped") {
    next = { ...next, taskList: [], taskCount: 0, activeTask: null };
  }

  if (event.type === "parallelism_decision_completed") {
    next = { ...next, parallelismDecision: data as ParallelismDecision };
  }

  if (event.type === "tool_call_started") {
    next = { ...next, toolCalls: upsertToolStarted(next.toolCalls, event, data) };
  }

  if (event.type === "tool_call_completed") {
    next = { ...next, toolCalls: completeToolCall(next.toolCalls, event, data) };
  }

  if (event.type === "task_started") {
    next = { ...next, subagentTasks: upsertSubagentTask(next.subagentTasks, toSubagentTask(data, "running")) };
  }

  if (event.type === "task_completed") {
    next = { ...next, subagentTasks: upsertSubagentTask(next.subagentTasks, toSubagentTask(data, "completed")) };
  }

  if (event.type === "task_failed") {
    next = { ...next, subagentTasks: upsertSubagentTask(next.subagentTasks, toSubagentTask(data, "failed")) };
  }

  if (event.type === "task_blocked") {
    next = { ...next, subagentTasks: upsertSubagentTask(next.subagentTasks, toSubagentTask(data, "blocked")) };
  }

  if (event.type === "run_completed" || event.type === "completed") {
    next = { ...next, status: "completed" };
  }

  if (event.type === "failed" || event.type === "error" || event.type === "cancelled") {
    next = { ...next, status: "failed" };
  }

  return next;
}

export function replayRunEvents(events: AgentEvent[], planMode = false): RunViewState {
  return events.reduce(
    (state, event) => runEventReducer(state, { type: "event", event }),
    runEventReducer(initialRunViewState, { type: "reset", planMode, status: "idle" }),
  );
}

export function agentData(event: AgentEvent): Record<string, unknown> {
  const payload = event.payload || {};
  const nested = payload.data;
  if (nested && typeof nested === "object" && !Array.isArray(nested)) {
    return nested as Record<string, unknown>;
  }
  return payload;
}

function upsertToolStarted(toolCalls: ToolCallView[], event: AgentEvent, data: Record<string, unknown>): ToolCallView[] {
  const index = numeric(data.index, toolCalls.length + 1);
  const id = `${event.run_id || "run"}:${index}:${String(data.name || "tool")}`;
  const started: ToolCallView = {
    id,
    name: String(data.name || "tool"),
    status: "running",
    arguments: data.arguments,
    startedAt: event.created_at,
  };
  const existing = toolCalls.findIndex((call) => call.id === id);
  if (existing >= 0) {
    return toolCalls.map((call, callIndex) => (callIndex === existing ? { ...call, ...started } : call));
  }
  return [...toolCalls, started];
}

function completeToolCall(toolCalls: ToolCallView[], event: AgentEvent, data: Record<string, unknown>): ToolCallView[] {
  const name = String(data.name || "tool");
  const index = [...toolCalls].reverse().findIndex((call) => call.name === name && call.status === "running");
  const realIndex = index >= 0 ? toolCalls.length - 1 - index : -1;
  const completed: Partial<ToolCallView> = {
    name,
    status: data.blocked ? "blocked" : "completed",
    result: data.result,
    blocked: Boolean(data.blocked),
    reason: typeof data.reason === "string" ? data.reason : undefined,
    metadata: isRecord(data.metadata) ? data.metadata : undefined,
    completedAt: event.created_at,
  };
  if (realIndex >= 0) {
    return toolCalls.map((call, callIndex) => (callIndex === realIndex ? { ...call, ...completed } : call));
  }
  return [
    ...toolCalls,
    {
      id: `${event.run_id || "run"}:${toolCalls.length + 1}:${name}`,
      arguments: data.arguments,
      ...completed,
    } as ToolCallView,
  ];
}

function toSubagentTask(data: Record<string, unknown>, status: SubagentTaskView["status"]): SubagentTaskView {
  const id = String(data.task_id || data.id || `task-${Math.random().toString(16).slice(2)}`);
  return {
    id,
    description: String(data.description || data.title || "Scoped Task"),
    subagentType: String(data.subagent_type || "general-purpose"),
    status,
    result: typeof data.result === "string" ? data.result : undefined,
    error: typeof data.error === "string" ? data.error : undefined,
    reason: typeof data.reason === "string" ? data.reason : undefined,
    readPaths: Array.isArray(data.read_paths) ? data.read_paths.map(String) : undefined,
  };
}

function upsertSubagentTask(tasks: SubagentTaskView[], task: SubagentTaskView): SubagentTaskView[] {
  const existing = tasks.findIndex((item) => item.id === task.id);
  if (existing >= 0) {
    return tasks.map((item, index) => (index === existing ? { ...item, ...task } : item));
  }
  return [...tasks, task];
}

function numeric(value: unknown, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function isTask(value: unknown): value is TaskListItem {
  return isRecord(value) && typeof value.status === "string";
}
