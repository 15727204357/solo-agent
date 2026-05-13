export type RunStatus = "idle" | "queued" | "running" | "completed" | "failed" | "cancelled" | "awaiting_approval";

export type Session = {
  id: string;
  title: string;
  workspace_path: string | null;
  created_at: string;
  updated_at: string;
};

export type RunRecord = {
  id: string;
  session_id: string;
  prompt: string;
  status: string;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
  metadata: Record<string, unknown>;
  stream_url?: string;
};

export type SessionDetail = Session & {
  message_count: number;
  summary: Record<string, unknown> | null;
  runs: RunRecord[];
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system" | string;
  content: string;
  run_id?: string | null;
  sequence?: number;
  created_at?: string;
};

export type AgentEvent = {
  session_id?: string;
  run_id?: string;
  sequence?: number;
  type: string;
  message?: string;
  payload?: Record<string, unknown>;
  created_at?: string;
};

export type TaskListItem = {
  id?: string;
  subject?: string;
  title?: string;
  description?: string;
  status: "pending" | "in_progress" | "completed" | "blocked" | "deleted" | string;
  notes?: string;
  metadata?: Record<string, unknown>;
};

export type ParallelismDecision = {
  strategy?: string;
  suitable?: boolean;
  allowed?: boolean;
  reason?: string;
  task_count?: number;
  candidates?: unknown[];
  subagent_enabled?: boolean;
  subagent_policy?: "off" | "auto" | string;
  [key: string]: unknown;
};

export type ToolCallView = {
  id: string;
  name: string;
  status: "running" | "completed" | "blocked" | "failed";
  arguments?: unknown;
  result?: unknown;
  blocked?: boolean;
  reason?: string;
  metadata?: Record<string, unknown>;
  startedAt?: string;
  completedAt?: string;
};

export type SubagentTaskView = {
  id: string;
  description: string;
  subagentType: string;
  status: "running" | "completed" | "failed" | "blocked";
  result?: string;
  error?: string;
  reason?: string;
  metadata?: Record<string, unknown>;
  readPaths?: string[];
};

export type RunViewState = {
  status: RunStatus;
  responseText: string;
  taskList: TaskListItem[];
  taskCount: number;
  activeTask: TaskListItem | null;
  planMode: boolean;
  parallelismDecision: ParallelismDecision | null;
  toolCalls: ToolCallView[];
  subagentTasks: SubagentTaskView[];
  rawEvents: AgentEvent[];
  lastError?: string;
};

export type ComposerSettings = {
  planMode: boolean;
  memoryEnabled: boolean;
  conversationHistoryEnabled: boolean;
};
