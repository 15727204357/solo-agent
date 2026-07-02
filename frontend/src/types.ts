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

export type IntentRoutePlanView = {
  route_plan_schema_version?: string;
  route_id?: string;
  route_epoch?: number;
  intent?: string;
  intent_alternatives?: unknown[];
  confidence?: number;
  matched_terms?: string[];
  searched_scopes?: string[];
  constraints?: Record<string, unknown>;
  context_plan?: Record<string, unknown>;
  tool_plan?: Record<string, unknown>;
  skill_plan?: Record<string, unknown>;
  recipe_plan?: Record<string, unknown>;
  approval_plan?: Record<string, unknown>;
  verification_plan?: Record<string, unknown>;
  decision_trace?: unknown[];
  reroute_triggers?: unknown[];
  model_advisor?: Record<string, unknown>;
  tool_candidates?: unknown[];
  proposed_tool_calls?: unknown[];
  skill_candidates?: unknown[];
  recipe_candidates?: unknown[];
  evidence?: unknown[];
  risk_summary?: Record<string, unknown>;
  next_actions?: string[];
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

export type VerificationCommandView = {
  command?: string;
  args?: string[];
  target?: string | null;
  tool?: string | null;
  purpose?: string;
  [key: string]: unknown;
};

export type VerificationPlanView = {
  commands?: VerificationCommandView[];
  required?: boolean;
  reason?: string;
  [key: string]: unknown;
};

export type StopGateView = {
  status?: "passed" | "failed" | "missing" | "waived" | string;
  approval_ready?: boolean;
  reason?: string;
  missing_evidence?: string[];
  [key: string]: unknown;
};

export type PatchProposalView = {
  id?: string;
  status?: string;
  summary?: string;
  diff?: string;
  verification_plan?: VerificationPlanView;
  stop_gate?: StopGateView;
  verification?: Record<string, unknown>;
  [key: string]: unknown;
};

export type RunViewState = {
  status: RunStatus;
  responseText: string;
  taskList: TaskListItem[];
  taskCount: number;
  activeTask: TaskListItem | null;
  planMode: boolean;
  intentRoute: IntentRoutePlanView | null;
  routeHistory: IntentRoutePlanView[];
  routeRerouteRequests: unknown[];
  parallelismDecision: ParallelismDecision | null;
  patchProposal: PatchProposalView | null;
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
