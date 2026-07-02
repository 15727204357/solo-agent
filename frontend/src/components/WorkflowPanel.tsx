import { ChevronDown, Clipboard, PanelRightClose, PanelRightOpen, Search } from "lucide-react";
import type { ReactNode } from "react";
import { useMemo, useState } from "react";
import type { AgentEvent, IntentRoutePlanView, PatchProposalView, RunViewState, TaskListItem } from "../types";

type WorkflowPanelProps = {
  state: RunViewState;
  collapsed: boolean;
  onToggleCollapsed: () => void;
};

const tabs = ["Plan", "Route", "Patch Gate", "Tools", "Subagents", "Parallelism", "Raw Events"] as const;
type Tab = (typeof tabs)[number];

export function WorkflowPanel({ state, collapsed, onToggleCollapsed }: WorkflowPanelProps) {
  const [activeTab, setActiveTab] = useState<Tab>("Plan");
  const [typeFilter, setTypeFilter] = useState("");
  const [nodeFilter, setNodeFilter] = useState("");
  const [search, setSearch] = useState("");

  const filteredEvents = useMemo(() => {
    return state.rawEvents.filter((event) => {
      const node = String(event.payload?.node || "");
      const raw = JSON.stringify(event).toLowerCase();
      return (
        (!typeFilter || event.type.includes(typeFilter)) &&
        (!nodeFilter || node.includes(nodeFilter)) &&
        (!search || raw.includes(search.toLowerCase()))
      );
    });
  }, [nodeFilter, search, state.rawEvents, typeFilter]);

  if (collapsed) {
    return (
      <aside className="hidden border-l border-slate-200 bg-white/70 p-2 dark:border-slate-800 dark:bg-slate-950/70 lg:block">
        <button className="icon-button" type="button" onClick={onToggleCollapsed} title="展开工作流面板">
          <PanelRightOpen size={18} />
        </button>
      </aside>
    );
  }

  return (
    <aside className="workflow-panel">
      <header className="flex items-center justify-between gap-3 border-b border-slate-200 px-4 py-3 dark:border-slate-800">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">Workflow</p>
          <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">运行检查器</h2>
        </div>
        <button className="icon-button" type="button" onClick={onToggleCollapsed} title="折叠工作流面板">
          <PanelRightClose size={18} />
        </button>
      </header>

      <div className="flex gap-1 overflow-x-auto border-b border-slate-200 px-3 py-2 dark:border-slate-800">
        {tabs.map((tab) => (
          <button
            key={tab}
            type="button"
            className={`tab-button ${activeTab === tab ? "tab-active" : ""}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab}
          </button>
        ))}
      </div>

      <div className="min-h-0 flex-1 overflow-auto p-4">
        {activeTab === "Plan" && <PlanPanel state={state} />}
        {activeTab === "Route" && <RoutePanel route={state.intentRoute} history={state.routeHistory} rerouteRequests={state.routeRerouteRequests} />}
        {activeTab === "Patch Gate" && <PatchGatePanel proposal={state.patchProposal} />}
        {activeTab === "Tools" && <ToolsPanel state={state} />}
        {activeTab === "Subagents" && <SubagentsPanel state={state} />}
        {activeTab === "Parallelism" && <ParallelismPanel state={state} />}
        {activeTab === "Raw Events" && (
          <RawEventsPanel
            events={filteredEvents}
            typeFilter={typeFilter}
            nodeFilter={nodeFilter}
            search={search}
            onTypeFilter={setTypeFilter}
            onNodeFilter={setNodeFilter}
            onSearch={setSearch}
          />
        )}
      </div>
    </aside>
  );
}

function PlanPanel({ state }: { state: RunViewState }) {
  const visibleTasks = state.taskList.filter((task) => task.status !== "deleted");
  return (
    <div className="space-y-4">
      <div className="info-strip">
        <span>计划模式</span>
        <strong>{state.planMode ? "已启用" : "未启用"}</strong>
      </div>
      <div className="grid grid-cols-2 gap-2 text-sm">
        <Metric label="task_count" value={String(state.taskCount || visibleTasks.length)} />
        <Metric label="active_task" value={taskTitle(state.activeTask) || "-"} />
      </div>
      {!visibleTasks.length ? (
        <EmptyState text={state.planMode ? "等待 Agent 创建任务列表" : "Agent 模式未启用计划模式"} />
      ) : (
        <div className="space-y-2">
          {visibleTasks.map((task, index) => (
            <TaskItem key={task.id || `${taskTitle(task)}-${index}`} task={task} />
          ))}
        </div>
      )}
    </div>
  );
}

function RoutePanel({
  route,
  history,
  rerouteRequests,
}: {
  route: IntentRoutePlanView | null;
  history: IntentRoutePlanView[];
  rerouteRequests: unknown[];
}) {
  if (!route) {
    return <EmptyState text="Waiting for intent_route_completed event" />;
  }
  const scopes = asArray(route.searched_scopes);
  const contextScopes = recordArray(route.context_plan, "scopes");
  const selectedTools = recordArray(route.tool_plan, "selected_tools");
  const rejectedTools = recordArray(route.tool_plan, "rejected_tools");
  const tools = selectedTools.length ? selectedTools : asArray(route.tool_candidates);
  const skills = recordArray(route.skill_plan, "candidates").length ? recordArray(route.skill_plan, "candidates") : asArray(route.skill_candidates);
  const recipes = recordArray(route.recipe_plan, "candidates").length ? recordArray(route.recipe_plan, "candidates") : asArray(route.recipe_candidates);
  const alternatives = asArray(route.intent_alternatives);
  const decisionTrace = asArray(route.decision_trace);
  const rerouteTriggers = asArray(route.reroute_triggers);
  const actions = asArray(route.next_actions);
  const risk = route.risk_summary || {};
  const approval = route.approval_plan || {};
  const advisor = route.model_advisor || {};
  return (
    <div className="space-y-4">
      <div className="rounded-md border border-slate-200 bg-white p-3 text-sm dark:border-slate-800 dark:bg-slate-950">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h3 className="font-semibold text-slate-900 dark:text-slate-100">{String(route.intent || "unknown")}</h3>
            <p className="mt-1 text-xs text-slate-500">{String(risk.boundary || approval.approval_boundary || "No routing boundary recorded")}</p>
            <p className="mt-1 font-mono text-[11px] text-slate-400">
              {String(route.route_plan_schema_version || "v1")} · epoch {String(route.route_epoch ?? 0)} · {String(route.route_id || "route")}
            </p>
          </div>
          <span className="status-chip">{formatConfidence(route.confidence)}</span>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2 text-sm">
        <Metric label="context_scopes" value={String(contextScopes.length || scopes.length)} />
        <Metric label="tools" value={String(tools.length)} />
        <Metric label="skills" value={String(skills.length)} />
        <Metric label="recipes" value={String(recipes.length)} />
        <Metric label="alternatives" value={String(alternatives.length)} />
        <Metric label="reroutes" value={String(Math.max(0, history.length - 1) + rerouteRequests.length)} />
        <Metric label="max_risk" value={String(risk.max_risk_level || "-")} />
        <Metric label="approval" value={String(Boolean(risk.requires_approval || approval.requires_approval))} />
      </div>
      {alternatives.length ? <CandidateList title="intent alternatives" candidates={alternatives} /> : null}
      {contextScopes.length ? <CandidateList title="context plan" candidates={contextScopes} /> : <ChipList title="searched scopes" items={scopes} />}
      <CandidateList title="tool candidates" candidates={tools} />
      {rejectedTools.length ? <CandidateList title="rejected tools" candidates={rejectedTools} /> : null}
      {skills.length ? <CandidateList title="skill candidates" candidates={skills} /> : null}
      {recipes.length ? <CandidateList title="recipe candidates" candidates={recipes} /> : null}
      {decisionTrace.length ? <CandidateList title="decision trace" candidates={decisionTrace} /> : null}
      {rerouteTriggers.length ? <CandidateList title="reroute triggers" candidates={rerouteTriggers} /> : null}
      {history.length > 1 ? <CandidateList title="reroute history" candidates={history.map((item) => ({ route_epoch: item.route_epoch, intent: item.intent, confidence: item.confidence, route_id: item.route_id }))} /> : null}
      {rerouteRequests.length ? <JsonDetails title="reroute requests" value={rerouteRequests} /> : null}
      <ChipList title="next actions" items={actions} />
      {Object.keys(advisor).length ? <JsonDetails title="model advisor" value={advisor} /> : null}
      <JsonDetails title="route JSON" value={route} />
    </div>
  );
}

function TaskItem({ task }: { task: TaskListItem }) {
  const statusClass =
    task.status === "in_progress"
      ? "border-signal bg-cyan-50 text-slate-900 dark:bg-cyan-950/40"
      : task.status === "completed"
        ? "border-slate-200 bg-slate-50 text-slate-500 opacity-75 dark:border-slate-800 dark:bg-slate-900"
        : task.status === "blocked"
          ? "border-red-300 bg-red-50 text-red-900 dark:border-red-900 dark:bg-red-950/30 dark:text-red-200"
          : "border-slate-200 bg-white text-slate-700 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-200";
  return (
    <article className={`rounded-md border p-3 ${statusClass}`}>
      <div className="flex items-start justify-between gap-3">
        <h3 className="text-sm font-medium">{taskTitle(task)}</h3>
        <span className="status-chip">{task.status}</span>
      </div>
      {task.notes ? <p className="mt-2 text-xs opacity-80">{task.notes}</p> : null}
    </article>
  );
}

function ParallelismPanel({ state }: { state: RunViewState }) {
  const decision = state.parallelismDecision;
  if (!decision) {
    return <EmptyState text="等待 parallelism_decision_completed 事件" />;
  }
  const suitable = Boolean(decision.suitable ?? decision.allowed);
  const subagentEnabled = Boolean(decision.subagent_enabled);
  const subagentPolicy = String(decision.subagent_policy || (state.planMode ? "auto" : "off"));
  let headline = String(decision.reason || "串行执行");
  if (subagentPolicy === "off") {
    headline = "子代理策略关闭";
  } else if (state.planMode) {
    headline = "Auto 子代理策略已启用，是否执行由并行门控决定";
  } else if (suitable && !subagentEnabled) {
    headline = "任务适合并行，但子代理工具未启用";
  } else if (suitable && subagentEnabled) {
    headline = "主 Agent 可以选择 task 工具";
  }
  return (
    <div className="space-y-4">
      <div className="rounded-md border border-slate-200 bg-white p-3 text-sm dark:border-slate-800 dark:bg-slate-950">
        <p className="font-medium text-slate-900 dark:text-slate-100">{headline}</p>
        {!suitable ? <p className="mt-1 text-slate-500">{String(decision.reason || "不适合并行")}</p> : null}
      </div>
      <div className="grid grid-cols-2 gap-2 text-sm">
        <Metric label="strategy" value={String(decision.strategy || "-")} />
        <Metric label="suitable" value={String(suitable)} />
        <Metric label="task_count" value={String(decision.task_count ?? 0)} />
        <Metric label="subagent_policy" value={subagentPolicy} />
        <Metric label="subagent_enabled" value={String(subagentEnabled)} />
      </div>
      <JsonDetails title="candidates" value={decision.candidates || []} />
      <JsonDetails title="decision JSON" value={decision} />
    </div>
  );
}

function PatchGatePanel({ proposal }: { proposal: PatchProposalView | null }) {
  if (!proposal) {
    return <EmptyState text="Waiting for a patch approval event" />;
  }
  const plan = proposal.verification_plan || {};
  const gate = proposal.stop_gate || {};
  const commands = Array.isArray(plan.commands) ? plan.commands : [];
  const missingEvidence = Array.isArray(gate.missing_evidence) ? gate.missing_evidence : [];
  const status = String(gate.status || "missing");
  const approvalReady = Boolean(gate.approval_ready);
  const statusClass =
    status === "passed"
      ? "chip-success"
      : status === "failed"
        ? "chip-danger"
        : status === "waived"
          ? "chip-warn"
          : "chip-warn";

  return (
    <div className="space-y-4">
      <div className="rounded-md border border-slate-200 bg-white p-3 text-sm dark:border-slate-800 dark:bg-slate-950">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h3 className="font-semibold text-slate-900 dark:text-slate-100">{proposal.summary || proposal.id || "Patch proposal"}</h3>
            <p className="mt-1 text-xs text-slate-500">{String(gate.reason || plan.reason || "No gate reason recorded")}</p>
          </div>
          <span className={`status-chip ${statusClass}`}>{status}</span>
        </div>
        {!approvalReady ? (
          <p className="mt-3 rounded-md border border-amber-200 bg-amber-50 p-2 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950/30 dark:text-amber-200">
            Gate is not approval-ready. Run the planned verification or record an explicit waiver.
          </p>
        ) : null}
      </div>
      <div className="grid grid-cols-2 gap-2 text-sm">
        <Metric label="approval_ready" value={String(approvalReady)} />
        <Metric label="required" value={String(plan.required ?? true)} />
        <Metric label="commands" value={String(commands.length)} />
        <Metric label="patch_status" value={String(proposal.status || "-")} />
      </div>
      <div className="space-y-2">
        {commands.length ? (
          commands.map((command, index) => (
            <article key={`${command.command || "command"}-${index}`} className="rounded-md border border-slate-200 bg-white p-3 text-sm dark:border-slate-800 dark:bg-slate-950">
              <div className="flex items-start justify-between gap-3">
                <code className="break-words text-xs text-slate-800 dark:text-slate-100">{String(command.command || command.tool || "verification")}</code>
                <span className="status-chip">{String(command.tool || "command")}</span>
              </div>
              {command.target ? <p className="mt-2 text-xs text-slate-500">target: {String(command.target)}</p> : null}
              {command.purpose ? <p className="mt-1 text-xs text-slate-500">{String(command.purpose)}</p> : null}
            </article>
          ))
        ) : (
          <EmptyState text="No verification command is required for this patch" />
        )}
      </div>
      {missingEvidence.length ? <JsonDetails title="missing evidence" value={missingEvidence} /> : null}
      <JsonDetails title="patch gate JSON" value={{ verification_plan: plan, stop_gate: gate, verification: proposal.verification || null }} />
    </div>
  );
}

function ToolsPanel({ state }: { state: RunViewState }) {
  if (!state.toolCalls.length) {
    return <EmptyState text="等待工具调用事件" />;
  }
  return (
    <div className="space-y-3">
      {state.toolCalls.map((call) => (
        <article key={call.id} className="rounded-md border border-slate-200 bg-white p-3 dark:border-slate-800 dark:bg-slate-950">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="font-mono text-sm font-semibold text-slate-900 dark:text-slate-100">{call.name}</h3>
              {call.reason ? <p className="mt-1 text-xs text-amber-700 dark:text-amber-300">{call.reason}</p> : null}
            </div>
            <span className={`status-chip ${call.status === "blocked" ? "chip-warn" : ""}`}>{call.status}</span>
          </div>
          {call.metadata?.truncated ? <p className="mt-2 text-xs text-amber-700 dark:text-amber-300">结果已截断</p> : null}
          <JsonDetails title="arguments" value={call.arguments || {}} />
          <JsonDetails title="result" value={call.result || {}} />
        </article>
      ))}
    </div>
  );
}

function SubagentsPanel({ state }: { state: RunViewState }) {
  if (!state.subagentTasks.length) {
    return <EmptyState text="等待 Scoped Task / Subagent Task 事件" />;
  }
  return (
    <div className="space-y-3">
      {state.subagentTasks.map((task) => (
        <article key={task.id} className="rounded-md border border-slate-200 bg-white p-3 dark:border-slate-800 dark:bg-slate-950">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">{task.description}</h3>
              <p className="mt-1 font-mono text-xs text-slate-500">{task.id}</p>
            </div>
            <span className={`status-chip ${task.status === "failed" ? "chip-danger" : task.status === "blocked" ? "chip-warn" : ""}`}>{task.status}</span>
          </div>
          <p className="mt-2 text-xs text-slate-500">Scoped Task · {task.subagentType} · {subagentModeLabel(task.metadata)}</p>
          {task.result ? <p className="mt-3 whitespace-pre-wrap text-sm text-slate-700 dark:text-slate-200">{task.result}</p> : null}
          {task.reason ? <p className="mt-3 text-sm text-amber-700 dark:text-amber-300">{task.reason}</p> : null}
          {task.error ? <p className="mt-3 text-sm text-red-600 dark:text-red-300">{task.error}</p> : null}
        </article>
      ))}
    </div>
  );
}

type RawEventsProps = {
  events: AgentEvent[];
  typeFilter: string;
  nodeFilter: string;
  search: string;
  onTypeFilter: (value: string) => void;
  onNodeFilter: (value: string) => void;
  onSearch: (value: string) => void;
};

function RawEventsPanel({ events, typeFilter, nodeFilter, search, onTypeFilter, onNodeFilter, onSearch }: RawEventsProps) {
  return (
    <div className="space-y-3">
      <div className="grid gap-2">
        <FilterInput icon={<Search size={14} />} value={search} placeholder="搜索 JSON" onChange={onSearch} />
        <div className="grid grid-cols-2 gap-2">
          <FilterInput value={typeFilter} placeholder="type 过滤" onChange={onTypeFilter} />
          <FilterInput value={nodeFilter} placeholder="node 过滤" onChange={onNodeFilter} />
        </div>
      </div>
      <p className="text-xs text-slate-500">显示最近 {events.length} 条匹配事件，原始缓存最多 300 条。</p>
      <div className="space-y-2">
        {events.map((event, index) => (
          <details key={`${event.sequence || index}-${event.type}`} className="rounded-md border border-slate-200 bg-white p-2 dark:border-slate-800 dark:bg-slate-950">
            <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-xs">
              <span className="font-mono font-semibold text-slate-800 dark:text-slate-100">{event.type}</span>
              <span className="text-slate-500">{String(event.payload?.node || "")}</span>
            </summary>
            <div className="mt-2 flex justify-end">
              <button
                type="button"
                className="mini-button"
                onClick={() => navigator.clipboard?.writeText(JSON.stringify(event, null, 2))}
              >
                <Clipboard size={13} />
                复制 JSON
              </button>
            </div>
            <pre className="mt-2 max-h-80 overflow-auto rounded bg-slate-950 p-3 text-xs text-slate-100">
              {JSON.stringify(event, null, 2)}
            </pre>
          </details>
        ))}
      </div>
    </div>
  );
}

function ChipList({ title, items }: { title: string; items: unknown[] }) {
  if (!items.length) {
    return <EmptyState text={`No ${title} recorded`} />;
  }
  return (
    <div>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">{title}</h3>
      <div className="flex flex-wrap gap-2">
        {items.map((item, index) => (
          <span key={`${String(item)}-${index}`} className="status-chip">
            {String(item)}
          </span>
        ))}
      </div>
    </div>
  );
}

function CandidateList({ title, candidates }: { title: string; candidates: unknown[] }) {
  if (!candidates.length) {
    return <EmptyState text={`No ${title} recorded`} />;
  }
  return (
    <div className="space-y-2">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">{title}</h3>
      {candidates.map((candidate, index) => {
        const item = isRecord(candidate) ? candidate : { value: candidate };
        return (
          <article key={`${String(item.name || item.id || item.value || title)}-${index}`} className="rounded-md border border-slate-200 bg-white p-3 text-sm dark:border-slate-800 dark:bg-slate-950">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h4 className="font-mono text-xs font-semibold text-slate-900 dark:text-slate-100">
                  {String(item.name || item.id || item.value || "candidate")}
                </h4>
                {item.reason || item.recommendation_reason ? (
                  <p className="mt-1 text-xs text-slate-500">{String(item.reason || item.recommendation_reason)}</p>
                ) : null}
              </div>
              <span className="status-chip">{String(item.risk_level || item.category || item.run_policy || "-")}</span>
            </div>
            {item.confidence ? <p className="mt-2 text-xs text-slate-500">confidence: {formatConfidence(item.confidence)}</p> : null}
            <JsonDetails title="candidate JSON" value={item} />
          </article>
        );
      })}
    </div>
  );
}

function JsonDetails({ title, value }: { title: string; value: unknown }) {
  return (
    <details className="mt-3 rounded-md border border-slate-200 bg-slate-50 p-2 text-xs dark:border-slate-800 dark:bg-slate-900/60">
      <summary className="flex cursor-pointer list-none items-center gap-2 font-medium text-slate-700 dark:text-slate-200">
        <ChevronDown size={14} />
        {title}
      </summary>
      <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap font-mono text-slate-600 dark:text-slate-300">
        {JSON.stringify(value, null, 2)}
      </pre>
    </details>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-slate-200 bg-white p-3 dark:border-slate-800 dark:bg-slate-950">
      <dt className="text-xs text-slate-500">{label}</dt>
      <dd className="mt-1 break-words font-medium text-slate-900 dark:text-slate-100">{value}</dd>
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return <div className="rounded-md border border-dashed border-slate-300 p-6 text-center text-sm text-slate-500 dark:border-slate-700">{text}</div>;
}

function FilterInput({ icon, value, placeholder, onChange }: { icon?: ReactNode; value: string; placeholder: string; onChange: (value: string) => void }) {
  return (
    <label className="flex items-center gap-2 rounded-md border border-slate-200 bg-white px-2 py-1.5 text-sm dark:border-slate-800 dark:bg-slate-950">
      {icon}
      <input
        className="min-w-0 flex-1 bg-transparent text-sm outline-none"
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}

function taskTitle(task?: TaskListItem | null) {
  if (!task) {
    return "";
  }
  return task.subject || task.title || task.description || task.id || "Untitled task";
}

function subagentModeLabel(metadata?: Record<string, unknown>) {
  const mode = String(metadata?.mode || "sync_readonly");
  if (mode === "sync_child_agent") {
    return "只读子代理分析";
  }
  if (mode === "sync_readonly") {
    return "只读证据收集";
  }
  return mode;
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function recordArray(value: unknown, key: string): unknown[] {
  return isRecord(value) ? asArray(value[key]) : [];
}

function formatConfidence(value: unknown): string {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "-";
  }
  return `${Math.round(numeric * 100)}%`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}
