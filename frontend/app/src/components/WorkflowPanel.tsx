import { ChevronDown, Clipboard, PanelRightClose, PanelRightOpen, Search } from "lucide-react";
import type { ReactNode } from "react";
import { useMemo, useState } from "react";
import type { AgentEvent, RunViewState, TaskListItem } from "../types";

type WorkflowPanelProps = {
  state: RunViewState;
  collapsed: boolean;
  onToggleCollapsed: () => void;
};

const tabs = ["Plan", "Tools", "Subagents", "Parallelism", "Raw Events"] as const;
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
          <p className="mt-2 text-xs text-slate-500">Scoped Task · {task.subagentType}</p>
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
