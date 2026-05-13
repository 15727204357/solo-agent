import {
  CheckSquare,
  ChevronDown,
  Circle,
  Folder,
  History,
  Plus,
  Send,
  Settings,
  Square,
  StopCircle,
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import {
  buildCreateRunPayload,
  createRun,
  createSession,
  getHealth,
  getRunEventHistory,
  getSession,
  listMessages,
  listSessions,
} from "./api";
import { Markdown } from "./components/Markdown";
import { WorkflowPanel } from "./components/WorkflowPanel";
import { initialRunViewState, runEventReducer } from "./runReducer";
import type { AgentEvent, ChatMessage, ComposerSettings, RunRecord, Session, SessionDetail } from "./types";

const sseEventNames = [
  "started",
  "run_started",
  "receive_user_turn",
  "task_list_loaded",
  "task_list_updated",
  "task_list_skipped",
  "parallelism_decision_completed",
  "tool_selection_completed",
  "tool_call_started",
  "tool_call_completed",
  "tool_progress",
  "task_started",
  "task_completed",
  "task_failed",
  "response_started",
  "response_delta",
  "response_completed",
  "run_completed",
  "completed",
  "failed",
  "error",
  "cancelled",
  "patch_approval_required",
];

const defaultComposer: ComposerSettings = {
  planMode: false,
  subagentEnabled: false,
  memoryEnabled: true,
  conversationHistoryEnabled: true,
};

export function App() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSession, setActiveSession] = useState<SessionDetail | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [activeRun, setActiveRun] = useState<RunRecord | null>(null);
  const [workspacePath, setWorkspacePath] = useState("");
  const [prompt, setPrompt] = useState("");
  const [title, setTitle] = useState("");
  const [composer, setComposer] = useState<ComposerSettings>(defaultComposer);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [panelCollapsed, setPanelCollapsed] = useState(false);
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const [notice, setNotice] = useState("加载会话中...");
  const [runState, dispatchRun] = useReducer(runEventReducer, initialRunViewState);
  const sourceRef = useRef<EventSource | null>(null);

  const running = runState.status === "running" || runState.status === "queued";
  const latestRun = activeSession?.runs?.[0] || null;

  const refreshSessions = useCallback(async () => {
    const items = await listSessions();
    setSessions(items);
  }, []);

  useEffect(() => {
    getHealth()
      .then((data) => {
        setHealth(data);
        if (typeof data.workspace_root === "string") {
          setWorkspacePath((current) => current || data.workspace_root as string);
        }
      })
      .catch((error: Error) => setNotice(error.message));
    refreshSessions()
      .then(() => setNotice("准备就绪"))
      .catch((error: Error) => setNotice(error.message));
  }, [refreshSessions]);

  useEffect(() => {
    return () => sourceRef.current?.close();
  }, []);

  const applyEvent = useCallback((event: AgentEvent) => {
    dispatchRun({ type: "event", event });
    if (["run_completed", "completed", "failed", "error", "cancelled", "patch_approval_required"].includes(event.type)) {
      sourceRef.current?.close();
      sourceRef.current = null;
      refreshSessions().catch(() => undefined);
    }
  }, [refreshSessions]);

  const connectSse = useCallback((run: RunRecord) => {
    const streamUrl = run.stream_url || `/api/sessions/${run.session_id}/runs/${run.id}/events`;
    sourceRef.current?.close();
    const source = new EventSource(streamUrl);
    sourceRef.current = source;

    const handle = (message: MessageEvent) => {
      try {
        applyEvent(JSON.parse(message.data) as AgentEvent);
      } catch {
        setNotice("收到无法解析的事件");
      }
    };

    source.onmessage = handle;
    sseEventNames.forEach((name) => source.addEventListener(name, handle));
    source.onerror = () => {
      if (source.readyState !== EventSource.CLOSED) {
        setNotice("事件流暂时不可用，保留当前已恢复状态");
      }
    };
  }, [applyEvent]);

  const loadSession = useCallback(async (sessionId: string) => {
    sourceRef.current?.close();
    sourceRef.current = null;
    setNotice("正在加载会话历史...");
    const [detail, loadedMessages] = await Promise.all([getSession(sessionId), listMessages(sessionId)]);
    setActiveSession(detail);
    setMessages(loadedMessages);
    setWorkspacePath(detail.workspace_path || workspacePath);
    setTitle(detail.title || "");

    const run = detail.runs?.[0];
    setActiveRun(run || null);
    const planMode = run?.metadata?.run_mode === "plan";
    dispatchRun({ type: "reset", planMode, status: run?.status === "running" ? "running" : "idle" });
    if (run?.id) {
      const history = (await getRunEventHistory(detail.id, run.id)) as AgentEvent[];
      for (const event of history) {
        dispatchRun({ type: "event", event });
      }
      if (run.status === "running" || run.status === "queued") {
        connectSse(run);
      }
      setNotice(`已恢复最近一次运行的 ${history.length} 条事件`);
    } else {
      setNotice("会话已加载，可以继续追问");
    }
  }, [connectSse, workspacePath]);

  const startNewSession = () => {
    sourceRef.current?.close();
    sourceRef.current = null;
    setActiveSession(null);
    setActiveRun(null);
    setMessages([]);
    setTitle("");
    setPrompt("");
    dispatchRun({ type: "reset", planMode: composer.planMode });
    setNotice("新会话已准备好");
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const cleanPrompt = prompt.trim();
    if (!cleanPrompt || running) {
      return;
    }

    setNotice(activeSession ? "正在发送..." : "正在创建会话...");
    let session = activeSession;
    if (!session) {
      const created = await createSession(title.trim() || "New coding session", workspacePath.trim() || null);
      session = { ...created, runs: [], message_count: 0, summary: null };
      setActiveSession(session);
      setSessions((items) => [created, ...items]);
    }

    const payload = buildCreateRunPayload(cleanPrompt, composer);
    dispatchRun({ type: "reset", planMode: composer.planMode, status: "running" });
    setMessages((items) => [
      ...items,
      {
        id: `local-${Date.now()}`,
        role: "user",
        content: cleanPrompt,
        created_at: new Date().toISOString(),
      },
    ]);
    setPrompt("");

    try {
      const run = await createRun(session.id, payload);
      setActiveRun(run);
      setActiveSession((current) =>
        current ? { ...current, runs: [run, ...(current.runs || [])], message_count: current.message_count + 1 } : current,
      );
      setNotice("消息已发送，正在等待 Agent 事件");
      connectSse(run);
    } catch (error) {
      dispatchRun({ type: "event", event: { type: "failed", message: (error as Error).message } });
      setNotice((error as Error).message);
    }
  };

  const displayMessages = useMemo(() => {
    const hasAssistantForRun = Boolean(activeRun && messages.some((message) => message.run_id === activeRun.id && message.role === "assistant"));
    if (!activeRun || !runState.responseText || hasAssistantForRun) {
      return messages;
    }
    return [
      ...messages,
      {
        id: `assistant-${activeRun.id}`,
        role: "assistant",
        run_id: activeRun.id,
        content: runState.responseText,
        created_at: new Date().toISOString(),
      },
    ];
  }, [activeRun, messages, runState.responseText]);

  return (
    <div className="app-shell">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSession?.id || null}
        workspacePath={workspacePath}
        health={health}
        onWorkspacePath={setWorkspacePath}
        onNewSession={startNewSession}
        onSelectSession={(id) => loadSession(id).catch((error: Error) => setNotice(error.message))}
      />

      <main className="chat-column">
        <header className="workspace-header">
          <div className="min-w-0">
            <p className="text-xs font-medium uppercase text-slate-500">Solo Agent Workspace</p>
            <h1 className="truncate text-xl font-semibold text-slate-950 dark:text-slate-50">
              {activeSession?.title || "新会话"}
            </h1>
            <p className="truncate text-sm text-slate-500">{workspacePath || "默认工作区"}</p>
          </div>
          <div className="header-meta">
            <StatusPill status={runState.status} />
            <span className="hidden text-xs text-slate-500 sm:inline">{activeRun ? shortId(activeRun.id) : "尚未运行"}</span>
            <span className="hidden text-xs text-slate-500 xl:inline">{String(health?.model || health?.environment || "")}</span>
          </div>
        </header>

        <section className="message-list" aria-label="消息">
          {!displayMessages.length ? (
            <div className="empty-thread">
              <h2>开始一次真实工作流</h2>
              <p>用一个输入框描述任务；需要计划时勾选计划模式，Scoped Task 能力放在高级设置里。</p>
            </div>
          ) : (
            displayMessages.map((message) => <MessageBubble key={message.id} message={message} />)
          )}
        </section>

        <div className="composer-zone">
          <TaskListDock state={runState} />
          <form className="composer" onSubmit={submit}>
            {!activeSession ? (
              <input
                className="title-input"
                value={title}
                placeholder="会话标题"
                maxLength={120}
                onChange={(event) => setTitle(event.target.value)}
              />
            ) : null}
            <textarea
              value={prompt}
              rows={3}
              maxLength={8000}
              placeholder="输入任务或追问..."
              onChange={(event) => setPrompt(event.target.value)}
              onKeyDown={(event) => {
                if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
                  event.currentTarget.form?.requestSubmit();
                }
              }}
            />
            <div className="composer-actions">
              <label className="inline-flex items-center gap-2 rounded-md border border-slate-200 px-3 py-2 text-sm dark:border-slate-800">
                <input
                  type="checkbox"
                  checked={composer.planMode}
                  onChange={(event) => setComposer((current) => ({ ...current, planMode: event.target.checked }))}
                />
                <CheckSquare size={16} />
                计划模式
              </label>
              <button className="secondary-button" type="button" onClick={() => setAdvancedOpen((value) => !value)}>
                <Settings size={16} />
                高级设置
                <ChevronDown size={14} />
              </button>
              <div className="ml-auto flex items-center gap-2">
                <button className="secondary-button" type="button" disabled title="后端 stop API 尚未实现">
                  <StopCircle size={16} />
                  停止
                </button>
                <button className="primary-button" type="submit" disabled={running || !prompt.trim()}>
                  {running ? <Circle size={16} className="animate-pulse" /> : <Send size={16} />}
                  发送
                </button>
              </div>
            </div>
            {advancedOpen ? <AdvancedSettings composer={composer} onChange={setComposer} /> : null}
            <p className="text-xs text-slate-500" role="status">{notice}</p>
          </form>
        </div>
      </main>

      <WorkflowPanel state={runState} collapsed={panelCollapsed} onToggleCollapsed={() => setPanelCollapsed((value) => !value)} />
    </div>
  );
}

function Sidebar({
  sessions,
  activeSessionId,
  workspacePath,
  health,
  onWorkspacePath,
  onNewSession,
  onSelectSession,
}: {
  sessions: Session[];
  activeSessionId: string | null;
  workspacePath: string;
  health: Record<string, unknown> | null;
  onWorkspacePath: (value: string) => void;
  onNewSession: () => void;
  onSelectSession: (id: string) => void;
}) {
  return (
    <aside className="sidebar">
      <header className="brand">
        <div className="brand-mark">SA</div>
        <div className="min-w-0">
          <h2>Solo Agent</h2>
          <p>{String(health?.environment || "local")}</p>
        </div>
      </header>
      <button className="primary-button w-full" type="button" onClick={onNewSession}>
        <Plus size={16} />
        新建会话
      </button>
      <section className="sidebar-section">
        <div className="section-label">
          <Folder size={14} />
          Workspace
        </div>
        <input className="sidebar-input" value={workspacePath} onChange={(event) => onWorkspacePath(event.target.value)} />
      </section>
      <section className="sidebar-section min-h-0 flex-1">
        <div className="section-label">
          <History size={14} />
          会话列表
        </div>
        <div className="session-list">
          {sessions.map((session) => (
            <button
              key={session.id}
              type="button"
              className={`session-button ${activeSessionId === session.id ? "active" : ""}`}
              onClick={() => onSelectSession(session.id)}
            >
              <span className="truncate font-medium">{session.title}</span>
              <span className="truncate text-xs text-slate-500">{session.workspace_path || shortId(session.id)}</span>
            </button>
          ))}
        </div>
      </section>
    </aside>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <article className={`message-bubble ${isUser ? "message-user" : "message-agent"}`}>
      <div className="message-meta">
        <span>{isUser ? "你" : "Solo Agent"}</span>
        <span>{message.created_at ? new Date(message.created_at).toLocaleString() : ""}</span>
      </div>
      <Markdown content={message.content} />
    </article>
  );
}

function TaskListDock({ state }: { state: ReturnType<typeof runEventReducer> }) {
  const activeTasks = state.taskList.filter((task) => task.status !== "deleted");
  if (!activeTasks.length) {
    return null;
  }
  return (
    <div className="todo-dock">
      <div className="flex items-center gap-2 text-sm font-medium text-slate-700 dark:text-slate-200">
        <CheckSquare size={16} />
        TaskList
        <span className="status-chip">{activeTasks.length}</span>
      </div>
      <div className="mt-2 flex flex-wrap gap-2">
        {activeTasks.slice(0, 5).map((task) => (
          <span key={task.id || task.subject || task.title} className={`todo-pill todo-${task.status}`}>
            {task.status === "completed" ? <CheckSquare size={13} /> : <Square size={13} />}
            {task.subject || task.title || task.description || task.id}
          </span>
        ))}
      </div>
    </div>
  );
}

function AdvancedSettings({
  composer,
  onChange,
}: {
  composer: ComposerSettings;
  onChange: React.Dispatch<React.SetStateAction<ComposerSettings>>;
}) {
  return (
    <div className="advanced-settings">
      <ToggleRow
        label="启用 Scoped Task 工具"
        description="只作为高级能力开关；是否调用由后端 parallelism_gate/select_tools/execute_tools 决定。"
        checked={composer.subagentEnabled}
        onChange={(checked) => onChange((current) => ({ ...current, subagentEnabled: checked }))}
      />
      <ToggleRow
        label="Memory"
        description="继续传递 memory_enabled。"
        checked={composer.memoryEnabled}
        onChange={(checked) => onChange((current) => ({ ...current, memoryEnabled: checked }))}
      />
      <ToggleRow
        label="Conversation history"
        description="继续传递 conversation_history_enabled。"
        checked={composer.conversationHistoryEnabled}
        onChange={(checked) => onChange((current) => ({ ...current, conversationHistoryEnabled: checked }))}
      />
    </div>
  );
}

function ToggleRow({
  label,
  description,
  checked,
  onChange,
}: {
  label: string;
  description: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className="flex items-start justify-between gap-4 rounded-md border border-slate-200 bg-white p-3 dark:border-slate-800 dark:bg-slate-950">
      <span>
        <span className="block text-sm font-medium text-slate-900 dark:text-slate-100">{label}</span>
        <span className="mt-1 block text-xs text-slate-500">{description}</span>
      </span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
    </label>
  );
}

function StatusPill({ status }: { status: string }) {
  return <span className={`status-pill status-${status}`}>{status === "idle" ? "ready" : status}</span>;
}

function shortId(id: string) {
  return id.length <= 12 ? id : `${id.slice(0, 7)}...${id.slice(-4)}`;
}
