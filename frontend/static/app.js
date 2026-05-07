const dom = {
  form: document.querySelector("#session-form"),
  titleField: document.querySelector("#title-field"),
  newSession: document.querySelector("#new-session"),
  clearThread: document.querySelector("#clear-thread"),
  startRun: document.querySelector("#start-run"),
  formMessage: document.querySelector("#form-message"),
  workspacePath: document.querySelector("#workspace-path"),
  folderPicker: document.querySelector("#folder-picker"),
  folderHint: document.querySelector("#folder-hint"),
  workspaceList: document.querySelector("#workspace-list"),
  workspaceCount: document.querySelector("#workspace-count"),
  healthDot: document.querySelector("#health-dot"),
  healthText: document.querySelector("#health-text"),
  threadTitle: document.querySelector("#thread-title"),
  threadSubtitle: document.querySelector("#thread-subtitle"),
  threadBody: document.querySelector("#thread-body"),
  composerMode: document.querySelector("#composer-mode"),
  runStatus: document.querySelector("#run-status"),
  currentRun: document.querySelector("#current-run"),
  pipelineState: document.querySelector("#pipeline-state"),
  eventCount: document.querySelector("#event-count"),
  toolCount: document.querySelector("#tool-count"),
  planCount: document.querySelector("#plan-count"),
  responseCount: document.querySelector("#response-count"),
  planSummary: document.querySelector("#plan-summary"),
  toolList: document.querySelector("#tool-list"),
  events: document.querySelector("#events"),
  timelineState: document.querySelector("#timeline-state"),
  payloadType: document.querySelector("#payload-type"),
  rawCount: document.querySelector("#raw-count"),
  latestPayload: document.querySelector("#latest-payload"),
  rawEvents: document.querySelector("#raw-events"),
  memoryCount: document.querySelector("#memory-count"),
  memoryStatus: document.querySelector("#memory-status"),
  memoryRefresh: document.querySelector("#memory-refresh"),
  memoryList: document.querySelector("#memory-list"),
  memoryEntries: document.querySelector("#memory-entries"),
};

const state = {
  sessions: [],
  workspaces: [],
  activeWorkspace: "",
  activeSession: null,
  activeRun: null,
  threadMessages: [],
  monitorState: {
    stageEvents: 0,
    rawEvents: 0,
    toolCalls: 0,
    planChars: 0,
    responseChars: 0,
  },
  currentSource: null,
  activeAssistantMessage: null,
  memoryCandidates: [],
  memoryEntries: [],
};

const terminalTypes = new Set(["completed", "run_completed", "failed", "cancelled"]);
const debugOnlyEvents = new Set(["plan_delta", "response_delta"]);
const timelineEvents = new Set([
  "started",
  "run_started",
  "plan_started",
  "plan_completed",
  "context_started",
  "context_completed",
  "inspect_started",
  "inspect_completed",
  "tool_call_started",
  "tool_call_completed",
  "response_started",
  "response_completed",
  "persist_started",
  "persist_completed",
  "persist_snapshot_completed",
  "completed",
  "run_completed",
  "failed",
  "cancelled",
]);
const agentEventNames = [...timelineEvents, ...debugOnlyEvents];
const stageByEvent = {
  started: "received",
  run_started: "received",
  plan_started: "plan",
  plan_delta: "plan",
  plan_completed: "plan",
  context_started: "context",
  context_completed: "context",
  inspect_started: "inspect",
  inspect_completed: "inspect",
  tool_call_started: "tool",
  tool_call_completed: "tool",
  response_started: "response",
  response_delta: "response",
  response_completed: "response",
  persist_started: "done",
  persist_completed: "done",
  persist_snapshot_completed: "done",
  completed: "done",
  run_completed: "done",
  failed: "done",
  cancelled: "done",
};
const completedEvents = new Set([
  "started",
  "run_started",
  "plan_completed",
  "context_completed",
  "inspect_completed",
  "tool_call_completed",
  "response_completed",
  "persist_completed",
  "persist_snapshot_completed",
  "completed",
  "run_completed",
]);

function prettyJson(value) {
  return JSON.stringify(value ?? {}, null, 2);
}

function shortId(id) {
  return id ? `${id.slice(0, 8)}...${id.slice(-4)}` : "无";
}

function workspaceKey(path) {
  return (path && String(path).trim()) || "默认工作区";
}

function workspaceName(path) {
  const value = workspaceKey(path);
  if (value === "默认工作区") {
    return value;
  }
  return value.split(/[\\/]/).filter(Boolean).pop() || value;
}

function agentData(event) {
  const data = event.payload?.data;
  return data && typeof data === "object" ? data : {};
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `请求失败：${response.status}`);
  }
  return response.json();
}

function postJson(url, body) {
  return requestJson(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function patchJson(url, body) {
  return requestJson(url, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function setHealth(ok, text) {
  dom.healthDot.className = `status-dot ${ok ? "status-ok" : "status-bad"}`;
  dom.healthText.textContent = text;
}

function setRunStatus(status, label) {
  dom.runStatus.className = `status-pill status-${status}`;
  dom.runStatus.textContent = label;
}

function setStage(stage, status) {
  const item = document.querySelector(`[data-stage="${stage}"]`);
  if (!item) {
    return;
  }
  const labels = {
    waiting: "等待",
    running: "运行中",
    done: "完成",
    failed: "异常",
  };
  item.className = `stage-${status}`;
  item.querySelector("em").textContent = labels[status];
}

function resetPipeline() {
  document.querySelectorAll("[data-stage]").forEach((item) => {
    item.className = "stage-waiting";
    item.querySelector("em").textContent = "等待";
  });
  dom.pipelineState.textContent = "等待";
}

function updatePipeline(event) {
  const stage = stageByEvent[event.type];
  if (stage) {
    setStage(stage, completedEvents.has(event.type) ? "done" : "running");
    const stageLabel = document.querySelector(`[data-stage="${stage}"] span`)?.textContent;
    dom.pipelineState.textContent = stageLabel || "运行中";
  }

  if (event.type === "failed" || event.type === "cancelled") {
    document.querySelectorAll(".stage-running").forEach((item) => {
      item.className = "stage-failed";
      item.querySelector("em").textContent = "异常";
    });
    dom.pipelineState.textContent = "异常";
  }

  if (event.type === "completed" || event.type === "run_completed") {
    document.querySelectorAll(".stage-running").forEach((item) => {
      item.className = "stage-done";
      item.querySelector("em").textContent = "完成";
    });
    dom.pipelineState.textContent = "完成";
  }
}

function resetMonitor() {
  state.monitorState = {
    stageEvents: 0,
    rawEvents: 0,
    toolCalls: 0,
    planChars: 0,
    responseChars: 0,
  };
  dom.eventCount.textContent = "0";
  dom.toolCount.textContent = "0";
  dom.planCount.textContent = "0";
  dom.responseCount.textContent = "0";
  dom.timelineState.textContent = "0 条";
  dom.payloadType.textContent = "无事件";
  dom.rawCount.textContent = "0 raw";
  dom.latestPayload.textContent = "{}";
  dom.events.replaceChildren();
  dom.rawEvents.replaceChildren();
  dom.toolList.replaceChildren();
  dom.planSummary.textContent = "等待规划完成。";
  dom.planSummary.classList.add("empty-state");
  resetPipeline();
}

function createMessage(role, content, meta = "") {
  return {
    id: `msg-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    role,
    content,
    meta,
  };
}

function renderThread() {
  dom.threadBody.replaceChildren();
  if (!state.threadMessages.length) {
    const empty = document.createElement("article");
    const title = document.createElement("h3");
    const copy = document.createElement("p");
    empty.className = "empty-thread";
    title.textContent = "开始一次真实的项目问询";
    copy.textContent = "输入项目路径、会话标题和任务描述。运行后，助手回复会在这里流式输出，右侧只显示必要链路状态。";
    empty.append(title, copy);
    dom.threadBody.append(empty);
    return;
  }

  state.threadMessages.forEach((message) => dom.threadBody.append(createMessageNode(message)));
  dom.threadBody.scrollTop = dom.threadBody.scrollHeight;
}

function createMessageNode(message) {
  const article = document.createElement("article");
  const label = document.createElement("div");
  const labelText = document.createElement("span");
  const meta = document.createElement("span");
  const content = document.createElement("div");
  const roleLabel = {
    user: "你",
    assistant: "Solo Agent",
    plan: "计划摘要",
    system: "系统",
  };

  article.className = `message-card ${message.role}`;
  article.dataset.messageId = message.id;
  label.className = "message-label";
  labelText.textContent = roleLabel[message.role] || message.role;
  meta.textContent = message.meta || "";
  content.className = "message-content";
  if (!message.content) {
    content.classList.add("empty-state");
    content.textContent = "等待输出...";
  } else {
    content.textContent = message.content;
  }
  label.append(labelText, meta);
  article.append(label, content);
  return article;
}

function updateMessageContent(message, content) {
  message.content = content;
  const node = dom.threadBody.querySelector(`[data-message-id="${message.id}"] .message-content`);
  if (!node) {
    renderThread();
    return;
  }
  node.classList.toggle("empty-state", !content);
  node.textContent = content || "等待输出...";
  dom.threadBody.scrollTop = dom.threadBody.scrollHeight;
}

function appendMessageContent(message, chunk) {
  updateMessageContent(message, `${message.content || ""}${chunk || ""}`);
}

function renderMemoryInbox() {
  if (!dom.memoryList || !dom.memoryEntries) {
    return;
  }
  dom.memoryList.replaceChildren();
  dom.memoryEntries.replaceChildren();
  dom.memoryCount.textContent = `${state.memoryCandidates.length} pending`;
  dom.memoryStatus.textContent = state.memoryCandidates.length
    ? "Review, edit, approve, or reject before memory is published."
    : "No pending memory candidates.";

  if (!state.memoryCandidates.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "Inbox is clear.";
    dom.memoryList.append(empty);
  } else {
    state.memoryCandidates.forEach((candidate) => dom.memoryList.append(createMemoryCandidateNode(candidate)));
  }

  if (!state.memoryEntries.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No published memory entries yet.";
    dom.memoryEntries.append(empty);
  } else {
    state.memoryEntries.forEach((entry) => dom.memoryEntries.append(createMemoryEntryNode(entry)));
  }
}

function createMemoryCandidateNode(candidate) {
  const card = document.createElement("article");
  const header = document.createElement("div");
  const title = document.createElement("strong");
  const meta = document.createElement("span");
  const textarea = document.createElement("textarea");
  const details = document.createElement("p");
  const actions = document.createElement("div");
  const resolution = document.createElement("select");
  const approve = document.createElement("button");
  const reject = document.createElement("button");

  card.className = "memory-candidate";
  header.className = "memory-candidate-header";
  title.textContent = `${candidate.target || "memory"} candidate`;
  meta.textContent = `confidence ${Math.round(Number(candidate.confidence || 0) * 100)}%`;
  textarea.value = candidate.content || "";
  textarea.rows = candidate.target === "skill" ? 8 : 4;
  details.className = "memory-candidate-details";
  details.textContent = memoryCandidateDetails(candidate);
  actions.className = "memory-candidate-actions";

  ["add", "replace", "merge"].forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    resolution.append(option);
  });
  if (candidate.conflict_ids?.length) {
    resolution.value = "replace";
  }

  approve.className = "primary-button";
  approve.type = "button";
  approve.textContent = "Approve";
  approve.addEventListener("click", async () => {
    approve.disabled = true;
    try {
      await postJson(`/api/memory/candidates/${candidate.id}/approve`, {
        resolution: resolution.value,
        content: textarea.value.trim(),
      });
      dom.memoryStatus.textContent = "Memory candidate approved.";
      await loadMemoryInbox();
    } catch (error) {
      dom.memoryStatus.textContent = error.message;
    } finally {
      approve.disabled = false;
    }
  });

  reject.className = "secondary-button";
  reject.type = "button";
  reject.textContent = "Reject";
  reject.addEventListener("click", async () => {
    reject.disabled = true;
    try {
      await postJson(`/api/memory/candidates/${candidate.id}/reject`, { reason: "rejected in inbox" });
      dom.memoryStatus.textContent = "Memory candidate rejected.";
      await loadMemoryInbox();
    } catch (error) {
      dom.memoryStatus.textContent = error.message;
    } finally {
      reject.disabled = false;
    }
  });

  header.append(title, meta);
  actions.append(resolution, approve, reject);
  card.append(header, textarea, details, actions);
  return card;
}

function createMemoryEntryNode(entry) {
  const item = document.createElement("article");
  const header = document.createElement("div");
  const title = document.createElement("strong");
  const status = document.createElement("span");
  const content = document.createElement("p");
  const revoke = document.createElement("button");

  item.className = "memory-entry";
  header.className = "memory-candidate-header";
  title.textContent = entry.target || "memory";
  status.textContent = entry.status || "active";
  content.textContent = entry.content || "";
  revoke.className = "secondary-button";
  revoke.type = "button";
  revoke.textContent = "Revoke";
  revoke.addEventListener("click", async () => {
    revoke.disabled = true;
    try {
      await postJson(`/api/memory/entries/${entry.id}/revoke`, { reason: "revoked in inbox" });
      dom.memoryStatus.textContent = "Memory entry revoked.";
      await loadMemoryInbox();
    } catch (error) {
      dom.memoryStatus.textContent = error.message;
    } finally {
      revoke.disabled = false;
    }
  });
  header.append(title, status);
  item.append(header, content, revoke);
  return item;
}

function memoryCandidateDetails(candidate) {
  const parts = [];
  if (candidate.safety_flags?.length) {
    parts.push(`flags: ${candidate.safety_flags.join(", ")}`);
  }
  if (candidate.conflict_ids?.length) {
    parts.push(`conflicts: ${candidate.conflict_ids.length}`);
  }
  if (candidate.source_excerpt) {
    parts.push(`source: ${String(candidate.source_excerpt).slice(0, 160)}`);
  }
  return parts.join(" | ") || "No warnings.";
}

async function loadMemoryInbox() {
  if (!dom.memoryList) {
    return;
  }
  try {
    const [inbox, entries] = await Promise.all([
      requestJson("/api/memory/inbox?status=pending&limit=50"),
      requestJson("/api/memory/entries?limit=50"),
    ]);
    state.memoryCandidates = inbox.items || [];
    state.memoryEntries = entries.items || [];
    renderMemoryInbox();
  } catch (error) {
    dom.memoryStatus.textContent = error.message;
  }
}

function renderWorkspaces() {
  const grouped = new Map();
  state.sessions.forEach((session) => {
    const key = workspaceKey(session.workspace_path);
    if (!grouped.has(key)) {
      grouped.set(key, []);
    }
    grouped.get(key).push(session);
  });

  state.workspaces = [...grouped.keys()];
  dom.workspaceCount.textContent = String(state.workspaces.length);
  dom.workspaceList.replaceChildren();

  if (!state.workspaces.length) {
    const empty = document.createElement("div");
    empty.className = "workspace-group";
    const button = document.createElement("button");
    button.className = "workspace-button";
    button.type = "button";
    button.textContent = "暂无历史工作区";
    empty.append(button);
    dom.workspaceList.append(empty);
    return;
  }

  state.workspaces.forEach((workspace) => {
    const group = document.createElement("section");
    const workspaceButton = document.createElement("button");
    const name = document.createElement("span");
    const path = document.createElement("span");
    const list = document.createElement("ul");

    group.className = "workspace-group";
    workspaceButton.className = `workspace-button ${workspace === state.activeWorkspace ? "active" : ""}`;
    workspaceButton.type = "button";
    name.className = "workspace-name";
    path.className = "workspace-path";
    name.textContent = workspaceName(workspace);
    path.textContent = workspace;
    workspaceButton.append(name, path);
    workspaceButton.addEventListener("click", () => {
      state.activeWorkspace = workspace;
      if (workspace !== "默认工作区") {
        dom.workspacePath.value = workspace;
      }
      renderWorkspaces();
    });

    list.className = "session-list";
    grouped.get(workspace).forEach((session) => list.append(createSessionButton(session)));
    group.append(workspaceButton, list);
    dom.workspaceList.append(group);
  });
}

function createSessionButton(session) {
  const item = document.createElement("li");
  const button = document.createElement("button");
  const title = document.createElement("span");
  const meta = document.createElement("span");

  button.className = `session-button ${state.activeSession?.id === session.id ? "active" : ""}`;
  button.type = "button";
  title.className = "session-title";
  meta.className = "session-meta";
  title.textContent = session.title || "新的编码会话";
  meta.textContent = sessionMeta(session);
  button.append(title, meta);
  button.addEventListener("click", () => selectSession(session.id));
  item.append(button);
  return item;
}

function messageMeta(message) {
  if (message.created_at) {
    return new Date(message.created_at).toLocaleString("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  }
  return message.run_id ? shortId(String(message.run_id)) : "";
}

function sessionMeta(session) {
  if (typeof session.message_count === "number") {
    return `${shortId(session.id)} · ${session.message_count} 条消息`;
  }
  if (session.updated_at) {
    return `${shortId(session.id)} · ${new Date(session.updated_at).toLocaleDateString("zh-CN")}`;
  }
  return shortId(session.id);
}

function hydrateThreadFromMessages(messages) {
  state.threadMessages = messages
    .filter((message) => message.role === "user" || message.role === "assistant")
    .map((message) =>
      createMessage(
        message.role,
        String(message.content || ""),
        messageMeta(message),
      ),
    );
}

function hydrateThreadFromRuns(sessionDetail) {
  const runs = sessionDetail.runs || [];
  state.threadMessages = [];
  runs.slice().reverse().forEach((run) => {
    if (run.prompt) {
      state.threadMessages.push(createMessage("user", run.prompt, run.status || "历史运行"));
      const statusText = run.status === "completed" ? "已完成" : run.status || "历史";
      state.threadMessages.push(
        createMessage(
          "assistant",
          "该历史运行的完整回答未由当前接口返回；新的运行会在这里实时流式显示。",
          statusText,
        ),
      );
    }
  });
}

function updateSessionInCache(sessionDetail) {
  const index = state.sessions.findIndex((session) => session.id === sessionDetail.id);
  if (index >= 0) {
    state.sessions[index] = { ...state.sessions[index], ...sessionDetail };
  } else {
    state.sessions.unshift(sessionDetail);
  }
}

async function selectSession(sessionId) {
  closeStream();
  resetMonitor();
  setRunStatus("idle", "空闲");
  dom.currentRun.textContent = "尚未运行";
  dom.formMessage.textContent = "已加载历史会话，可以继续追问。";

  try {
    const detail = await requestJson(`/api/sessions/${sessionId}`);
    const messages = await requestJson(`/api/sessions/${sessionId}/messages?limit=200`);
    state.activeSession = detail;
    state.activeWorkspace = workspaceKey(detail.workspace_path);
    if (detail.workspace_path) {
      dom.workspacePath.value = detail.workspace_path;
    }
    updateSessionInCache(detail);
    if (messages.items?.length) {
      hydrateThreadFromMessages(messages.items);
    } else {
      hydrateThreadFromRuns(detail);
    }
    renderThread();
    renderWorkspaces();
    updateThreadHeader();
  } catch (error) {
    dom.formMessage.textContent = error.message;
  }
}

function updateThreadHeader() {
  if (state.activeSession) {
    dom.threadTitle.textContent = state.activeSession.title || "未命名会话";
    dom.threadSubtitle.textContent = state.activeSession.workspace_path || "默认工作区";
    dom.composerMode.textContent = "继续追问";
    dom.titleField.hidden = true;
    dom.form.classList.add("continue-mode");
    return;
  }
  dom.threadTitle.textContent = "新的编码会话";
  dom.threadSubtitle.textContent = "选择历史会话，或输入第一条消息开始对话。";
  dom.composerMode.textContent = "新会话";
  dom.titleField.hidden = false;
  dom.form.classList.remove("continue-mode");
}

function appendTimelineEvent(event) {
  if (!timelineEvents.has(event.type) || debugOnlyEvents.has(event.type)) {
    return;
  }
  state.monitorState.stageEvents += 1;
  dom.eventCount.textContent = String(state.monitorState.stageEvents);
  dom.timelineState.textContent = `${state.monitorState.stageEvents} 条`;

  const item = document.createElement("li");
  const type = document.createElement("span");
  const message = document.createElement("p");
  const payload = document.createElement("code");
  type.className = "event-type";
  type.textContent = eventLabel(event.type);
  message.textContent = event.message || event.type;
  payload.textContent = compactPayload(event);
  item.append(type, message, payload);
  dom.events.append(item);
}

function appendRawEvent(event) {
  state.monitorState.rawEvents += 1;
  dom.rawCount.textContent = `${state.monitorState.rawEvents} raw`;
  dom.payloadType.textContent = event.type;
  dom.latestPayload.textContent = prettyJson(event.payload);

  const item = document.createElement("li");
  const type = document.createElement("span");
  const message = document.createElement("p");
  const payload = document.createElement("code");
  type.className = "raw-type";
  type.textContent = event.type;
  message.textContent = event.message || "";
  payload.textContent = prettyJson(event.payload);
  item.append(type, message, payload);
  dom.rawEvents.append(item);
}

function eventLabel(type) {
  const labels = {
    started: "已接收",
    run_started: "开始",
    plan_started: "规划中",
    plan_completed: "规划完成",
    context_started: "上下文",
    context_completed: "上下文完成",
    inspect_started: "安全检查",
    inspect_completed: "检查完成",
    tool_call_started: "工具开始",
    tool_call_completed: "工具完成",
    response_started: "生成回复",
    response_completed: "回复完成",
    persist_started: "持久化",
    persist_completed: "持久化完成",
    persist_snapshot_completed: "快照完成",
    completed: "完成",
    run_completed: "完成",
    failed: "失败",
    cancelled: "取消",
  };
  return labels[type] || type;
}

function compactPayload(event) {
  const data = agentData(event);
  if (event.type === "plan_completed" && data.plan) {
    return prettyJson({ plan: data.plan });
  }
  if (event.type === "response_completed" && data.response) {
    return prettyJson({ response_chars: data.response.length });
  }
  if (event.type.includes("tool_call")) {
    return prettyJson({
      name: data.name,
      arguments: data.arguments,
      ok: data.result?.ok,
    });
  }
  if (Object.keys(data).length) {
    return prettyJson(data);
  }
  return prettyJson(event.payload);
}

function appendToolEvent(event) {
  if (!event.type.includes("tool_call")) {
    return;
  }
  const data = agentData(event);
  const item = document.createElement("li");
  const name = document.createElement("span");
  const message = document.createElement("p");
  const payload = document.createElement("code");
  state.monitorState.toolCalls += event.type === "tool_call_started" ? 1 : 0;
  dom.toolCount.textContent = String(state.monitorState.toolCalls);
  name.className = "tool-name";
  name.textContent = data.name || "tool";
  message.textContent = event.type === "tool_call_started" ? "开始调用" : "调用完成";
  payload.textContent = prettyJson(data.arguments || data.result || data);
  item.append(name, message, payload);
  dom.toolList.append(item);
}

function applyRunEvent(event) {
  const data = agentData(event);
  updatePipeline(event);
  appendRawEvent(event);
  appendTimelineEvent(event);
  appendToolEvent(event);

  if (event.type === "started" || event.type === "run_started") {
    setRunStatus("running", "运行中");
  }

  if (event.type === "plan_completed" && data.plan) {
    state.monitorState.planChars = data.plan.length;
    dom.planCount.textContent = String(data.plan.length);
    dom.planSummary.classList.remove("empty-state");
    dom.planSummary.textContent = data.plan;
    state.threadMessages.push(createMessage("plan", data.plan, "本次运行"));
    renderThread();
  }

  if (event.type === "response_started" && !state.activeAssistantMessage) {
    state.activeAssistantMessage = createMessage("assistant", "", "流式输出");
    state.threadMessages.push(state.activeAssistantMessage);
    renderThread();
  }

  if (event.type === "response_delta") {
    if (!state.activeAssistantMessage) {
      state.activeAssistantMessage = createMessage("assistant", "", "流式输出");
      state.threadMessages.push(state.activeAssistantMessage);
      renderThread();
    }
    appendMessageContent(state.activeAssistantMessage, event.message || "");
    state.monitorState.responseChars = state.activeAssistantMessage.content.length;
    dom.responseCount.textContent = String(state.monitorState.responseChars);
  }

  if (event.type === "response_completed" && data.response) {
    if (!state.activeAssistantMessage) {
      state.activeAssistantMessage = createMessage("assistant", "", "已完成");
      state.threadMessages.push(state.activeAssistantMessage);
      renderThread();
    }
    state.activeAssistantMessage.meta = "已完成";
    updateMessageContent(state.activeAssistantMessage, data.response);
    state.monitorState.responseChars = data.response.length;
    dom.responseCount.textContent = String(data.response.length);
  }

  if (terminalTypes.has(event.type)) {
    const labels = { completed: "已完成", run_completed: "已完成", failed: "失败", cancelled: "已取消" };
    setRunStatus(event.type, labels[event.type]);
    closeStream();
    state.activeAssistantMessage = null;
    loadSessions();
    loadMemoryInbox();
  }
}

function closeStream() {
  if (state.currentSource) {
    state.currentSource.close();
    state.currentSource = null;
  }
}

function connectStream(run) {
  closeStream();
  state.activeRun = run;
  state.activeAssistantMessage = null;
  resetMonitor();
  setRunStatus("running", "运行中");
  dom.currentRun.textContent = shortId(run.id);

  state.currentSource = new EventSource(run.stream_url);
  state.currentSource.addEventListener("message", (message) => applyRunEvent(JSON.parse(message.data)));
  agentEventNames.forEach((name) => {
    state.currentSource.addEventListener(name, (message) => applyRunEvent(JSON.parse(message.data)));
  });
  state.currentSource.onerror = () => {
    if (state.currentSource?.readyState === EventSource.CLOSED) {
      return;
    }
    dom.formMessage.textContent = "事件流暂时不可用。";
  };
}

function newSessionView() {
  closeStream();
  state.activeSession = null;
  state.activeRun = null;
  state.activeAssistantMessage = null;
  state.threadMessages = [];
  resetMonitor();
  setRunStatus("idle", "空闲");
  dom.currentRun.textContent = "尚未运行";
  dom.formMessage.textContent = "选择历史会话后可继续追问；新会话可先取一个标题。";
  updateThreadHeader();
  renderThread();
  renderWorkspaces();
}

async function submitRun(event) {
  event.preventDefault();
  const data = new FormData(dom.form);
  const title = String(data.get("title") || "新的编码会话").trim() || "新的编码会话";
  const prompt = String(data.get("prompt") || "").trim();
  const workspacePath = dom.workspacePath.value.trim();

  if (!prompt) {
    dom.formMessage.textContent = "请先输入消息。";
    return;
  }

  dom.startRun.disabled = true;
  dom.formMessage.textContent = state.activeSession ? "正在发送追问..." : "正在创建会话...";

  try {
    let session = state.activeSession;
    if (!session) {
      session = await postJson("/api/sessions", {
        title,
        workspace_path: workspacePath || null,
      });
      state.activeSession = session;
      state.activeWorkspace = workspaceKey(session.workspace_path);
      state.sessions.unshift(session);
    }

    state.threadMessages.push(createMessage("user", prompt, "刚刚"));
    renderThread();
    updateThreadHeader();
    renderWorkspaces();

    const run = await postJson(`/api/sessions/${session.id}/runs`, { prompt });
    dom.form.reset();
    if (workspacePath) {
      dom.workspacePath.value = workspacePath;
    }
    dom.formMessage.textContent = "消息已发送，等待 Agent 回复。";
    connectStream(run);
  } catch (error) {
    setRunStatus("failed", "失败");
    dom.formMessage.textContent = error.message;
  } finally {
    dom.startRun.disabled = false;
  }
}

async function loadHealth() {
  try {
    const health = await requestJson("/api/health");
    setHealth(Boolean(health.ok), `${health.service} 已就绪`);
    if (!dom.workspacePath.value && health.workspace_root) {
      dom.workspacePath.placeholder = health.workspace_root;
    }
  } catch (error) {
    setHealth(false, error.message);
  }
}

async function loadSessions() {
  try {
    const data = await requestJson("/api/sessions");
    state.sessions = data.items || [];
    if (!state.activeWorkspace && state.sessions[0]) {
      state.activeWorkspace = workspaceKey(state.sessions[0].workspace_path);
    }
    renderWorkspaces();
  } catch (error) {
    state.sessions = [];
    renderWorkspaces();
    dom.formMessage.textContent = error.message;
  }
}

async function chooseFolder() {
  if (!("showDirectoryPicker" in window)) {
    dom.folderHint.textContent = "当前浏览器不支持目录选择，请直接输入项目路径。";
    return;
  }

  try {
    const handle = await window.showDirectoryPicker();
    let previewCount = 0;
    for await (const _entry of handle.values()) {
      previewCount += 1;
      if (previewCount >= 20) {
        break;
      }
    }
    dom.folderHint.textContent = `已选择“${handle.name}”，预览到 ${previewCount} 个条目；真实路径仍需手动填写。`;
  } catch (error) {
    if (error.name !== "AbortError") {
      dom.folderHint.textContent = error.message;
    }
  }
}

dom.form.addEventListener("submit", submitRun);
dom.newSession.addEventListener("click", newSessionView);
dom.clearThread.addEventListener("click", newSessionView);
dom.folderPicker.addEventListener("click", chooseFolder);
dom.memoryRefresh?.addEventListener("click", loadMemoryInbox);

resetMonitor();
renderThread();
renderMemoryInbox();
setRunStatus("idle", "空闲");
loadHealth();
loadSessions();
loadMemoryInbox();
