import type { ChatMessage, ComposerSettings, RunRecord, Session, SessionDetail } from "./types";

export type CreateRunPayload = {
  prompt: string;
  run_mode: "agent" | "plan";
  memory_enabled: boolean;
  conversation_history_enabled: boolean;
  subagent_policy: "off" | "auto";
};

async function requestJson<T>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function buildCreateRunPayload(prompt: string, settings: ComposerSettings): CreateRunPayload {
  const payload: CreateRunPayload = {
    prompt,
    run_mode: settings.planMode ? "plan" : "agent",
    subagent_policy: settings.planMode ? "auto" : "off",
    memory_enabled: settings.memoryEnabled,
    conversation_history_enabled: settings.conversationHistoryEnabled,
  };
  return payload;
}

export async function getHealth(): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>("/api/health");
}

export async function listSessions(): Promise<Session[]> {
  const data = await requestJson<{ items: Session[] }>("/api/sessions");
  return data.items || [];
}

export async function createSession(title: string, workspacePath: string | null): Promise<Session> {
  return requestJson<Session>("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, workspace_path: workspacePath }),
  });
}

export async function getSession(sessionId: string): Promise<SessionDetail> {
  return requestJson<SessionDetail>(`/api/sessions/${sessionId}`);
}

export async function listMessages(sessionId: string, limit = 200): Promise<ChatMessage[]> {
  const data = await requestJson<{ items: ChatMessage[] }>(`/api/sessions/${sessionId}/messages?limit=${limit}`);
  return data.items || [];
}

export async function createRun(sessionId: string, payload: CreateRunPayload): Promise<RunRecord> {
  return requestJson<RunRecord & { stream_url: string }>(`/api/sessions/${sessionId}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function cancelRun(sessionId: string, runId: string): Promise<RunRecord> {
  return requestJson<RunRecord>(`/api/sessions/${sessionId}/runs/${runId}/cancel`, {
    method: "POST",
  });
}

export async function getRunEventHistory(sessionId: string, runId: string, limit = 1000) {
  const data = await requestJson<{ items: unknown[] }>(
    `/api/sessions/${sessionId}/runs/${runId}/events/history?limit=${limit}`,
  );
  return data.items;
}
