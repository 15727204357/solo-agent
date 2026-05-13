import { describe, expect, it } from "vitest";
import { buildCreateRunPayload } from "../api";

describe("buildCreateRunPayload", () => {
  it("uses plan run_mode when planMode is true", () => {
    const payload = buildCreateRunPayload("plan it", {
      planMode: true,
      memoryEnabled: true,
      conversationHistoryEnabled: true,
    });

    expect(payload.run_mode).toBe("plan");
    expect(payload.subagent_policy).toBe("auto");
  });

  it("uses agent run_mode when planMode is false", () => {
    const payload = buildCreateRunPayload("do it", {
      planMode: false,
      memoryEnabled: true,
      conversationHistoryEnabled: true,
    });

    expect(payload.run_mode).toBe("agent");
    expect(payload.subagent_policy).toBe("off");
  });

  it("does not send the legacy subagent_enabled flag", () => {
    const payload = buildCreateRunPayload("parallel task", {
      planMode: true,
      memoryEnabled: false,
      conversationHistoryEnabled: false,
    });

    expect(payload).not.toHaveProperty("subagent_enabled");
  });

  it("passes memory and conversation history flags", () => {
    const payload = buildCreateRunPayload("remember this", {
      planMode: false,
      memoryEnabled: false,
      conversationHistoryEnabled: true,
    });

    expect(payload.memory_enabled).toBe(false);
    expect(payload.conversation_history_enabled).toBe(true);
  });
});
