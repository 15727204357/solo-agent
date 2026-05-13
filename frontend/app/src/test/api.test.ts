import { describe, expect, it } from "vitest";
import { buildCreateRunPayload } from "../api";

describe("buildCreateRunPayload", () => {
  it("uses plan run_mode when planMode is true", () => {
    const payload = buildCreateRunPayload("plan it", {
      planMode: true,
      subagentEnabled: false,
      memoryEnabled: true,
      conversationHistoryEnabled: true,
    });

    expect(payload.run_mode).toBe("plan");
  });

  it("uses agent run_mode when planMode is false", () => {
    const payload = buildCreateRunPayload("do it", {
      planMode: false,
      subagentEnabled: false,
      memoryEnabled: true,
      conversationHistoryEnabled: true,
    });

    expect(payload.run_mode).toBe("agent");
  });

  it("includes subagent_enabled only when enabled", () => {
    const enabled = buildCreateRunPayload("parallel task", {
      planMode: false,
      subagentEnabled: true,
      memoryEnabled: false,
      conversationHistoryEnabled: false,
    });
    const disabled = buildCreateRunPayload("serial task", {
      planMode: false,
      subagentEnabled: false,
      memoryEnabled: false,
      conversationHistoryEnabled: false,
    });

    expect(enabled.subagent_enabled).toBe(true);
    expect(disabled).not.toHaveProperty("subagent_enabled");
  });
});
