import { AgentRunResponse, AgentStreamEvent, ApiError } from "../types";

function normalizeDetail(detail: unknown): string[] {
  if (Array.isArray(detail)) {
    return detail.map(String);
  }
  if (typeof detail === "string") {
    return [detail];
  }
  return ["Request failed."];
}

function parseSseChunk(chunk: string): AgentStreamEvent[] {
  const events: AgentStreamEvent[] = [];
  for (const line of chunk.split("\n")) {
    if (!line.startsWith("data: ")) {
      continue;
    }
    try {
      events.push(JSON.parse(line.slice(6)) as AgentStreamEvent);
    } catch {
      continue;
    }
  }
  return events;
}

export async function runAgentStream(
  query: string,
  files: File[],
  onEvent: (event: AgentStreamEvent) => void,
): Promise<AgentRunResponse> {
  const formData = new FormData();
  formData.append("query", query.trim());
  files.forEach((file) => {
    formData.append("files", file);
  });

  const response = await fetch("/api/v1/agent/run/stream", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let payload: unknown = null;
    try {
      payload = await response.json();
    } catch {
      throw new ApiError(response.status, [`Request failed with status ${response.status}.`]);
    }
    const detail =
      payload && typeof payload === "object" && "detail" in payload
        ? (payload as { detail: unknown }).detail
        : "Request failed.";
    throw new ApiError(response.status, normalizeDetail(detail));
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new ApiError(response.status, ["Streaming response body was empty."]);
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let finalResponse: AgentRunResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";

    for (const part of parts) {
      for (const event of parseSseChunk(part)) {
        onEvent(event);
        if (event.type === "complete" && event.response) {
          finalResponse = event.response;
        }
      }
    }
  }

  if (buffer.trim()) {
    for (const event of parseSseChunk(buffer)) {
      onEvent(event);
      if (event.type === "complete" && event.response) {
        finalResponse = event.response;
      }
    }
  }

  if (!finalResponse) {
    throw new ApiError(response.status, ["Stream ended without a final response."]);
  }

  return finalResponse;
}

export async function runAgent(
  query: string,
  files: File[],
): Promise<AgentRunResponse> {
  const formData = new FormData();
  formData.append("query", query.trim());
  files.forEach((file) => {
    formData.append("files", file);
  });

  const response = await fetch("/api/v1/agent/run", {
    method: "POST",
    body: formData,
  });

  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    if (!response.ok) {
      throw new ApiError(response.status, [`Request failed with status ${response.status}.`]);
    }
    throw new ApiError(response.status, ["Received an invalid response from the server."]);
  }

  if (!response.ok) {
    const detail =
      payload && typeof payload === "object" && "detail" in payload
        ? (payload as { detail: unknown }).detail
        : "Request failed.";
    throw new ApiError(response.status, normalizeDetail(detail));
  }

  return payload as AgentRunResponse;
}
