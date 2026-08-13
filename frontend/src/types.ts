export type AgentRunStatus =
  | "success"
  | "clarification_required"
  | "error"
  | "partial";

export type StreamEventType =
  | "node_update"
  | "tool_update"
  | "status_update"
  | "final_answer"
  | "complete"
  | "error";

export interface AgentRunResponse {
  request_id: string;
  status: AgentRunStatus;
  final_answer: string;
  extracted_content: Array<Record<string, unknown>>;
  trace: Array<Record<string, unknown>>;
  evidence: Array<Record<string, unknown>>;
  errors: string[];
}

export interface AgentStreamEvent {
  type: StreamEventType;
  request_id?: string;
  node?: string;
  tool?: string;
  status?: string;
  message?: string;
  latency_ms?: number;
  final_answer?: string;
  response?: AgentRunResponse;
  errors?: string[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "agent";
  query: string;
  files: string[];
  response?: AgentRunResponse;
  errorMessages?: string[];
}

export class ApiError extends Error {
  status: number;
  details: string[];

  constructor(status: number, details: string[]) {
    super(details.join(" "));
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}
