import type { ReactNode } from "react";

import { AgentRunResponse, AgentStreamEvent } from "../types";

interface ResultPanelsProps {
  response: AgentRunResponse | null;
  errorMessages: string[];
  streamEvents: AgentStreamEvent[];
  streamingFinalAnswer: string;
}

export function ResultPanels({
  response,
  errorMessages,
  streamEvents,
  streamingFinalAnswer,
}: ResultPanelsProps) {
  const displayErrors = [
    ...errorMessages,
    ...(response?.errors ?? []),
  ];
  const traceItems = buildTraceItems(streamEvents, response);
  const liveAnswer = streamingFinalAnswer || response?.final_answer || "";

  return (
    <section className="results-grid" aria-live="polite">
      <Panel title="Final Answer" className="panel-answer">
        <AnswerContent response={response} liveAnswer={liveAnswer} />
      </Panel>

      <CollapsiblePanel title="Processing details & Agent steps" count={traceItems.length}>
        <StreamTraceList items={traceItems} />
      </CollapsiblePanel>

      <CollapsiblePanel title="Extracted content" count={response?.extracted_content?.length ?? 0}>
        <JsonList
          items={response?.extracted_content ?? []}
          emptyMessage={
            response ? "No extracted content returned." : "Waiting for agent execution..."
          }
        />
      </CollapsiblePanel>

      <CollapsiblePanel title="Grounded evidence" count={response?.evidence?.length ?? 0}>
        <JsonList
          items={response?.evidence ?? []}
          emptyMessage={response ? "No evidence returned." : "Waiting for evidence..."}
        />
      </CollapsiblePanel>

      {displayErrors.length > 0 && (
        <CollapsiblePanel title="Errors" count={displayErrors.length} defaultOpen className="panel-errors">
          <ul className="error-list">
            {displayErrors.map((error, index) => (
              <li key={`${error}-${index}`}>{error}</li>
            ))}
          </ul>
        </CollapsiblePanel>
      )}
    </section>
  );
}

function Panel({
  title,
  children,
  className = "",
}: {
  title: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <article className={`panel ${className}`.trim()}>
      <header className="panel-header">
        <h2>{title}</h2>
      </header>
      <div className="panel-body">{children}</div>
    </article>
  );
}

function CollapsiblePanel({
  title,
  count,
  defaultOpen = false,
  children,
  className = "",
}: {
  title: string;
  count?: number;
  defaultOpen?: boolean;
  children: ReactNode;
  className?: string;
}) {
  return (
    <details className={`panel collapsible-panel ${className}`.trim()} open={defaultOpen}>
      <summary className="panel-header accordion-summary">
        <h2>
          {title}
          {typeof count === "number" && count > 0 ? (
            <span className="count-badge">{count}</span>
          ) : null}
        </h2>
      </summary>
      <div className="panel-body">{children}</div>
    </details>
  );
}

function AnswerContent({
  response,
  liveAnswer,
}: {
  response: AgentRunResponse | null;
  liveAnswer: string;
}) {
  if (!liveAnswer.trim() && !response) {
    return <p className="empty-state">Run the agent to see the final answer.</p>;
  }

  if (!liveAnswer.trim()) {
    if (response?.status === "clarification_required") {
      return (
        <p className="clarification-note">
          Clarification is required. Review the question above and submit a more
          specific follow-up query.
        </p>
      );
    }
    return <p className="empty-state">The agent returned an empty answer.</p>;
  }

  return (
    <div className="answer-block">
      {response && (
        <p className="status-badge" data-status={response.status}>
          {formatStatus(response.status)}
        </p>
      )}
      <pre className="scrollable-text">{liveAnswer}</pre>
      {response?.request_id && (
        <p className="meta-line">Request ID: {response.request_id}</p>
      )}
    </div>
  );
}

function StreamTraceList({ items }: { items: Array<Record<string, unknown>> }) {
  if (items.length === 0) {
    return <p className="empty-state">Execution trace will appear here in real time.</p>;
  }

  return (
    <div className="stream-trace-list">
      {items.map((item, index) => (
        <div key={index} className="stream-trace-item">
          <div className="stream-trace-header">
            <strong>{String(item.label ?? "update")}</strong>
            {item.status ? (
              <span className="trace-status">{String(item.status)}</span>
            ) : null}
          </div>
          {item.message ? <p className="stream-trace-message">{String(item.message)}</p> : null}
          {item.latency_ms != null ? (
            <p className="meta-line">{String(item.latency_ms)} ms</p>
          ) : null}
        </div>
      ))}
    </div>
  );
}

function JsonList({
  items,
  emptyMessage,
}: {
  items: Array<Record<string, unknown>>;
  emptyMessage: string;
}) {
  if (items.length === 0) {
    return <p className="empty-state">{emptyMessage}</p>;
  }

  return (
    <div className="json-list">
      {items.map((item, index) => (
        <pre key={index} className="scrollable-text json-item">
          {JSON.stringify(item, null, 2)}
        </pre>
      ))}
    </div>
  );
}

function buildTraceItems(
  streamEvents: AgentStreamEvent[],
  response: AgentRunResponse | null,
): Array<Record<string, unknown>> {
  if (streamEvents.length > 0) {
    return streamEvents
      .filter((event) =>
        ["node_update", "tool_update", "status_update", "error"].includes(event.type),
      )
      .map((event) => ({
        type: event.type,
        label: event.node ?? event.tool ?? event.type,
        status: event.status,
        message: event.message,
        latency_ms: event.latency_ms,
      }));
  }

  return response?.trace ?? [];
}

function formatStatus(status: AgentRunResponse["status"]): string {
  switch (status) {
    case "success":
      return "Success";
    case "clarification_required":
      return "Clarification Required";
    case "partial":
      return "Partial Result";
    case "error":
      return "Error";
    default:
      return status;
  }
}
