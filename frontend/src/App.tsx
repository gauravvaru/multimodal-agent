import { useMemo, useRef, useState } from "react";

import { runAgentStream } from "./api/agent";
import { ResultPanels } from "./components/ResultPanels";
import { ApiError, AgentRunResponse, AgentStreamEvent, ChatMessage } from "./types";
import { getMaxUploadMb, validateSubmission } from "./validation";

function createId(): string {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export default function App() {
  const [query, setQuery] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [requestErrors, setRequestErrors] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [latestResponse, setLatestResponse] = useState<AgentRunResponse | null>(
    null,
  );
  const [streamEvents, setStreamEvents] = useState<AgentStreamEvent[]>([]);
  const [streamingFinalAnswer, setStreamingFinalAnswer] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const acceptedTypes = useMemo(
    () => ".pdf,.jpg,.jpeg,.png,.mp3,.wav,.m4a",
    [],
  );

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const incoming = Array.from(event.target.files ?? []);
    if (incoming.length === 0) {
      return;
    }
    setSelectedFiles((current) => mergeFiles(current, incoming));
    event.target.value = "";
  };

  const removeFile = (index: number) => {
    setSelectedFiles((current) => current.filter((_, itemIndex) => itemIndex !== index));
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setValidationErrors([]);
    setRequestErrors([]);

    const errors = validateSubmission(query, selectedFiles);
    if (errors.length > 0) {
      setValidationErrors(errors);
      return;
    }

    const userMessage: ChatMessage = {
      id: createId(),
      role: "user",
      query: query.trim(),
      files: selectedFiles.map((file) => file.name),
    };

    setMessages((current) => [...current, userMessage]);
    setIsLoading(true);
    setStreamEvents([]);
    setStreamingFinalAnswer("");
    setLatestResponse(null);

    try {
      const response = await runAgentStream(query, selectedFiles, (event) => {
        setStreamEvents((current) => [...current, event]);
        if (event.type === "final_answer" && event.final_answer) {
          setStreamingFinalAnswer(event.final_answer);
        }
        if (event.type === "error" && event.errors) {
          setRequestErrors((current) => [...current, ...event.errors!]);
        }
      });
      setLatestResponse(response);
      if (response.final_answer) {
        setStreamingFinalAnswer(response.final_answer);
      }

      const agentMessage: ChatMessage = {
        id: createId(),
        role: "agent",
        query: query.trim(),
        files: selectedFiles.map((file) => file.name),
        response,
      };
      setMessages((current) => [...current, agentMessage]);
    } catch (error) {
      const details =
        error instanceof ApiError
          ? error.details
          : ["Unable to reach the agent service. Check that the backend is running."];

      setRequestErrors(details);
      setLatestResponse(null);

      const agentMessage: ChatMessage = {
        id: createId(),
        role: "agent",
        query: query.trim(),
        files: selectedFiles.map((file) => file.name),
        errorMessages: details,
      };
      setMessages((current) => [...current, agentMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const displayErrors = [...validationErrors, ...requestErrors];

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Multimodal Agent</p>
          <h1>Agent Workspace</h1>
          <p className="subtitle">
            Submit a query with optional files. The backend orchestrates tools,
            validation, evidence, and synthesis.
          </p>
        </div>
      </header>

      <main className="app-main">
        <section className="chat-column">
          <div className="chat-thread" aria-label="Conversation history">
            {messages.length === 0 ? (
              <div className="chat-empty">
                <p>Start by describing what you want the agent to do.</p>
                <p className="hint">
                  Supported files: PDF, JPG, JPEG, PNG, MP3, WAV, M4A. Max{" "}
                  {getMaxUploadMb()} MB per file.
                </p>
              </div>
            ) : (
              messages.map((message) => <ChatBubble key={message.id} message={message} />)
            )}
          </div>

          <form className="composer" onSubmit={handleSubmit}>
            <label className="field-label" htmlFor="query">
              Query
            </label>
            <textarea
              id="query"
              className="query-input"
              rows={4}
              placeholder="Example: Summarize the uploaded PDF and highlight key risks."
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              disabled={isLoading}
            />

            <div className="composer-actions">
              <div className="file-controls">
                <input
                  ref={fileInputRef}
                  id="files"
                  type="file"
                  className="file-input"
                  accept={acceptedTypes}
                  multiple
                  onChange={handleFileChange}
                  disabled={isLoading}
                />
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isLoading}
                >
                  Add Files
                </button>
              </div>

              <button
                type="submit"
                className="primary-button"
                disabled={isLoading}
              >
                {isLoading ? "Running Agent..." : "Run Agent"}
              </button>
            </div>

            {selectedFiles.length > 0 && (
              <SelectedFilesList files={selectedFiles} onRemove={removeFile} />
            )}

            {displayErrors.length > 0 && (
              <div className="inline-errors" role="alert">
                <ul>
                  {displayErrors.map((error, index) => (
                    <li key={`${error}-${index}`}>{error}</li>
                  ))}
                </ul>
              </div>
            )}

            {isLoading && (
              <p className="loading-banner" aria-live="polite">
                Agent is running. Execution trace updates will appear in real
                time while tools execute and results are validated.
              </p>
            )}
          </form>
        </section>

        <ResultPanels
          response={latestResponse}
          errorMessages={requestErrors}
          streamEvents={streamEvents}
          streamingFinalAnswer={streamingFinalAnswer}
        />
      </main>
    </div>
  );
}

function ChatBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <article className={`chat-bubble ${isUser ? "user" : "agent"}`}>
      <div className="bubble-meta">
        <span>{isUser ? "You" : "Agent"}</span>
        {message.response && (
          <span className="status-badge" data-status={message.response.status}>
            {message.response.status.replace(/_/g, " ")}
          </span>
        )}
      </div>
      <p className="bubble-query">{message.query}</p>
      {message.files.length > 0 && (
        <ul className="bubble-files">
          {message.files.map((filename) => (
            <li key={filename}>{filename}</li>
          ))}
        </ul>
      )}
      {!isUser && message.response?.final_answer && (
        <pre className="bubble-answer scrollable-text">
          {message.response.final_answer}
        </pre>
      )}
      {!isUser && message.errorMessages && (
        <ul className="bubble-errors">
          {message.errorMessages.map((error, index) => (
            <li key={`${error}-${index}`}>{error}</li>
          ))}
        </ul>
      )}
    </article>
  );
}

function SelectedFilesList({
  files,
  onRemove,
}: {
  files: File[];
  onRemove: (index: number) => void;
}) {
  return (
    <div className="selected-files">
      <p className="field-label">Selected Files</p>
      <ul>
        {files.map((file, index) => (
          <li key={`${file.name}-${file.size}-${index}`}>
            <div>
              <strong>{file.name}</strong>
              <span>{formatFileSize(file.size)}</span>
            </div>
            <button
              type="button"
              className="text-button"
              onClick={() => onRemove(index)}
              aria-label={`Remove ${file.name}`}
            >
              Remove
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function mergeFiles(existing: File[], incoming: File[]): File[] {
  const merged = [...existing];
  incoming.forEach((file) => {
    const duplicate = merged.some(
      (current) => current.name === file.name && current.size === file.size,
    );
    if (!duplicate) {
      merged.push(file);
    }
  });
  return merged;
}

function formatFileSize(sizeBytes: number): string {
  if (sizeBytes < 1024) {
    return `${sizeBytes} B`;
  }
  if (sizeBytes < 1024 * 1024) {
    return `${(sizeBytes / 1024).toFixed(1)} KB`;
  }
  return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
}
