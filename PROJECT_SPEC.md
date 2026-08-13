# Multimodal Agentic Intelligence Platform

## Project Objective

Build a production-oriented multimodal agent that accepts:

- text
- images
- PDFs
- audio
- multiple files simultaneously

The system must understand the user's goal, determine whether clarification is required, create the minimum required tool sequence, execute tools through LangGraph, combine results, validate outputs, and return a text-only final answer.

The application must be deployed publicly using FastAPI and Docker.

## Mandatory Capabilities

1. Image/PDF extraction
2. OCR with confidence
3. YouTube URL detection and transcript fetching
4. Conversational answering
5. Summarization
6. Sentiment analysis
7. Code explanation
8. Audio transcription and summarization
9. Cross-input reasoning
10. Multiple files in one request
11. Follow-up question when intent is ambiguous
12. Autonomous multi-tool chaining
13. Execution trace
14. Error handling and graceful degradation

## Required Technology

Backend:
FastAPI

Agent orchestration:
LangGraph

Agent/tool abstractions:
LangChain

Retrieval:
Qdrant

RAG:
Hybrid dense + sparse retrieval with reranking

Document extraction:
PyMuPDF

OCR:
Tesseract

Audio:
faster-whisper

Frontend:
React/Vite

Deployment:
Docker + public cloud

Testing:
pytest

Observability:
LangSmith

## Architecture Principle

Use deterministic code whenever the task can be reliably solved without an LLM.

Use LLMs only for:
- intent understanding
- semantic planning
- semantic analysis
- summarization
- reasoning
- synthesis

Do NOT use an LLM for:
- file type detection
- URL extraction
- PDF parsing
- OCR when deterministic OCR is sufficient
- simple routing rules
- timing
- validation that can be expressed deterministically

## Agent Architecture

Input
→ Validation
→ Normalization
→ Intent Detection
→ Clarification Gate
→ Planner
→ Plan Validation
→ Tool Execution
→ Result Validation
→ Retry/Fallback if necessary
→ Evidence Construction
→ Final Synthesis
→ Final Response

## LangGraph State

The state should contain:

request_id
user_query
input_artifacts
normalized_contents
intent
constraints
clarification_required
clarification_question
plan
current_step
tool_results
evidence
errors
trace
final_response

## Tool Contract

Every tool must return:

tool_name
status
data
confidence
latency_ms
error

## Reliability

Implement:
- file size limits
- MIME validation
- timeout handling
- retry limits
- maximum graph steps
- graceful fallback
- low-confidence warnings
- no-evidence handling

## RAG

RAG must use:

1. document parsing
2. intelligent chunking
3. metadata
4. dense retrieval
5. sparse/BM25 retrieval
6. result fusion
7. reranking
8. evidence selection
9. grounded generation

Every retrieved evidence item should preserve:
document
page
chunk
text
retrieval score

If evidence is insufficient, the system must not hallucinate an answer.

## Important

Do not build a generic chatbot.

Do not build a simple RAG pipeline.

Do not make every operation an LLM call.

Do not use an uncontrolled agent loop.

The goal is a production-oriented agentic workflow with deterministic tools, explicit orchestration, reliable RAG, observability and explainability.

## Coding Standards

Python 3.11

Use type hints.

Use Pydantic models for contracts.

Keep business logic outside FastAPI routes.

Use dependency injection where appropriate.

Write tests for important functionality.

Use small modules.

Do not duplicate logic.

Do not add dependencies unless necessary.

Before adding a library, explain why it is required.

Every implementation must be explainable by the developer.

## AI Coding Rules

AI assistants may assist with implementation, but must not redesign the architecture without approval.

Do not rewrite unrelated files.

Do not introduce unnecessary frameworks.

Do not hide logic inside prompts when deterministic Python logic is more appropriate.

After implementation:
- run tests
- run linting
- inspect diff
- explain important changes

## Success Criteria

The system should score strongly on:

Correctness
Autonomy and planning
Robustness
Explainability
Code quality
UX
RAG accuracy
RAG latency
Production readiness