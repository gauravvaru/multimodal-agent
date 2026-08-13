# multimodal-agent

A production-ready multimodal AI agent built with **FastAPI**, **LangGraph**, and **React (Vite)**. The agent autonomously ingests text, images, PDFs, audio files, and YouTube URLs, formulates execution plans, invokes specialized tools, and synthesizes grounded final answers.

---

## Architecture Diagram

```
User Query + Uploads (PDF, Image, Audio)
                 │
                 ▼
       ┌──────────────────┐
       │   FastAPI UI     │
       └─────────┬────────┘
                 │
                 ▼
       ┌──────────────────┐
       │ Input Processing │  (MIME detection, size validation, temp storage)
       └─────────┬────────┘
                 │
                 ▼
       ┌──────────────────┐
       │ Agent Orchestrator│ (LangGraph State Graph)
       │                  │
       │ 1. Validation    │
       │ 2. Normalization │
       │ 3. Intent Detect │ -> [Ambiguous?] ──> Clarification Question
       │ 4. Planner       │
       └─────────┬────────┘
                 │
                 ▼
       ┌──────────────────┐
       │  Tool Registry   │
       │                  │
       │ - PyMuPDF / OCR  │ (PDF text & layout extraction)
       │ - Whisper / Audio│ (Audio transcription)
       │ - Code Analyzer  │ (Image/code explanation)
       │ - YouTube API    │ (Transcript fetching)
       │ - RAG / Vector   │ (Hybrid retrieval & reranking)
       │ - Sentiment      │ (Sentiment analysis)
       │ - Comparison     │ (Cross-input reasoning)
       └─────────┬────────┘
                 │
                 ▼
       ┌──────────────────┐
       │ Result Validation│ -> [Success / Retry / Fallback]
       └─────────┬────────┘
                 │
                 ▼
       ┌──────────────────┐
       │ Evidence & Synthesis│
       └─────────┬────────┘
                 │
                 ▼
          Final Answer
```

---

## Features & Supported Capabilities

1. **Multimodal File Support**: Ingests PDF documents, images (PNG, JPG), audio files (MP3, WAV, M4A), and YouTube URLs in a single request.
2. **Intent Detection & Ambiguity Handling**: Classifies user queries into task categories. If a query is ambiguous, it requests clarification before executing tools.
3. **Autonomous Planning**: Constructs minimal tool execution sequences using LangGraph and validates plan safety before tool invocation.
4. **Document Extraction & OCR**: Uses PyMuPDF for fast text extraction with automatic Tesseract OCR fallback for scanned or image-based PDFs.
5. **Audio Transcription**: Uses `faster-whisper` and `ffmpeg` for segment-level audio transcription and summarization.
6. **YouTube Integration**: Automatically extracts YouTube links from input text or PDFs and fetches transcript text for downstream analysis.
7. **Cross-Input Reasoning**: Compares content across multiple files (e.g., matching audio transcripts against PDF reports).
8. **Real-time Execution Trace**: Streams graph updates via Server-Sent Events (SSE) so tool calls and status changes display in real time.

---

## Tech Stack

* **Backend**: Python 3.11, FastAPI, Uvicorn, Pydantic v2
* **Agent Framework**: LangGraph, LangChain Core, LangChain Google GenAI
* **Media & Processing**: PyMuPDF, pytesseract, pdf2image, faster-whisper, youtube-transcript-api
* **Frontend**: React, TypeScript, Vite
* **Deployment**: Docker, Docker Compose

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+ & npm
- Tesseract OCR (`tesseract-ocr`)
- Poppler (`poppler-utils`)
- FFmpeg (`ffmpeg`)

### Environment Variables

Copy `.env.example` to `.env` and set your Gemini API key:

```bash
cp .env.example .env
```

```env
LLM_PROVIDER=google
GEMINI_API_KEY=your_gemini_api_key_here
LLM_MODEL=gemini-3.5-flash
```

---

## Running Locally

### 1. Backend Server

```bash
pip install -e ".[media]"
uvicorn multimodal_agent.main:app --reload --port 8000
```

### 2. Frontend Development UI

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Docker Deployment

Build and run the single-container deployment (includes all native OCR and media dependencies):

```bash
# Build Docker image
docker build -t multimodal-agent .

# Run container
docker run -p 8000:8000 -e GEMINI_API_KEY="your_api_key_here" multimodal-agent
```

---

## Test Suite

Run the automated pytest suite:

```bash
pytest tests/
```

### Primary Assignment Scenarios Verified

1. **Audio Transcription + Summary**: Transcribing audio clips and summarizing key topics.
2. **PDF Action Items & Query Answering**: Extracting text from multi-page PDFs to retrieve specific sections or action items.
3. **Code Image Explanation**: Extracting code snippets from image uploads and generating step-by-step explanations.
4. **YouTube URL in PDF**: Parsing YouTube links embedded within uploaded PDFs, fetching transcripts, and summarizing video content.
5. **Unified Audio & PDF Comparison**: Performing cross-input reasoning over audio transcripts and document text simultaneously.
