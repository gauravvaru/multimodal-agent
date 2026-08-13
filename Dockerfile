FROM python:3.11-slim

# Install system dependencies for OCR and audio processing
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast dependency management (or we can just use pip)
RUN pip install --no-cache-dir uv

# Copy project configuration
COPY pyproject.toml ./

# Install dependencies including media extras
RUN uv pip install --system -e ".[media]"

# Copy the rest of the application
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "multimodal_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
