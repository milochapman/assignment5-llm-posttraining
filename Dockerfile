# Assignment 5 — Post-training an LLM
#
# This Dockerfile builds a FastAPI application that exposes an endpoint
# `/generate_with_llm` backed by a (fine-tuned) GPT-2 model.
#
# Usage:
#   docker build -t assignment5-llm-api .
#   docker run --rm -p 8000:8000 assignment5-llm-api

FROM python:3.11-slim

WORKDIR /app

# Install system utilities (optional but useful for debugging)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]