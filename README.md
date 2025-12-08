# Assignment 5 — LLM Post-Training with Reinforcement Learning  
Columbia University — Applied Analytics  
Author: Mingyuan Li  

---

## 1. Project Overview

This project implements the requirements for **Assignment 5: Post-training an LLM**:

1. **Post-training a GPT-2 model** so that it follows a specific answer format.  
2. **Implementing and exposing a text generation API** using FastAPI.  
3. **Answering reinforcement learning theory questions** in a separate write-up.

At a high level:

- The model side uses **GPT-2 (via Hugging Face Transformers)**, optionally fine-tuned on SQuAD.  
- The serving side uses **FastAPI + Uvicorn**, wrapped in a **Docker** image so the instructor can run it easily.  
- The RL theory part is documented in `theory_answers.md`.

---

## 2. Repository Structure

```
.
├── Dockerfile
├── README.md
├── requirements.txt
├── theory_answers.md
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── inference.py
│   └── schemas.py
└── training/
    └── fine_tune_gpt2_squad.py
```

---

## 3. Run Locally (Python)

### 3.1 Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 3.3 Start server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 4. Run with Docker

### 4.1 Build

```bash
docker build -t assignment5-llm-api .
```

### 4.2 Run

```bash
docker run --rm -p 8000:8000 assignment5-llm-api
```

---

## 5. API Usage

### Health check

```bash
curl http://127.0.0.1:8000/
```

### Generate text

```bash
curl -X POST "http://127.0.0.1:8000/generate_with_llm"   -H "Content-Type: application/json"   -d '{"start_word": "What is reinforcement learning?", "length": 60}'
```

---

## 6. Training Details

- Base model: `openai-community/gpt2`
- Optional fine-tune script: `training/fine_tune_gpt2_squad.py`
- Attempts to load model from:
  ```
  models/fine_tuned_gpt2_format/
  ```
  Falls back to base GPT‑2 if not present.

---

## 7. RL Theory Answers

See: `theory_answers.md`

---

## 8. Notes

- Python version: 3.11  
- Docker image: `python:3.11-slim`  
- All dependencies pinned for reproducibility.
