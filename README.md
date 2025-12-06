# Assignment 5 — LLM Post-Training with Reinforcement Learning  
### Columbia University — Applied Analytics  
### Author: Mingyuan Li  

---

## Overview

This project implements **LLM post-training using Reinforcement Learning (RL)**.  
It consists of two major parts required by Assignment 5:

1. **Post-training a GPT‑2 model** to follow a specific response format  
2. **Answering reinforcement learning theoretical questions**  

The final deliverable includes:

- A functioning **FastAPI server**
- A **Dockerized deployment** runnable on instructor machines
- A **fine‑tuned GPT‑2 model**
- A clean, well‑organized repo
- Theoretical answers (`theory_answers.md`)

This README contains all instructions needed to build, run, and test the API.

---

## Repository Structure

```
assignment5-llm-posttraining/
│
├── app/
│   ├── main.py                # FastAPI entrypoint
│   ├── inference.py           # Model loading + generation logic
│
├── training/
│   └── fine_tune_gpt2_squad.py  # GPT‑2 fine‑tuning script
│
├── models/                    # Fine‑tuned model folder (loaded at runtime)
│
├── theory_answers.md          # Reinforcement Learning theory answers
├── requirements.txt           # Python dependencies for API
├── Dockerfile                 # Docker deployment file
├── README.md                  # This file
└── .gitignore
```

---

## Running the API Locally (No Docker)

### Create virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies
```
pip install -r requirements.txt
```

### Start the FastAPI server
```
uvicorn app.main:app --reload --port 8000
```

### Test API
**Health check**
```
curl http://127.0.0.1:8000/
```

**Generate text**
```
curl -X POST "http://127.0.0.1:8000/generate_with_llm"   -H "Content-Type: application/json"   -d '{
        "start_word": "What is reinforcement learning?",
        "length": 60
      }'
```

### Swagger UI
Open in browser:

`http://127.0.0.1:8000/docs`

---

## Running with Docker

### Build image
```
docker build -t assignment5-llm-api .
```

### Run container
```
docker run --rm -p 8000:8000 assignment5-llm-api
```

### Test API (same commands as above)
```
curl http://127.0.0.1:8000/
```

```
curl -X POST "http://127.0.0.1:8000/generate_with_llm"   -H "Content-Type: application/json"   -d '{"start_word": "Hello", "length": 40}'
```

---

## Fine‑Tuning the Model (Optional Re‑Run)

Your model is already fine‑tuned and stored under `models/fine_tuned_gpt2_format/`.

But if you ever want to rerun:

```
python training/fine_tune_gpt2_squad.py
```

Outputs will be saved automatically into the `models/` directory.

---

## Theory Answers

All answers to Part 2 are included in:

`theory_answers.md`
---

## Notes

This assignment demonstrates:

- Practical LLM post‑training  
- RL‑based behavior shaping  
- Real‑world model deployment with FastAPI  
- Proper Dockerization  
- Clean, reproducible ML workflow  

If any issues occur when running the API or Docker image, please ensure that Python 3.11+ and Docker Desktop are up to date.

---
