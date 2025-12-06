# Assignment 5 — Post-training an LLM

This repository implements the **practice** part of Assignment 5:

- Update the text generation API from Modules 3 and 7 to use a
  **fine-tuned GPT‑2 model** (`openai-community/gpt2`).
- Fine-tune the model on a QA dataset so that answers follow a
  **specific format**, e.g.:

  > That is a great question. ... Let me know if you have any other questions.

It also includes a separate file with written answers for the
**Reinforcement Learning theory** questions.

## Project Structure

- `app/`
  - `main.py` — FastAPI app with `/generate_with_llm` endpoint.
  - `inference.py` — model loading and generation helper functions.
  - `schemas.py` — Pydantic request/response models.
- `training/`
  - `fine_tune_gpt2_squad.py` — script to fine-tune GPT‑2 on SQuAD with a
    fixed answer template.
- `theory/`
  - `rl_theory_answers.md` — detailed solutions for the RL questions.
- `Dockerfile` — builds a container that serves the FastAPI app.
- `requirements.txt` — Python dependencies.

The API is designed so that it can be run **locally** and **inside
Docker** without changing the code. The same `app.main:app` object is
used in both cases, avoiding the kind of mismatch that caused issues in
Assignment 4.

---

## 1. Local Setup (MacBook Intel, CPU)

### 1.1. Create and activate a virtual environment

```bash
cd assignment5_llm_posttraining

python -m venv .venv
source .venv/bin/activate  # On macOS / Linux
# For Windows PowerShell:
#   .venv\Scripts\Activate.ps1
```

### 1.2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Fine-tune GPT‑2 on SQuAD

> You can reduce the number of training samples and epochs if you only
> need a quick demonstration run on CPU.

```bash
cd assignment5_llm_posttraining
source .venv/bin/activate

python training/fine_tune_gpt2_squad.py
```

By default this:

- Downloads `rajpurkar/squad` from HuggingFace.
- Builds training texts of the form:

  ```text
  Question: <question>
  Answer: That is a great question. <answer> Let me know if you have any other questions.
  ```

- Fine-tunes `openai-community/gpt2` for 1 epoch on up to 2000 samples.
- Saves the fine-tuned model and tokenizer into:

  ```text
  models/fine_tuned_gpt2_format/
  ```

The FastAPI app automatically tries to load the model from this
directory. If it does **not** exist, the app falls back to the base
GPT‑2 model and logs a warning.

---

## 3. Run the FastAPI Server Locally

Make sure your virtual environment is active:

```bash
cd assignment5_llm_posttraining
source .venv/bin/activate

uvicorn app.main:app --reload --port 8000
```

You should see a log message like:

```text
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Loading LLM at startup...
INFO:     Model loaded and ready.
```

### 3.1. Test the API locally

Use `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/generate_with_llm" \
  -H "Content-Type: application/json" \
  -d '{
        "start_word": "What is reinforcement learning?",
        "length": 80
      }'
```

Expected JSON structure:

```json
{
  "generated_text": "Question: What is reinforcement learning? Answer: That is a great question. ..."
}
```

(Exact text will vary, but the model should learn to use the desired
answer template after fine-tuning.)

---

## 4. Build and Run with Docker

> Important for the rubric: the **same app** used locally is used
> inside Docker. There is **no separate local server**, which avoids the
> “calling local model instead of Docker” problem from Assignment 4.

### 4.1. Make sure the fine-tuned model is included

After running the training script, you should have:

```text
models/fine_tuned_gpt2_format/
```

inside the project folder. This directory is part of the Docker build
context, so it will be copied into the image automatically by:

```dockerfile
COPY . .
```

### 4.2. Build the Docker image

From the project root:

```bash
cd assignment5_llm_posttraining

docker build -t assignment5-llm-api .
```

### 4.3. Run the Docker container

```bash
docker run --rm -p 8000:8000 assignment5-llm-api
```

You should see logs similar to:

```text
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Loading LLM at startup...
INFO:     Model loaded and ready.
```

### 4.4. Test the API inside Docker

In a **separate terminal**, run:

```bash
curl -X POST "http://127.0.0.1:8000/generate_with_llm" \
  -H "Content-Type: application/json" \
  -d '{
        "start_word": "Explain the idea behind Q-learning.",
        "length": 80
      }'
```

If you get a valid JSON response with a `generated_text` field, the
Dockerized API is working and can be used by the instructor for grading.

> Tip: Make sure **no local uvicorn server** is running on port 8000
> when testing the Docker container, so you are definitely talking to
> the containerized app (this avoids the exact confusion mentioned in
> the Assignment 4 feedback).

---

## 5. GitHub Submission Checklist (Rubric Alignment)

To satisfy all rubric items:

1. **Commit new code to GitHub (10 pts)**

   - Create a new repository, e.g. `assignment5-llm-posttraining`.
   - Add all files in this folder.
   - Commit with a clear message, e.g. `feat: add GPT-2 fine-tuning and FastAPI server`.
   - Push to GitHub.

2. **Docker deployment runs a FastAPI server (20 pts)**

   - Ensure the instructor can run:

     ```bash
     docker build -t assignment5-llm-api .
     docker run --rm -p 8000:8000 assignment5-llm-api
     ```

   - And then successfully call `/generate_with_llm`.

3. **API can be successfully queried for inference (20 pts)**

   - `/generate_with_llm` should return a JSON object with the key
     `generated_text`.
   - No crashes or missing-model errors when the container starts.

4. **Code organization and architecture (20 pts)**

   - App code is under `app/`, training code under `training/`.
   - Clear separation of concerns: schemas, inference helpers, main API.
   - README explains the full workflow end-to-end.

5. **Conceptual/calculation questions (30 pts)**

   - The file `theory/rl_theory_answers.md` contains complete, clearly
     explained answers with all intermediate steps.

---

## 6. Minimal Smoke Test Summary

After following the steps above, you should be able to:

1. Run `uvicorn app.main:app --reload --port 8000` locally.
2. `curl` the `/` endpoint and see a status message.
3. `curl` the `/generate_with_llm` endpoint and get a JSON response.
4. Build and run the Docker image and repeat steps 2–3 successfully.

If all of these checks pass, your submission should satisfy all parts
of the Assignment 5 rubric.