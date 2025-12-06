import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import TextGenerationRequest, TextGenerationResponse
from .inference import load_model, generate_with_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Assignment 5 — Post-training an LLM",
    description=(
        "FastAPI server exposing a text generation endpoint backed by "
        "a fine-tuned GPT-2 model."
    ),
)

# Allow simple local testing from browser tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model objects, loaded once at startup
tokenizer = None
model = None


@app.on_event("startup")
def load_llm_on_startup():
    global tokenizer, model
    logger.info("Loading LLM at startup...")
    tokenizer, model = load_model()
    logger.info("Model loaded and ready.")


@app.get("/")
def root():
    return {
        "message": "Assignment 5 LLM Text Generation API is running.",
        "endpoints": ["/generate_with_llm"],
    }


@app.post("/generate_with_llm", response_model=TextGenerationResponse)
def generate_with_llm_endpoint(request: TextGenerationRequest):
    """Generate text using the (fine-tuned) GPT‑2 model.

    For backward compatibility with earlier modules, the request object
    uses the fields `start_word` and `length`, but in this assignment
    `start_word` can contain a full *question* or *prompt*.

    Example request body:

    {
        "start_word": "What is reinforcement learning?",
        "length": 80
    }
    """
    global tokenizer, model

    if tokenizer is None or model is None:
        # This should not happen in normal usage because we load at startup,
        # but we keep the check for robustness.
        tokenizer, model = load_model()

    # Treat `start_word` as a full prompt/question.
    prompt = request.start_word.strip()
    if not prompt:
        prompt = "That is a great question."

    generated = generate_with_llm(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        max_new_tokens=request.length,
    )

    return TextGenerationResponse(generated_text=generated)