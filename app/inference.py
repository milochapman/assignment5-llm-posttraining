import logging
from pathlib import Path
from typing import Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("models") / "fine_tuned_gpt2_format"
BASE_MODEL_NAME = "openai-community/gpt2"


def load_model(
    model_dir: Path = DEFAULT_MODEL_DIR,
    base_model_name: str = BASE_MODEL_NAME,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load the fine-tuned GPT‑2 model if available, otherwise fall back
    to the base GPT‑2 model.

    This makes the Docker image robust even if the fine‑tuned weights
    were not copied into the image yet: the API will still start and
    generate text, but without the post‑training behavior.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_dir.exists():
        logger.info(f"Loading fine-tuned model from {model_dir} ...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForCausalLM.from_pretrained(str(model_dir))
    else:
        logger.warning(
            "Fine-tuned model directory not found at %s. "
            "Falling back to base model %s. "
            "Make sure to run training/fine_tune_gpt2_squad.py "
            "and rebuild the Docker image before submission.",
            model_dir,
            base_model_name,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # GPT‑2 sometimes has no explicit pad token; we map it to EOS for safety.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()
    return tokenizer, model


def generate_with_llm(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 50,
) -> str:
    """Generate text using the (fine‑tuned) GPT‑2 model.

    The assignment asks that the model respond in a specific format.
    During fine‑tuning (see training/fine_tune_gpt2_squad.py) we wrap
    answers in a template like:

        "That is a great question. {answer} Let me know if you have any other questions."

    Here we simply let the model autoregressively complete the prompt.
    """
    device = next(model.parameters()).device

    # We treat the incoming `prompt` as a full question or sentence.
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
    ).input_ids.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated