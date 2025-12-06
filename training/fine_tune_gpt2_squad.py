"""Fine-tune GPT-2 on SQuAD to learn a specific answer format.

This script implements the *practice* part of Assignment 5:

- We start from the base model `openai-community/gpt2`.
- We fine-tune it on the SQuAD question–answer dataset.
- For each (question, answer) pair, we build a training text where the
  answer is wrapped in a fixed template, for example:

    That is a great question. {answer} Let me know if you have any other questions.

After training, we save the fine-tuned weights into
`models/fine_tuned_gpt2_format/`. The FastAPI app will load the model
from this directory at runtime.

You can reduce `max_train_samples` and `num_train_epochs` if you only
need a quick demo run on CPU.
"""

from pathlib import Path
from typing import Dict

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "openai-community/gpt2"
OUTPUT_DIR = Path("models") / "fine_tuned_gpt2_format"


def build_training_text(example: Dict) -> str:
    """Construct a single training string from a SQuAD example.

    We only use the first answer for simplicity.
    """
    question = example.get("question", "").strip()
    answers = example.get("answers", {}).get("text", [])
    if answers:
        answer = answers[0].strip()
    else:
        answer = "I am not sure."

    template = (
        "Question: {q}\n"
        "Answer: That is a great question. {a} "
        "Let me know if you have any other questions."
    )
    return template.format(q=question, a=answer)


def main(
    output_dir: Path = OUTPUT_DIR,
    max_train_samples: int = 2000,
    num_train_epochs: float = 1.0,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SQuAD dataset...")
    dataset = load_dataset("rajpurkar/squad")

    print("Preparing training corpus...")
    def map_fn(batch):
        texts = [build_training_text(ex) for ex in batch["data"]]
        return {"text": texts}

    # The SQuAD dataset schema in HuggingFace is usually:
    # dataset["train"] with fields: id, title, context, question, answers
    train_ds = dataset["train"].select(range(min(max_train_samples, len(dataset["train"]))))
    train_ds = train_ds.map(lambda ex: {"text": build_training_text(ex)})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
        )

    tokenized_train = train_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_ds.column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        learning_rate=5e-5,
        weight_decay=0.01,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
    )

    print("Starting fine-tuning...")
    trainer.train()
    print("Saving model to", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("Done.")


if __name__ == "__main__":
    main()