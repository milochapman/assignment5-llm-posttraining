from pydantic import BaseModel

class TextGenerationRequest(BaseModel):
    """Request schema for LLM-based text generation.

    We keep the same field names as in the Module 9 activity
    (start_word, length) so that the instructor's tests can re-use
    the same client code. In practice, `start_word` can contain a full
    question or prompt, not just a single word.
    """

    start_word: str
    length: int = 50


class TextGenerationResponse(BaseModel):
    generated_text: str