"""Text helpers: one normaliser for references and model outputs."""
from functools import lru_cache
from typing import Literal


@lru_cache(maxsize=1)
def _tokenizer():
    # Whisper's English normaliser: lower-cases, removes punctuation,
    # standardises spellings and numbers ("seven" -> "7").
    from transformers import WhisperTokenizer

    return WhisperTokenizer.from_pretrained("openai/whisper-tiny")


def normalize(s: str) -> str:
    """Normalise text the same way for references and Whisper outputs."""
    if not s or not s.strip():
        return ""
    return " ".join(_tokenizer().normalize(s).split())


def prompt_type(norm: str) -> Literal["word", "sentence"]:
    """A prompt with one word is a 'word'; anything longer is a 'sentence'."""
    return "word" if len(norm.split()) <= 1 else "sentence"
