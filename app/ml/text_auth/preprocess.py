import re

from app.ml.text_auth.config import TextAuthSettings


_WHITESPACE_RE = re.compile(r"\s+")


def preprocess_text(text: str, settings: TextAuthSettings) -> str:
    """Normalize text consistently for training and inference without overcleaning."""
    if text is None:
        raise ValueError("Text is required")

    cleaned = _WHITESPACE_RE.sub(" ", text.strip())

    if not cleaned:
        raise ValueError("Text must not be empty")
    if len(cleaned) < settings.min_text_length:
        raise ValueError(f"Text is too short. Minimum length is {settings.min_text_length} characters.")
    if len(cleaned) > settings.max_text_length:
        cleaned = cleaned[: settings.max_text_length].rstrip()

    return cleaned
