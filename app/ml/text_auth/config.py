from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class TextAuthSettings(BaseSettings):
    """Configuration for the text authentication pipeline."""

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    classifier_path: str = "./app/ml/text_auth/saved_models/classifier.pkl"
    label_encoder_path: str = "./app/ml/text_auth/saved_models/label_encoder.pkl"
    min_text_length: int = 20
    max_text_length: int = 20_000
    batch_size: int = 32
    device: str = "cpu"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="TEXT_AUTH_",
        extra="ignore",
    )

    @property
    def resolved_classifier_path(self) -> Path:
        """Classifier path resolved from the ml-service working directory."""
        return self._resolve_path(self.classifier_path)

    @property
    def resolved_label_encoder_path(self) -> Path:
        """Label encoder path resolved from the ml-service working directory."""
        return self._resolve_path(self.label_encoder_path)

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        service_root = Path(__file__).resolve().parents[3]
        return service_root / path


@lru_cache
def get_text_auth_settings() -> TextAuthSettings:
    """Return cached text-auth settings."""
    return TextAuthSettings()
