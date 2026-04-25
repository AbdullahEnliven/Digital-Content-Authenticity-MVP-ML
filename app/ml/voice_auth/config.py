from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class VoiceAuthSettings(BaseSettings):
    """Configuration for the voice authentication pipeline."""

    sample_rate: int = 16_000
    clip_duration_seconds: float = 4.0
    allowed_extensions: set[str] = {".flac", ".m4a", ".mp3", ".ogg", ".wav"}
    allowed_content_types: set[str] = {
        "application/octet-stream",
        "video/mp4",
        "audio/x-m4a",
        "audio/mp4",
        "audio/mpeg",
        "audio/flac",
        "audio/x-flac",
        "audio/wav",
        "audio/x-wav",
    }

    speechbrain_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    speechbrain_savedir: str = "./models/speechbrain_ecapa"
    classifier_path: str = "./app/ml/voice_auth/saved_models/classifier.pkl"
    label_encoder_path: str = "./app/ml/voice_auth/saved_models/label_encoder.pkl"
    device: str = "cpu"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="VOICE_AUTH_",
        extra="ignore",
    )

    @property
    def target_num_samples(self) -> int:
        """Number of waveform samples after standardization."""
        return int(self.sample_rate * self.clip_duration_seconds)

    @property
    def resolved_classifier_path(self) -> Path:
        """Classifier path resolved from the ml-service working directory."""
        return self._resolve_path(self.classifier_path)

    @property
    def resolved_label_encoder_path(self) -> Path:
        """Label encoder path resolved from the ml-service working directory."""
        return self._resolve_path(self.label_encoder_path)

    @property
    def resolved_speechbrain_savedir(self) -> Path:
        """Local cache directory for the SpeechBrain model."""
        return self._resolve_path(self.speechbrain_savedir)

    def extension_from_filename(self, filename: str) -> str:
        """Return a normalized file extension including the leading dot."""
        return Path(filename).suffix.lower()

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        service_root = Path(__file__).resolve().parents[3]
        return service_root / path


@lru_cache
def get_voice_auth_settings() -> VoiceAuthSettings:
    """Return cached voice-auth settings."""
    return VoiceAuthSettings()
