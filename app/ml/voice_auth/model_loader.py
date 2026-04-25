from pathlib import Path
from threading import Lock

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from app.ml.voice_auth.config import VoiceAuthSettings, get_voice_auth_settings
from app.ml.voice_auth.embedder import ECAPAEmbedder


class VoiceAuthModelLoader:
    """Lazy, reusable loader for the ECAPA embedder and sklearn classifier."""

    def __init__(self, settings: VoiceAuthSettings | None = None) -> None:
        self.settings = settings or get_voice_auth_settings()
        self._lock = Lock()
        self._embedder: ECAPAEmbedder | None = None
        self._classifier: LogisticRegression | None = None
        self._label_encoder: LabelEncoder | None = None

    @property
    def embedder(self) -> ECAPAEmbedder:
        self.load_models()
        assert self._embedder is not None
        return self._embedder

    @property
    def classifier(self) -> LogisticRegression:
        self.load_models()
        assert self._classifier is not None
        return self._classifier

    @property
    def label_encoder(self) -> LabelEncoder:
        self.load_models()
        assert self._label_encoder is not None
        return self._label_encoder

    def load_models(self, required: bool = True) -> None:
        """Load all model assets once. If required=False, missing classifier files are tolerated."""
        if self._embedder and self._classifier and self._label_encoder:
            return

        with self._lock:
            if self._embedder is None:
                self._embedder = ECAPAEmbedder(self.settings)

            if self._classifier is None or self._label_encoder is None:
                classifier_path = self.settings.resolved_classifier_path
                label_encoder_path = self.settings.resolved_label_encoder_path

                if not classifier_path.exists() or not label_encoder_path.exists():
                    message = (
                        "Voice-auth classifier files are missing. Train the classifier first: "
                        f"{classifier_path} and {label_encoder_path}"
                    )
                    if required:
                        raise FileNotFoundError(message)
                    return

                self._classifier = joblib.load(classifier_path)
                self._label_encoder = joblib.load(label_encoder_path)

    def is_ready(self) -> bool:
        """Return whether the saved classifier and label encoder exist."""
        return (
            Path(self.settings.resolved_classifier_path).exists()
            and Path(self.settings.resolved_label_encoder_path).exists()
        )


_VOICE_AUTH_LOADER: VoiceAuthModelLoader | None = None


def get_voice_auth_model_loader() -> VoiceAuthModelLoader:
    """Return the process-wide voice-auth model loader."""
    global _VOICE_AUTH_LOADER
    if _VOICE_AUTH_LOADER is None:
        _VOICE_AUTH_LOADER = VoiceAuthModelLoader()
    return _VOICE_AUTH_LOADER
