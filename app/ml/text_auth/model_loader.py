from pathlib import Path
from threading import Lock

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from app.ml.text_auth.config import TextAuthSettings, get_text_auth_settings
from app.ml.text_auth.embedder import MiniLMEmbedder


class TextAuthModelLoader:
    """Lazy, reusable loader for the MiniLM embedder and sklearn classifier."""

    def __init__(self, settings: TextAuthSettings | None = None) -> None:
        self.settings = settings or get_text_auth_settings()
        self._lock = Lock()
        self._embedder: MiniLMEmbedder | None = None
        self._classifier: LogisticRegression | None = None
        self._label_encoder: LabelEncoder | None = None

    @property
    def embedder(self) -> MiniLMEmbedder:
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
        """Load all text-auth assets once. If required=False, missing classifier files are tolerated."""
        if self._embedder and self._classifier and self._label_encoder:
            return

        with self._lock:
            if self._embedder is None:
                self._embedder = MiniLMEmbedder(self.settings)

            if self._classifier is None or self._label_encoder is None:
                classifier_path = self.settings.resolved_classifier_path
                label_encoder_path = self.settings.resolved_label_encoder_path

                if not classifier_path.exists() or not label_encoder_path.exists():
                    message = (
                        "Text-auth classifier files are missing. Train the classifier first: "
                        f"{classifier_path} and {label_encoder_path}"
                    )
                    if required:
                        raise FileNotFoundError(message)
                    return

                self._classifier = _patch_logistic_regression_compat(joblib.load(classifier_path))
                self._label_encoder = joblib.load(label_encoder_path)

    def is_ready(self) -> bool:
        """Return whether the saved classifier and label encoder exist."""
        return (
            Path(self.settings.resolved_classifier_path).exists()
            and Path(self.settings.resolved_label_encoder_path).exists()
        )


_TEXT_AUTH_LOADER: TextAuthModelLoader | None = None


def get_text_auth_model_loader() -> TextAuthModelLoader:
    """Return the process-wide text-auth model loader."""
    global _TEXT_AUTH_LOADER
    if _TEXT_AUTH_LOADER is None:
        _TEXT_AUTH_LOADER = TextAuthModelLoader()
    return _TEXT_AUTH_LOADER


def _patch_logistic_regression_compat(classifier: LogisticRegression) -> LogisticRegression:
    """Repair sklearn LogisticRegression pickles saved without legacy attrs."""
    if isinstance(classifier, LogisticRegression) and not hasattr(classifier, "multi_class"):
        classifier.multi_class = "auto"
    return classifier
