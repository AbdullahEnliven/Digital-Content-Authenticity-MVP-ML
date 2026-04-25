import os

import numpy as np

from app.ml.text_auth.config import TextAuthSettings

# Keep Transformers/SentenceTransformers in PyTorch-only mode. This avoids importing
# TensorFlow/Keras from global Python environments where Keras 3 may be installed.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from sentence_transformers import SentenceTransformer


class MiniLMEmbedder:
    """SentenceTransformers MiniLM embedding extractor."""

    def __init__(self, settings: TextAuthSettings) -> None:
        self.settings = settings
        self.model = SentenceTransformer(settings.embedding_model_name, device=settings.device)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a list of preprocessed texts into normalized embeddings."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.settings.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32, copy=False)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode one preprocessed text into a single embedding vector."""
        return self.encode_texts([text])[0]
