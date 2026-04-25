from pathlib import Path

import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier

from app.ml.voice_auth.config import VoiceAuthSettings


class ECAPAEmbedder:
    """SpeechBrain ECAPA-TDNN embedding extractor."""

    def __init__(self, settings: VoiceAuthSettings) -> None:
        self.settings = settings
        self.device = torch.device(settings.device)
        self.classifier = self._load_classifier(settings)

    def _load_classifier(self, settings: VoiceAuthSettings) -> EncoderClassifier:
        savedir = Path(settings.resolved_speechbrain_savedir)
        savedir.mkdir(parents=True, exist_ok=True)
        return EncoderClassifier.from_hparams(
            source=settings.speechbrain_source,
            savedir=str(savedir),
            run_opts={"device": settings.device},
        )

    def extract_embedding(self, waveform: np.ndarray) -> np.ndarray:
        """Extract a single 1D embedding vector from a standardized waveform."""
        if waveform.ndim != 1:
            raise ValueError("Expected a mono waveform with shape [num_samples]")

        tensor = torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.classifier.encode_batch(tensor)

        return embedding.squeeze().detach().cpu().numpy().astype(np.float32)
