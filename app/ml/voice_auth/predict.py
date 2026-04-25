import numpy as np

from app.ml.voice_auth.model_loader import VoiceAuthModelLoader, get_voice_auth_model_loader
from app.ml.voice_auth.preprocess import preprocess_audio_file
from app.ml.voice_auth.utils import temporary_audio_file


def predict_voice_auth(
    file_bytes: bytes,
    suffix: str,
    loader: VoiceAuthModelLoader | None = None,
) -> dict:
    """Run the full voice-auth inference pipeline on uploaded audio bytes."""
    model_loader = loader or get_voice_auth_model_loader()
    model_loader.load_models(required=True)

    with temporary_audio_file(file_bytes, suffix=suffix) as path:
        waveform = preprocess_audio_file(path, model_loader.settings)

    embedding = model_loader.embedder.extract_embedding(waveform).reshape(1, -1)
    probabilities = model_loader.classifier.predict_proba(embedding)[0]
    predicted_index = int(np.argmax(probabilities))
    labels = model_loader.label_encoder.inverse_transform(np.arange(len(probabilities)))

    class_probabilities = {
        str(label): round(float(probability), 6)
        for label, probability in zip(labels, probabilities, strict=True)
    }
    predicted_label = str(labels[predicted_index])

    return {
        "predicted_label": predicted_label,
        "confidence": round(float(probabilities[predicted_index]), 6),
        "class_probabilities": class_probabilities,
    }
