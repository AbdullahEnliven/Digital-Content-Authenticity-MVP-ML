import numpy as np

from app.ml.text_auth.model_loader import TextAuthModelLoader, get_text_auth_model_loader
from app.ml.text_auth.preprocess import preprocess_text


def predict_text_auth(text: str, loader: TextAuthModelLoader | None = None) -> dict:
    """Run the full text-auth inference pipeline."""
    model_loader = loader or get_text_auth_model_loader()
    model_loader.load_models(required=True)

    cleaned_text = preprocess_text(text, model_loader.settings)
    embedding = model_loader.embedder.encode_text(cleaned_text).reshape(1, -1)

    if hasattr(model_loader.classifier, "predict_proba"):
        probabilities = model_loader.classifier.predict_proba(embedding)[0]
    else:
        decision = model_loader.classifier.decision_function(embedding)
        probabilities = _softmax(np.atleast_2d(decision))[0]

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


def _softmax(values: np.ndarray) -> np.ndarray:
    """Convert decision scores to probabilities for classifiers without predict_proba."""
    if values.shape[1] == 1:
        values = np.hstack([-values, values])
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)
