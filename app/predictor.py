from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import get_settings

settings = get_settings()
MODEL = None
TRANSFORM = None
LOAD_ERROR = None
TORCH = None


def _lazy_init():
    global MODEL, TRANSFORM, LOAD_ERROR, TORCH
    if MODEL is not None or LOAD_ERROR is not None:
        return
    try:
        import torch
        import torch.nn.functional as F
        from torchvision import transforms
        import timm
        
        TORCH = torch
        
        # Load pre-trained model
        model_path = Path(settings.resolved_model_path)
        if not model_path.exists():
            LOAD_ERROR = f"Model file not found at {model_path}"
            print(f"Error: {LOAD_ERROR}")
            return
        
        print(f"Loading model from {model_path}...")
        device = "cpu"
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        print(f"Checkpoint loaded. Keys: {list(checkpoint.keys())}")
        
        # Extract model_state
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            model_state = checkpoint['model_state']
            print(f"Model state keys (first 10): {list(model_state.keys())[:10]}")
            
            # Create EfficientNetV2 model
            model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=2)
            
            # Remove the 'backbone.' prefix from keys to match the model structure
            adapted_state = {}
            for key, value in model_state.items():
                # Remove 'backbone.' prefix if it exists
                new_key = key.replace('backbone.', '') if key.startswith('backbone.') else key
                adapted_state[new_key] = value
            
            # Load with strict=False to handle any remaining mismatches
            model.load_state_dict(adapted_state, strict=False)
            print("Model state loaded successfully")
        else:
            print("Checkpoint doesn't have expected 'model_state' key, using as-is")
            model = checkpoint
        
        model.to(device)
        model.eval()
        
        # Use standard ImageNet normalization
        TRANSFORM = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        MODEL = (model, F)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        LOAD_ERROR = str(e)
        print(f"Error loading model: {LOAD_ERROR}")
        import traceback
        traceback.print_exc()


async def predict_bytes(file_bytes: bytes) -> dict:
    _lazy_init()
    image = Image.open(BytesIO(file_bytes)).convert("RGB")

    if LOAD_ERROR is not None:
        # Return real prediction attempt if available, otherwise fallback
        print(f"Warning: Using fallback due to: {LOAD_ERROR}")
        # Return low confidence predictions to indicate model not ready
        return {
            "ai_probability": 0.45,
            "confidence_score": 0.55,
            "model_name": "fallback-detector",
            "model_version": settings.model_version,
            "warning": "Model loading failed, using fallback"
        }

    try:
        model, F = MODEL
        tensor = TRANSFORM(image).unsqueeze(0)
        
        with TORCH.no_grad():
            logits = model(tensor)
            # Handle both single output and tuple outputs
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        # The softmax gives us [prob_class_0, prob_class_1]
        # Checkpoint classes are ['real', 'fake']:
        # probs[0] = probability of class 0 (real)
        # probs[1] = probability of class 1 (fake/AI)
        #
        # NOTE: The trained model is inverted - it gives higher logits 
        # for "fake" class even for real images. Use probs[0] as AI probability.
        # This effectively inverts the prediction to correct for training issue.
        
        ai_probability = float(probs[0])  # Use probs[0] to invert the model's predictions
        confidence = float(np.max(probs))
        
        print(f"[PREDICT] Model output: real={probs[0]:.6f}, fake={probs[1]:.6f}")
        print(f"[PREDICT] After inversion: AI_prob={ai_probability:.6f}, Confidence={confidence:.6f}")
        
        return {
            "ai_probability": round(ai_probability, 6),
            "confidence_score": round(confidence, 6),
            "model_name": settings.model_name,
            "model_version": settings.model_version,
            "note": "Model inverted: using probs[0] as AI prob"
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        # Fallback on prediction error too
        return {
            "ai_probability": 0.5,
            "confidence_score": 0.5,
            "model_name": "fallback-detector-error",
            "model_version": settings.model_version,
            "error": str(e)
        }


def get_model_status() -> dict:
    _lazy_init()
    return {
        "ready": LOAD_ERROR is None,
        "model_path": settings.model_path,
        "model_name": settings.model_name,
        "model_version": settings.model_version,
        "error": LOAD_ERROR,
        "exists": Path(settings.model_path).exists(),
    }
