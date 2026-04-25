from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.ml.voice_auth.config import get_voice_auth_settings
from app.ml.voice_auth.model_loader import get_voice_auth_model_loader
from app.ml.voice_auth.predict import predict_voice_auth
from app.ml.voice_auth.schemas import VoiceAuthResponse

router = APIRouter(prefix="/predict", tags=["voice-auth"])
settings = get_voice_auth_settings()


@router.post("/voice-auth", response_model=VoiceAuthResponse)
async def predict_voice_auth_endpoint(file: UploadFile = File(...)) -> VoiceAuthResponse:
    """Classify an uploaded voice clip as human or AI-generated."""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio file is required")

    suffix = settings.extension_from_filename(file.filename)
    if suffix not in settings.allowed_extensions:
        allowed = ", ".join(sorted(settings.allowed_extensions))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format '{suffix or 'unknown'}'. Allowed formats: {allowed}",
        )

    if file.content_type and not (
        file.content_type.startswith("audio/")
        or file.content_type in settings.allowed_content_types
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported content type '{file.content_type}'. Please upload an audio file.",
        )

    try:
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded audio file is empty")

        loader = get_voice_auth_model_loader()
        result = predict_voice_auth(payload, suffix=suffix, loader=loader)
        return VoiceAuthResponse(success=True, **result)
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice authentication failed: {exc}",
        ) from exc
