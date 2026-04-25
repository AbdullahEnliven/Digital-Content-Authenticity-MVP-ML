from fastapi import APIRouter, HTTPException, status

from app.ml.text_auth.config import get_text_auth_settings
from app.ml.text_auth.model_loader import get_text_auth_model_loader
from app.ml.text_auth.predict import predict_text_auth
from app.ml.text_auth.schemas import TextAuthRequest, TextAuthResponse

router = APIRouter(prefix="/predict", tags=["text-auth"])
settings = get_text_auth_settings()


@router.post("/text-auth", response_model=TextAuthResponse)
async def predict_text_auth_endpoint(request: TextAuthRequest) -> TextAuthResponse:
    """Classify text as human-written or AI-generated."""
    try:
        result = predict_text_auth(request.text, loader=get_text_auth_model_loader())
        return TextAuthResponse(success=True, **result)
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
            detail=f"Text authentication failed: {exc}",
        ) from exc
