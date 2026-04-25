from pydantic import BaseModel, Field


class VoiceAuthResponse(BaseModel):
    """Response returned by the voice authentication endpoint."""

    success: bool = True
    predicted_label: str = Field(..., examples=["ai"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.93])
    class_probabilities: dict[str, float] = Field(
        ...,
        examples=[{"human": 0.07, "ai": 0.93}],
    )
