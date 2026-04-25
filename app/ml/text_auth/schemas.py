from pydantic import BaseModel, Field


class TextAuthRequest(BaseModel):
    """Request body for text authentication."""

    text: str = Field(..., min_length=1, examples=["This is a sample paragraph to classify."])


class TextAuthResponse(BaseModel):
    """Response returned by the text authentication endpoint."""

    success: bool = True
    predicted_label: str = Field(..., examples=["ai"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.91])
    class_probabilities: dict[str, float] = Field(
        ...,
        examples=[{"human": 0.09, "ai": 0.91}],
    )
