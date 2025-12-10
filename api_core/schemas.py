from pydantic import BaseModel, HttpUrl, Field
from typing import Optional

class VideoRequest(BaseModel):
    video_url: HttpUrl = Field(..., description="URL of the video to predict")

class PredictionResponse(BaseModel):
    prediction: str

class TranslationResponse(BaseModel):
    translation: str
    summary: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
