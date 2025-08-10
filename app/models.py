# app/models.py

from typing import Optional
from pydantic import BaseModel

class SpeechRequest(BaseModel):
    """Request model for text-to-speech generation.

    Attributes:
        model: Optional model name (not currently used)
        input: The text to convert to speech
        voice: The voice to use for generation
        response_format: The format of the audio response (default: 'wav')
        speed: Speed adjustment for the generated speech (default: 1.0)
    """
    model: Optional[str] = ""
    input: str
    voice: str
    response_format: str = "wav"
    speed: float = 1.0

class SpeechResponse(BaseModel):
    """Response model for text-to-speech generation.

    Attributes:
        audio: The generated audio bytes
    """
    audio: bytes