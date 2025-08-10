# app/main.py

import argparse
import uvicorn
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from models import SpeechRequest, SpeechResponse
from tts import load_model, generate_audio_bytes, is_model_loaded
from utils import convert_audio_format
from config import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model when the application starts"""
    logger.info("TTS Server: Initializing application...")
    load_model()
    yield

app = FastAPI(
    title="Kyutai TTS API", 
    description="OpenAI-compatible TTS API server",
    lifespan=lifespan
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/audio/speech", response_model=bytes)
async def generate_speech(request: SpeechRequest) -> Response:
    """
    Generate speech from text using the TTS model.

    This endpoint is compatible with OpenAI's /v1/audio/speech API.

    Args:
        request: SpeechRequest object containing input parameters

    Returns:
        Response containing generated audio in the requested format
    """
    try:
        # Generate audio in WAV format
        audio_data = generate_audio_bytes(request)

        # Convert to the requested format if different from WAV
        if request.response_format.lower() != 'wav':
            audio_data = convert_audio_format(audio_data, request.response_format)

        # Set appropriate content type
        content_types = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "flac": "audio/flac",
            "ogg": "audio/ogg"
        }

        return Response(
            content=audio_data,
            media_type=content_types.get(request.response_format.lower(), "audio/wav"),
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
            }
        )

    except Exception as e:
        logger.error(f"Speech generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the TTS API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload during development")

    args = parser.parse_args()

    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)