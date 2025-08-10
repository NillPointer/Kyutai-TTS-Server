import argparse
import sys
import os
from pathlib import Path
from typing import Optional
import logging
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO, TTSModel
from io import BytesIO
import soundfile as sf
from huggingface_hub import snapshot_download
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Kyutai TTS API", description="OpenAI-compatible TTS API server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (loaded once)
tts_model = None
model_loaded = False

# Get voices directory from environment variable or use default
VOICES_DIR = os.getenv("VOICES_DIR", "./voices")

class SpeechRequest(BaseModel):
    model: Optional[str] = ""
    input: str
    voice: str
    response_format: str = "wav"
    speed: float = 1.0

class SpeechResponse(BaseModel):
    audio: bytes

def load_model():
    """Load the TTS model once at startup"""
    global tts_model, model_loaded

    if model_loaded:
        return tts_model

    try:
        logger.info("Loading TTS model...")
        checkpoint_info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
        tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info,
            n_q=32,
            temp=0.6,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info("TTS model loaded successfully")
        
        logger.info("Loading Voices...")
        snapshot_download(repo_id=DEFAULT_DSM_TTS_VOICE_REPO)
        logger.info("Voices voices loaded successfully")

        model_loaded = True
        return tts_model
    except Exception as e:
        logger.info(f"Error loading TTS model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load TTS model: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts"""
    global tts_model
    tts_model = load_model()

def generate_audio_bytes(request: SpeechRequest) -> bytes:
    """Generate audio stream for the given request in WAV format"""
    if not tts_model:
        raise HTTPException(status_code=500, detail="TTS model not loaded")

    try:
        logger.info(f"Generating audio for: '{request.input[:50]}{'...' if len(request.input) > 50 else ''}'")
        # Prepare the script with the input text
        entries = tts_model.prepare_script([request.input], padding_between=1)

        # Get voice path
        voice_path = tts_model.get_voice_path(request.voice)
        logger.info(f"Used voice path: {voice_path}")

        # Create condition attributes
        condition_attributes = tts_model.make_condition_attributes(
            [voice_path], cfg_coef=2.0
        )

        # Generate audio
        pcms = []

        def _on_frame(frame):
            if (frame != -1).all():
                pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                pcms.append(torch.clamp(torch.from_numpy(pcm[0, 0]), -1, 1).numpy())

        # Generate audio with streaming
        all_entries = [entries]
        all_condition_attributes = [condition_attributes]

        logger.info("Starting voice generation")
        with tts_model.mimi.streaming(len(all_entries)):
            result = tts_model.generate(all_entries, all_condition_attributes, on_frame=_on_frame)

        # Concatenate all PCM frames
        if not pcms:
            raise HTTPException(status_code=500, detail="Failed to generate audio")

        audio = np.concatenate(pcms, axis=-1)

        # Apply speed adjustment if needed
        if request.speed != 1.0:
            # Simple speed adjustment by resampling
            from scipy import signal
            audio_length = len(audio)
            new_length = int(audio_length / request.speed)
            audio = signal.resample(audio, new_length)

        # Convert to bytes (WAV format)
        audio_bytes = BytesIO()
        sf.write(audio_bytes, audio, samplerate=tts_model.mimi.sample_rate, format='WAV')
        audio_bytes.seek(0)

        logger.info("Audio generated successfully")
        return audio_bytes.read()

    except Exception as e:
        logger.info(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate speech: {str(e)}")

def convert_audio_format(audio_bytes: bytes, target_format: str) -> bytes:
    """
    Convert audio bytes from WAV format to the target format using pydub.

    Args:
        audio_bytes: The audio data in WAV format
        target_format: The target audio format (e.g., 'mp3', 'flac', 'ogg')

    Returns:
        The converted audio bytes
    """
    try:
        # Create a BytesIO object for the input audio
        audio_file = BytesIO(audio_bytes)

        # Read the WAV audio using pydub
        audio = AudioSegment.from_wav(audio_file)

        # Convert to the target format
        output_file = BytesIO()
        if target_format.lower() == 'mp3':
            audio.export(output_file, format='mp3')
        elif target_format.lower() == 'flac':
            audio.export(output_file, format='flac')
        elif target_format.lower() == 'ogg':
            audio.export(output_file, format='ogg')
        else:
            # Default to WAV if format is not supported
            audio.export(output_file, format='wav')

        output_file.seek(0)
        return output_file.read()
    except Exception as e:
        logger.error(f"Error converting audio format: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to convert audio format: {str(e)}")

@app.post("/v1/audio/speech")
async def generate_speech(request: SpeechRequest) -> bytes:
    """
    Generate speech from text using the TTS model.

    This endpoint is compatible with OpenAI's /v1/audio/speech API.

    Args:
        request: SpeechRequest object containing input parameters

    Returns:
        StreamingResponse containing generated audio in the requested format
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
    return {"status": "healthy", "model_loaded": model_loaded}

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the TTS API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload during development")

    args = parser.parse_args()

    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)