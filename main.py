import os
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from openai import APIError, AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

WHISPER_MODEL = "whisper-1"
MAX_AUDIO_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}

app = FastAPI(title="Speech to text")


class Transcript(BaseModel):
    text: str


def _client() -> AsyncOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
    return AsyncOpenAI()


@app.post("/transcribe", response_model=Transcript)
async def transcribe(file: UploadFile = File(...)) -> Transcript:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported audio type")

    data = await file.read(MAX_AUDIO_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 25 MB")

    audio = BytesIO(data)
    audio.name = f"upload{suffix}"
    try:
        result = await _client().audio.transcriptions.create(model=WHISPER_MODEL, file=audio)
    except APIError:
        raise HTTPException(status_code=502, detail="Transcription failed") from None
    return Transcript(text=result.text)
