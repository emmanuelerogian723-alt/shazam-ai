"""
SHAZAM AI — Chat API Routes
Handles text chat, streaming, voice input/output.
"""
import base64
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

from app.agents.orchestrator import get_orchestrator
from app.core.ai_engine import ai_engine
from app.api.auth import get_current_user

router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Request / Response Models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    history: list = Field(default_factory=list)
    session_id: Optional[str] = None
    stream: bool = False


class ChatResponse(BaseModel):
    type: str
    content: str
    latency_ms: Optional[int] = None
    plan: Optional[dict] = None


class VoiceResponse(BaseModel):
    transcription: str
    response: str
    audio_base64: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────
@router.post("/message", response_model=ChatResponse)
async def chat_message(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """Send a text message and get a response."""
    orchestrator = get_orchestrator(user_id=str(current_user["id"]))
    result = await orchestrator.handle(
        message=req.message,
        context={"history": req.history},
    )
    return ChatResponse(
        type=result.get("type", "text"),
        content=result.get("content", ""),
        latency_ms=result.get("latency_ms"),
        plan=result.get("plan"),
    )


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """Stream a chat response token by token."""
    orchestrator = get_orchestrator(user_id=str(current_user["id"]))

    async def generate():
        async for chunk in orchestrator.stream_handle(
            message=req.message,
            context={"history": req.history},
        ):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/voice", response_model=VoiceResponse)
async def voice_chat(
    audio: UploadFile = File(...),
    respond_with_audio: bool = Form(default=False),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload voice → transcribe (Groq Whisper) → respond → optionally return audio.
    All using FREE Groq models.
    """
    if audio.content_type not in ["audio/wav","audio/mpeg","audio/ogg","audio/webm","audio/mp4","audio/flac","audio/m4a"]:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    audio_bytes = await audio.read()
    if len(audio_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Audio file too large (max 25MB)")

    # Step 1: Speech → Text (Groq Whisper — FREE)
    transcription = await ai_engine.transcribe(
        audio_bytes=audio_bytes,
        filename=audio.filename or "audio.wav",
    )

    # Step 2: Generate text response
    orchestrator = get_orchestrator(user_id=str(current_user["id"]))
    result = await orchestrator.handle(message=transcription)
    text_response = result.get("content", "")

    # Step 3: Text → Speech (Groq PlayAI — FREE)
    audio_b64 = None
    if respond_with_audio and text_response:
        tts_bytes = await ai_engine.text_to_speech(text=text_response[:2000])
        audio_b64 = base64.b64encode(tts_bytes).decode()

    return VoiceResponse(
        transcription=transcription,
        response=text_response,
        audio_base64=audio_b64,
    )


@router.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    voice: str = Form(default="Arista-PlayAI"),
    current_user: dict = Depends(get_current_user),
):
    """Convert text to speech using Groq PlayAI (FREE tier)."""
    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long (max 5000 chars)")

    audio_bytes = await ai_engine.text_to_speech(text=text, voice=voice)
    return Response(content=audio_bytes, media_type="audio/wav")
