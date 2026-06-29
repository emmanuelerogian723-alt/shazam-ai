"""
SHAZAM AI — Multi-Provider AI Engine
Automatically falls back to next provider if one fails.

FREE MODELS USED:
- Groq:        llama-3.3-70b-versatile (text), llama-3.1-8b-instant (fast)
               whisper-large-v3-turbo (STT), playai-tts-arabic (TTS)
- OpenRouter:  meta-llama/llama-3.3-70b-instruct:free
               meta-llama/llama-3.1-8b-instruct:free
               mistralai/mistral-7b-instruct:free
               google/gemma-3-27b-it:free
               deepseek/deepseek-r1:free (reasoning)
- Pollinations: FREE image generation (no key)
- HuggingFace: stabilityai/stable-diffusion-xl-base-1.0 (free tier)
"""
import asyncio
import base64
import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx
import structlog
from groq import AsyncGroq
from openai import AsyncOpenAI

from app.core.config import settings

log = structlog.get_logger(__name__)

# ── Model Registry ────────────────────────────────────────────────────────────
PROVIDER_MODELS = {
    "groq": {
        "default":  "llama-3.3-70b-versatile",
        "fast":     "llama-3.1-8b-instant",
        "vision":   "meta-llama/llama-4-scout-17b-16e-instruct",
        "reasoning":"llama-3.3-70b-versatile",
        "stt":      "whisper-large-v3-turbo",
        "tts":      "playai-tts",
    },
    "openrouter": {
        "default":  "meta-llama/llama-3.3-70b-instruct:free",
        "fast":     "meta-llama/llama-3.1-8b-instruct:free",
        "vision":   "google/gemma-3-27b-it:free",
        "reasoning":"deepseek/deepseek-r1:free",
        "coding":   "qwen/qwen-2.5-coder-32b-instruct:free",
    },
    "openai": {
        "default":  "gpt-4o-mini",
        "fast":     "gpt-4o-mini",
        "vision":   "gpt-4o",
        "reasoning":"o1-mini",
    },
    "anthropic": {
        "default":  "claude-3-haiku-20240307",
        "reasoning":"claude-3-5-sonnet-20241022",
    },
    "ollama": {
        "default":  "llama3.2",
        "coding":   "codellama",
        "fast":     "phi3",
    },
}

# ── AI Engine ─────────────────────────────────────────────────────────────────
class AIEngine:
    """Central AI engine. Tries providers in order, auto-falls back on failure."""

    def __init__(self):
        self._groq: Optional[AsyncGroq] = None
        self._openrouter: Optional[AsyncOpenAI] = None
        self._openai: Optional[AsyncOpenAI] = None

    @property
    def groq(self) -> AsyncGroq:
        if not self._groq and settings.GROQ_API_KEY:
            self._groq = AsyncGroq(api_key=settings.GROQ_API_KEY)
        return self._groq

    @property
    def openrouter(self) -> AsyncOpenAI:
        if not self._openrouter and settings.OPENROUTER_API_KEY:
            self._openrouter = AsyncOpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
            )
        return self._openrouter

    @property
    def openai_client(self) -> AsyncOpenAI:
        if not self._openai and settings.OPENAI_API_KEY:
            self._openai = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        return self._openai

    # ── Chat ──────────────────────────────────────────────────────────────────
    async def chat(
        self,
        messages: List[Dict[str, str]],
        mode: str = "default",  # default | fast | reasoning | coding | vision
        max_tokens: int = 4096,
        temperature: float = 0.7,
        system: Optional[str] = None,
    ) -> str:
        """Send chat, auto-fallback across providers."""
        if system:
            messages = [{"role": "system", "content": system}] + messages

        for provider in settings.available_providers:
            try:
                return await self._chat_with_provider(
                    provider, messages, mode, max_tokens, temperature
                )
            except Exception as e:
                log.warning("provider_failed", provider=provider, error=str(e))
                continue

        raise RuntimeError("All AI providers failed. Check your API keys.")

    async def _chat_with_provider(
        self,
        provider: str,
        messages: List[Dict],
        mode: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        model = PROVIDER_MODELS.get(provider, {}).get(mode) or \
                PROVIDER_MODELS.get(provider, {}).get("default", "")

        if provider == "groq" and self.groq:
            resp = await self.groq.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content

        if provider == "openrouter" and self.openrouter:
            resp = await self.openrouter.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_headers={
                    "HTTP-Referer": "https://shazamai.app",
                    "X-Title": "SHAZAM AI",
                },
            )
            return resp.choices[0].message.content

        if provider == "openai" and self.openai_client:
            resp = await self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content

        if provider == "ollama":
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/api/chat",
                    json={"model": model, "messages": messages, "stream": False},
                )
                return resp.json()["message"]["content"]

        raise ValueError(f"Provider {provider} not configured")

    # ── Streaming Chat ────────────────────────────────────────────────────────
    async def stream_chat(
        self,
        messages: List[Dict],
        mode: str = "default",
        system: Optional[str] = None,
    ) -> AsyncIterator[str]:
        if system:
            messages = [{"role": "system", "content": system}] + messages

        for provider in settings.available_providers:
            try:
                async for chunk in self._stream_with_provider(provider, messages, mode):
                    yield chunk
                return
            except Exception as e:
                log.warning("stream_provider_failed", provider=provider, error=str(e))
                continue

    async def _stream_with_provider(
        self, provider: str, messages: List[Dict], mode: str
    ) -> AsyncIterator[str]:
        model = PROVIDER_MODELS.get(provider, {}).get(mode) or \
                PROVIDER_MODELS.get(provider, {}).get("default", "")

        if provider == "groq" and self.groq:
            stream = await self.groq.chat.completions.create(
                model=model, messages=messages, stream=True, max_tokens=4096
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        elif provider == "openrouter" and self.openrouter:
            stream = await self.openrouter.chat.completions.create(
                model=model, messages=messages, stream=True, max_tokens=4096,
                extra_headers={"HTTP-Referer": "https://shazamai.app", "X-Title": "SHAZAM AI"},
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    # ── Speech to Text (Groq Whisper — FREE) ─────────────────────────────────
    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: str = "en",
    ) -> str:
        """Transcribe audio using Groq Whisper (whisper-large-v3-turbo — free tier)."""
        if not self.groq:
            raise RuntimeError("GROQ_API_KEY required for transcription")

        transcription = await self.groq.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model="whisper-large-v3-turbo",
            language=language,
            response_format="text",
            temperature=0.0,
        )
        return transcription

    # ── Text to Speech (Groq PlayAI — FREE tier) ─────────────────────────────
    async def text_to_speech(
        self,
        text: str,
        voice: str = "Arista-PlayAI",  # free voices on Groq
        model: str = "playai-tts",
    ) -> bytes:
        """Convert text to speech using Groq's PlayAI TTS (free tier)."""
        if not self.groq:
            raise RuntimeError("GROQ_API_KEY required for TTS")

        response = await self.groq.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format="wav",
        )
        return response.read()

    # ── Image Generation (FREE — Pollinations.ai, no key needed) ─────────────
    async def generate_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        model: str = "flux",  # flux | turbo | dall-e-3
        enhance: bool = True,
    ) -> bytes:
        """
        Generate image using Pollinations.ai (FREE, no API key needed).
        Falls back to HuggingFace if available.
        """
        if settings.POLLINATIONS_ENABLED:
            try:
                return await self._pollinations_generate(prompt, width, height, model)
            except Exception as e:
                log.warning("pollinations_failed", error=str(e))

        if settings.HUGGINGFACE_API_KEY:
            return await self._huggingface_generate(prompt, width, height)

        raise RuntimeError("No image generation provider available")

    async def _pollinations_generate(
        self, prompt: str, width: int, height: int, model: str
    ) -> bytes:
        """Pollinations.ai — completely FREE, no API key."""
        import urllib.parse
        encoded = urllib.parse.quote(prompt)
        url = f"{settings.POLLINATIONS_BASE_URL}/prompt/{encoded}?width={width}&height={height}&model={model}&nologo=true"
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

    async def _huggingface_generate(
        self, prompt: str, width: int, height: int
    ) -> bytes:
        """HuggingFace Inference API — free tier available."""
        url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}
        payload = {"inputs": prompt, "parameters": {"width": width, "height": height}}
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.content

    # ── Web Search (Serper.dev — free tier 2500/month) ────────────────────────
    async def web_search(self, query: str, num: int = 5) -> List[Dict]:
        """Search the web using Serper.dev (2500 free searches/month)."""
        if not settings.SERPER_API_KEY:
            return [{"error": "SERPER_API_KEY not configured"}]

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": settings.SERPER_API_KEY, "Content-Type": "application/json"},
                json={"q": query, "num": num},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("organic", [])[:num]

    # ── Reasoning (DeepSeek R1 via OpenRouter — FREE) ────────────────────────
    async def reason(self, problem: str, context: str = "") -> Dict[str, str]:
        """Deep reasoning using DeepSeek R1 (free on OpenRouter)."""
        messages = [
            {
                "role": "user",
                "content": f"{context}\n\nProblem: {problem}" if context else problem,
            }
        ]
        result = await self.chat(messages, mode="reasoning")
        return {"reasoning": result, "model": "deepseek/deepseek-r1:free"}


# ── Singleton ─────────────────────────────────────────────────────────────────
ai_engine = AIEngine()
