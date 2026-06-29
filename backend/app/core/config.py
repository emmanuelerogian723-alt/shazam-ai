"""
SHAZAM AI — Core Configuration
All settings loaded from environment variables. Never hardcode secrets.
"""
from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "SHAZAM AI"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"  # development | staging | production
    DEBUG: bool = False
    SECRET_KEY: str = "change-this-in-production-use-openssl-rand-hex-32"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:3001"]

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://shazam:shazam@localhost:5432/shazam_ai"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 3600

    # ── Vector DB (Qdrant) ────────────────────────────────────────────────────
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "shazam_memory"

    # ── Storage (S3-compatible) ───────────────────────────────────────────────
    STORAGE_ENDPOINT: Optional[str] = None      # e.g. https://s3.amazonaws.com
    STORAGE_BUCKET: str = "shazam-ai-uploads"
    STORAGE_ACCESS_KEY: Optional[str] = None
    STORAGE_SECRET_KEY: Optional[str] = None
    STORAGE_REGION: str = "us-east-1"
    MAX_UPLOAD_MB: int = 50

    # ── JWT ───────────────────────────────────────────────────────────────────
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # ── AI Providers ──────────────────────────────────────────────────────────
    # Groq — FREE tier: llama-3.3-70b, whisper-large-v3-turbo, playai-tts
    GROQ_API_KEY: Optional[str] = None

    # OpenRouter — many free models (meta-llama/llama-3.1-8b-instruct:free etc)
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # Optional paid providers — system falls back gracefully if not set
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    # Ollama — local models, no key needed
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_ENABLED: bool = False

    # Serper — web search (free tier: 2500 searches/month)
    SERPER_API_KEY: Optional[str] = None

    # ── Image Generation (FREE options) ──────────────────────────────────────
    # Hugging Face Inference API — free tier, open source models
    HUGGINGFACE_API_KEY: Optional[str] = None
    # Stability AI — fallback
    STABILITY_API_KEY: Optional[str] = None
    # Pollinations.ai — completely FREE, no key needed
    POLLINATIONS_ENABLED: bool = True
    POLLINATIONS_BASE_URL: str = "https://image.pollinations.ai"

    # ── Voice / TTS (FREE options) ─────────────────────────────────────────────
    # Groq TTS — playai-tts model (free tier)
    # ElevenLabs — free tier (10k chars/month)
    ELEVENLABS_API_KEY: Optional[str] = None
    # Kokoro — completely open-source TTS (self-hosted)
    KOKORO_ENABLED: bool = False

    # ── Telegram ─────────────────────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_WEBHOOK_URL: Optional[str] = None

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_DAY: int = 1000

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json | text

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def available_providers(self) -> List[str]:
        """Returns list of configured AI providers in priority order."""
        providers = []
        if self.GROQ_API_KEY:
            providers.append("groq")
        if self.OPENROUTER_API_KEY:
            providers.append("openrouter")
        if self.OPENAI_API_KEY:
            providers.append("openai")
        if self.ANTHROPIC_API_KEY:
            providers.append("anthropic")
        if self.GEMINI_API_KEY:
            providers.append("gemini")
        if self.OLLAMA_ENABLED:
            providers.append("ollama")
        return providers


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
