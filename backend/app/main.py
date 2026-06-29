"""
SHAZAM AI — FastAPI Application Entry Point
"""
import time
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.api.auth  import router as auth_router
from app.api.chat  import router as chat_router
from app.api.media import router as media_router
from app.api.video import router as video_router   # ← NEW

log = structlog.get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "shazam_ai_starting",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        providers=settings.available_providers,
    )
    yield
    log.info("shazam_ai_shutdown")


app = FastAPI(
    title="SHAZAM AI",
    description=(
        "Production autonomous AI platform. "
        "Chat, voice, code, research, image generation, image editing, "
        "video creation, AI voiceover, subtitles — all free & open source."
    ),
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    log.info("http", method=request.method, path=request.url.path,
             status=response.status_code, ms=round((time.time()-start)*1000))
    return response


# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(auth_router,  prefix="/api/v1")
app.include_router(chat_router,  prefix="/api/v1")
app.include_router(media_router, prefix="/api/v1")
app.include_router(video_router, prefix="/api/v1")   # ← NEW


@app.get("/", tags=["Health"])
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "docs": "/docs",
        "providers": settings.available_providers,
        "capabilities": [
            "chat", "streaming_chat", "voice_input", "text_to_speech",
            "image_generation", "image_editing", "background_removal",
            "upscaling", "video_from_text", "video_from_image",
            "slideshow", "ai_voiceover", "subtitles", "video_filters",
            "pdf_analysis", "web_research", "code_generation",
            "skill_auto_generation",
        ],
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "environment": settings.ENVIRONMENT}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("unhandled_exception", error=str(exc), path=request.url.path)
    return JSONResponse(status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__})
