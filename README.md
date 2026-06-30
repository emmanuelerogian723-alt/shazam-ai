# ⚡ SHAZAM AI

> The open-source, free alternative to Higgsfield AI, Runway, ElevenLabs, and Midjourney — all in one platform.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![Stars](https://img.shields.io/github/stars/emmanuelerogian723-alt/shazam-ai?style=social)](https://github.com/emmanuelerogian723-alt/shazam-ai)

---

## What SHAZAM AI Can Do

| Feature | Tool Used | Cost |
|---|---|---|
| 💬 Chat & Reasoning | Groq llama-3.3-70b | FREE |
| 🎙 Voice → Text (STT) | Groq Whisper | FREE |
| 🔊 Text → Voice (TTS) | Groq PlayAI | FREE |
| 🎨 Image Generation | Pollinations.ai | 100% FREE |
| 🖼 Background Removal | rembg + U2Net | FREE (open source) |
| 📐 Image Upscaling 2x/4x | Real-ESRGAN / Pillow | FREE |
| ✨ Auto Image Enhance | Pillow | FREE |
| 🎬 Text → Video | Wan2.1 + Pollinations | FREE |
| 🎥 Image → Video | MoviePy + FFmpeg | FREE |
| 📽 Slideshow → Video | MoviePy | FREE |
| 🎤 Add AI Voiceover | Groq PlayAI TTS | FREE |
| 📝 Add Subtitles | FFmpeg SRT | FREE |
| 🎵 Add Background Music | FFmpeg | FREE |
| 🎨 Video Color Filters | FFmpeg | FREE |
| ✂️ Trim & Merge Clips | FFmpeg | FREE |
| 📊 GIF Export | FFmpeg | FREE |
| 🔍 Web Research | Serper.dev | 2500/month FREE |
| 💻 Code Generation | Qwen2.5-Coder | FREE |
| 🤖 Auto Skill Generator | LLM + Python | FREE |

---

## Quick Start (3 minutes)

### 1. Clone the repo

```bash
git clone https://github.com/emmanuelerogian723-alt/shazam-ai
cd shazam-ai/backend
cp .env.example .env
```

### 2. Get your FREE API keys (takes 2 minutes)

| Service | URL | What you get |
|---|---|---|
| Groq | https://console.groq.com | Chat + Voice STT + TTS |
| OpenRouter | https://openrouter.ai/keys | Free LLMs (DeepSeek, Llama, Mistral) |
| Serper | https://serper.dev | Web search (2500/month) |

Add them to your `.env` file:
```
GROQ_API_KEY=gsk_xxxxx
OPENROUTER_API_KEY=sk-or-xxxxx
SERPER_API_KEY=xxxxx
```

**Image generation (Pollinations.ai) requires NO API key — it just works.**

### 3. Run with Docker

```bash
cd docker
docker-compose up -d
```

### 4. Or run directly

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 5. Test it

```bash
curl http://localhost:8000/health
# Full API docs: http://localhost:8000/docs
```

---

## Telegram Bot Setup

1. Open Telegram → search `@BotFather`
2. Send `/newbot` → follow prompts → copy your token
3. Add to `.env`:
```
TELEGRAM_BOT_TOKEN=7123456789:AAxxxxxxxxxx
```
4. Run:
```bash
cd telegram
python bot.py
```

### Bot Commands
```
/start          — Welcome menu
/image [prompt] — Generate an image
/code [task]    — Write code
/research [query] — Web research
/clear          — Clear chat history
```

Or just send:
- A text message → AI responds
- A voice note → Auto-transcribed + AI responds
- A PDF or .txt file → Auto-summarized

---

## API Endpoints

### Chat
```
POST /api/v1/chat/message      — Text chat
POST /api/v1/chat/stream       — Streaming chat (SSE)
POST /api/v1/chat/voice        — Voice input/output
POST /api/v1/chat/tts          — Text to speech
```

### Image Generation & Editing
```
POST /api/v1/media/image/generate          — Generate image
POST /api/v1/media/document/analyze        — Analyze PDF/doc
POST /api/v1/create/image/remove-background — Remove background
POST /api/v1/create/image/upscale          — Upscale 2x/4x
POST /api/v1/create/image/enhance          — Auto enhance
POST /api/v1/create/image/filter           — Apply filter
```

### Video Creation & Editing
```
POST /api/v1/create/video/from-text        — Text → Video
POST /api/v1/create/video/from-image       — Image → Video (Ken Burns)
POST /api/v1/create/video/slideshow        — Images → Slideshow
POST /api/v1/create/video/add-voiceover    — Add AI voice
POST /api/v1/create/video/add-subtitles    — Burn subtitles
POST /api/v1/create/video/filter           — Color grade
POST /api/v1/create/video/gif              — Export GIF
```

### Auth
```
POST /api/v1/auth/register
POST /api/v1/auth/login
POST /api/v1/auth/refresh
GET  /api/v1/auth/me
```

---

## Project Structure

```
shazam-ai/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app entry point
│   │   ├── core/
│   │   │   ├── config.py            # All settings (env vars)
│   │   │   └── ai_engine.py         # Multi-provider AI (auto-fallback)
│   │   ├── agents/
│   │   │   ├── orchestrator.py      # Routes tasks to correct agent
│   │   │   ├── planner.py           # Breaks tasks into steps
│   │   │   ├── coding_agent.py      # Code generation & debugging
│   │   │   ├── research_agent.py    # Web search + document analysis
│   │   │   ├── image_agent.py       # Image generation
│   │   │   ├── image_edit_agent.py  # Image editing
│   │   │   ├── video_agent.py       # Video creation & editing
│   │   │   ├── writing_agent.py     # Content creation
│   │   │   └── skill_generator.py   # Auto-creates new skills
│   │   ├── skills/
│   │   │   ├── video/
│   │   │   │   └── generator.py     # Full video engine (MoviePy + FFmpeg)
│   │   │   └── image_edit/
│   │   │       └── editor.py        # Full image edit engine (rembg + Pillow)
│   │   └── api/
│   │       ├── auth.py              # JWT authentication
│   │       ├── chat.py              # Chat + voice endpoints
│   │       ├── media.py             # Image gen + document analysis
│   │       └── video.py             # Video + image edit endpoints
│   ├── requirements.txt
│   └── .env.example
├── telegram/
│   └── bot.py                       # Full-featured Telegram bot
├── docker/
│   ├── Dockerfile.backend
│   └── docker-compose.yml
└── README.md
```

---

## Environment Variables Reference

```bash
# ── REQUIRED (all free) ──────────────────────────────────
GROQ_API_KEY=          # console.groq.com — chat, voice, TTS
OPENROUTER_API_KEY=    # openrouter.ai — free LLMs
SERPER_API_KEY=        # serper.dev — web search
SECRET_KEY=            # any random 32-char string (for JWT)

# ── OPTIONAL (unlock more features) ──────────────────────
HUGGINGFACE_API_KEY=   # huggingface.co — Real-ESRGAN upscaling, Wan2.1 video
TELEGRAM_BOT_TOKEN=    # from @BotFather — run the Telegram bot
OPENAI_API_KEY=        # optional paid fallback
ANTHROPIC_API_KEY=     # optional paid fallback

# ── IMAGE GENERATION (no key needed) ─────────────────────
POLLINATIONS_ENABLED=true   # 100% free, default ON

# ── DATABASE (optional, use Neon.tech free tier) ─────────
DATABASE_URL=postgresql+asyncpg://user:pass@host/shazam_ai
REDIS_URL=redis://default:pass@host:6379
```

---

## Deploy on Render (Free Tier)

1. Fork this repo
2. Go to https://render.com → New Web Service
3. Connect your GitHub repo
4. Set:
   - Root directory: `backend`
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables from `.env.example`
6. Deploy → your URL will be `https://your-app.onrender.com`

---

## Contributing

Pull requests are welcome! Areas that need help:

- Frontend (Next.js web UI)
- More video effects and transitions
- WhatsApp integration
- Discord bot
- More HuggingFace video models

```bash
git clone https://github.com/emmanuelerogian723-alt/shazam-ai
cd shazam-ai
git checkout -b feature/your-feature
# make changes
git commit -m "feat: your feature"
git push origin feature/your-feature
# open a Pull Request
```

---

## License

MIT License — free to use, modify, and distribute.

---

## Star History

If SHAZAM AI saved you money or helped your project, please ⭐ star the repo.
It helps others discover it and keeps the project alive.

https://github.com/emmanuelerogian723-alt/shazam-ai

---

*Built with ❤️ — Because powerful AI tools should be free for everyone.*
