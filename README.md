# ⚡ SHAZAM AI

> Production-ready autonomous AI platform. Multi-agent. Multi-provider. Free-first.

---

## Free AI Models Used

| Capability | Provider | Model | Cost |
|---|---|---|---|
| Chat / Reasoning | Groq | llama-3.3-70b-versatile | FREE tier |
| Fast responses | Groq | llama-3.1-8b-instant | FREE tier |
| Speech to Text | Groq | whisper-large-v3-turbo | FREE tier |
| Text to Speech | Groq | playai-tts | FREE tier |
| Advanced reasoning | OpenRouter | deepseek/deepseek-r1:free | FREE |
| Coding | OpenRouter | qwen/qwen-2.5-coder-32b:free | FREE |
| Vision | OpenRouter | google/gemma-3-27b-it:free | FREE |
| Image generation | Pollinations.ai | flux | 100% FREE, no key |
| Web search | Serper.dev | - | 2500/month FREE |

---

## Quick Start

### 1. Clone & configure

```bash
git clone https://github.com/your-username/shazam-ai
cd shazam-ai/backend
cp .env.example .env
# Fill in your API keys
```

### 2. Add your API keys in .env

Minimum required (all FREE):
```
GROQ_API_KEY=gsk_xxx          # console.groq.com (free)
OPENROUTER_API_KEY=sk-or-xxx  # openrouter.ai (free models available)
SERPER_API_KEY=xxx             # serper.dev (2500 free/month)
```

### 3. Run with Docker

```bash
cd docker
docker-compose up -d
```

### 4. Test it

```bash
curl http://localhost:8000/health
# API docs: http://localhost:8000/docs
```

### 5. Telegram Bot

```
TELEGRAM_BOT_TOKEN=xxx  # from @BotFather
```
```bash
cd telegram && python bot.py
```

---

## Architecture

```
shazam-ai/
  backend/
    app/
      main.py              FastAPI app
      core/
        config.py          All settings
        ai_engine.py       Multi-provider AI (Groq, OpenRouter, etc.)
      agents/
        orchestrator.py    Routes tasks to correct agents
        planner.py         Breaks tasks into steps
        coding_agent.py    Code generation & debugging
        research_agent.py  Web search & document analysis
        image_agent.py     Image generation (Pollinations FREE)
        writing_agent.py   Content creation
        skill_generator.py Auto-creates new skills
      api/
        auth.py            JWT authentication
        chat.py            Chat, voice, streaming
        media.py           Images, documents
  telegram/
    bot.py                 Full Telegram bot
  docker/
    docker-compose.yml
    Dockerfile.backend
  .github/workflows/
    ci.yml                 GitHub Actions CI/CD
```

---

## API Keys You Need

| Service | URL | Free Tier | Used For |
|---|---|---|---|
| Groq | console.groq.com | Yes, generous | Chat, Voice STT, TTS |
| OpenRouter | openrouter.ai | Yes (free models) | Reasoning, Coding |
| Serper | serper.dev | 2500/month | Web search |
| Pollinations | pollinations.ai | 100% free | Image generation |
| Neon | neon.tech | 0.5GB free | PostgreSQL |
| Upstash | upstash.com | 10K req/day | Redis |
| Telegram | @BotFather | Free | Telegram bot |

---

## Render Deployment

1. New Web Service → connect GitHub repo
2. Root directory: `backend`
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add all env vars from `.env.example`
