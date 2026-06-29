"""
SHAZAM AI — Telegram Bot
Full-featured bot with voice, images, files, and streaming responses.
"""
import asyncio
import base64
import logging
from pathlib import Path
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters,
)
from app.agents.orchestrator import get_orchestrator
from app.core.ai_engine import ai_engine
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Session store (in-memory — use Redis in production) ──────────────────────
user_sessions: dict = {}


def get_session(user_id: int) -> dict:
    if user_id not in user_sessions:
        user_sessions[user_id] = {"history": [], "mode": "default"}
    return user_sessions[user_id]


# ── Commands ──────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("💬 Chat", callback_data="mode_chat"),
         InlineKeyboardButton("💻 Code", callback_data="mode_code")],
        [InlineKeyboardButton("🔍 Research", callback_data="mode_research"),
         InlineKeyboardButton("🎨 Image", callback_data="mode_image")],
        [InlineKeyboardButton("✍️ Write", callback_data="mode_write"),
         InlineKeyboardButton("❓ Help", callback_data="help")],
    ]
    await update.message.reply_text(
        "⚡ *SHAZAM AI* — Your Autonomous AI Platform\n\n"
        "I can:\n"
        "• Think, reason, and solve complex problems\n"
        "• Write and debug code in any language\n"
        "• Generate images, logos, banners & thumbnails\n"
        "• Research anything on the internet\n"
        "• Transcribe voice messages\n"
        "• Summarize PDFs and documents\n"
        "• Create content, scripts, emails, reports\n\n"
        "Just send me a message or pick a mode:",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *SHAZAM AI Commands*\n\n"
        "/start — Welcome menu\n"
        "/image [prompt] — Generate an image\n"
        "/code [task] — Write code\n"
        "/research [query] — Search & research\n"
        "/voice — Enable voice responses\n"
        "/clear — Clear conversation history\n"
        "/help — This message\n\n"
        "Or just send a message, voice note, or file!",
        parse_mode="Markdown",
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    user_sessions.pop(uid, None)
    await update.message.reply_text("✅ Conversation cleared.")


# ── Text Messages ─────────────────────────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    text = update.message.text
    session = get_session(uid)

    # Typing indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        orchestrator = get_orchestrator(user_id=str(uid))
        result = await orchestrator.handle(
            message=text,
            context={"history": session["history"][-10:]},
        )
        response = result.get("content", "I could not process that request.")

        # Update history
        session["history"].append({"role": "user", "content": text})
        session["history"].append({"role": "assistant", "content": response})

        # Split long messages (Telegram limit = 4096 chars)
        if len(response) <= 4096:
            await update.message.reply_text(response)
        else:
            chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
            for chunk in chunks:
                await update.message.reply_text(chunk)

    except Exception as e:
        log.error(f"Message error: {e}")
        await update.message.reply_text(f"⚠️ Error: {str(e)[:200]}")


# ── Voice Messages ────────────────────────────────────────────────────────────
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        # Download voice file
        voice = update.message.voice
        file = await context.bot.get_file(voice.file_id)
        audio_bytes = await file.download_as_bytearray()

        # Transcribe (Groq Whisper — FREE)
        transcription = await ai_engine.transcribe(bytes(audio_bytes), "voice.ogg")
        await update.message.reply_text(f"🎙 _Transcription:_ {transcription}", parse_mode="Markdown")

        # Get AI response
        orchestrator = get_orchestrator(user_id=str(uid))
        result = await orchestrator.handle(message=transcription)
        response = result.get("content", "")

        await update.message.reply_text(response)

    except Exception as e:
        await update.message.reply_text(f"⚠️ Voice error: {str(e)[:200]}")


# ── Document/File Messages ─────────────────────────────────────────────────────
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc = update.message.document
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    if doc.mime_type not in ["application/pdf", "text/plain"]:
        await update.message.reply_text("⚠️ Supported formats: PDF, TXT")
        return

    try:
        file = await context.bot.get_file(doc.file_id)
        content = await file.download_as_bytearray()

        text = ""
        if doc.mime_type == "application/pdf":
            import pypdf, io
            reader = pypdf.PdfReader(io.BytesIO(bytes(content)))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
        else:
            text = bytes(content).decode("utf-8", errors="replace")

        from app.agents.research_agent import ResearchAgent
        agent = ResearchAgent(user_id=str(update.effective_user.id))
        result = await agent.execute(
            "Summarize this document",
            context={"action": "summarize_document", "document": text[:8000], "doc_type": doc.file_name},
        )
        await update.message.reply_text(f"📄 *Summary of {doc.file_name}*\n\n{result['content'][:3500]}", parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text(f"⚠️ Document error: {str(e)[:200]}")


# ── Image Command ──────────────────────────────────────────────────────────────
async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = " ".join(context.args) if context.args else ""
    if not prompt:
        await update.message.reply_text("Usage: /image [your prompt]\nExample: /image a futuristic Lagos skyline at sunset")
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_photo")

    try:
        from app.agents.image_agent import ImageAgent
        agent = ImageAgent(user_id=str(update.effective_user.id))
        result = await agent.execute(prompt, context={"type": "general"})
        image_bytes = result.get("image_bytes", b"")

        await update.message.reply_photo(
            photo=image_bytes,
            caption=f"🎨 _{result.get('enhanced_prompt', prompt)[:200]}_",
            parse_mode="Markdown",
        )
    except Exception as e:
        await update.message.reply_text(f"⚠️ Image error: {str(e)[:200]}")


# ── Callback Buttons ──────────────────────────────────────────────────────────
async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data.startswith("mode_"):
        mode = data.replace("mode_", "")
        session = get_session(update.effective_user.id)
        session["mode"] = mode
        await query.edit_message_text(f"✅ Switched to *{mode.capitalize()}* mode. Send your message!", parse_mode="Markdown")
    elif data == "help":
        await query.edit_message_text(
            "📖 Just send any message!\n\nExamples:\n• Write Python code to scrape a website\n• Research latest AI news\n• Generate a logo for my startup\n• Summarize this document [attach file]"
        )


# ── Run Bot ────────────────────────────────────────────────────────────────────
def run_bot():
    if not settings.TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not configured")

    app = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(CommandHandler("image", image_command))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    log.info("SHAZAM AI Telegram Bot starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    run_bot()
