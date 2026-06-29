"""
SHAZAM AI — Video & Image Edit API Routes
"""
import base64
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from app.agents.video_agent import VideoAgent
from app.agents.image_edit_agent import ImageEditAgent
from app.api.auth import get_current_user

router = APIRouter(prefix="/create", tags=["Video & Image Edit"])


# ── Video Endpoints ───────────────────────────────────────────────────────────

class TextToVideoRequest(BaseModel):
    prompt: str
    platform: str = "youtube"   # youtube | youtube_short | tiktok | instagram | instagram_reel | gif
    duration: int = 10
    voiceover_script: Optional[str] = None
    filter: Optional[str] = None  # cinematic | vintage | vivid | bw


@router.post("/video/from-text")
async def create_video_from_text(
    req: TextToVideoRequest,
    current_user: dict = Depends(get_current_user),
):
    """Generate a video from a text prompt. Uses Wan2.1 (HuggingFace) or Pollinations slideshow."""
    agent = VideoAgent(user_id=str(current_user["id"]))
    result = await agent.execute(
        req.prompt,
        context={
            "action": "text_to_video",
            "platform": req.platform,
            "duration": req.duration,
            "voiceover_script": req.voiceover_script,
        },
    )
    video_bytes = result.get("video_bytes", b"")

    if req.filter:
        from app.skills.video.generator import video_generator
        video_bytes = await video_generator.apply_video_filter(video_bytes, req.filter)

    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={"X-Platform": req.platform, "X-Prompt": req.prompt[:100]},
    )


@router.post("/video/from-image")
async def create_video_from_image(
    image: UploadFile = File(...),
    effect: str = Form(default="ken_burns"),  # zoom_in | zoom_out | pan_left | pan_right | ken_burns
    duration: int = Form(default=6),
    platform: str = Form(default="youtube"),
    voiceover_script: Optional[str] = Form(default=None),
    current_user: dict = Depends(get_current_user),
):
    """Animate a still image into a video with motion effects."""
    image_bytes = await image.read()
    agent = VideoAgent(user_id=str(current_user["id"]))
    result = await agent.execute(
        f"Animate this image with {effect} effect",
        context={
            "action": "image_to_video",
            "image_bytes": image_bytes,
            "effect": effect,
            "duration": duration,
            "platform": platform,
            "voiceover_script": voiceover_script,
        },
    )
    return Response(content=result.get("video_bytes", b""), media_type="video/mp4")


@router.post("/video/slideshow")
async def create_slideshow(
    images: list[UploadFile] = File(...),
    transition: str = Form(default="fade"),   # fade | crossfade | zoom | none
    duration_per_image: float = Form(default=3.0),
    platform: str = Form(default="youtube"),
    music: Optional[UploadFile] = File(default=None),
    current_user: dict = Depends(get_current_user),
):
    """Create a slideshow video from multiple images."""
    images_bytes = [await img.read() for img in images]
    music_bytes  = await music.read() if music else None

    agent = VideoAgent(user_id=str(current_user["id"]))
    result = await agent.execute(
        "Create slideshow",
        context={
            "action": "slideshow",
            "images": images_bytes,
            "transition": transition,
            "duration_per_image": duration_per_image,
            "platform": platform,
            "music_bytes": music_bytes,
        },
    )
    return Response(content=result.get("video_bytes", b""), media_type="video/mp4")


@router.post("/video/add-voiceover")
async def add_voiceover(
    video: UploadFile = File(...),
    script: str = Form(...),
    voice: str = Form(default="Arista-PlayAI"),
    current_user: dict = Depends(get_current_user),
):
    """Add AI voiceover to a video using Groq PlayAI TTS (FREE)."""
    video_bytes = await video.read()
    agent = VideoAgent(user_id=str(current_user["id"]))
    result = await agent.execute(
        script,
        context={"action": "add_voiceover", "video_bytes": video_bytes, "script": script},
    )
    return Response(content=result.get("video_bytes", b""), media_type="video/mp4")


@router.post("/video/add-subtitles")
async def add_subtitles(
    video: UploadFile = File(...),
    transcript: str = Form(...),
    current_user: dict = Depends(get_current_user),
):
    """Burn subtitles into a video."""
    video_bytes = await video.read()
    agent = VideoAgent(user_id=str(current_user["id"]))
    result = await agent.execute(
        transcript,
        context={"action": "add_subtitles", "video_bytes": video_bytes, "transcript": transcript},
    )
    return Response(content=result.get("video_bytes", b""), media_type="video/mp4")


@router.post("/video/filter")
async def apply_video_filter(
    video: UploadFile = File(...),
    filter_name: str = Form(...),   # cinematic | vintage | bright | dark | vivid | bw | warm | cool
    current_user: dict = Depends(get_current_user),
):
    """Apply a color grade / filter to a video."""
    from app.skills.video.generator import video_generator
    video_bytes = await video.read()
    result = await video_generator.apply_video_filter(video_bytes, filter_name)
    return Response(content=result, media_type="video/mp4")


@router.post("/video/gif")
async def video_to_gif(
    video: UploadFile = File(...),
    fps: int = Form(default=12),
    current_user: dict = Depends(get_current_user),
):
    """Convert video to an optimized GIF."""
    from app.skills.video.generator import video_generator
    video_bytes = await video.read()
    gif = await video_generator.video_to_gif(video_bytes, fps=fps)
    return Response(content=gif, media_type="image/gif")


# ── Image Edit Endpoints ──────────────────────────────────────────────────────

@router.post("/image/remove-background")
async def remove_background(
    image: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Remove image background using rembg (open source U2Net — FREE)."""
    image_bytes = await image.read()
    agent = ImageEditAgent(user_id=str(current_user["id"]))
    result = await agent.execute("Remove background", context={"action": "remove_background", "image_bytes": image_bytes})
    return Response(content=result.get("image_bytes", b""), media_type="image/png")


@router.post("/image/upscale")
async def upscale_image(
    image: UploadFile = File(...),
    scale: int = Form(default=2),   # 2 or 4
    current_user: dict = Depends(get_current_user),
):
    """Upscale image 2x or 4x using Real-ESRGAN / Pillow Lanczos."""
    image_bytes = await image.read()
    agent = ImageEditAgent(user_id=str(current_user["id"]))
    result = await agent.execute("Upscale image", context={"action": "upscale", "image_bytes": image_bytes, "scale": scale})
    return Response(content=result.get("image_bytes", b""), media_type="image/png")


@router.post("/image/enhance")
async def enhance_image(
    image: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Auto-enhance image: brightness, contrast, sharpness, color."""
    image_bytes = await image.read()
    agent = ImageEditAgent(user_id=str(current_user["id"]))
    result = await agent.execute("Enhance image", context={"action": "enhance", "image_bytes": image_bytes})
    return Response(content=result.get("image_bytes", b""), media_type="image/png")


@router.post("/image/filter")
async def apply_image_filter(
    image: UploadFile = File(...),
    filter_name: str = Form(...),  # blur | sharpen | grayscale | sepia | vintage | sketch | invert
    current_user: dict = Depends(get_current_user),
):
    """Apply artistic filter to an image."""
    image_bytes = await image.read()
    agent = ImageEditAgent(user_id=str(current_user["id"]))
    result = await agent.execute(f"Apply {filter_name} filter", context={"action": "filter", "image_bytes": image_bytes, "filter": filter_name})
    return Response(content=result.get("image_bytes", b""), media_type="image/png")
