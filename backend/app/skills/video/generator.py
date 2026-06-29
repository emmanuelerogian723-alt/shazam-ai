"""
SHAZAM AI — Video Generator (Open Source, Free)
Capabilities:
  - Text → Video (HuggingFace free models: Wan2.1, AnimateDiff)
  - Image → Video (animate a still image)
  - Image slideshow → Video with transitions
  - Add AI voiceover to video (Groq TTS)
  - Add subtitles / captions (auto-generated)
  - Add background music
  - Add motion effects (zoom, pan, ken burns)
  - Video transitions (fade, slide, zoom)
  - Combine multiple clips
  - Add text overlays and lower thirds
  - Export for YouTube, TikTok, Instagram Reels, Shorts
  - GIF generation
"""
import asyncio
import io
import os
import tempfile
import base64
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import httpx
import structlog

log = structlog.get_logger(__name__)


# ── Platform Presets ──────────────────────────────────────────────────────────
VIDEO_PRESETS = {
    "youtube":       {"width": 1920, "height": 1080, "fps": 30, "duration": 60},
    "youtube_short": {"width": 1080, "height": 1920, "fps": 30, "duration": 60},
    "tiktok":        {"width": 1080, "height": 1920, "fps": 30, "duration": 60},
    "instagram":     {"width": 1080, "height": 1080, "fps": 30, "duration": 60},
    "instagram_reel":{"width": 1080, "height": 1920, "fps": 30, "duration": 90},
    "twitter":       {"width": 1280, "height": 720,  "fps": 30, "duration": 140},
    "ad_landscape":  {"width": 1920, "height": 1080, "fps": 30, "duration": 30},
    "ad_square":     {"width": 1080, "height": 1080, "fps": 30, "duration": 30},
    "gif":           {"width": 480,  "height": 480,  "fps": 12, "duration": 10},
}


class VideoGenerator:
    """
    Higgsfield-style AI video platform using 100% free & open-source tools.
    Uses: MoviePy, FFmpeg, HuggingFace free inference, Groq TTS.
    """

    # ── Text → Video (HuggingFace free models) ────────────────────────────────
    async def text_to_video(
        self,
        prompt: str,
        duration: int = 4,
        fps: int = 8,
        width: int = 512,
        height: int = 512,
        platform: str = "youtube",
    ) -> bytes:
        """
        Generate video from text prompt.
        Uses Wan2.1-T2V-14B (best free open-source text-to-video) via HuggingFace.
        Falls back to AnimateDiff, then slideshow from generated images.
        """
        from app.core.config import settings

        # Try HuggingFace Wan2.1 (best free text-to-video 2025)
        if settings.HUGGINGFACE_API_KEY:
            try:
                return await self._hf_text_to_video(prompt, settings.HUGGINGFACE_API_KEY)
            except Exception as e:
                log.warning("hf_t2v_failed", error=str(e))

        # Fallback: Generate images → slideshow video
        log.info("falling_back_to_slideshow", prompt=prompt[:50])
        return await self._prompt_to_slideshow(prompt, duration, fps, platform)

    async def _hf_text_to_video(self, prompt: str, hf_key: str) -> bytes:
        """Call HuggingFace Wan2.1 text-to-video model."""
        # Wan2.1 is best free open-source T2V model as of 2025
        models_to_try = [
            "Wan-AI/Wan2.1-T2V-14B",
            "guoyww/animatediff-motion-adapter-v1-5-2",
        ]
        for model in models_to_try:
            try:
                url = f"https://api-inference.huggingface.co/models/{model}"
                async with httpx.AsyncClient(timeout=300) as client:
                    resp = await client.post(
                        url,
                        headers={"Authorization": f"Bearer {hf_key}"},
                        json={"inputs": prompt},
                    )
                    if resp.status_code == 200 and len(resp.content) > 1000:
                        return resp.content
            except Exception as e:
                log.warning("hf_model_failed", model=model, error=str(e))
                continue
        raise RuntimeError("All HuggingFace video models failed")

    async def _prompt_to_slideshow(
        self,
        prompt: str,
        duration: int,
        fps: int,
        platform: str,
    ) -> bytes:
        """
        Generate 4-6 images from prompt → slideshow with transitions.
        100% free using Pollinations.ai.
        """
        from app.core.ai_engine import ai_engine

        preset = VIDEO_PRESETS.get(platform, VIDEO_PRESETS["youtube"])
        w, h = preset["width"], preset["height"]

        # Generate scene prompts
        scene_prompts = await self._generate_scene_prompts(prompt, num_scenes=5)

        # Generate images concurrently
        tasks = [
            ai_engine.generate_image(p, width=min(w, 1024), height=min(h, 1024))
            for p in scene_prompts
        ]
        images_bytes = await asyncio.gather(*tasks, return_exceptions=True)
        valid_images = [b for b in images_bytes if isinstance(b, bytes)]

        if not valid_images:
            raise RuntimeError("Could not generate images for video")

        return await self.images_to_video(
            images=valid_images,
            duration_per_image=max(2, duration // len(valid_images)),
            transition="fade",
            platform=platform,
        )

    async def _generate_scene_prompts(self, main_prompt: str, num_scenes: int = 5) -> List[str]:
        """Use AI to create scene-by-scene image prompts."""
        from app.core.ai_engine import ai_engine
        result = await ai_engine.chat(
            messages=[{
                "role": "user",
                "content": f"""
Create {num_scenes} sequential image prompts for a video about: {main_prompt}

Each prompt should be a different scene/shot that tells a visual story.
Return ONLY a numbered list, one prompt per line, no explanation.
Make each prompt vivid and cinematic.
""",
            }],
            mode="fast",
            max_tokens=500,
        )
        lines = [l.strip().lstrip("0123456789.-) ") for l in result.strip().split("\n") if l.strip()]
        return lines[:num_scenes] if lines else [main_prompt] * num_scenes

    # ── Image → Video (Ken Burns effect + motion) ────────────────────────────
    async def image_to_video(
        self,
        image_bytes: bytes,
        duration: int = 5,
        effect: str = "zoom_in",  # zoom_in | zoom_out | pan_left | pan_right | ken_burns
        fps: int = 24,
        platform: str = "youtube",
    ) -> bytes:
        """
        Animate a still image using motion effects (Ken Burns, zoom, pan).
        Pure MoviePy — 100% offline, no API needed.
        """
        try:
            from moviepy import ImageClip, CompositeVideoClip
            from moviepy.video.fx import Resize, Crop
            import numpy as np
            from PIL import Image
        except ImportError:
            raise RuntimeError("Install moviepy: pip install moviepy")

        preset = VIDEO_PRESETS.get(platform, VIDEO_PRESETS["youtube"])
        w, h = preset["width"], preset["height"]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save input image
            img_path = os.path.join(tmpdir, "input.png")
            with open(img_path, "wb") as f:
                f.write(image_bytes)

            # Load and resize
            from PIL import Image as PILImage
            img = PILImage.open(img_path).convert("RGB")
            # Make slightly larger than output for pan/zoom room
            scale_w, scale_h = int(w * 1.3), int(h * 1.3)
            img = img.resize((scale_w, scale_h), PILImage.LANCZOS)
            scaled_path = os.path.join(tmpdir, "scaled.png")
            img.save(scaled_path)

            clip = ImageClip(scaled_path, duration=duration)
            total_frames = duration * fps

            # Apply motion effect
            def make_frame(t):
                progress = t / duration
                if effect == "zoom_in":
                    zoom = 1.0 + progress * 0.3
                    cw, ch = int(w / zoom), int(h / zoom)
                    cx, cy = (scale_w - cw) // 2, (scale_h - ch) // 2
                elif effect == "zoom_out":
                    zoom = 1.3 - progress * 0.3
                    cw, ch = int(w / zoom), int(h / zoom)
                    cx, cy = (scale_w - cw) // 2, (scale_h - ch) // 2
                elif effect == "pan_left":
                    cx = int(progress * (scale_w - w))
                    cy = (scale_h - h) // 2
                    cw, ch = w, h
                elif effect == "pan_right":
                    cx = int((1 - progress) * (scale_w - w))
                    cy = (scale_h - h) // 2
                    cw, ch = w, h
                else:  # ken_burns: zoom + slight pan
                    zoom = 1.0 + progress * 0.2
                    cw, ch = int(w / zoom), int(h / zoom)
                    cx = int(progress * 20)
                    cy = (scale_h - ch) // 2

                frame_img = img.crop((cx, cy, cx + cw, cy + ch)).resize((w, h), PILImage.LANCZOS)
                return np.array(frame_img)

            from moviepy import VideoClip
            motion_clip = VideoClip(make_frame, duration=duration).with_fps(fps)

            out_path = os.path.join(tmpdir, "output.mp4")
            motion_clip.write_videofile(out_path, fps=fps, codec="libx264", audio=False, logger=None)

            with open(out_path, "rb") as f:
                return f.read()

    # ── Images → Slideshow Video ───────────────────────────────────────────────
    async def images_to_video(
        self,
        images: List[bytes],
        duration_per_image: float = 3.0,
        transition: str = "fade",  # fade | slide | zoom | none
        fps: int = 24,
        platform: str = "youtube",
        audio_bytes: Optional[bytes] = None,
    ) -> bytes:
        """
        Create a professional slideshow video from multiple images.
        Supports fade, slide, zoom transitions. Optionally adds music.
        """
        try:
            import moviepy
            from moviepy import ImageClip, concatenate_videoclips, AudioFileClip
            from moviepy.video.fx import FadeIn, FadeOut, CrossFadeIn, CrossFadeOut
        except ImportError:
            raise RuntimeError("Install moviepy: pip install moviepy")

        preset = VIDEO_PRESETS.get(platform, VIDEO_PRESETS["youtube"])
        w, h = preset["width"], preset["height"]

        with tempfile.TemporaryDirectory() as tmpdir:
            clips = []
            for i, img_bytes in enumerate(images):
                img_path = os.path.join(tmpdir, f"img_{i}.png")

                # Resize image to target dimensions
                from PIL import Image as PILImage
                img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                img_resized = self._fit_image(img, w, h)
                img_resized.save(img_path)

                clip = ImageClip(img_path, duration=duration_per_image)

                if transition == "fade" and i > 0:
                    clip = clip.with_effects([FadeIn(0.5), FadeOut(0.5)])
                elif transition == "zoom":
                    # Ken Burns on each slide
                    orig_w, orig_h = img_resized.size
                    import numpy as np
                    def make_zoom_frame(t, _clip=clip, _img=img_resized):
                        progress = t / duration_per_image
                        zoom = 1.0 + progress * 0.1
                        cw, ch = int(w / zoom), int(h / zoom)
                        cx, cy = (w - cw) // 2, (h - ch) // 2
                        frame = _img.crop((cx, cy, cx+cw, cy+ch)).resize((w, h), PILImage.LANCZOS)
                        return np.array(frame)
                    from moviepy import VideoClip
                    clip = VideoClip(make_zoom_frame, duration=duration_per_image)

                clips.append(clip)

            if transition == "crossfade":
                final = concatenate_videoclips(clips, method="compose", padding=-0.5)
            else:
                final = concatenate_videoclips(clips, method="chain")

            # Add audio if provided
            if audio_bytes:
                audio_path = os.path.join(tmpdir, "audio.mp3")
                with open(audio_path, "wb") as f:
                    f.write(audio_bytes)
                audio = AudioFileClip(audio_path)
                if audio.duration > final.duration:
                    audio = audio.subclipped(0, final.duration)
                final = final.with_audio(audio)

            out_path = os.path.join(tmpdir, "slideshow.mp4")
            final.write_videofile(out_path, fps=fps, codec="libx264",
                                  audio_codec="aac" if audio_bytes else None,
                                  logger=None)
            with open(out_path, "rb") as f:
                return f.read()

    # ── Add Voiceover ─────────────────────────────────────────────────────────
    async def add_voiceover(
        self,
        video_bytes: bytes,
        script: str,
        voice: str = "Arista-PlayAI",
    ) -> bytes:
        """Generate AI voiceover (Groq PlayAI — FREE) and add to video."""
        from app.core.ai_engine import ai_engine

        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate TTS audio
            audio_bytes = await ai_engine.text_to_speech(text=script[:4000], voice=voice)

            video_path = os.path.join(tmpdir, "video.mp4")
            audio_path = os.path.join(tmpdir, "voice.wav")
            out_path   = os.path.join(tmpdir, "output.mp4")

            with open(video_path, "wb") as f: f.write(video_bytes)
            with open(audio_path, "wb") as f: f.write(audio_bytes)

            # Use FFmpeg to merge
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                out_path,
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            if not os.path.exists(out_path):
                raise RuntimeError("FFmpeg failed to add voiceover")

            with open(out_path, "rb") as f:
                return f.read()

    # ── Add Subtitles ─────────────────────────────────────────────────────────
    async def add_subtitles(
        self,
        video_bytes: bytes,
        transcript: str,
        font_size: int = 40,
        color: str = "white",
        bg: bool = True,
    ) -> bytes:
        """Burn subtitles into video using FFmpeg drawtext filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4")
            out_path   = os.path.join(tmpdir, "output.mp4")
            srt_path   = os.path.join(tmpdir, "subs.srt")

            with open(video_path, "wb") as f: f.write(video_bytes)

            # Simple SRT from transcript (split into chunks)
            srt_content = self._transcript_to_srt(transcript)
            with open(srt_path, "w") as f: f.write(srt_content)

            bg_filter = f":box=1:boxcolor=black@0.5:boxborderw=5" if bg else ""
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", f"subtitles={srt_path}:force_style='FontSize={font_size},PrimaryColour=&H00ffffff{bg_filter}'",
                "-c:a", "copy",
                out_path,
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            with open(out_path, "rb") as f:
                return f.read()

    # ── Add Background Music ───────────────────────────────────────────────────
    async def add_background_music(
        self,
        video_bytes: bytes,
        music_bytes: bytes,
        music_volume: float = 0.3,
    ) -> bytes:
        """Mix background music with video at reduced volume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "video.mp4")
            music_path = os.path.join(tmpdir, "music.mp3")
            out_path   = os.path.join(tmpdir, "output.mp4")

            with open(video_path, "wb") as f: f.write(video_bytes)
            with open(music_path, "wb") as f: f.write(music_bytes)

            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", music_path,
                "-filter_complex",
                f"[1:a]volume={music_volume},aloop=loop=-1:size=2e+09[music];"
                "[0:a][music]amix=inputs=2:duration=first[aout]",
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                out_path,
            ]
            proc = await asyncio.create_subprocess_exec(*cmd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await proc.communicate()

            with open(out_path, "rb") as f:
                return f.read()

    # ── Trim / Cut ────────────────────────────────────────────────────────────
    async def trim_video(
        self,
        video_bytes: bytes,
        start_sec: float,
        end_sec: float,
    ) -> bytes:
        """Trim video to a specific time range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "input.mp4")
            out = os.path.join(tmpdir, "output.mp4")
            with open(inp, "wb") as f: f.write(video_bytes)

            cmd = [
                "ffmpeg", "-y",
                "-i", inp,
                "-ss", str(start_sec),
                "-to", str(end_sec),
                "-c", "copy",
                out,
            ]
            proc = await asyncio.create_subprocess_exec(*cmd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await proc.communicate()

            with open(out, "rb") as f:
                return f.read()

    # ── Merge Videos ─────────────────────────────────────────────────────────
    async def merge_videos(self, videos: List[bytes], transition: str = "fade") -> bytes:
        """Concatenate multiple video clips into one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i, v in enumerate(videos):
                p = os.path.join(tmpdir, f"clip_{i}.mp4")
                with open(p, "wb") as f: f.write(v)
                paths.append(p)

            list_path = os.path.join(tmpdir, "list.txt")
            with open(list_path, "w") as f:
                for p in paths:
                    f.write(f"file '{p}'\n")

            out = os.path.join(tmpdir, "merged.mp4")
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                   "-c", "copy", out]
            proc = await asyncio.create_subprocess_exec(*cmd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await proc.communicate()

            with open(out, "rb") as f:
                return f.read()

    # ── Video → GIF ───────────────────────────────────────────────────────────
    async def video_to_gif(self, video_bytes: bytes, fps: int = 12, scale: int = 480) -> bytes:
        """Convert video to optimized GIF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "input.mp4")
            out = os.path.join(tmpdir, "output.gif")
            with open(inp, "wb") as f: f.write(video_bytes)

            palette = os.path.join(tmpdir, "palette.png")
            cmd1 = ["ffmpeg", "-y", "-i", inp, f"-vf", f"fps={fps},scale={scale}:-1:flags=lanczos,palettegen", palette]
            cmd2 = ["ffmpeg", "-y", "-i", inp, "-i", palette,
                    "-filter_complex", f"fps={fps},scale={scale}:-1:flags=lanczos[x];[x][1:v]paletteuse",
                    out]

            for cmd in [cmd1, cmd2]:
                proc = await asyncio.create_subprocess_exec(*cmd,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                await proc.communicate()

            with open(out, "rb") as f:
                return f.read()

    # ── Apply Video Filter ────────────────────────────────────────────────────
    async def apply_video_filter(
        self,
        video_bytes: bytes,
        filter_name: str,  # cinematic | vintage | bright | dark | vivid | bw
    ) -> bytes:
        """Apply color grade / filter to entire video using FFmpeg."""
        filters = {
            "cinematic": "curves=preset=cross_process,vignette=PI/4",
            "vintage":   "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131,vignette=PI/3",
            "bright":    "eq=brightness=0.1:contrast=1.1:saturation=1.2",
            "dark":      "eq=brightness=-0.1:contrast=1.3:saturation=0.9,vignette=PI/4",
            "vivid":     "eq=contrast=1.2:saturation=1.5:brightness=0.05",
            "bw":        "hue=s=0,eq=contrast=1.2",
            "warm":      "colorbalance=rs=0.1:gs=0.05:bs=-0.1",
            "cool":      "colorbalance=rs=-0.1:gs=0.0:bs=0.1",
        }
        vf = filters.get(filter_name.lower(), "")
        if not vf:
            return video_bytes

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "input.mp4")
            out = os.path.join(tmpdir, "output.mp4")
            with open(inp, "wb") as f: f.write(video_bytes)

            cmd = ["ffmpeg", "-y", "-i", inp, "-vf", vf, "-c:a", "copy", out]
            proc = await asyncio.create_subprocess_exec(*cmd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await proc.communicate()

            with open(out, "rb") as f:
                return f.read()

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _fit_image(self, img, target_w: int, target_h: int):
        from PIL import Image as PILImage
        aspect = img.width / img.height
        target_aspect = target_w / target_h
        if aspect > target_aspect:
            new_w = target_w
            new_h = int(target_w / aspect)
        else:
            new_h = target_h
            new_w = int(target_h * aspect)
        resized = img.resize((new_w, new_h), PILImage.LANCZOS)
        canvas = PILImage.new("RGB", (target_w, target_h), (0, 0, 0))
        x = (target_w - new_w) // 2
        y = (target_h - new_h) // 2
        canvas.paste(resized, (x, y))
        return canvas

    def _transcript_to_srt(self, text: str) -> str:
        """Convert plain transcript to basic SRT format."""
        words = text.split()
        chunk_size = 8
        chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        srt = ""
        for i, chunk in enumerate(chunks):
            start = i * 3
            end = start + 3
            srt += f"{i+1}\n"
            srt += f"{self._sec_to_srt(start)} --> {self._sec_to_srt(end)}\n"
            srt += f"{chunk}\n\n"
        return srt

    def _sec_to_srt(self, sec: float) -> str:
        h, rem = divmod(int(sec), 3600)
        m, s = divmod(rem, 60)
        ms = int((sec - int(sec)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"


# Singleton
video_generator = VideoGenerator()
