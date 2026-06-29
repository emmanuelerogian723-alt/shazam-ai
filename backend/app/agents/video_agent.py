"""
SHAZAM AI — Video Agent
Orchestrates all video creation and editing tasks.
Similar to Higgsfield AI — but 100% open source and free.
"""
from typing import Any, Dict, Optional
from app.agents.base_agent import BaseAgent
from app.skills.video.generator import video_generator
import structlog

log = structlog.get_logger(__name__)


class VideoAgent(BaseAgent):
    name = "Video Agent"
    description = (
        "AI video creator and editor. Creates videos from text prompts or images, "
        "adds voiceovers, subtitles, music, transitions, and exports for any platform."
    )

    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        action = context.get("action", "auto")

        if action == "text_to_video" or (action == "auto" and not context.get("images")):
            return await self._text_to_video(task, context)

        elif action == "image_to_video":
            return await self._image_to_video(task, context)

        elif action == "slideshow":
            return await self._slideshow(task, context)

        elif action == "add_voiceover":
            return await self._add_voiceover(task, context)

        elif action == "add_subtitles":
            return await self._add_subtitles(task, context)

        elif action == "add_music":
            return await self._add_music(task, context)

        elif action == "trim":
            return await self._trim(task, context)

        elif action == "merge":
            return await self._merge(task, context)

        elif action == "filter":
            return await self._apply_filter(task, context)

        elif action == "gif":
            return await self._to_gif(task, context)

        else:
            # Auto-detect from task description
            return await self._auto_route(task, context)

    async def _auto_route(self, task: str, context: Dict) -> Dict:
        """Use AI to decide the best video action."""
        prompt = f"""
The user wants this video task: {task}

Available actions: text_to_video, image_to_video, slideshow, add_voiceover, add_subtitles, add_music, trim, merge, filter, gif

Reply with ONLY the action name:
"""
        action = (await self.think(prompt, mode="fast")).strip().lower()
        context["action"] = action
        return await self.execute(task, context)

    async def _text_to_video(self, task: str, context: Dict) -> Dict:
        platform = context.get("platform", "youtube")
        duration = context.get("duration", 10)

        # Enhance the video prompt
        enhanced = await self._enhance_video_prompt(task)
        self.log.info("generating_video", prompt=enhanced[:80], platform=platform)

        video_bytes = await video_generator.text_to_video(
            prompt=enhanced,
            duration=duration,
            platform=platform,
        )

        # Optionally add voiceover
        script = context.get("voiceover_script")
        if script:
            video_bytes = await video_generator.add_voiceover(video_bytes, script)

        return {
            "type": "video",
            "action": "text_to_video",
            "video_bytes": video_bytes,
            "size_bytes": len(video_bytes),
            "platform": platform,
            "prompt": enhanced,
        }

    async def _image_to_video(self, task: str, context: Dict) -> Dict:
        image_bytes = context.get("image_bytes")
        if not image_bytes:
            raise ValueError("image_bytes required for image_to_video")

        effect = context.get("effect", "ken_burns")
        duration = context.get("duration", 6)
        platform = context.get("platform", "youtube")

        video_bytes = await video_generator.image_to_video(
            image_bytes=image_bytes,
            duration=duration,
            effect=effect,
            platform=platform,
        )

        script = context.get("voiceover_script")
        if script:
            video_bytes = await video_generator.add_voiceover(video_bytes, script)

        return {
            "type": "video",
            "action": "image_to_video",
            "video_bytes": video_bytes,
            "size_bytes": len(video_bytes),
            "effect": effect,
        }

    async def _slideshow(self, task: str, context: Dict) -> Dict:
        images = context.get("images", [])
        if not images:
            raise ValueError("images list required for slideshow")

        duration_per = context.get("duration_per_image", 3.0)
        transition   = context.get("transition", "fade")
        platform     = context.get("platform", "youtube")
        music        = context.get("music_bytes")

        video_bytes = await video_generator.images_to_video(
            images=images,
            duration_per_image=duration_per,
            transition=transition,
            platform=platform,
            audio_bytes=music,
        )

        return {
            "type": "video",
            "action": "slideshow",
            "video_bytes": video_bytes,
            "size_bytes": len(video_bytes),
            "image_count": len(images),
        }

    async def _add_voiceover(self, task: str, context: Dict) -> Dict:
        video_bytes = context.get("video_bytes")
        script = context.get("script", task)
        if not video_bytes:
            raise ValueError("video_bytes required")

        result = await video_generator.add_voiceover(video_bytes, script)
        return {"type": "video", "action": "voiceover_added", "video_bytes": result}

    async def _add_subtitles(self, task: str, context: Dict) -> Dict:
        video_bytes = context.get("video_bytes")
        transcript  = context.get("transcript", task)
        if not video_bytes:
            raise ValueError("video_bytes required")

        result = await video_generator.add_subtitles(video_bytes, transcript)
        return {"type": "video", "action": "subtitles_added", "video_bytes": result}

    async def _add_music(self, task: str, context: Dict) -> Dict:
        video_bytes = context.get("video_bytes")
        music_bytes = context.get("music_bytes")
        volume      = context.get("volume", 0.3)
        if not video_bytes or not music_bytes:
            raise ValueError("video_bytes and music_bytes required")

        result = await video_generator.add_background_music(video_bytes, music_bytes, volume)
        return {"type": "video", "action": "music_added", "video_bytes": result}

    async def _trim(self, task: str, context: Dict) -> Dict:
        video_bytes = context.get("video_bytes")
        start = context.get("start_sec", 0)
        end   = context.get("end_sec", 30)
        if not video_bytes:
            raise ValueError("video_bytes required")

        result = await video_generator.trim_video(video_bytes, start, end)
        return {"type": "video", "action": "trimmed", "video_bytes": result}

    async def _merge(self, task: str, context: Dict) -> Dict:
        videos = context.get("videos", [])
        if not videos:
            raise ValueError("videos list required")

        result = await video_generator.merge_videos(videos)
        return {"type": "video", "action": "merged", "video_bytes": result}

    async def _apply_filter(self, task: str, context: Dict) -> Dict:
        video_bytes  = context.get("video_bytes")
        filter_name  = context.get("filter", "cinematic")
        if not video_bytes:
            raise ValueError("video_bytes required")

        result = await video_generator.apply_video_filter(video_bytes, filter_name)
        return {"type": "video", "action": "filter_applied", "video_bytes": result}

    async def _to_gif(self, task: str, context: Dict) -> Dict:
        video_bytes = context.get("video_bytes")
        if not video_bytes:
            raise ValueError("video_bytes required")

        result = await video_generator.video_to_gif(video_bytes)
        return {"type": "gif", "action": "gif_created", "video_bytes": result}

    async def _enhance_video_prompt(self, prompt: str) -> str:
        result = await self.think(
            f"""Enhance this video prompt for AI generation. Make it cinematic, vivid, and detailed.
Original: {prompt}
Return ONLY the enhanced prompt, no explanation:""",
            mode="fast",
        )
        return result.strip().strip('"')
