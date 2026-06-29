"""
SHAZAM AI — Image Agent
Generates logos, banners, thumbnails, posters, social media graphics.
Uses Pollinations.ai (FREE, no key) with HuggingFace fallback.
"""
from typing import Any, Dict
from app.agents.base_agent import BaseAgent
from app.core.ai_engine import ai_engine


IMAGE_PRESETS = {
    "logo":       {"width": 512,  "height": 512,  "suffix": "clean logo design, minimal, vector style, white background"},
    "banner":     {"width": 1280, "height": 400,  "suffix": "wide banner, professional design, high quality"},
    "thumbnail":  {"width": 1280, "height": 720,  "suffix": "YouTube thumbnail, bold text, eye-catching, vibrant"},
    "poster":     {"width": 800,  "height": 1200, "suffix": "movie poster style, dramatic lighting, professional"},
    "social":     {"width": 1080, "height": 1080, "suffix": "social media post, clean design, engaging"},
    "avatar":     {"width": 512,  "height": 512,  "suffix": "profile picture, centered, clean background"},
    "ad":         {"width": 1200, "height": 628,  "suffix": "advertisement banner, marketing, professional, call to action"},
    "wallpaper":  {"width": 1920, "height": 1080, "suffix": "desktop wallpaper, high quality, stunning"},
}


class ImageAgent(BaseAgent):
    name = "Image Agent"
    description = (
        "Creative AI visual artist. Generates logos, banners, thumbnails, "
        "posters, social media graphics, advertisements, and any custom images."
    )

    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        image_type = context.get("type", "general")
        style = context.get("style", "")
        preset = IMAGE_PRESETS.get(image_type, {"width": 1024, "height": 1024, "suffix": ""})

        # Enhance the prompt with AI
        enhanced_prompt = await self._enhance_prompt(task, image_type, style)

        # Full prompt with preset suffix
        full_prompt = f"{enhanced_prompt}, {preset['suffix']}" if preset["suffix"] else enhanced_prompt

        # Generate image
        image_bytes = await ai_engine.generate_image(
            prompt=full_prompt,
            width=preset["width"],
            height=preset["height"],
            model="flux",
        )

        return {
            "type": "image",
            "image_type": image_type,
            "prompt": full_prompt,
            "enhanced_prompt": enhanced_prompt,
            "width": preset["width"],
            "height": preset["height"],
            "image_bytes": image_bytes,
            "size_bytes": len(image_bytes),
        }

    async def _enhance_prompt(self, task: str, image_type: str, style: str) -> str:
        """Use LLM to enhance/refine the image prompt for better results."""
        prompt = f"""
Create an optimized image generation prompt for this request:

REQUEST: {task}
IMAGE TYPE: {image_type}
STYLE PREFERENCE: {style or "professional, high quality"}

Rules:
- Be specific about visual elements, colors, composition
- Include art style, lighting, mood
- Keep it under 200 words
- Make it vivid and detailed
- Return ONLY the prompt, no explanation

Enhanced prompt:"""
        result = await self.think(prompt, mode="fast")
        return result.strip().strip('"')
