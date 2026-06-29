"""
SHAZAM AI — Image Editing Agent
Orchestrates all image editing tasks.
"""
from typing import Any, Dict
from app.agents.base_agent import BaseAgent
from app.skills.image_edit.editor import image_editor
import structlog

log = structlog.get_logger(__name__)


class ImageEditAgent(BaseAgent):
    name = "Image Edit Agent"
    description = (
        "Professional AI image editor. Removes backgrounds, upscales, enhances, "
        "applies filters, adds text, creates collages, and edits any image with AI."
    )

    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        action = context.get("action", "auto")
        image_bytes = context.get("image_bytes")

        actions = {
            "remove_background": self._remove_bg,
            "upscale":           self._upscale,
            "enhance":           self._enhance,
            "filter":            self._filter,
            "add_text":          self._add_text,
            "resize":            self._resize,
            "crop":              self._crop,
            "rotate":            self._rotate,
            "flip":              self._flip,
            "collage":           self._collage,
            "remove_object":     self._remove_object,
        }

        if action in actions:
            return await actions[action](task, context)
        else:
            return await self._auto_route(task, context)

    async def _auto_route(self, task: str, context: Dict) -> Dict:
        prompt = f"""
Image editing task: {task}

Available actions: remove_background, upscale, enhance, filter, add_text, resize, crop, rotate, flip, collage, remove_object

Reply with ONLY the action name:
"""
        action = (await self.think(prompt, mode="fast")).strip().lower()
        context["action"] = action
        return await self.execute(task, context)

    async def _remove_bg(self, task: str, ctx: Dict) -> Dict:
        result = await image_editor.remove_background(ctx["image_bytes"])
        return {"type": "image", "action": "background_removed", "image_bytes": result}

    async def _upscale(self, task: str, ctx: Dict) -> Dict:
        scale = ctx.get("scale", 2)
        result = await image_editor.upscale(ctx["image_bytes"], scale=scale)
        return {"type": "image", "action": "upscaled", "image_bytes": result, "scale": scale}

    async def _enhance(self, task: str, ctx: Dict) -> Dict:
        result = await image_editor.auto_enhance(
            ctx["image_bytes"],
            brightness=ctx.get("brightness", 1.1),
            contrast=ctx.get("contrast", 1.2),
            sharpness=ctx.get("sharpness", 1.3),
            color=ctx.get("color", 1.1),
        )
        return {"type": "image", "action": "enhanced", "image_bytes": result}

    async def _filter(self, task: str, ctx: Dict) -> Dict:
        filter_name = ctx.get("filter", "enhance")
        result = await image_editor.apply_filter(ctx["image_bytes"], filter_name)
        return {"type": "image", "action": "filter_applied", "filter": filter_name, "image_bytes": result}

    async def _add_text(self, task: str, ctx: Dict) -> Dict:
        text = ctx.get("text", task)
        result = await image_editor.add_text(
            ctx["image_bytes"],
            text=text,
            position=tuple(ctx.get("position", [50, 50])),
            font_size=ctx.get("font_size", 48),
            color=ctx.get("color", "white"),
        )
        return {"type": "image", "action": "text_added", "image_bytes": result}

    async def _resize(self, task: str, ctx: Dict) -> Dict:
        w, h = ctx.get("width", 1024), ctx.get("height", 1024)
        result = await image_editor.resize(ctx["image_bytes"], w, h)
        return {"type": "image", "action": "resized", "image_bytes": result}

    async def _crop(self, task: str, ctx: Dict) -> Dict:
        result = await image_editor.crop(
            ctx["image_bytes"],
            ctx.get("left",0), ctx.get("top",0),
            ctx.get("right",100), ctx.get("bottom",100),
        )
        return {"type": "image", "action": "cropped", "image_bytes": result}

    async def _rotate(self, task: str, ctx: Dict) -> Dict:
        result = await image_editor.rotate(ctx["image_bytes"], ctx.get("degrees", 90))
        return {"type": "image", "action": "rotated", "image_bytes": result}

    async def _flip(self, task: str, ctx: Dict) -> Dict:
        result = await image_editor.flip(ctx["image_bytes"], ctx.get("direction", "horizontal"))
        return {"type": "image", "action": "flipped", "image_bytes": result}

    async def _collage(self, task: str, ctx: Dict) -> Dict:
        images = ctx.get("images", [])
        if not images:
            raise ValueError("images list required for collage")
        result = await image_editor.create_collage(images, cols=ctx.get("cols", 2))
        return {"type": "image", "action": "collage_created", "image_bytes": result}

    async def _remove_object(self, task: str, ctx: Dict) -> Dict:
        mask = ctx.get("mask_bytes")
        if not mask:
            raise ValueError("mask_bytes required for object removal")
        result = await image_editor.remove_object(ctx["image_bytes"], mask)
        return {"type": "image", "action": "object_removed", "image_bytes": result}
