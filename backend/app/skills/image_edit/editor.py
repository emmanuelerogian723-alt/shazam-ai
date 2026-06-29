"""
SHAZAM AI — Image Editor (Open Source, 100% Free)
Capabilities:
  - Background removal (rembg — open source)
  - AI upscaling 2x/4x (Pillow + Real-ESRGAN HuggingFace)
  - Face enhancement
  - Object removal (inpainting via HuggingFace)
  - Color grading & filters
  - Auto-enhance (brightness, contrast, sharpness)
  - Add text overlays, watermarks
  - Resize, crop, rotate, flip
  - Convert formats
  - Blur / sharpen / denoise
  - Create collages / grids
"""
import io
import base64
import asyncio
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import httpx
import structlog
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont

log = structlog.get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────
def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGBA")


def bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode()


# ── Image Editor ──────────────────────────────────────────────────────────────
class ImageEditor:
    """
    Full image editing toolkit using 100% open-source libraries.
    All methods are async and return bytes (PNG/JPEG).
    """

    # ── Background Removal (rembg — open source) ──────────────────────────────
    async def remove_background(self, image_bytes: bytes) -> bytes:
        """Remove background using rembg (open source U2Net model)."""
        try:
            from rembg import remove
            result = await asyncio.get_event_loop().run_in_executor(
                None, remove, image_bytes
            )
            return result
        except ImportError:
            # Fallback: HuggingFace BRIA RMBG model (free API)
            return await self._remove_bg_huggingface(image_bytes)

    async def _remove_bg_huggingface(self, image_bytes: bytes) -> bytes:
        from app.core.config import settings
        if not settings.HUGGINGFACE_API_KEY:
            raise RuntimeError("Install rembg or set HUGGINGFACE_API_KEY for background removal")
        url = "https://api-inference.huggingface.co/models/briaai/RMBG-1.4"
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"},
                content=image_bytes,
            )
            resp.raise_for_status()
            return resp.content

    # ── AI Upscale (Real-ESRGAN via HuggingFace — free) ──────────────────────
    async def upscale(self, image_bytes: bytes, scale: int = 2) -> bytes:
        """Upscale image 2x or 4x using Real-ESRGAN (open source)."""
        from app.core.config import settings

        if settings.HUGGINGFACE_API_KEY:
            try:
                model = "ai-forever/Real-ESRGAN" if scale == 4 else "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"
                url = f"https://api-inference.huggingface.co/models/{model}"
                async with httpx.AsyncClient(timeout=120) as client:
                    resp = await client.post(
                        url,
                        headers={"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"},
                        content=image_bytes,
                    )
                    if resp.status_code == 200:
                        return resp.content
            except Exception as e:
                log.warning("hf_upscale_failed", error=str(e))

        # Fallback: Pillow Lanczos (still high quality)
        img = bytes_to_pil(image_bytes)
        new_size = (img.width * scale, img.height * scale)
        upscaled = img.resize(new_size, Image.LANCZOS)
        return pil_to_bytes(upscaled)

    # ── Auto Enhance ──────────────────────────────────────────────────────────
    async def auto_enhance(
        self,
        image_bytes: bytes,
        brightness: float = 1.1,
        contrast: float = 1.2,
        sharpness: float = 1.3,
        color: float = 1.1,
    ) -> bytes:
        """Auto-enhance image: brightness, contrast, sharpness, color."""
        img = bytes_to_pil(image_bytes).convert("RGB")
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
        img = ImageEnhance.Color(img).enhance(color)
        return pil_to_bytes(img)

    # ── Filters ───────────────────────────────────────────────────────────────
    async def apply_filter(self, image_bytes: bytes, filter_name: str) -> bytes:
        """
        Apply a filter. Options:
        blur, sharpen, edge_enhance, emboss, smooth,
        grayscale, sepia, vintage, invert, sketch
        """
        img = bytes_to_pil(image_bytes).convert("RGB")

        filters = {
            "blur":         lambda i: i.filter(ImageFilter.GaussianBlur(radius=3)),
            "sharpen":      lambda i: i.filter(ImageFilter.SHARPEN),
            "edge_enhance": lambda i: i.filter(ImageFilter.EDGE_ENHANCE_MORE),
            "emboss":       lambda i: i.filter(ImageFilter.EMBOSS),
            "smooth":       lambda i: i.filter(ImageFilter.SMOOTH_MORE),
            "grayscale":    lambda i: i.convert("L").convert("RGB"),
            "invert":       lambda i: Image.eval(i, lambda x: 255 - x),
            "sepia":        self._apply_sepia,
            "vintage":      self._apply_vintage,
            "sketch":       self._apply_sketch,
        }

        fn = filters.get(filter_name.lower())
        if fn:
            img = fn(img)

        return pil_to_bytes(img)

    def _apply_sepia(self, img: Image.Image) -> Image.Image:
        r, g, b = img.split()
        r2 = Image.eval(r, lambda x: min(255, int(x * 0.393 + 0.769 * x + 0.189 * x)))
        result = Image.merge("RGB", [
            Image.eval(img, lambda x: min(255, int(x * 0.393 + 0.349 * x + 0.272 * x))).split()[0],
            Image.eval(img, lambda x: min(255, int(x * 0.349 + 0.686 * x + 0.168 * x))).split()[0],
            Image.eval(img, lambda x: min(255, int(x * 0.131 + 0.168 * x + 0.131 * x))).split()[0],
        ])
        return result

    def _apply_vintage(self, img: Image.Image) -> Image.Image:
        img = ImageEnhance.Color(img).enhance(0.7)
        img = ImageEnhance.Contrast(img).enhance(0.9)
        img = ImageEnhance.Brightness(img).enhance(1.1)
        return img

    def _apply_sketch(self, img: Image.Image) -> Image.Image:
        gray = img.convert("L")
        inverted = Image.eval(gray, lambda x: 255 - x)
        blurred = inverted.filter(ImageFilter.GaussianBlur(radius=10))
        sketch = Image.eval(
            Image.blend(gray.convert("RGB"), blurred.convert("RGB"), alpha=0.5),
            lambda x: min(255, int(x * 1.5))
        )
        return sketch

    # ── Text Overlay ──────────────────────────────────────────────────────────
    async def add_text(
        self,
        image_bytes: bytes,
        text: str,
        position: Tuple[int, int] = (50, 50),
        font_size: int = 48,
        color: str = "white",
        stroke: bool = True,
    ) -> bytes:
        """Add text overlay to image."""
        img = bytes_to_pil(image_bytes).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        if stroke:
            for dx, dy in [(-2,-2),(2,-2),(-2,2),(2,2),(0,-2),(0,2),(-2,0),(2,0)]:
                draw.text((position[0]+dx, position[1]+dy), text, font=font, fill=(0,0,0,200))

        draw.text(position, text, font=font, fill=color)
        result = Image.alpha_composite(img, overlay)
        return pil_to_bytes(result)

    # ── Resize / Crop ─────────────────────────────────────────────────────────
    async def resize(
        self,
        image_bytes: bytes,
        width: int,
        height: int,
        maintain_aspect: bool = True,
    ) -> bytes:
        img = bytes_to_pil(image_bytes).convert("RGB")
        if maintain_aspect:
            img.thumbnail((width, height), Image.LANCZOS)
        else:
            img = img.resize((width, height), Image.LANCZOS)
        return pil_to_bytes(img)

    async def crop(
        self,
        image_bytes: bytes,
        left: int, top: int, right: int, bottom: int,
    ) -> bytes:
        img = bytes_to_pil(image_bytes).convert("RGB")
        cropped = img.crop((left, top, right, bottom))
        return pil_to_bytes(cropped)

    async def rotate(self, image_bytes: bytes, degrees: float) -> bytes:
        img = bytes_to_pil(image_bytes).convert("RGB")
        rotated = img.rotate(degrees, expand=True)
        return pil_to_bytes(rotated)

    async def flip(self, image_bytes: bytes, direction: str = "horizontal") -> bytes:
        img = bytes_to_pil(image_bytes).convert("RGB")
        if direction == "horizontal":
            result = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            result = img.transpose(Image.FLIP_TOP_BOTTOM)
        return pil_to_bytes(result)

    # ── Collage / Grid ────────────────────────────────────────────────────────
    async def create_collage(
        self,
        images: list[bytes],
        cols: int = 2,
        padding: int = 10,
        bg_color: str = "white",
    ) -> bytes:
        """Create a grid collage from multiple images."""
        pil_images = [bytes_to_pil(b).convert("RGB") for b in images]
        thumb_size = (400, 400)
        thumbs = []
        for im in pil_images:
            im.thumbnail(thumb_size, Image.LANCZOS)
            thumbs.append(im)

        rows = (len(thumbs) + cols - 1) // cols
        w = cols * (thumb_size[0] + padding) + padding
        h = rows * (thumb_size[1] + padding) + padding

        canvas = Image.new("RGB", (w, h), bg_color)
        for i, thumb in enumerate(thumbs):
            row, col = divmod(i, cols)
            x = padding + col * (thumb_size[0] + padding)
            y = padding + row * (thumb_size[1] + padding)
            canvas.paste(thumb, (x, y))

        return pil_to_bytes(canvas)

    # ── AI Inpainting — Object Removal (HuggingFace stable-diffusion-inpainting) ──
    async def remove_object(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str = "empty background, clean, seamless",
    ) -> bytes:
        """
        Remove an object from image using inpainting.
        mask_bytes = white where object is, black everywhere else.
        """
        from app.core.config import settings
        if not settings.HUGGINGFACE_API_KEY:
            raise RuntimeError("HUGGINGFACE_API_KEY required for object removal")

        img_b64 = bytes_to_base64(image_bytes)
        mask_b64 = bytes_to_base64(mask_bytes)

        url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-inpainting"
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"},
                json={
                    "inputs": {
                        "prompt": prompt,
                        "image": img_b64,
                        "mask_image": mask_b64,
                    }
                },
            )
            resp.raise_for_status()
            return resp.content


# Singleton
image_editor = ImageEditor()
