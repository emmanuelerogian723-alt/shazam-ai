"""
SHAZAM AI — Media API
Image generation, upload, document processing.
"""
import base64
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from app.core.ai_engine import ai_engine
from app.api.auth import get_current_user

router = APIRouter(prefix="/media", tags=["Media"])


class ImageGenerateRequest(BaseModel):
    prompt: str
    type: str = "general"   # logo | banner | thumbnail | poster | social | ad
    style: Optional[str] = None
    width: int = 1024
    height: int = 1024
    enhance_prompt: bool = True


@router.post("/image/generate")
async def generate_image(
    req: ImageGenerateRequest,
    current_user: dict = Depends(get_current_user),
):
    """Generate an image using Pollinations.ai (FREE, no API key needed)."""
    from app.agents.image_agent import ImageAgent, IMAGE_PRESETS

    agent = ImageAgent(user_id=str(current_user["id"]))
    result = await agent.execute(
        req.prompt,
        context={"type": req.type, "style": req.style or ""},
    )
    image_bytes = result.get("image_bytes", b"")
    return Response(
        content=image_bytes,
        media_type="image/png",
        headers={
            "X-Prompt": result.get("enhanced_prompt", req.prompt)[:200],
            "X-Width": str(result.get("width", req.width)),
            "X-Height": str(result.get("height", req.height)),
        },
    )


@router.post("/document/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    action: str = Form(default="summarize"),  # summarize | extract | qa
    question: Optional[str] = Form(default=None),
    current_user: dict = Depends(get_current_user),
):
    """Upload a PDF or text document and analyze it."""
    allowed_types = [
        "application/pdf", "text/plain", "text/markdown",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    content = await file.read()

    # Extract text from PDF
    text = ""
    if file.content_type == "application/pdf":
        try:
            import pypdf
            import io
            reader = pypdf.PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not extract PDF text: {e}")
    else:
        text = content.decode("utf-8", errors="replace")

    # Run research agent
    from app.agents.research_agent import ResearchAgent
    agent = ResearchAgent(user_id=str(current_user["id"]))

    if action == "summarize":
        result = await agent.execute(
            "Summarize this document",
            context={"action": "summarize_document", "document": text, "doc_type": file.filename},
        )
    elif action == "qa" and question:
        result = await agent.execute(
            question,
            context={"action": "summarize_document", "document": f"Answer this question: {question}\n\nDocument:\n{text}", "doc_type": "Q&A"},
        )
    else:
        result = await agent.execute(
            "Extract all key information",
            context={"action": "summarize_document", "document": text},
        )

    return result
