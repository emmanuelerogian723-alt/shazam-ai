"""
SHAZAM AI — Writing Agent
Technical writing, content creation, documentation, scripts.
"""
from typing import Any, Dict
from app.agents.base_agent import BaseAgent


CONTENT_TYPES = {
    "blog": "SEO-optimized blog post with headers, subheaders, and engaging content",
    "social": "engaging social media post optimized for the platform",
    "email": "professional email with clear subject, body, and call to action",
    "script": "video script with scene descriptions and natural dialogue",
    "docs": "technical documentation with code examples and clear explanations",
    "readme": "GitHub README with badges, installation, usage, and contributing sections",
    "proposal": "business proposal with executive summary, scope, timeline, and pricing",
    "marketing": "compelling marketing copy that converts",
    "report": "professional report with executive summary, findings, and recommendations",
    "thread": "Twitter/X thread with numbered tweets, hooks, and engagement",
}


class WritingAgent(BaseAgent):
    name = "Writing Agent"
    description = (
        "Professional content creator and technical writer. Creates blogs, "
        "documentation, scripts, emails, marketing copy, and any written content."
    )

    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        content_type = context.get("content_type", "general")
        tone = context.get("tone", "professional")
        length = context.get("length", "medium")
        audience = context.get("audience", "general")

        type_desc = CONTENT_TYPES.get(content_type, "well-structured written content")

        prompt = f"""
Create {type_desc}.

TASK: {task}
TONE: {tone}
LENGTH: {length} (short=300 words, medium=600 words, long=1200+ words)
AUDIENCE: {audience}

Requirements:
- Engaging opening that hooks the reader
- Clear structure with proper formatting
- Strong conclusion
- Natural, human-sounding language
- No filler phrases or generic statements

Write the content now:
"""
        content = await self.think(prompt, mode="default")
        return {
            "type": "written_content",
            "content_type": content_type,
            "content": content,
            "word_count": len(content.split()),
            "tone": tone,
        }
