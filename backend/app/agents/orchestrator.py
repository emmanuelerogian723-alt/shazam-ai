"""
SHAZAM AI — Agent Orchestrator
Central coordinator. Routes tasks to correct agents, handles multi-agent flows.
"""
import asyncio
import time
from typing import Any, AsyncIterator, Dict, Optional
import structlog

from app.agents.planner import PlannerAgent
from app.agents.coding_agent import CodingAgent
from app.agents.research_agent import ResearchAgent
from app.agents.image_agent import ImageAgent
from app.agents.writing_agent import WritingAgent
from app.agents.skill_generator import SkillGeneratorAgent
from app.core.ai_engine import ai_engine

log = structlog.get_logger(__name__)

AGENT_REGISTRY = {
    "coding":       CodingAgent,
    "research":     ResearchAgent,
    "image":        ImageAgent,
    "writing":      WritingAgent,
    "skill":        SkillGeneratorAgent,
    "reasoning":    None,  # handled inline via ai_engine.reason()
    "data_analysis":None,  # handled inline
    "workflow":     None,  # future
    "video":        None,  # future
}


class Orchestrator:
    """Routes user requests to the right agent(s) and returns unified results."""

    def __init__(self, user_id: Optional[str] = None):
        self.user_id = user_id
        self.planner = PlannerAgent(user_id=user_id)

    async def handle(
        self,
        message: str,
        context: Optional[Dict] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        start = time.time()
        ctx = context or {}
        log.info("request_received", user=self.user_id, message=message[:80])

        # 1. Classify intent
        intent = await self._classify(message)
        log.info("intent_classified", intent=intent)

        # 2. Simple tasks: skip planner, go direct
        if intent in ("chat", "question"):
            result = await self._direct_chat(message, ctx)
            result["latency_ms"] = int((time.time() - start) * 1000)
            return result

        # 3. Complex tasks: run planner
        plan = await self.planner.execute(message, ctx)
        results = {}

        # 4. Execute each step
        for step in plan.get("steps", []):
            agent_name = step.get("agent", "reasoning")
            task = step.get("task", message)
            step_num = step.get("step", 1)

            agent_result = await self._run_agent(agent_name, task, ctx)
            results[f"step_{step_num}"] = agent_result

        # 5. Synthesize final answer
        final = await self._synthesize(message, results, plan)
        final["plan"] = plan
        final["latency_ms"] = int((time.time() - start) * 1000)

        log.info("request_completed", latency_ms=final["latency_ms"])
        return final

    async def stream_handle(self, message: str, context: Optional[Dict] = None) -> AsyncIterator[str]:
        """Streaming response for chat interface."""
        messages = [{"role": "user", "content": message}]
        async for chunk in ai_engine.stream_chat(messages, mode="default"):
            yield chunk

    async def _classify(self, message: str) -> str:
        prompt = f"""
Classify this user message into exactly ONE category:

chat        - casual conversation, greetings, general chat
question    - asking for information, explanation, definitions
coding      - write code, debug, refactor, explain code
research    - search internet, find information, news
image       - generate image, logo, banner, thumbnail, poster
writing     - write blog, email, script, documentation, content
analysis    - analyze data, chart, statistics, insights
workflow    - automation, multi-step process
skill       - needs a new capability not listed above

Message: {message}

Reply with ONLY the category name, nothing else:"""

        result = await ai_engine.chat(
            messages=[{"role": "user", "content": prompt}],
            mode="fast",
            max_tokens=20,
        )
        intent = result.strip().lower().split()[0]
        valid = {"chat","question","coding","research","image","writing","analysis","workflow","skill"}
        return intent if intent in valid else "chat"

    async def _direct_chat(self, message: str, context: Dict) -> Dict:
        history = context.get("history", [])
        messages = history + [{"role": "user", "content": message}]
        response = await ai_engine.chat(
            messages=messages,
            mode="default",
            system=(
                "You are SHAZAM AI — a powerful, friendly, knowledgeable AI assistant. "
                "Be helpful, precise, and personable. Never say you can't help — always find a way."
            ),
        )
        return {"type": "chat", "content": response}

    async def _run_agent(self, agent_name: str, task: str, context: Dict) -> Dict:
        agent_class = AGENT_REGISTRY.get(agent_name)

        if agent_class:
            agent = agent_class(user_id=self.user_id)
            return await agent.execute(task, context)

        # Fallback: handle inline
        if agent_name == "reasoning":
            result = await ai_engine.reason(task)
            return {"type": "reasoning", "content": result["reasoning"]}

        if agent_name == "data_analysis":
            response = await ai_engine.chat(
                messages=[{"role": "user", "content": f"Analyze this data and provide insights:\n{task}"}],
                mode="reasoning",
            )
            return {"type": "analysis", "content": response}

        # Unknown agent — use default chat
        response = await ai_engine.chat(
            messages=[{"role": "user", "content": task}],
            mode="default",
        )
        return {"type": "text", "content": response}

    async def _synthesize(self, original: str, results: Dict, plan: Dict) -> Dict:
        if not results:
            return {"type": "text", "content": "Task completed with no output."}

        # Single step — return directly
        if len(results) == 1:
            return list(results.values())[0]

        # Multiple steps — summarize
        steps_summary = "\n".join([
            f"Step {k}: {v.get('content', str(v))[:500]}"
            for k, v in results.items()
        ])

        summary = await ai_engine.chat(
            messages=[{
                "role": "user",
                "content": (
                    f"Original request: {original}\n\n"
                    f"Completed steps:\n{steps_summary}\n\n"
                    "Write a cohesive, complete response that addresses the original request "
                    "using all the results above."
                ),
            }],
            mode="default",
        )
        return {"type": "synthesized", "content": summary, "steps": results}


# ── Singleton factory ─────────────────────────────────────────────────────────
def get_orchestrator(user_id: Optional[str] = None) -> Orchestrator:
    return Orchestrator(user_id=user_id)
