"""
SHAZAM AI — Planner Agent
Breaks complex tasks into subtasks and assigns to specialist agents.
"""
import json
from typing import Any, Dict, List
import structlog
from app.agents.base_agent import BaseAgent
from app.core.ai_engine import ai_engine

log = structlog.get_logger(__name__)

AVAILABLE_AGENTS = [
    "coding", "research", "image", "video",
    "writing", "reasoning", "memory", "workflow", "data_analysis"
]


class PlannerAgent(BaseAgent):
    name = "Planner"
    description = (
        "You decompose complex user requests into clear subtasks, "
        "assign each subtask to the most capable specialist agent, "
        "and coordinate the overall execution plan."
    )

    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        plan = await self._create_plan(task, context)
        self.log.info("plan_created", task=task[:100], steps=len(plan.get("steps", [])))
        return plan

    async def _create_plan(self, task: str, context: Dict) -> Dict:
        prompt = f"""
Analyze this user request and create an execution plan.

USER REQUEST: {task}

CONTEXT: {json.dumps(context, indent=2) if context else "none"}

AVAILABLE AGENTS: {", ".join(AVAILABLE_AGENTS)}

Return a JSON object with this exact structure:
{{
  "summary": "One sentence summary of the request",
  "complexity": "simple | moderate | complex",
  "estimated_time_seconds": 30,
  "steps": [
    {{
      "step": 1,
      "agent": "coding",
      "task": "specific task for this agent",
      "depends_on": [],
      "output_type": "code | text | image | video | data"
    }}
  ],
  "final_output_type": "text | code | image | video | report"
}}

Rules:
- Simple tasks = 1 step
- Complex tasks = up to 5 steps  
- Always use the most specific agent available
- steps that can run in parallel should have the same depends_on values
- Only return valid JSON, no markdown
"""
        result = await self.think(prompt, mode="fast")
        try:
            # Strip markdown code blocks if present
            clean = result.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            # Fallback: single step plan
            return {
                "summary": task[:100],
                "complexity": "simple",
                "estimated_time_seconds": 15,
                "steps": [{"step": 1, "agent": "reasoning", "task": task, "depends_on": [], "output_type": "text"}],
                "final_output_type": "text",
            }
