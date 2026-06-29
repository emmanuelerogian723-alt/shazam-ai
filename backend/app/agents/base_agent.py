"""
SHAZAM AI — Base Agent
All specialized agents inherit from this.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import structlog
from app.core.ai_engine import ai_engine

log = structlog.get_logger(__name__)


class BaseAgent(ABC):
    name: str = "base"
    description: str = ""

    def __init__(self, user_id: Optional[str] = None):
        self.user_id = user_id
        self.history: List[Dict] = []
        self.log = structlog.get_logger(self.__class__.__name__)

    @abstractmethod
    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task and return structured result."""
        ...

    async def think(self, prompt: str, mode: str = "default") -> str:
        """Quick LLM call for this agent."""
        return await ai_engine.chat(
            messages=[{"role": "user", "content": prompt}],
            mode=mode,
            system=self._system_prompt(),
        )

    def _system_prompt(self) -> str:
        return (
            f"You are {self.name}, a specialized AI agent within SHAZAM AI. "
            f"{self.description} "
            "Be precise, structured, and professional. "
            "Return actionable results, not vague answers."
        )

    def add_to_history(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        # Keep last 20 messages
        if len(self.history) > 20:
            self.history = self.history[-20:]
