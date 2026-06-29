"""
SHAZAM AI — Coding Agent
Writes, debugs, refactors, explains code across all languages.
Uses Groq llama-3.3-70b or Qwen2.5-Coder via OpenRouter (free).
"""
import re
from typing import Any, Dict
from app.agents.base_agent import BaseAgent
from app.core.ai_engine import ai_engine


class CodingAgent(BaseAgent):
    name = "Coding Agent"
    description = (
        "Expert software engineer. Writes production-grade code, debugs errors, "
        "refactors messy code, creates full applications, and explains technical concepts clearly."
    )

    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        action = context.get("action", "generate")  # generate | debug | refactor | explain

        if action == "debug":
            return await self._debug(task, context)
        elif action == "refactor":
            return await self._refactor(task, context)
        elif action == "explain":
            return await self._explain(task, context)
        else:
            return await self._generate(task, context)

    async def _generate(self, task: str, context: Dict) -> Dict:
        language = context.get("language", "python")
        prompt = f"""
Write production-quality {language} code for the following:

TASK: {task}

Requirements:
- Clean, readable, well-commented code
- Error handling included
- Follow {language} best practices
- Include usage example at the end

Return the code wrapped in a proper code block.
"""
        code = await self.think(prompt, mode="coding")
        return {
            "type": "code",
            "language": language,
            "content": code,
            "blocks": self._extract_code_blocks(code),
        }

    async def _debug(self, task: str, context: Dict) -> Dict:
        code = context.get("code", "")
        error = context.get("error", "")
        prompt = f"""
Debug this code and fix all errors.

CODE:
{code}

ERROR MESSAGE:
{error}

TASK DESCRIPTION: {task}

Provide:
1. Root cause of the error
2. Fixed code
3. Explanation of what was wrong
"""
        result = await self.think(prompt, mode="coding")
        return {"type": "debug_result", "content": result, "blocks": self._extract_code_blocks(result)}

    async def _refactor(self, task: str, context: Dict) -> Dict:
        code = context.get("code", "")
        prompt = f"""
Refactor this code to be cleaner, more efficient, and production-ready.

ORIGINAL CODE:
{code}

REQUIREMENTS: {task}

Provide the refactored code with comments explaining key improvements.
"""
        result = await self.think(prompt, mode="coding")
        return {"type": "refactored_code", "content": result, "blocks": self._extract_code_blocks(result)}

    async def _explain(self, task: str, context: Dict) -> Dict:
        code = context.get("code", task)
        prompt = f"""
Explain this code clearly for someone learning programming.

CODE:
{code}

Cover:
1. What it does (in plain English)
2. How it works step by step
3. Any important patterns or concepts used
4. Potential improvements
"""
        result = await self.think(prompt, mode="default")
        return {"type": "explanation", "content": result}

    def _extract_code_blocks(self, text: str) -> list:
        pattern = r"```(\w+)?\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [{"language": lang or "text", "code": code.strip()} for lang, code in matches]
