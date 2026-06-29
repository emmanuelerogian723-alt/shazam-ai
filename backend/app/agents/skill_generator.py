"""
SHAZAM AI — Skill Generator Agent
When Shazam AI encounters a task it cannot perform,
this agent automatically generates a new Python skill module,
tests it, documents it, and registers it.
"""
import json
import os
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from app.agents.base_agent import BaseAgent
import structlog

log = structlog.get_logger(__name__)
SKILLS_DIR = Path("app/skills/generated")


class SkillGeneratorAgent(BaseAgent):
    name = "Skill Generator"
    description = (
        "Automatically detects missing capabilities, generates new Python skill modules, "
        "writes unit tests, creates documentation, validates the code, and registers skills."
    )

    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        skill_name = context.get("skill_name") or self._derive_skill_name(task)
        log.info("generating_skill", name=skill_name, task=task[:80])

        # Step 1: Generate skill code
        skill_code = await self._generate_skill_code(task, skill_name)

        # Step 2: Generate unit tests
        test_code = await self._generate_tests(skill_name, skill_code)

        # Step 3: Generate documentation
        docs = await self._generate_docs(skill_name, skill_code, task)

        # Step 4: Save skill
        result = await self._save_skill(skill_name, skill_code, test_code, docs)

        log.info("skill_registered", name=skill_name, path=result["path"])
        return result

    async def _generate_skill_code(self, task: str, skill_name: str) -> str:
        prompt = f"""
Generate a production-quality Python skill module for SHAZAM AI.

SKILL NAME: {skill_name}
CAPABILITY NEEDED: {task}

Requirements:
1. Create a class named {self._to_class_name(skill_name)}Skill
2. Implement an async execute(self, params: dict) -> dict method
3. Include proper type hints
4. Add comprehensive error handling with try/except
5. Use structlog for logging
6. Return a dict with 'success', 'result', and optionally 'error' keys
7. Add detailed docstrings

Example structure:
```python
import structlog
from typing import Dict, Any

log = structlog.get_logger(__name__)

class ExampleSkill:
    \"\"\"Skill description.\"\"\"
    name = "example"
    description = "What this skill does"
    version = "1.0.0"

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # implementation
            return {{"success": True, "result": result}}
        except Exception as e:
            log.error("skill_error", error=str(e))
            return {{"success": False, "error": str(e)}}
```

Generate ONLY the Python code for the {skill_name} skill, no explanation:
"""
        return await self.think(prompt, mode="coding")

    async def _generate_tests(self, skill_name: str, skill_code: str) -> str:
        prompt = f"""
Write pytest unit tests for this Python skill:

SKILL CODE:
{skill_code[:3000]}

Write 3-5 test cases covering:
- Happy path (successful execution)
- Error handling (bad input)
- Edge cases

Use pytest and pytest-asyncio. Return only the test code:
"""
        return await self.think(prompt, mode="coding")

    async def _generate_docs(self, skill_name: str, skill_code: str, task: str) -> str:
        prompt = f"""
Write concise documentation for this skill module.

SKILL: {skill_name}
PURPOSE: {task}

Include:
- Description (2-3 sentences)
- Parameters (what params dict should contain)
- Returns (what the result dict contains)
- Example usage

Keep it under 300 words. Return only the documentation:
"""
        return await self.think(prompt, mode="fast")

    async def _save_skill(
        self, skill_name: str, code: str, tests: str, docs: str
    ) -> Dict:
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)

        skill_path = SKILLS_DIR / f"{skill_name}.py"
        test_path = SKILLS_DIR / f"test_{skill_name}.py"
        docs_path = SKILLS_DIR / f"{skill_name}.md"
        registry_path = SKILLS_DIR / "registry.json"

        # Clean code (remove markdown)
        clean_code = code.strip()
        for marker in ["```python", "```py", "```"]:
            clean_code = clean_code.replace(marker, "")
        clean_code = clean_code.strip()

        skill_path.write_text(clean_code)
        test_path.write_text(tests.strip().replace("```python", "").replace("```", "").strip())
        docs_path.write_text(docs)

        # Update registry
        registry = {}
        if registry_path.exists():
            try:
                registry = json.loads(registry_path.read_text())
            except Exception:
                registry = {}

        registry[skill_name] = {
            "name": skill_name,
            "path": str(skill_path),
            "description": docs[:200],
            "created_at": datetime.utcnow().isoformat(),
            "version": "1.0.0",
        }
        registry_path.write_text(json.dumps(registry, indent=2))

        return {
            "type": "skill_created",
            "skill_name": skill_name,
            "path": str(skill_path),
            "test_path": str(test_path),
            "docs_path": str(docs_path),
            "success": True,
            "message": f"Skill '{skill_name}' created and registered successfully.",
        }

    def _derive_skill_name(self, task: str) -> str:
        words = task.lower().split()[:4]
        name = "_".join(w for w in words if w.isalpha())
        return name or "custom_skill"

    def _to_class_name(self, skill_name: str) -> str:
        return "".join(word.capitalize() for word in skill_name.split("_"))
