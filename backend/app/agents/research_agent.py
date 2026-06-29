"""
SHAZAM AI — Research Agent
Web search + document analysis + report generation.
"""
import asyncio
from typing import Any, Dict, List
from app.agents.base_agent import BaseAgent
from app.core.ai_engine import ai_engine


class ResearchAgent(BaseAgent):
    name = "Research Agent"
    description = (
        "Deep research specialist. Searches the web, analyzes documents, "
        "synthesizes information, fact-checks, and generates structured reports."
    )

    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        action = context.get("action", "research")

        if action == "summarize_document":
            return await self._summarize_document(task, context)
        elif action == "fact_check":
            return await self._fact_check(task, context)
        else:
            return await self._research(task, context)

    async def _research(self, task: str, context: Dict) -> Dict:
        # Step 1: Search web
        search_results = await ai_engine.web_search(task, num=5)

        # Step 2: Synthesize with LLM
        sources_text = "\n".join([
            f"- {r.get('title', '')}: {r.get('snippet', '')} ({r.get('link', '')})"
            for r in search_results
        ])

        prompt = f"""
Based on the following search results, write a comprehensive, well-structured research report.

QUERY: {task}

SEARCH RESULTS:
{sources_text}

Format the report with:
1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points)
3. Detailed Analysis
4. Conclusion
5. Sources

Be factual. Only include information from the provided sources.
"""
        report = await self.think(prompt, mode="default")
        return {
            "type": "research_report",
            "query": task,
            "content": report,
            "sources": [r.get("link", "") for r in search_results],
            "result_count": len(search_results),
        }

    async def _summarize_document(self, task: str, context: Dict) -> Dict:
        document = context.get("document", task)
        doc_type = context.get("doc_type", "document")

        prompt = f"""
Summarize the following {doc_type} comprehensively.

DOCUMENT:
{document[:8000]}  

Provide:
1. TL;DR (1-2 sentences)
2. Key Points (5-10 bullet points)
3. Important Details
4. Action Items (if applicable)
5. Conclusion
"""
        summary = await self.think(prompt, mode="default")
        return {"type": "summary", "content": summary, "original_length": len(document)}

    async def _fact_check(self, task: str, context: Dict) -> Dict:
        claim = context.get("claim", task)
        search_results = await ai_engine.web_search(f"fact check: {claim}", num=5)

        sources_text = "\n".join([
            f"- {r.get('title', '')}: {r.get('snippet', '')}"
            for r in search_results
        ])

        prompt = f"""
Fact-check the following claim using the provided sources.

CLAIM: {claim}

SOURCES:
{sources_text}

Provide:
1. Verdict: TRUE / FALSE / PARTIALLY TRUE / UNVERIFIED
2. Evidence supporting or refuting the claim
3. Nuances or important context
4. Confidence level (low/medium/high)
"""
        result = await self.think(prompt, mode="reasoning")
        return {"type": "fact_check", "claim": claim, "content": result}
