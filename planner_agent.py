"""
Planner Agent Module.
Responsible for breaking down complex queries into specific retrieval tasks.
Implements the "Explicit Planner" pattern.
"""
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from llm_manager import get_llm_manager

@dataclass
class PlanStep:
    id: int
    tool: str
    query: str
    reasoning: str

class PlannerAgent:
    """
    Decomposes queries into execution plans.
    """
    
    def __init__(self):
        self.llm = get_llm_manager()
    
    def _fallback_plan(self, user_query: str) -> List[PlanStep]:
        """Fallback plan when JSON parsing fails - search all collections."""
        print(f"   🔄 Using fallback plan...")
        return [
            PlanStep(id=1, tool="search_tactical_memory", query=user_query, reasoning="Fallback"),
            PlanStep(id=2, tool="search_audio_memory", query=user_query, reasoning="Fallback"),
            PlanStep(id=3, tool="search_visual_memory", query=user_query, reasoning="Fallback")
        ]
    
    def create_plan(self, user_query: str, context: str = "") -> List[PlanStep]:
        """
        Analyze the query and create a retrieval plan.
        Returns a list of PlanSteps.
        """
        # System prompt to force JSON plan
        system_prompt = """You are the Aegis Planner Agent.
Your goal is to break down crisis queries into specific retrieval tasks.

AVAILABLE TOOLS:
- search_visual_memory(query): For images, video, visual evidence.
- search_audio_memory(query): For transcripts, radio calls, spoken words.
- search_tactical_memory(query): For written reports, stats, locations.

RULES:
1. For greetings like "Hello" or "Hi", return: {"steps": []}
2. For crisis queries, return 1-3 retrieval steps.
3. ONLY output valid JSON. No explanation, no markdown.

EXAMPLE OUTPUT:
{"steps": [{"tool": "search_tactical_memory", "query": "flood chennai", "reasoning": "Get reports"}]}"""
        

        full_prompt = f"""CONTEXT FROM PREVIOUS TURN:
{context}

USER QUERY: {user_query}

INSTRUCTIONS:
1. If the query uses pronouns like "it", "there", "that", "status", REPLACE them with the specific location/topic from the CONTEXT.
2. Example: If Context says "floods in Bengaluru" and Query is "status there", search for "Bengaluru flood status".
3. RETURN JSON ONLY."""

        
        response = self.llm.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.0  # Zero temp for deterministic JSON
        )
        
        if not response or len(response.strip()) < 5:
            print(f"   ⚠️ Planner returned empty response")
            return self._fallback_plan(user_query)
            
        try:
            # Parse JSON - try multiple cleanup strategies
            clean_resp = response.strip()
            # Remove markdown code blocks
            clean_resp = clean_resp.replace("```json", "").replace("```", "").strip()
            # Find JSON object in response
            start_idx = clean_resp.find("{")
            end_idx = clean_resp.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                clean_resp = clean_resp[start_idx:end_idx]
            
            plan_data = json.loads(clean_resp)
            
            steps = []
            for i, step in enumerate(plan_data.get("steps", [])):
                steps.append(PlanStep(
                    id=i+1,
                    tool=step.get("tool"),
                    query=step.get("query"),
                    reasoning=step.get("reasoning")
                ))
            
            print(f"   📋 Plan Created: {len(steps)} steps")
            return steps
            
        except Exception as e:
            print(f"   ⚠️ Planner JSON Error: {e}")
            return self._fallback_plan(user_query)

# Singleton
_planner = None
def get_planner_agent():
    global _planner
    if _planner is None:
        _planner = PlannerAgent()
    return _planner
