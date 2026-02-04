"""
Critic/Evaluator Agent for Aegis Crisis Command Center
Reviews Analyst Agent responses for hallucinations, missing citations, and quality.
Implements self-correction through re-search and rephrase mechanisms.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from llm_manager import get_llm_manager


@dataclass
class CriticResult:
    """Result of a Critic evaluation."""
    status: str  # "approved" | "needs_research" | "needs_rephrase"
    confidence: float  # 0.0 - 1.0
    feedback: str  # Detailed explanation
    issues: List[str] = field(default_factory=list)  # Specific problems found
    suggested_query: Optional[str] = None  # Refined query for re-search
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CriticAgent:
    """
    Evaluates Analyst Agent responses for quality and accuracy.
    
    Responsibilities:
    1. Citation Verification - Ensures response cites provided evidence
    2. Hallucination Detection - Identifies claims not grounded in evidence
    3. Confidence Scoring - Assesses overall response quality
    4. Actionable Feedback - Provides specific improvement suggestions
    """
    
    # Thresholds for evaluation
    CONFIDENCE_THRESHOLD_HIGH = 0.75  # Above this = approved
    CONFIDENCE_THRESHOLD_LOW = 0.4    # Below this = needs_rephrase
    MIN_CITATIONS_REQUIRED = 1        # Minimum citations for approval
    
    def __init__(self):
        self.llm_manager = get_llm_manager()
        print("🔍 CriticAgent initialized")
    
    def evaluate(
        self, 
        query: str, 
        response: str, 
        evidence: List[Dict[str, Any]]
    ) -> CriticResult:
        """
        Evaluate the Analyst's response against the query and evidence.
        
        Args:
            query: The user's original question
            response: The Analyst Agent's generated response
            evidence: List of retrieved evidence points with source, content, score
            
        Returns:
            CriticResult with status, confidence, feedback, and issues
        """
        if not response:
            return CriticResult(
                status="needs_research",
                confidence=0.0,
                feedback="No response generated. Need to search for evidence.",
                issues=["Empty response"]
            )
        
        if not evidence:
            return CriticResult(
                status="needs_research",
                confidence=0.2,
                feedback="No evidence available to verify the response.",
                issues=["No evidence retrieved"],
                suggested_query=self._suggest_broader_query(query)
            )
        
        # Run evaluation checks
        citation_result = self._check_citations(response, evidence)
        hallucination_result = self._check_hallucinations(query, response, evidence)
        
        # Aggregate issues
        all_issues = citation_result["issues"] + hallucination_result["issues"]
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(
            citation_score=citation_result["score"],
            hallucination_score=hallucination_result["score"],
            evidence_quality=self._assess_evidence_quality(evidence)
        )
        
        # Determine status based on confidence and issues
        status = self._determine_status(confidence, all_issues, citation_result)
        
        # Generate feedback
        feedback = self._generate_feedback(status, all_issues, confidence)
        
        # Suggest refined query if needed
        suggested_query = None
        if status == "needs_research":
            suggested_query = self._suggest_refined_query(query, all_issues, evidence)
        
        return CriticResult(
            status=status,
            confidence=confidence,
            feedback=feedback,
            issues=all_issues,
            suggested_query=suggested_query
        )
    
    def _check_citations(
        self, 
        response: str, 
        evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check if the response properly cites the evidence sources."""
        issues = []
        cited_sources = []
        response_lower = response.lower()
        
        for item in evidence:
            source = item.get("source", "")
            disaster_type = item.get("disaster_type", "")
            content = item.get("content_preview", "")
            
            # Check if source is mentioned
            source_cited = source.lower() in response_lower if source else False
            
            # Check if disaster type is mentioned
            disaster_cited = disaster_type.lower() in response_lower if disaster_type else False
            
            # Check if specific content is referenced
            content_words = [w for w in content.split()[:5] if len(w) > 4]
            content_cited = any(w.lower() in response_lower for w in content_words)
            
            if source_cited or disaster_cited or content_cited:
                cited_sources.append(source)
        
        # Calculate citation score
        if not evidence:
            score = 0.0
        else:
            score = len(cited_sources) / len(evidence)
        
        if len(cited_sources) < self.MIN_CITATIONS_REQUIRED and len(evidence) > 0:
            issues.append(f"Missing citations: Only {len(cited_sources)}/{len(evidence)} sources cited")
        
        return {
            "score": score,
            "cited_sources": cited_sources,
            "issues": issues
        }
    
    def _check_hallucinations(
        self, 
        query: str, 
        response: str, 
        evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use LLM to detect potential hallucinations in the response."""
        issues = []
        
        if not self.llm_manager:
            # Fallback: Basic keyword check
            return {"score": 0.5, "issues": ["LLM unavailable for hallucination check"]}
        
        # Build evidence summary
        evidence_summary = "\n".join([
            f"- Source: {e.get('source', 'Unknown')} | "
            f"Type: {e.get('disaster_type', 'Unknown')} | "
            f"Content: {e.get('content_preview', 'N/A')[:100]}"
            for e in evidence[:5]
        ])
        
        critic_prompt = f"""You are a STRICT FACT-CHECKER for a crisis response system.

TASK: Analyze if the RESPONSE contains claims NOT supported by the EVIDENCE.

QUERY: {query}

EVIDENCE:
{evidence_summary}

RESPONSE TO CHECK:
{response}

INSTRUCTIONS:
1. Check if every factual claim in the RESPONSE is grounded in the EVIDENCE.
2. Flag specific phrases that appear to be hallucinated or invented.
3. Rate the grounding quality from 0.0 (completely hallucinated) to 1.0 (fully grounded).

OUTPUT FORMAT (JSON only, no markdown):
{{"grounding_score": 0.0-1.0, "hallucinated_claims": ["claim1", "claim2"], "verdict": "grounded|partial|hallucinated"}}"""

        try:
            result = self.llm_manager.chat_completion(
                messages=[{"role": "user", "content": critic_prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            # Parse JSON response
            if result:
                # Clean potential markdown formatting
                result = result.strip()
                if result.startswith("```"):
                    result = re.sub(r'^```json?\s*', '', result)
                    result = re.sub(r'\s*```$', '', result)
                
                parsed = json.loads(result)
                score = float(parsed.get("grounding_score", 0.5))
                hallucinated = parsed.get("hallucinated_claims", [])
                
                if hallucinated:
                    issues.append(f"Potential hallucinations detected: {', '.join(hallucinated[:3])}")
                
                return {"score": score, "issues": issues}
                
        except (json.JSONDecodeError, Exception) as e:
            # Fallback on parse error
            print(f"   ⚠️ Critic parse error: {e}")
            return {"score": 0.6, "issues": []}
        
        return {"score": 0.5, "issues": issues}
    
    def _assess_evidence_quality(self, evidence: List[Dict[str, Any]]) -> float:
        """Assess the quality of retrieved evidence based on scores."""
        if not evidence:
            return 0.0
        
        scores = [e.get("score", 0.5) for e in evidence]
        avg_score = sum(scores) / len(scores)
        
        # Bonus for having multiple high-quality sources
        high_quality_count = sum(1 for s in scores if s > 0.7)
        quality_bonus = min(0.2, high_quality_count * 0.05)
        
        return min(1.0, avg_score + quality_bonus)
    
    def _calculate_confidence(
        self, 
        citation_score: float, 
        hallucination_score: float, 
        evidence_quality: float
    ) -> float:
        """Calculate overall confidence score."""
        # Weighted average
        weights = {
            "citation": 0.3,
            "hallucination": 0.4,
            "evidence": 0.3
        }
        
        confidence = (
            citation_score * weights["citation"] +
            hallucination_score * weights["hallucination"] +
            evidence_quality * weights["evidence"]
        )
        
        return round(confidence, 2)
    
    def _determine_status(
        self, 
        confidence: float, 
        issues: List[str],
        citation_result: Dict[str, Any]
    ) -> str:
        """Determine the evaluation status based on confidence and issues."""
        # High confidence and no major issues = approved
        if confidence >= self.CONFIDENCE_THRESHOLD_HIGH:
            if len(citation_result.get("cited_sources", [])) >= self.MIN_CITATIONS_REQUIRED:
                return "approved"
        
        # Very low confidence = needs more research
        if confidence < self.CONFIDENCE_THRESHOLD_LOW:
            return "needs_research"
        
        # Missing citations specifically = needs research
        if not citation_result.get("cited_sources"):
            return "needs_research"
        
        # Low-medium confidence with some citations = needs rephrase
        if confidence < self.CONFIDENCE_THRESHOLD_HIGH:
            return "needs_rephrase"
        
        return "approved"
    
    def _generate_feedback(
        self, 
        status: str, 
        issues: List[str], 
        confidence: float
    ) -> str:
        """Generate human-readable feedback based on evaluation."""
        if status == "approved":
            return f"Response verified with {confidence:.0%} confidence. Citations present and claims grounded."
        
        if status == "needs_research":
            issue_summary = "; ".join(issues[:2]) if issues else "Insufficient evidence"
            return f"Additional research needed. Issues: {issue_summary}"
        
        if status == "needs_rephrase":
            issue_summary = "; ".join(issues[:2]) if issues else "Low confidence"
            return f"Response needs improvement. Issues: {issue_summary}"
        
        return "Evaluation complete."
    
    def _suggest_broader_query(self, query: str) -> str:
        """Suggest a broader version of the query for re-search."""
        # Remove specific location markers for broader search
        broader = re.sub(r'\b(in|at|near|around)\s+\w+', '', query, flags=re.IGNORECASE)
        return broader.strip() if broader.strip() != query else query + " crisis alert"
    
    def _suggest_refined_query(
        self, 
        query: str, 
        issues: List[str], 
        evidence: List[Dict[str, Any]]
    ) -> str:
        """Suggest a refined query based on issues found."""
        # If no evidence, try broadening
        if not evidence:
            return self._suggest_broader_query(query)
        
        # If citations missing, try more specific terms from evidence
        if any("citation" in issue.lower() for issue in issues):
            # Extract key terms from evidence
            evidence_terms = []
            for e in evidence[:2]:
                disaster = e.get("disaster_type", "")
                if disaster and disaster != "Unknown":
                    evidence_terms.append(disaster)
            
            if evidence_terms:
                return f"{query} {' '.join(evidence_terms)}"
        
        return query
    
    def get_rephrase_prompt(self, original_response: str, feedback: str) -> str:
        """Generate a prompt to help the Analyst rephrase the response."""
        return f"""The previous response needs improvement.

ORIGINAL RESPONSE:
{original_response}

CRITIC FEEDBACK:
{feedback}

Please rewrite the response to:
1. Explicitly cite the evidence sources by name
2. Only include claims that are directly supported by the evidence
3. If evidence is limited, acknowledge uncertainty
4. Maintain the helpful, tactical tone

IMPROVED RESPONSE:"""


# Singleton Pattern
_critic_agent = None

def get_critic_agent() -> CriticAgent:
    """Get singleton CriticAgent instance."""
    global _critic_agent
    if _critic_agent is None:
        _critic_agent = CriticAgent()
    return _critic_agent


if __name__ == "__main__":
    # Test the Critic Agent
    critic = get_critic_agent()
    
    # Test case: Response with evidence
    test_evidence = [
        {"source": "flood_alert.jpg", "disaster_type": "flood", "content_preview": "Major flooding in Chennai area", "score": 0.85},
        {"source": "sensor_data.txt", "disaster_type": "flood", "content_preview": "Water level rising rapidly", "score": 0.72}
    ]
    
    test_response = "Based on flood_alert.jpg, there is major flooding in Chennai. The sensor data confirms rising water levels."
    
    result = critic.evaluate(
        query="Is it safe to travel to Chennai?",
        response=test_response,
        evidence=test_evidence
    )
    
    print(f"\n🔍 Critic Evaluation Result:")
    print(f"   Status: {result.status}")
    print(f"   Confidence: {result.confidence:.0%}")
    print(f"   Feedback: {result.feedback}")
    print(f"   Issues: {result.issues}")
