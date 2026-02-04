"""
Retrieval Logger Module for Aegis Crisis Command Center
Logs RAG provenance: queries, retrieved points, scores, and LLM responses.
"""

import os
import json
import datetime
from typing import List, Dict, Any, Optional

LOG_FILE = "retrieval_logs.json"


class RetrievalLogger:
    """Logs RAG retrieval provenance for traceability and audit."""
    
    def __init__(self, log_file: str = LOG_FILE):
        self.log_file = log_file
        self.logs = self._load_logs()
    
    def _load_logs(self) -> List[Dict]:
        """Load existing logs from file."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_logs(self):
        """Save logs to file."""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.logs[-1000:], f, indent=2)  # Keep last 1000 logs
    
    def log_retrieval(
        self,
        query: str,
        retrieved_points: List[Dict[str, Any]],
        llm_prompt: str,
        llm_response: str,
        model: str = "llama-3.3-70b-versatile",
        session_id: Optional[str] = None,
        critic_evaluation: Optional[Dict] = None
    ) -> str:
        """
        Log a RAG retrieval operation.
        
        Args:
            query: User's search query
            retrieved_points: List of {id, score, source, disaster_type, content_preview}
            llm_prompt: Full prompt sent to LLM
            llm_response: LLM's response
            model: Model name used
            session_id: Optional session identifier
            critic_evaluation: Optional Critic Agent evaluation result
        
        Returns:
            Log entry ID
        """
        log_id = f"log_{int(datetime.datetime.now().timestamp() * 1000)}"
        
        entry = {
            "id": log_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "query": query,
            "session_id": session_id,
            "retrieved_points": retrieved_points,
            "retrieval_count": len(retrieved_points),
            "llm_model": model,
            "llm_prompt_length": len(llm_prompt),
            "llm_response": llm_response,
            "citations": self._extract_citations(llm_response, retrieved_points),
            # Critic Agent fields
            "critic_status": critic_evaluation.get("status") if critic_evaluation else None,
            "critic_confidence": critic_evaluation.get("confidence") if critic_evaluation else None,
            "critic_issues": critic_evaluation.get("issues", []) if critic_evaluation else [],
            "critic_feedback": critic_evaluation.get("feedback") if critic_evaluation else None
        }
        
        self.logs.append(entry)
        self._save_logs()
        
        return log_id
    
    def _extract_citations(
        self,
        response: str,
        retrieved_points: List[Dict]
    ) -> List[Dict]:
        """Extract which retrieved points are cited in the response."""
        citations = []
        response_lower = response.lower()
        
        for point in retrieved_points:
            source = point.get("source", "")
            disaster = point.get("disaster_type", "")
            
            # Check if source or disaster type is mentioned
            is_cited = (
                source.lower() in response_lower or
                disaster.lower() in response_lower
            )
            
            if is_cited:
                citations.append({
                    "point_id": point.get("id"),
                    "source": source,
                    "score": point.get("score")
                })
        
        return citations
    
    def get_recent_logs(self, limit: int = 10) -> List[Dict]:
        """Get most recent retrieval logs."""
        return self.logs[-limit:][::-1]
    
    def get_log_by_id(self, log_id: str) -> Optional[Dict]:
        """Get specific log by ID."""
        for log in self.logs:
            if log.get("id") == log_id:
                return log
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        if not self.logs:
            return {"total_logs": 0}
        
        total = len(self.logs)
        avg_retrieved = sum(log.get("retrieval_count", 0) for log in self.logs) / total
        avg_citations = sum(len(log.get("citations", [])) for log in self.logs) / total
        
        return {
            "total_logs": total,
            "avg_points_retrieved": round(avg_retrieved, 2),
            "avg_citations_per_response": round(avg_citations, 2),
            "first_log": self.logs[0].get("timestamp") if self.logs else None,
            "last_log": self.logs[-1].get("timestamp") if self.logs else None
        }


def format_evidence_for_llm(retrieved_points: List[Any]) -> str:
    """
    Format retrieved points for LLM prompt with citations.
    
    Args:
        retrieved_points: List of Qdrant ScoredPoint objects
    
    Returns:
        Formatted evidence string for LLM context
    """
    evidence_lines = []
    
    for i, point in enumerate(retrieved_points, 1):
        p = point.payload if hasattr(point, 'payload') else point
        
        source = p.get("source", "Unknown")
        disaster = p.get("detected_disaster", p.get("disaster_type", "Unknown"))
        score = point.score if hasattr(point, 'score') else p.get("score", 0)
        
        # Get content based on modality
        content = ""
        if p.get("ocr_text"):
            content = f"OCR: {p['ocr_text'][:200]}"
        elif p.get("transcript"):
            content = f"Transcript: {p['transcript'][:200]}"
        elif p.get("content"):
            content = f"Content: {p['content'][:200]}"
        
        location = p.get("location", {})
        loc_name = location.get("name", "") if isinstance(location, dict) else ""
        
        evidence_lines.append(
            f"[Evidence {i}] Source: {source} | Disaster: {disaster} | "
            f"Confidence: {score:.2f} | Location: {loc_name}\n{content}"
        )
    
    return "\n\n".join(evidence_lines)


def create_grounded_prompt(
    query: str,
    evidence: str,
    system_instruction: str = None
) -> str:
    """
    Create an evidence-grounded LLM prompt with citation instructions.
    
    Args:
        query: User's question
        evidence: Formatted evidence string
        system_instruction: Optional custom system instruction
    
    Returns:
        Complete prompt with grounding instructions
    """
    if not system_instruction:
        system_instruction = """You are the Aegis Crisis Analyst. 
Analyze the provided evidence and answer the user's question.

CRITICAL RULES:
1. ONLY use facts from the provided evidence.
2. CITE your sources by mentioning the source name.
3. If evidence is insufficient, say "Based on available evidence..." 
4. Do NOT hallucinate or invent information.
5. Be concise and tactical (10-30 words per point)."""

    prompt = f"""{system_instruction}

=== EVIDENCE ===
{evidence}

=== USER QUESTION ===
{query}

=== YOUR ANALYSIS (cite sources) ==="""

    return prompt


# Singleton instance
_logger = None

def get_retrieval_logger() -> RetrievalLogger:
    """Get singleton RetrievalLogger instance."""
    global _logger
    if _logger is None:
        _logger = RetrievalLogger()
    return _logger


if __name__ == "__main__":
    # Test the logger
    logger = get_retrieval_logger()
    print("📊 Retrieval Stats:")
    print(json.dumps(logger.get_stats(), indent=2))
