"""
Critic Memory Module for Aegis Crisis Command Center
Stores Critic evaluations and feedback for continuous improvement.
Enables learning from past corrections and performance tracking.
"""

import datetime
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from qdrant_client import models
from config import get_qdrant_client
from fastembed import TextEmbedding


class CriticMemory:
    """
    Stores Critic evaluations in Qdrant for continuous improvement.
    
    Features:
    1. Evaluation Storage - Persists every Critic decision
    2. Pattern Recognition - Retrieves similar past corrections
    3. Performance Tracking - Aggregates statistics for monitoring
    """
    
    COLLECTION_NAME = "critic_feedback"
    VECTOR_DIM = 384  # BGE small embedding dimension
    
    def __init__(self):
        self.client = get_qdrant_client()
        self._ensure_collection()
        
        # Lazy load embedding model
        self._embed_model = None
        print("📝 CriticMemory initialized")
    
    @property
    def embed_model(self):
        """Lazy load embedding model."""
        if self._embed_model is None:
            self._embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return self._embed_model
    
    def _ensure_collection(self):
        """Create the critic feedback collection if it doesn't exist."""
        if not self.client.collection_exists(self.COLLECTION_NAME):
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=self.VECTOR_DIM,
                    distance=models.Distance.COSINE
                )
            )
            print(f"   📁 Created collection: {self.COLLECTION_NAME}")
    
    def log_evaluation(
        self,
        query: str,
        response: str,
        evaluation: Dict[str, Any],
        evidence_count: int,
        attempt_number: int = 1,
        final_status: str = "shown"
    ) -> str:
        """
        Log a Critic evaluation to memory.
        
        Args:
            query: Original user query
            response: The evaluated response
            evaluation: CriticResult as dict
            evidence_count: Number of evidence items used
            attempt_number: Which attempt this was (1, 2, etc.)
            final_status: "shown" | "rejected" | "improved"
            
        Returns:
            Point ID of the stored evaluation
        """
        import time
        point_id = int(time.time() * 1000)
        
        # Create embedding from query + response summary
        embed_text = f"{query} | {response[:200]}"
        vector = list(self.embed_model.embed([embed_text]))[0].tolist()
        
        payload = {
            "query": query,
            "response_preview": response[:500],
            "status": evaluation.get("status", "unknown"),
            "confidence": evaluation.get("confidence", 0.0),
            "feedback": evaluation.get("feedback", ""),
            "issues": evaluation.get("issues", []),
            "evidence_count": evidence_count,
            "attempt_number": attempt_number,
            "final_status": final_status,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        
        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )
        
        print(f"   📝 Logged evaluation: {evaluation.get('status')} ({evaluation.get('confidence', 0):.0%})")
        return str(point_id)
    
    def get_similar_corrections(
        self, 
        query: str, 
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar past corrections for learning.
        
        Useful for:
        - Understanding common failure patterns
        - Pre-emptively addressing known issues
        - Providing context to the Analyst
        """
        # Embed the query
        vector = list(self.embed_model.embed([query]))[0].tolist()
        
        # Search for similar evaluations that had issues
        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="status",
                        match=models.MatchAny(any=["needs_research", "needs_rephrase"])
                    )
                ]
            ),
            limit=limit,
            with_payload=True
        )
        
        corrections = []
        for point in results.points:
            corrections.append({
                "query": point.payload.get("query"),
                "issues": point.payload.get("issues", []),
                "feedback": point.payload.get("feedback"),
                "score": point.score
            })
        
        return corrections
    
    def get_improvement_stats(self) -> Dict[str, Any]:
        """
        Get aggregate statistics on Critic evaluations.
        
        Returns:
            Dictionary with counts, averages, and trends
        """
        try:
            # Get all points
            all_points = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                limit=1000,
                with_payload=True
            )[0]
            
            if not all_points:
                return {
                    "total_evaluations": 0,
                    "message": "No evaluations logged yet"
                }
            
            # Calculate stats
            total = len(all_points)
            statuses = {}
            confidences = []
            multi_attempts = 0
            
            for point in all_points:
                payload = point.payload
                status = payload.get("status", "unknown")
                statuses[status] = statuses.get(status, 0) + 1
                
                conf = payload.get("confidence", 0)
                if conf:
                    confidences.append(conf)
                
                if payload.get("attempt_number", 1) > 1:
                    multi_attempts += 1
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            approval_rate = statuses.get("approved", 0) / total if total > 0 else 0
            
            return {
                "total_evaluations": total,
                "status_breakdown": statuses,
                "average_confidence": round(avg_confidence, 2),
                "approval_rate": round(approval_rate, 2),
                "multi_attempt_count": multi_attempts,
                "improvement_rate": round(multi_attempts / total, 2) if total > 0 else 0
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_recent_evaluations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent evaluations for debugging/monitoring."""
        try:
            points = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                limit=limit,
                with_payload=True
            )[0]
            
            evaluations = []
            for point in points:
                p = point.payload
                evaluations.append({
                    "timestamp": p.get("timestamp", ""),
                    "query": p.get("query", "")[:50],
                    "status": p.get("status"),
                    "confidence": p.get("confidence"),
                    "issues_count": len(p.get("issues", []))
                })
            
            # Sort by timestamp (newest first)
            evaluations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return evaluations
            
        except Exception as e:
            return [{"error": str(e)}]


# Singleton Pattern
_critic_memory = None

def get_critic_memory() -> CriticMemory:
    """Get singleton CriticMemory instance."""
    global _critic_memory
    if _critic_memory is None:
        _critic_memory = CriticMemory()
    return _critic_memory


if __name__ == "__main__":
    # Test the Critic Memory
    memory = get_critic_memory()
    
    print("\n📊 Critic Memory Stats:")
    stats = memory.get_improvement_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n📋 Recent Evaluations:")
    recent = memory.get_recent_evaluations(5)
    for eval_item in recent:
        print(f"   {eval_item}")
