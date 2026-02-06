"""
Episodic Memory Module for Aegis.
Stores past interaction episodes to provide context for follow-up queries.
"""
import datetime
import time
from typing import List, Dict, Any
from qdrant_client import models
from config import get_qdrant_client
from fastembed import TextEmbedding

class EpisodicMemory:
    """
    Manages episodic memory of user interactions.
    Enables the agent to remember context like "User is focused on Chennai".
    """
    
    COLLECTION_NAME = "episodic_memory"
    VECTOR_DIM = 384
    
    def __init__(self):
        self.client = get_qdrant_client()
        self._ensure_collection()
        self._embed_model = None # Lazy load
        
    @property
    def embed_model(self):
        if self._embed_model is None:
            self._embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return self._embed_model

    def _ensure_collection(self):
        if not self.client.collection_exists(self.COLLECTION_NAME):
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=self.VECTOR_DIM,
                    distance=models.Distance.COSINE
                )
            )
            print(f"   🧠 Created Episodic Memory: {self.COLLECTION_NAME}")

    def add_episode(self, query: str, response: str, summary: str):
        """
        Store a new interaction episode.
        
        Args:
            query: User's query
            response: Agent's response
            summary: Extracted context/focus (e.g. "User asked about Chennai Floods")
        """
        point_id = int(time.time() * 1000)
        
        # Embed the SUMMARY (this is what we want to match against future intent)
        # OR Embed Query + Summary?
        # Embedding current query against past summary works well for "status of that".
        vector = list(self.embed_model.embed([summary]))[0].tolist()
        
        payload = {
            "query": query,
            "response": response[:500], # Store preview
            "summary": summary,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
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
        print(f"   🧠 Stored Episode: {summary}")

    def retrieve_context(self, current_query: str, limit: int = 3) -> List[str]:
        """
        Retrieve relevant past context for the current query.
        Returns a list of summary strings.
        """
        # Strategy:
        # 1. Get MOST RECENT episode (Immediate context is critical for "it", "that", "the status")
        # 2. Search for RELEVANT episodes (Long-term recall)
        
        context_items = []
        
        # 1. Recent (Last 2)
        try:
            # Scroll provides storage order (ID ascending = oldest first).
            # To get "Recent", we must fetch enough points to reach the end, or use a better query.
            # FIX: Increase limit to 100 to check more history.
            recent_points = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                limit=100, 
                with_payload=True
            )[0]
            
            # Sort by ID (timestamp) descending
            recent_points.sort(key=lambda x: x.id, reverse=True)
            
            for p in recent_points[:2]:
                timestamp = p.payload.get("timestamp", "").split("T")[1][:5] # Time only
                context_items.append(f"[Recent - {timestamp}] {p.payload.get('summary')}")
                
        except Exception as e:
            print(f"Error fetching recent context: {e}")

        # 2. Relevant (Vector Search)
        try:
            vector = list(self.embed_model.embed([current_query]))[0].tolist()
            search_results = self.client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=vector,
                limit=2,
                with_payload=True
            )
            
            for p in search_results.points:
                if p.score > 0.6: # High relevance threshold
                    summary = p.payload.get("summary")
                    # Dedup with recent
                    if not any(summary in c for c in context_items):
                        context_items.append(f"[Relevant] {summary}")
                        
        except Exception as e:
             print(f"Error fetching relevant context: {e}")
             
        return context_items

# Singleton
_episodic_memory = None
def get_episodic_memory():
    global _episodic_memory
    if _episodic_memory is None:
        _episodic_memory = EpisodicMemory()
    return _episodic_memory
