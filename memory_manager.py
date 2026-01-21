"""
Memory Manager Module for Aegis Crisis Command Center
Centralized memory lifecycle management with eviction policies and provenance tracking.
"""

import os
import time
import datetime
from typing import Optional, Dict, List, Any
from qdrant_client import QdrantClient, models
from config import get_qdrant_client


class MemoryManager:
    """Centralized memory lifecycle manager for Aegis."""
    
    # Collection schemas
    COLLECTIONS = {
        "visual_memory": {"dim": 512, "description": "Images and video frames"},
        "audio_memory": {"dim": 384, "description": "Audio transcripts"},
        "tactical_memory": {"dim": 384, "description": "Text reports and social data"},
        "session_memory": {"dim": 384, "description": "Short-term working memory"},
    }
    
    # Standard payload schema
    PAYLOAD_SCHEMA = {
        "source": str,           # Original filename
        "modality": str,         # image, video, audio, text
        "timestamp": str,        # ISO format
        "disaster_type": str,    # Detected disaster class
        "confidence": float,     # Detection confidence
        "location": dict,        # {lat, lon, name}
        "content": str,          # Text content (transcript, OCR, etc)
        "revision": int,         # Update counter
        "session_id": str,       # For session memory
    }
    
    def __init__(self):
        self.client = get_qdrant_client()
        print("📦 MemoryManager initialized")
    
    def ensure_collections(self):
        """Create all collections if they don't exist."""
        for name, config in self.COLLECTIONS.items():
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(
                        size=config["dim"],
                        distance=models.Distance.COSINE
                    )
                )
                print(f"   📁 Created collection: {name}")
            else:
                print(f"   ✅ Collection exists: {name}")
    
    def upsert_point(
        self,
        collection: str,
        point_id: int,
        vector: List[float],
        payload: Dict[str, Any]
    ) -> bool:
        """Insert or update a point with standardized payload."""
        # Add metadata
        payload["revision"] = payload.get("revision", 0) + 1
        payload["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        try:
            self.client.upsert(
                collection_name=collection,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            return True
        except Exception as e:
            print(f"❌ Upsert failed: {e}")
            return False
    
    def update_point(
        self,
        collection: str,
        point_id: int,
        payload_updates: Dict[str, Any]
    ) -> bool:
        """Update payload fields without changing vector."""
        try:
            self.client.set_payload(
                collection_name=collection,
                payload=payload_updates,
                points=[point_id]
            )
            return True
        except Exception as e:
            print(f"❌ Update failed: {e}")
            return False
    
    def delete_point(self, collection: str, point_id: int) -> bool:
        """Delete a specific point."""
        try:
            self.client.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=[point_id])
            )
            return True
        except Exception as e:
            print(f"❌ Delete failed: {e}")
            return False
    
    def evict_by_age(self, collection: str, max_age_hours: int = 24) -> int:
        """Remove points older than max_age_hours."""
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=max_age_hours)
        cutoff_str = cutoff.isoformat()
        
        try:
            # Find old points
            old_points = self.client.scroll(
                collection_name=collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="timestamp",
                            range=models.Range(lt=cutoff_str)
                        )
                    ]
                ),
                limit=100,
                with_payload=False
            )[0]
            
            if old_points:
                ids = [p.id for p in old_points]
                self.client.delete(
                    collection_name=collection,
                    points_selector=models.PointIdsList(points=ids)
                )
                print(f"🗑️ Evicted {len(ids)} old points from {collection}")
                return len(ids)
            return 0
        except Exception as e:
            print(f"❌ Eviction failed: {e}")
            return 0
    
    def evict_by_confidence(self, collection: str, min_confidence: float = 0.3) -> int:
        """Remove low-confidence points."""
        try:
            low_conf_points = self.client.scroll(
                collection_name=collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="confidence",
                            range=models.Range(lt=min_confidence)
                        )
                    ]
                ),
                limit=100,
                with_payload=False
            )[0]
            
            if low_conf_points:
                ids = [p.id for p in low_conf_points]
                self.client.delete(
                    collection_name=collection,
                    points_selector=models.PointIdsList(points=ids)
                )
                print(f"🗑️ Evicted {len(ids)} low-confidence points from {collection}")
                return len(ids)
            return 0
        except Exception as e:
            print(f"❌ Eviction failed: {e}")
            return 0
    
    def clear_session_memory(self, session_id: Optional[str] = None) -> int:
        """Clear session memory (optionally by session_id)."""
        try:
            if session_id:
                points = self.client.scroll(
                    collection_name="session_memory",
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="session_id",
                                match=models.MatchValue(value=session_id)
                            )
                        ]
                    ),
                    limit=100
                )[0]
                ids = [p.id for p in points]
            else:
                # Clear all session memory
                count = self.client.count(collection_name="session_memory").count
                self.client.delete_collection("session_memory")
                self.ensure_collections()
                return count
            
            if ids:
                self.client.delete(
                    collection_name="session_memory",
                    points_selector=models.PointIdsList(points=ids)
                )
            return len(ids) if ids else 0
        except Exception as e:
            print(f"❌ Session clear failed: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics for all collections."""
        stats = {}
        for name in self.COLLECTIONS.keys():
            try:
                if self.client.collection_exists(name):
                    info = self.client.get_collection(name)
                    stats[name] = {
                        "points_count": getattr(info, "points_count", 0),
                        "indexed_vectors_count": getattr(info, "indexed_vectors_count", 0),
                    }
            except Exception as e:
                stats[name] = {"error": f"Could not retrieve: {str(e)}"}
        return stats
    
    def search_with_filters(
        self,
        collection: str,
        query_vector: List[float],
        limit: int = 5,
        disaster_type: Optional[str] = None,
        min_confidence: Optional[float] = None,
        modality: Optional[str] = None
    ) -> List[Any]:
        """Search with payload filters for hybrid retrieval."""
        filters = []
        
        if disaster_type:
            filters.append(
                models.FieldCondition(
                    key="disaster_type",
                    match=models.MatchValue(value=disaster_type)
                )
            )
        
        if min_confidence:
            filters.append(
                models.FieldCondition(
                    key="confidence",
                    range=models.Range(gte=min_confidence)
                )
            )
        
        if modality:
            filters.append(
                models.FieldCondition(
                    key="modality",
                    match=models.MatchValue(value=modality)
                )
            )
        
        search_filter = models.Filter(must=filters) if filters else None
        
        results = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=search_filter,
            limit=limit,
            with_payload=True
        )
        
        return results.points


# Singleton instance
_manager = None

def get_memory_manager() -> MemoryManager:
    """Get singleton MemoryManager instance."""
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager


if __name__ == "__main__":
    # Test the memory manager
    manager = get_memory_manager()
    manager.ensure_collections()
    print("\n📊 Memory Stats:")
    for name, stats in manager.get_stats().items():
        print(f"   {name}: {stats}")
