"""
Retrieval Tools for Strategic Retrieval Upgrade.
Encapsulates Qdrant search logic as callable tools for the Analyst Agent.
"""
import json
from fastembed import TextEmbedding
from config import get_qdrant_client

class RetrievalToolkit:
    """
    Provides search tools for the Analyst Agent.
    """
    
    def __init__(self, client, text_model, clip_text_model):
        self.client = client
        self.text_model = text_model
        self.clip_text_model = clip_text_model
        
        # Collection Names (from dashboard/config)
        self.VISUAL_COLLECTION = "visual_memory"
        self.AUDIO_COLLECTION = "audio_memory"
        self.TACTICAL_COLLECTION = "tactical_memory"
        self.score_threshold = 0.30  # Lowered to allow more matches

    def search_visual_memory(self, query: str):
        """
        Search for images and videos using natural language.
        Use this when the user asks to 'see', 'show', or check for visual evidence like 'fire', 'flood', etc.
        """
        print(f"   🔎 Tool Call: search_visual_memory('{query}')")
        if not self.client.collection_exists(self.VISUAL_COLLECTION):
            return {"error": "Visual memory collection does not exist."}
            
        try:
            # Embed using CLIP text model
            vector = list(self.clip_text_model.embed([query]))[0].tolist()
            
            results = self.client.query_points(
                collection_name=self.VISUAL_COLLECTION, 
                query=vector, 
                limit=5, 
                with_payload=True
            )
            
            hits = []
            for hit in results.points:
                if hit.score > self.score_threshold:
                    hits.append({
                        "type": "visual",
                        "source": hit.payload.get("source"),
                        "disaster": hit.payload.get("detected_disaster"),
                        "text_content": hit.payload.get("ocr_text", "")[:150],
                        "location": hit.payload.get("location", {}).get("name", "Unknown"),
                        "score": hit.score,
                        "raw_payload": hit.payload # Internal use
                    })
            
            if not hits:
                return {"result": []}  # Return empty list for consistent parsing
                
            return {"result": hits}
        except Exception as e:
            return {"error": str(e)}

    def search_tactical_memory(self, query: str):
        """
        Search for text reports, sensor data, and tactical updates.
        Use this for specific details, locations, or reported incidents.
        """
        print(f"   🔎 Tool Call: search_tactical_memory('{query}')")
        if not self.client.collection_exists(self.TACTICAL_COLLECTION):
            return {"error": "Tactical memory collection does not exist."}
            
        try:
            # Embed using BGE text model
            vector = list(self.text_model.embed([query]))[0].tolist()
            
            results = self.client.query_points(
                collection_name=self.TACTICAL_COLLECTION, 
                query=vector, 
                limit=5, 
                with_payload=True
            )
            
            hits = []
            for hit in results.points:
                if hit.score > self.score_threshold:
                    hits.append({
                        "type": "text",
                        "source": hit.payload.get("source"),
                        "content": hit.payload.get("content", "")[:200],
                        "location": hit.payload.get("location", {}).get("name", "Unknown"),
                        "score": hit.score,
                        "raw_payload": hit.payload
                    })
            
            if not hits:
                return {"result": []}  # Return empty list for consistent parsing
                
            return {"result": hits}
        except Exception as e:
            return {"error": str(e)}
            
    def search_audio_memory(self, query: str):
        """
        Search for audio transcripts and radio chatter.
        Use this to check for spoken reports, distress calls, or radio communications.
        """
        print(f"   🔎 Tool Call: search_audio_memory('{query}')")
        if not self.client.collection_exists(self.AUDIO_COLLECTION):
            return {"error": "Audio memory collection does not exist."}
            
        try:
            # Embed using BGE text model
            vector = list(self.text_model.embed([query]))[0].tolist()
            
            results = self.client.query_points(
                collection_name=self.AUDIO_COLLECTION, 
                query=vector, 
                limit=5, 
                with_payload=True
            )
            
            hits = []
            for hit in results.points:
                if hit.score > self.score_threshold:
                    hits.append({
                        "type": "audio",
                        "source": hit.payload.get("source"),
                        "transcript": hit.payload.get("transcript", "")[:200],
                        "location": hit.payload.get("location", {}).get("name", "Unknown"),
                        "score": hit.score,
                        "raw_payload": hit.payload
                    })
            
            if not hits:
                return {"result": []}  # Return empty list for consistent parsing
                
            return {"result": hits}
        except Exception as e:
            return {"error": str(e)}

    def get_tool_definitions(self):
        """Return the tool schema for the LLM."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_visual_memory",
                    "description": "Search for visual evidence (images/videos) of disasters, damages, or specific objects.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string", 
                                "description": "The search query (e.g. 'flooded streets', 'collapsed building')"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_tactical_memory",
                    "description": "Search text reports and tactical logs for locations, stats, and specific details.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string", 
                                "description": "The search query (e.g. 'Kochi casualty report')"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_audio_memory",
                    "description": "Search radio transcripts and audio logs for spoken reports.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string", 
                                "description": "The search query (e.g. 'distress call from sector 4')"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
