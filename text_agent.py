import time
import datetime
import random
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding

# Configuration
COLLECTION_NAME = "tactical_memory"
QDRANT_URL = "http://localhost:6333"

# Simulated social media posts and sensor reports
SAMPLE_POSTS = [
    {"content": "Help! Water rising near the main hospital entrance!", "source": "Twitter_User_123", "location_ref": "Sector-4"},
    {"content": "Fire spotted at Sector 7 warehouse district.", "source": "Twitter_User_456", "location_ref": "Sector-7"},
    {"content": "Roads are flooded near the central market area.", "source": "Sensor_Node_12", "location_ref": "Sector-2"},
    {"content": "HELP! Family trapped on rooftop, need rescue!", "source": "Twitter_User_789", "location_ref": "Sector-5"},
    {"content": "Power outage reported in residential block.", "source": "Sensor_Node_08", "location_ref": "Sector-3"},
    {"content": "Fire spreading rapidly towards school zone!", "source": "Twitter_User_321", "location_ref": "Sector-6"},
    {"content": "Bridge appears unstable, avoid crossing.", "source": "Sensor_Node_15", "location_ref": "Sector-1"},
    {"content": "Help needed! Elderly people stranded without food.", "source": "Twitter_User_654", "location_ref": "Sector-8"},
    {"content": "Gas leak detected near industrial area.", "source": "Sensor_Node_22", "location_ref": "Sector-9"},
    {"content": "All clear in downtown area, evacuation complete.", "source": "Twitter_User_111", "location_ref": "Sector-10"},
    {"content": "Fire engine stuck in floodwater, requesting backup.", "source": "Radio_Relay_01", "location_ref": "Sector-4"},
    {"content": "Medical supplies running low at shelter A.", "source": "Shelter_Admin", "location_ref": "Sector-2"},
]

def calculate_reliability(content: str) -> float:
    """Simple reliability scoring based on keywords."""
    high_priority_keywords = ["help", "fire", "trapped", "emergency", "rescue", "spreading"]
    content_lower = content.lower()
    
    for keyword in high_priority_keywords:
        if keyword in content_lower:
            return 0.9
    return 0.5

def init_qdrant():
    client = QdrantClient(url=QDRANT_URL)
    # Text embedding dimension for BAAI/bge-small-en-v1.5 is 384
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )
        print(f"Created collection: {COLLECTION_NAME}")
    return client

def main():
    print("Initializing Text Agent (Social & Sensors)...")
    client = init_qdrant()
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("Processing simulated social media and sensor data...")
    
    for i, sample in enumerate(SAMPLE_POSTS):
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        reliability = calculate_reliability(sample["content"])
        
        payload = {
            "source": sample["source"],
            "type": "text",
            "content": sample["content"],
            "timestamp": timestamp,
            "reliability_score": reliability,
            "location_ref": sample["location_ref"]
        }
        
        # Generate text embedding
        embeddings = list(embedding_model.embed([sample["content"]]))
        vector = embeddings[0]
        
        # Upsert to Qdrant
        point_id = int(time.time() * 1000) + i
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payload
                )
            ]
        )
        
        reliability_indicator = "🔴 HIGH" if reliability >= 0.9 else "🟡 MED"
        print(f"Upserted: [{reliability_indicator}] {sample['source']}: {sample['content'][:40]}...")
        
        time.sleep(0.3)
    
    print(f"\nText Agent finished. Ingested {len(SAMPLE_POSTS)} text reports.")

if __name__ == "__main__":
    main()
