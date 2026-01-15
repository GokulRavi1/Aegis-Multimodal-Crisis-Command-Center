import time
import datetime
import random
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding

# Configuration
COLLECTION_NAME = "audio_memory"
QDRANT_URL = "http://localhost:6333"

# Simulated emergency radio transcripts for demo purposes
SAMPLE_TRANSCRIPTS = [
    {"transcript": "Bridge collapsed near sector 4. Requesting immediate backup.", "urgency": "high", "unit_id": "Alpha-1"},
    {"transcript": "Flooding reported in residential block 7. Evacuations underway.", "urgency": "high", "unit_id": "Bravo-3"},
    {"transcript": "All clear in sector 2. No casualties reported.", "urgency": "low", "unit_id": "Charlie-5"},
    {"transcript": "Medical emergency at intersection 5th and Main. Ambulance dispatched.", "urgency": "medium", "unit_id": "Delta-2"},
    {"transcript": "Fire spotted near warehouse district. Engine 7 responding.", "urgency": "high", "unit_id": "Echo-1"},
    {"transcript": "Road blocked by debris on Highway 12. Traffic rerouting advised.", "urgency": "medium", "unit_id": "Foxtrot-4"},
    {"transcript": "Civilian group stranded on rooftop. Helicopter rescue requested.", "urgency": "high", "unit_id": "Golf-6"},
    {"transcript": "Power lines down in grid 9. Electricity shut off for safety.", "urgency": "medium", "unit_id": "Hotel-8"},
    {"transcript": "Water level rising rapidly near dam. Evacuation order issued.", "urgency": "high", "unit_id": "India-7"},
    {"transcript": "Search and rescue team deployed to collapsed building sector 3.", "urgency": "high", "unit_id": "Juliet-9"},
]

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
    print("Initializing Listener Agent (Audio)...")
    client = init_qdrant()
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("Processing simulated emergency radio transcripts...")
    
    for i, sample in enumerate(SAMPLE_TRANSCRIPTS):
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        payload = {
            "source": "Radio-Channel-1",
            "type": "audio",
            "transcript": sample["transcript"],
            "timestamp": timestamp,
            "urgency": sample["urgency"],
            "unit_id": sample["unit_id"]
        }
        
        # Generate text embedding
        embeddings = list(embedding_model.embed([sample["transcript"]]))
        vector = embeddings[0]
        
        # Upsert to Qdrant
        point_id = int(time.time() * 1000) + i  # Unique ID
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
        print(f"Upserted: [{sample['urgency'].upper()}] {sample['unit_id']}: {sample['transcript'][:50]}...")
        
        # Small delay to simulate real-time ingestion
        time.sleep(0.5)
    
    print(f"\nListener Agent finished. Ingested {len(SAMPLE_TRANSCRIPTS)} audio transcripts.")

if __name__ == "__main__":
    main()
