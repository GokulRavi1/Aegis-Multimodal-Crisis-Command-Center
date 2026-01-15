import random
import uuid
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configuration
COLLECTION_NAME = "civilian_memory"
QDRANT_URL = "http://localhost:6333"
CIVILIAN_COUNT = 20

# Drone starts at 28.7041, 77.1025. We want civilians near/around this path.
BASE_LAT = 28.7041
BASE_LON = 77.1025

def init_qdrant():
    client = QdrantClient(url=QDRANT_URL)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE), # Dummy vector size
        )
        print(f"Created collection: {COLLECTION_NAME}")
    return client

def generate_civilians(count=20):
    civilians = []
    for i in range(count):
        # Random location within ~5-10km of base
        lat_offset = random.uniform(-0.05, 0.05)
        lon_offset = random.uniform(-0.05, 0.05)
        
        civilians.append({
            "id": str(uuid.uuid4()),
            "payload": {
                "type": "civilian",
                "name": f"Civilian Group {i+1}",
                "location": {
                    "lat": BASE_LAT + lat_offset,
                    "lon": BASE_LON + lon_offset
                },
                "status": "unknown"
            }
        })
    return civilians

def main():
    print("Initializing Civilian Generator...")
    client = init_qdrant()
    
    civilians = generate_civilians(CIVILIAN_COUNT)
    
    points = []
    for civ in civilians:
        points.append(models.PointStruct(
            id=civ["id"],
            vector=[0.0], # Dummy vector as we rely on payload/geolocation
            payload=civ["payload"]
        ))
        
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"Successfully generated and upserted {len(points)} civilian records.")

if __name__ == "__main__":
    main()
