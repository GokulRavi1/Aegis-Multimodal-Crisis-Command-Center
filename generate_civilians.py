import random
import uuid
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config import get_qdrant_client, ensure_collection

# Configuration
COLLECTION_NAME = "civilian_memory"
CIVILIAN_COUNT = 20

# Drone starts at 28.7041, 77.1025. We want civilians near/around this path.
BASE_LAT = 28.7041
BASE_LON = 77.1025

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
    client = get_qdrant_client()
    ensure_collection(client, COLLECTION_NAME, 1) # Dummy vector size 1
    
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
