import datetime
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, ImageEmbedding
from config import get_qdrant_client
from PIL import Image
import numpy as np

# Hardcoded collection names
TACTICAL_COLLECTION = "tactical_memory"
VISUAL_COLLECTION = "visual_memory"
AUDIO_COLLECTION = "audio_memory"

# City Coordinates for Mock Data
CITY_COORDS = {
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
    "Kerala": {"lat": 10.8505, "lon": 76.2711}, # Generic Kerala center
    "Kochi": {"lat": 9.9312, "lon": 76.2673},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Wayanad": {"lat": 11.6854, "lon": 76.1320},
    "Madurai": {"lat": 9.9252, "lon": 78.1198},
    "Thiruvananthapuram": {"lat": 8.5241, "lon": 76.9366}
}

def ensure_collection(client, collection_name, vector_size=384):
    """Creates collection if it doesn't exist."""
    if not client.collection_exists(collection_name):
        print(f"📦 Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

def generate_mock_data():
    client = get_qdrant_client()
    text_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Ensure collections exist
    ensure_collection(client, TACTICAL_COLLECTION, 384)
    ensure_collection(client, AUDIO_COLLECTION, 384)
    
    # 1. TACTICAL REPORTS (Text)
    print("📝 Generating Tactical Reports...")
    reports = [
        ("URGENT: Major urban flooding reported in Silk Board Junction, Bengaluru. Traffic halted.", "flood", "Bengaluru"),
        ("Cyclone Warning for Kerala coast. Fishermen advised not to venture into sea. Red alert in Kochi.", "cyclone", "Kerala"),
        ("Infrastructure failure: Bridge collapse in Northern Kochi. Casualties feared. Rescue underway.", "collapsed building", "Kochi"),
        ("Fire hazard alert: Industrial fire near Chennai Port. Chemical fumes detected.", "fire", "Chennai"),
        ("Status Update: Flood waters receding in Chennai T. Nagar area. Relief camps operational.", "flood", "Chennai"),
        ("Emergency: Landslide reported in Wayanad, Kerala. Roads blocked.", "landslide", "Wayanad"),
        ("Medical Emergency: Dengue outbreak in Madurai confirmed. Hospitals on high alert.", "epidemic", "Madurai")
    ]
    
    embeddings = list(text_model.embed([r[0] for r in reports]))
    points = []
    
    for i, ((text, disaster, loc), vector) in enumerate(zip(reports, embeddings)):
        coords = CITY_COORDS.get(loc, {"lat": 28.7041, "lon": 77.1025}) # Default Delhi
        points.append({
            "id": i + 1000,
            "vector": vector.tolist(),
            "payload": {
                "content": text,
                "source": "simulated_report.txt",
                "type": "text",
                "detected_disaster": disaster,
                "timestamp": datetime.datetime.now().isoformat(),
                "location": {"name": loc, "lat": coords["lat"], "lon": coords["lon"]} 
            }
        })
        
    client.upsert(collection_name=TACTICAL_COLLECTION, points=points)
    print(f"✅ Indexed {len(points)} tactical reports.")

    # 2. AUDIO TRANSCRIPTS (Text Embedding of 'Audio')
    print("🎙️ Generating Audio Transcripts...")
    transcripts = [
        ("Control room, this is Unit 4. We are seeing massive flooding near the subway. Requesting backup.", "flood", "Chennai"),
        ("Panic in the streets here in Kochi! The bridge just gave way! Send ambulances!", "collapsed building", "Kochi"),
        ("Warning, fire spreading rapidly in the warehouse district. Wind is carrying smoke west.", "fire", "Chennai"),
        ("Heavy rain continuing in Thiruvananthapuram. Water levels rising in the dam.", "flood", "Thiruvananthapuram")
    ]
    
    aud_embeddings = list(text_model.embed([t[0] for t in transcripts]))
    aud_points = []
    
    for i, ((text, disaster, loc), vector) in enumerate(zip(transcripts, aud_embeddings)):
        coords = CITY_COORDS.get(loc, {"lat": 28.7041, "lon": 77.1025})
        aud_points.append({
            "id": i + 2000,
            "vector": vector.tolist(),
            "payload": {
                "transcript": text,
                "source": "simulated_radio.wav",
                "type": "audio",
                "detected_disaster": disaster,
                "timestamp": datetime.datetime.now().isoformat(),
                "location": {"name": loc, "lat": coords["lat"], "lon": coords["lon"]}
            }
        })
        
    client.upsert(collection_name=AUDIO_COLLECTION, points=aud_points)
    print(f"✅ Indexed {len(aud_points)} audio transcripts.")

    print("\n🎉 Mock Data Generation Complete! Run benchmark.py again.")

if __name__ == "__main__":
    generate_mock_data()
