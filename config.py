import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Qdrant Cloud Credentials
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Supported Disaster Categories for Detection
# Supported Disaster Categories for Detection
DISASTER_CLASSES = [
    # Geological
    "earthquake", "volcanic eruption", "tsunami", "landslide", "avalanche", 
    "sinkhole", "volcano", "lava",
    
    # Hydrological
    "flood", "flash flood", "storm surge", "mudflow", "debris flow", 
    "flooded street", "river overflow", "submerged car",
    
    # Meteorological
    "cyclone", "hurricane", "typhoon", "tornado", "thunderstorm", 
    "hailstorm", "blizzard", "ice storm", "heat wave", "cold wave", 
    "heavy rain", "strong wind",
    
    # Climatological
    "drought", "wildfire", "forest fire", "bushfire", "desertification", 
    "dry land", "parched earth", "smoke", "flames", "burning building",
    
    # Biological
    "epidemic", "pandemic", "insect infestation", "locust swarm", 
    "animal disease outbreak",
    
    # Extra-terrestrial
    "meteorite impact", "solar flare", "space weather storm",
    
    # Environmental / Atmospheric
    "dust storm", "sandstorm", "fog", "smog", "air pollution",
    
    # Human/Urban Context
    "explosion", "bomb blast", "fireball", "shattered glass", "collapsed building", "rubble",
    "medical emergency", "ambulance", "stretcher", "injured person",
    "person", "crowd", "car", "truck", "police car", "firetruck", "traffic jam"
]

if not QDRANT_URL:
    print("⚠️  WARNING: QDRANT_URL not set in .env. Defaulting to localhost.")
    QDRANT_URL = "http://localhost:6333"

def get_qdrant_client():
    """Returns a configured QdrantClient instance."""
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def ensure_collection(client, collection_name, vector_size=384):
    """Creates collection if it doesn't exist."""
    if not client.collection_exists(collection_name):
        print(f"📦 Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
