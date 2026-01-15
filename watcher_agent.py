import cv2
import time
import datetime
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import ImageEmbedding
from typing import Generator, Iterable

# Configuration
VIDEO_PATH = "flood_footage.mp4"
COLLECTION_NAME = "visual_memory"
QDRANT_URL = "http://localhost:6333"
FRAME_INTERVAL_SEC = 5

def init_qdrant():
    client = QdrantClient(url=QDRANT_URL)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
        )
        print(f"Created collection: {COLLECTION_NAME}")
    return client

def process_video(video_path: str) -> Generator[dict, None, None]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval_frames = int(fps * FRAME_INTERVAL_SEC)
    
    current_frame = 0
    simulated_lat = 28.7041
    simulated_lon = 77.1025

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % frame_interval_frames == 0:
            # Simulate movement
            simulated_lat += 0.001
            simulated_lon += 0.001
            
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            yield {
                "frame": frame,
                "payload": {
                    "source": "Drone-01",
                    "type": "visual",
                    "location": {"lat": simulated_lat, "lon": simulated_lon},
                    "timestamp": timestamp,
                    "hazard_type": "flood"
                }
            }
        
        current_frame += 1

    cap.release()

def main():
    print("Initializing Watcher Agent...")
    client = init_qdrant()
    embedding_model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")

    print(f"Processing video: {VIDEO_PATH}")
    for item in process_video(VIDEO_PATH):
        frame = item["frame"]
        payload = item["payload"]
        
        # Convert frame (BGR) to RGB for embedding
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array to PIL Image (fastembed expects PIL Image)
        pil_image = Image.fromarray(frame_rgb)
        
        # Generator for embedding (fastembed expects iterable)
        embeddings = list(embedding_model.embed([pil_image]))
        vector = embeddings[0]

        # Upsert to Qdrant
        point_id = int(time.time() * 1000) # Simple ID based on timestamp
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
        print(f"Upserted frame at {payload['timestamp']} | Loc: {payload['location']}")
        
        # Simulate real-time processing delay if needed, or just run through
        # time.sleep(1) 

if __name__ == "__main__":
    main()
