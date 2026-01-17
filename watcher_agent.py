import cv2
import time
import datetime
import os
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import ImageEmbedding
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ultralytics import YOLOWorld

# Configuration
VIDEO_INBOX = "video_inbox"
COLLECTION_NAME = "visual_memory"
QDRANT_URL = "http://localhost:6333"
MODEL_NAME = "yolov8s-worldv2.pt"
CUSTOM_CLASSES = ["person", "car", "flood", "cyclone", "hurricane"]

def init_qdrant():
    client = QdrantClient(url=QDRANT_URL)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
        )
    return client

class VideoHandler(FileSystemEventHandler):
    def __init__(self, client, embed_model, yolo_model):
        self.client = client
        self.embed_model = embed_model
        self.yolo_model = yolo_model

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"🎬 New video detected: {event.src_path}")
            self.process_video(event.src_path)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return

        frame_count = 0
        print(f"⚡ Processing Video: {video_path}...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 30 != 0:
                continue

            # Detection
            results = self.yolo_model.predict(frame, conf=0.1, verbose=False)
            result = results[0]
            
            detections = {}
            if result.boxes:
                for cls_id in result.boxes.cls:
                    name = self.yolo_model.names[int(cls_id)]
                    detections[name] = detections.get(name, 0) + 1
            
            if detections:
                print(f"   Frame {frame_count}: Found {detections}")

            # Save Latest Frame
            annotated_frame = result.plot()
            cv2.imwrite("latest_frame.jpg", annotated_frame)
            
            # Embedding
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            embeddings = list(self.embed_model.embed([pil_image]))
            vector = embeddings[0]

            payload = {
                "source": os.path.basename(video_path),
                "type": "visual",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "detections": detections,
                "location": {"lat": 28.7041, "lon": 77.1025}
            }
            
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=int(time.time() * 1000),
                        vector=vector.tolist(),
                        payload=payload
                    )
                ]
            )
            
        cap.release()
        print(f"✅ Finished Video.")

def main():
    print("Initializing Watcher Agent (Video Only)...")
    
    if not os.path.exists(VIDEO_INBOX):
        os.makedirs(VIDEO_INBOX)
        print(f"📂 Created {VIDEO_INBOX}")

    client = init_qdrant()
    
    print("⏳ Loading Models...")
    embed_model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
    yolo_model = YOLOWorld(MODEL_NAME)
    yolo_model.set_classes(CUSTOM_CLASSES)
    
    observer = Observer()
    handler = VideoHandler(client, embed_model, yolo_model)
    observer.schedule(handler, VIDEO_INBOX, recursive=False)
    observer.start()
    
    print(f"👀 Monitoring {VIDEO_INBOX} for videos...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
