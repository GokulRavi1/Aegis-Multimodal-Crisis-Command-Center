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
from config import get_qdrant_client, ensure_collection

# Configuration
IMAGE_INBOX = "image_inbox"
COLLECTION_NAME = "visual_memory"
MODEL_NAME = "yolov8s-worldv2.pt"
CUSTOM_CLASSES = ["person", "car", "flood", "cyclone", "hurricane"]

class ImageHandler(FileSystemEventHandler):
    def __init__(self, client, embed_model, yolo_model):
        self.client = client
        self.embed_model = embed_model
        self.yolo_model = yolo_model

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            print(f"🖼️ New image detected: {event.src_path}")
            self.process_image(event.src_path)

    def process_image(self, image_path):
        print(f"⚡ Processing Image: {image_path}...")
        
        # Retry mechanism for file locking
        frame = None
        for i in range(5):
            try:
                # Ensure file is readable
                with open(image_path, 'rb'):
                    pass
                frame = cv2.imread(image_path)
                if frame is not None:
                    break
            except (PermissionError, IOError):
                print(f"   ⏳ Waiting for file release... ({i+1}/5)")
                time.sleep(1)
        
        if frame is None:
            print("❌ Error reading image (locked or invalid)")
            return

        # Detection
        results = self.yolo_model.predict(frame, conf=0.1, verbose=False)
        result = results[0]
        
        detections = {}
        if result.boxes:
            for cls_id in result.boxes.cls:
                name = self.yolo_model.names[int(cls_id)]
                detections[name] = detections.get(name, 0) + 1
        
        if detections:
            print(f"   Found: {detections}")

        # Embedding
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        embeddings = list(self.embed_model.embed([pil_image]))
        vector = embeddings[0]

        # Upsert (Note: Type is 'image')
        payload = {
            "source": os.path.basename(image_path),
            "type": "image",
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
        print(f"✅ Indexed Image.")

def main():
    if not os.path.exists(IMAGE_INBOX):
        os.makedirs(IMAGE_INBOX)
        print(f"📂 Created {IMAGE_INBOX}")

    client = get_qdrant_client()
    ensure_collection(client, COLLECTION_NAME, 512)
    
    print("⏳ Loading Models for Image Agent...")
    embed_model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
    yolo_model = YOLOWorld(MODEL_NAME)
    yolo_model.set_classes(CUSTOM_CLASSES)
    
    observer = Observer()
    handler = ImageHandler(client, embed_model, yolo_model)
    observer.schedule(handler, IMAGE_INBOX, recursive=False)
    observer.start()
    
    print(f"👀 Image Agent active. Monitoring '{IMAGE_INBOX}'...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
