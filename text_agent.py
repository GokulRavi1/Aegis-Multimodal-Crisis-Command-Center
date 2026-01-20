import os
import time
import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from config import get_qdrant_client, ensure_collection, DISASTER_CLASSES

# ... (Config unchanged) ...

class TextHandler(FileSystemEventHandler):
    # ... (init/on_created/process_text read logic unchanged) ...
    # (assuming we are inside process_text after reading content)

    def process_text(self, file_path):
        # ... (Read logic) ...
        print(f"⚡ Reading {file_path}...")
        
        content = None
        for i in range(5):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if content:
                    break
            except (PermissionError, IOError):
                print(f"   ⏳ Waiting for file release... ({i+1}/5)")
                time.sleep(1)
        
        if not content:
            print("❌ Error reading text (locked or empty)")
            return

        if not content.strip():
            return

        print(f"📖 Content Preview: {content[:50]}...")
            
        try:
            # Embed
            embeddings = list(self.embed_model.embed([content]))
            vector = embeddings[0]
            
            # Detect Disaster via Keywords
            detected_disaster = "None"
            confidence = 0.0
            
            lower_text = content.lower()
            for disaster in DISASTER_CLASSES:
                if disaster in lower_text:
                    detected_disaster = disaster
                    confidence = 0.85 # Text is usually explicit
                    break
            
            # Alerts
            alerts = []
            if detected_disaster != "None":
                alerts.append(f"FUSED ALERT: {detected_disaster.upper()}")

            payload = {
                "source": os.path.basename(file_path),
                "type": "text",
                "detected_disaster": detected_disaster,
                "confidence": {"text": confidence},
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "content": content,
                "location_ref": "Unknown",
                "reliability_score": 0.8,
                "triggered_alerts": alerts
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
            print(f"✅ Indexed text document. Disaster: {detected_disaster}")
            
        except Exception as e:
            print(f"❌ Error processing text: {e}")

def main():
    if not os.path.exists(TEXT_INBOX):
        os.makedirs(TEXT_INBOX)
        print(f"📂 Created {TEXT_INBOX}")

    client = get_qdrant_client()
    ensure_collection(client, COLLECTION_NAME, 384)
    
    print("Loading Text Embedding Model...")
    embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    observer = Observer()
    handler = TextHandler(client, embed_model)
    observer.schedule(handler, TEXT_INBOX, recursive=False)
    observer.start()
    
    print(f"🧐 Text Agent active. Monitoring '{TEXT_INBOX}'...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
