import os
import time
import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from config import get_qdrant_client, ensure_collection

# Configuration
TEXT_INBOX = "text_inbox"
COLLECTION_NAME = "tactical_memory"

class TextHandler(FileSystemEventHandler):
    def __init__(self, client, embed_model):
        self.client = client
        self.embed_model = embed_model

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.txt', '.md', '.log', '.json')):
            print(f"📄 New text detected: {event.src_path}")
            self.process_text(event.src_path)

    def process_text(self, file_path):
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
            
            payload = {
                "source": os.path.basename(file_path),
                "type": "text",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "content": content,
                "location_ref": "Unknown",
                "reliability_score": 0.8
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
            print(f"✅ Indexed text document.")
            
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
