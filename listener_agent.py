import os
import time
import datetime
import speech_recognition as sr
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from config import get_qdrant_client, ensure_collection, DISASTER_CLASSES

# ... (Configuration unchanged) ...

class AudioHandler(FileSystemEventHandler):
    # ... (init/on_created unchanged) ...

    def process_audio(self, file_path):
        # ... (Transcription logic unchanged, up to line 63) ...
        print(f"📝 Transcript: {transcript}")
            
        try:
            # Embed the transcript
            embeddings = list(self.embed_model.embed([transcript]))
            vector = embeddings[0]
            
            # Detect Disaster via Keywords
            detected_disaster = "None"
            confidence = 0.0
            
            lower_text = transcript.lower()
            for disaster in DISASTER_CLASSES:
                if disaster in lower_text:
                    detected_disaster = disaster
                    confidence = 0.8 # High confidence if keyword explicit
                    break # Take first match
            
            # Alerts
            alerts = []
            if detected_disaster != "None":
                alerts.append(f"AUDIO ALERT: {detected_disaster.upper()}")

            # Upsert
            payload = {
                "source": os.path.basename(file_path),
                "type": "audio",
                "detected_disaster": detected_disaster,
                "confidence": {"audio": confidence},
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "transcript": transcript,
                "urgency": "high" if detected_disaster != "None" else "medium", 
                "content": transcript,
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
            print(f"✅ Indexed audio. Disaster: {detected_disaster}")
            
        except Exception as e:
            print(f"❌ Error indexing audio: {e}")

def main():
    if not os.path.exists(AUDIO_INBOX):
        os.makedirs(AUDIO_INBOX)
        print(f"📂 Created {AUDIO_INBOX}")
        
    client = get_qdrant_client()
    ensure_collection(client, COLLECTION_NAME, 384)
    
    print("Loading Text Embedding Model...")
    embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    observer = Observer()
    handler = AudioHandler(client, embed_model)
    observer.schedule(handler, AUDIO_INBOX, recursive=False)
    observer.start()
    
    print(f"👂 Listener Agent active. Monitoring '{AUDIO_INBOX}' for .wav/.flac files...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
