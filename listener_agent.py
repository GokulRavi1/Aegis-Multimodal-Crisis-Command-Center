import os
import time
import datetime
import speech_recognition as sr
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

# Configuration
AUDIO_INBOX = "audio_inbox"
COLLECTION_NAME = "audio_memory"
QDRANT_URL = "http://localhost:6333"

def init_qdrant():
    client = QdrantClient(url=QDRANT_URL)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )
    return client

class AudioHandler(FileSystemEventHandler):
    def __init__(self, client, embed_model):
        self.client = client
        self.embed_model = embed_model
        self.recognizer = sr.Recognizer()

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.wav', '.flac', '.aiff')):
            print(f"🎤 New audio detected: {event.src_path}")
            self.process_audio(event.src_path)

    def process_audio(self, file_path):
        print(f"⚡ Transcribing {file_path}...")
        try:
            with sr.AudioFile(file_path) as source:
                audio_data = self.recognizer.record(source)
                # 使用 Google Web Speech API (Default)
                transcript = self.recognizer.recognize_google(audio_data)
                
            print(f"📝 Transcript: {transcript}")
            
            # Embed the transcript
            embeddings = list(self.embed_model.embed([transcript]))
            vector = embeddings[0]
            
            # Upsert
            payload = {
                "source": os.path.basename(file_path),
                "type": "audio",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "transcript": transcript,
                "urgency": "medium", 
                "content": transcript # Unified field for search
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
            print(f"✅ Indexed audio.")
            
        except Exception as e:
            print(f"❌ Error processing audio: {e}")

def main():
    if not os.path.exists(AUDIO_INBOX):
        os.makedirs(AUDIO_INBOX)
        print(f"📂 Created {AUDIO_INBOX}")
        
    client = init_qdrant()
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
