import os
import time
import datetime
import speech_recognition as sr
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from config import get_qdrant_client, ensure_collection

# Configuration
AUDIO_INBOX = "audio_inbox"
COLLECTION_NAME = "audio_memory"

class AudioHandler(FileSystemEventHandler):
    def __init__(self, client, embed_model):
        self.client = client
        self.embed_model = embed_model
        self.recognizer = sr.Recognizer()

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.wav', '.flac', '.aiff','.mp3')):
            print(f"🎤 New audio detected: {event.src_path}")
            self.process_audio(event.src_path)

    def process_audio(self, file_path):
        print(f"⚡ Transcribing {file_path}...")
        
        transcript = None
        for i in range(5):
            try:
                # Ensure file is readable
                with open(file_path, 'rb'): 
                    pass
                    
                with sr.AudioFile(file_path) as source:
                    audio_data = self.recognizer.record(source)
                    transcript = self.recognizer.recognize_google(audio_data)
                
                if transcript:
                    break
            except (PermissionError, IOError):
                print(f"   ⏳ Waiting for file release... ({i+1}/5)")
                time.sleep(1)
            except ValueError:
                # Likely format error (e.g. mp3 without ffmpeg)
                print(f"⚠️ Format warning: Could not process {os.path.basename(file_path)}. Ensure it is a valid WAV/FLAC.")
                return

        if not transcript:
            print("❌ Error processing audio (locked or invalid format)")
            return

        print(f"📝 Transcript: {transcript}")
            
        try:
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
