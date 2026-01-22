import os
import time
import datetime
import speech_recognition as sr
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from groq import Groq
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
from config import get_qdrant_client, ensure_collection, DISASTER_CLASSES
from llm_manager import get_llm_manager

load_dotenv()

# Configuration
AUDIO_INBOX = "audio_inbox"
COLLECTION_NAME = "audio_memory"

class AudioHandler(FileSystemEventHandler):
    def __init__(self, client, embed_model):
        self.client = client
        self.embed_model = embed_model
        self.recognizer = sr.Recognizer()
        self.geolocator = Nominatim(user_agent="aegis_crisis_center")
        
        # Initialize LLM Manager
        self.llm_manager = get_llm_manager()

    def extract_location_with_llm(self, text):
        """Use LLM to extract location from text."""
        if not self.llm_manager or not text:
            return None
        
        try:
            prompt = f"""Extract ONLY the geographic location (city, state, country, or region name) from this text. 
            Return ONLY the location name, nothing else. If no location found, return 'NONE'.
            
            Text: "{text}"
            
            Location:"""
            
            location = self.llm_manager.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            if location and location.upper() != "NONE":
                print(f"   🤖 LLM Extracted Location: {location}")
                return location
            return None
        except Exception as e:
            print(f"   ⚠️ LLM Error: {e}")
            return None

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.wav', '.flac', '.mp3')):
            print(f"🎙️ New audio detected: {event.src_path}")
            # Wait for file to be fully written
            time.sleep(2)  # Simple wait
            self.process_audio(event.src_path)

    def wait_for_file_ready(self, file_path, timeout=10):
        """Wait for file to be fully written."""
        start = time.time()
        last_size = -1
        while time.time() - start < timeout:
            try:
                size = os.path.getsize(file_path)
                if size == last_size and size > 0:
                    return True
                last_size = size
            except:
                pass
            time.sleep(0.5)
        return False

    def process_audio(self, file_path):
        print(f"⚡ Processing Audio: {file_path}...")
        
        # Wait for file
        if not self.wait_for_file_ready(file_path):
            print("❌ File not ready")
            return
        
        # Convert MP3 to WAV if needed (SpeechRecognition works best with WAV)
        temp_wav = None
        audio_path = file_path
        
        if file_path.lower().endswith('.mp3'):
            try:
                from pydub import AudioSegment
                print("   🔄 Converting MP3 to WAV...")
                sound = AudioSegment.from_mp3(file_path)
                temp_wav = file_path.replace('.mp3', '_temp.wav')
                sound.export(temp_wav, format="wav")
                audio_path = temp_wav
            except Exception as e:
                print(f"   ⚠️ MP3 conversion failed: {e}")
                return
        
        # Transcription
        try:
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
            transcript = self.recognizer.recognize_google(audio)
        except Exception as e:
            print(f"❌ Transcription Error: {e}")
            if temp_wav and os.path.exists(temp_wav):
                os.remove(temp_wav)
            return
        
        # Clean up temp file
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)
            
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
                    confidence = 0.8
                    break
            
            # SMART LOCATION EXTRACTION
            detected_location = None
            location_name = self.extract_location_with_llm(transcript)
            if location_name:
                location = self.geolocator.geocode(location_name)
                if location:
                    detected_location = {"lat": location.latitude, "lon": location.longitude, "name": location.address}
                    print(f"   📍 Geotagged: {location.address}")
            
            # Alerts
            alerts = []
            if detected_disaster != "None":
                alerts.append(f"AUDIO ALERT: {detected_disaster.upper()}")
            if detected_location:
                alerts.append(f"📍 Location: {detected_location.get('name', 'Unknown')}")

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
                "triggered_alerts": alerts,
                "location": detected_location if detected_location else {"lat": 28.7041, "lon": 77.1025}
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
    
    print(f"👂 Listener Agent active. Monitoring '{AUDIO_INBOX}' for audio files...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
