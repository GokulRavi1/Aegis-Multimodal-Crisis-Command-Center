import os
import time
import datetime
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
TEXT_INBOX = "text_inbox"
COLLECTION_NAME = "tactical_memory"

class TextHandler(FileSystemEventHandler):
    def __init__(self, client, embed_model):
        self.client = client
        self.embed_model = embed_model
        self.geolocator = Nominatim(user_agent="aegis_crisis_center", timeout=5)
        
        # Initialize LLM
        # Initialize LLM & Memory Manager
        self.llm_manager = get_llm_manager()
        from memory_manager import get_memory_manager
        self.memory_manager = get_memory_manager()
        self.memory_manager.ensure_collections()

    def extract_location_with_llm(self, text):
        """Use LLM to extract location from text."""
        if not self.llm_manager or not text:
            return None
        
        try:
            prompt = f"""Extract a SPECIFIC geographic location (City, District, or State) from this text.
            CRITICAL RULES:
            1. REJECT BROAD COUNTRY NAMES like "India", "USA", "America", "UK". Return "NONE" if only country is found.
            2. REJECT common nouns not used as places.
            3. Return ONLY precise locations (e.g., "Mumbai", "Kerala", "Visakhapatnam").
            4. If text describes a general region without specific city, return "NONE".
            
            Text: "{text[:500]}"
            
            Location:"""
            
            location = self.llm_manager.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50, 
                temperature=0.1
            )
            
            noise_words = ["none", "india", "usa", "uk", "america", "region", "country"]
            
            if location:
                clean_loc = location.strip().lower()
                if clean_loc not in noise_words and len(clean_loc) > 3:
                    print(f"   🤖 LLM Extracted Location: {location}")
                    return location.strip()
            return None
        except Exception as e:
            print(f"   ⚠️ LLM Error: {e}")
            return None

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.txt', '.json', '.md')):
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
        
        if not content or not content.strip():
            print("❌ Error reading text (locked or empty)")
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
                    confidence = 0.85
                    break
            
            # SMART LOCATION EXTRACTION
            detected_location = None
            location_name = self.extract_location_with_llm(content)
            if location_name:
                location = self.geolocator.geocode(location_name)
                if location:
                    detected_location = {"lat": location.latitude, "lon": location.longitude, "name": location.address}
                    print(f"   📍 Geotagged: {location.address}")
            
            # Alerts
            alerts = []
            if detected_disaster != "None":
                alerts.append(f"TEXT ALERT: {detected_disaster.upper()}")
            if detected_location:
                alerts.append(f"📍 Location: {detected_location.get('name', 'Unknown')}")

            payload = {
                "agent": "text_agent",  # Attribution for hackathon
                "source": os.path.basename(file_path),
                "type": "text",
                "detected_disaster": detected_disaster,
                "confidence": {"text": confidence},
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "content": content,
                "location": detected_location if detected_location else {"lat": 28.7041, "lon": 77.1025},
                "reliability_score": 0.8,
                "triggered_alerts": alerts
            }
            
            self.memory_manager.upsert_point(
                collection=COLLECTION_NAME,
                vector=vector.tolist(),
                payload=payload,
                point_id=int(time.time() * 1000)
            )
            print(f"✅ Indexed text document. Disaster: {detected_disaster}")
            
        except Exception as e:
            print(f"❌ Error processing text: {e}")

def main():
    if not os.path.exists(TEXT_INBOX):
        os.makedirs(TEXT_INBOX)
        print(f"📂 Created {TEXT_INBOX}")

    client = get_qdrant_client()
    # ensure_collection(client, COLLECTION_NAME, 384)
    
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
