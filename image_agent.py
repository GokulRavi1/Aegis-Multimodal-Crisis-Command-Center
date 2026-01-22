import cv2
import time
import datetime
import os
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import ImageEmbedding, TextEmbedding
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ultralytics import YOLOWorld
import easyocr
from geopy.geocoders import Nominatim
import os
from groq import Groq
from dotenv import load_dotenv
from config import get_qdrant_client, ensure_collection, DISASTER_CLASSES
from llm_manager import get_llm_manager

load_dotenv()

# Configuration
IMAGE_INBOX = "image_inbox"
COLLECTION_NAME = "visual_memory"
MODEL_NAME = "yolov8s-worldv2.pt"

class ImageHandler(FileSystemEventHandler):
    # ... (init unchanged) ...
    def __init__(self, client, embed_model, yolo_model, text_model):
        self.client = client
        self.embed_model = embed_model
        self.yolo_model = yolo_model
        
        # Pre-compute text embeddings for Hybrid Detection
        print("🧠 Pre-computing disaster class embeddings for Hybrid Detection...")
        self.labels = DISASTER_CLASSES
        self.label_embeddings = list(text_model.embed(self.labels))
        self.bge_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5") # For tactical indexing
        print("✅ Hybrid Memory Ready.")
        
        # Initialize OCR and Geocoder
        print("🧠 Loading OCR Model (English)...")
        self.reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if CUDA available
        self.geolocator = Nominatim(user_agent="aegis_crisis_center")
        print("✅ OCR & Geolocation Ready.")
        
        # Initialize LLM for smart location extraction
        self.llm_manager = get_llm_manager()

    def extract_location_with_llm(self, ocr_text):
        """Use LLM to extract location/place names from OCR text."""
        if not self.llm_manager or not ocr_text:
            return None
        
        try:
            prompt = f"""Extract a SPECIFIC geographic location (City, District, or State) from this text.
            CRITICAL RULES:
            1. REJECT BROAD COUNTRY NAMES like "India", "USA", "America", "UK". Return "NONE" if only country is found.
            2. REJECT noise words and single letters.
            3. Return ONLY precise locations (e.g., "Mumbai", "Kerala", "Visakhapatnam").
            4. If unsure or no clear specific location found, return "NONE".
            
            Text: "{ocr_text[:300]}"
            
            Location:"""
            
            location = self.llm_manager.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            noise_words = ["none", "ila", "indian", "india", "the", "express", "usually", "we", "for", "news", "usa", "uk", "america"]
            
            if location:
                clean_loc = location.strip().lower()
                if clean_loc not in noise_words and len(clean_loc) > 3:
                     print(f"   🤖 LLM Extracted Location: {location}")
                     return location.strip()
            return None
        except Exception as e:
            print(f"   ⚠️ LLM Error: {e}")
            return None

    def generate_enriched_summary(self, disaster, location_name, ocr_text):
        """Generate a detailed tactical summary for better search indexing."""
        if not self.llm_manager:
            return f"Alert: {disaster} detected at {location_name}. {ocr_text}"
            
        try:
            prompt = f"""Generate a 20-30 word tactical alert summary for a crisis database. 
            Include the specific place name, the type of disaster, and a brief warning.
            
            Disaster: {disaster}
            Location: {location_name}
            OCR Evidence: {ocr_text}
            
            Format: [Location Name] alert: [Detailed description of issue] - [Warning/Action]."""
            
            return self.llm_manager.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7
            )
        except Exception as e:
            print(f"   ⚠️ Summary LLM Error: {e}")
            return f"Crisis Alert at {location_name}: {disaster} detected. OCR: {ocr_text[:50]}"

    def on_created(self, event):
        # ... logic unchanged ...
        if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            print(f"🖼️ New image detected: {event.src_path}")
            self.process_image(event.src_path)

    def wait_for_file_ready(self, file_path, timeout=5, stability_duration=1.0):
        """Waits until the file size assumes a stable value."""
        start_time = time.time()
        last_size = -1
        last_check = time.time()
        
        while time.time() - start_time < timeout:
            if not os.path.exists(file_path):
                time.sleep(0.1)
                continue
                
            current_size = os.path.getsize(file_path)
            if current_size == last_size and current_size > 0:
                if time.time() - last_check > stability_duration:
                    return True
            else:
                last_size = current_size
                last_check = time.time()
            
            time.sleep(0.5)
        return False

    def process_image(self, image_path):
        # ... (read frame logic unchanged) ...
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
        max_conf = 0.0
        primary_disaster = "None"
        
        if result.boxes:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = self.yolo_model.names[cls_id]
                
                detections[name] = detections.get(name, 0) + 1
                
                # Update primary disaster (Prioritize disasters over person/car)
                if name not in ["person", "car"]:
                    if conf > max_conf:
                        max_conf = conf
                        primary_disaster = name
                elif primary_disaster == "None" and conf > max_conf:
                    max_conf = conf
                    primary_disaster = name

        if detections:
            print(f"   Found (YOLO): {detections} (Primary: {primary_disaster})")
            
        # Embedding
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        embeddings = list(self.embed_model.embed([pil_image]))
        vector = embeddings[0]

        # --- HYBRID DETECTION (CLIP FALLBACK) ---
        # If YOLO found nothing significant, ask CLIP what it sees.
        detection_source = "YOLO"
        if primary_disaster in ["None", "person", "car"]:
            # Compute Cosine Similarity between Image and all Labels
            scores = []
            for i, label_emb in enumerate(self.label_embeddings):
                # Cosine Sim: (A . B) / (|A| * |B|) - FastEmbed returns normalized vectors, so just Dot Product
                score = sum(v * l for v, l in zip(vector, label_emb))
                scores.append((self.labels[i], score))
            
            # Sort by score
            scores.sort(key=lambda x: x[1], reverse=True)
            top_label, top_score = scores[0]
            
            # Threshold Check (e.g., 0.22 is a reasonable starting point for CLIP/BGE matches)
            if top_score > 0.22 and top_label not in ["person", "car", "truck"]:
                primary_disaster = top_label
                max_conf = float(top_score)
                detection_source = "CLIP (Hybrid)"
                print(f"   🧠 Hybrid Correction: Reclassified as '{primary_disaster}' (Score: {top_score:.2f})")
        
        # --- OCR & GEOLOCATION ---
        ocr_text = ""
        detected_location = None
        
        try:
            # Simple OCR
            results = self.reader.readtext(frame, detail=0)
            ocr_text = " ".join(results)
            
            if ocr_text:
                print(f"   📝 OCR Extracted: {ocr_text}")
                
            # SMART LOCATION EXTRACTION
            # Step 1: Try LLM to extract location intelligently
            location_name = self.extract_location_with_llm(ocr_text)
            
            # Step 2: Fallback to basic text cleaning if LLM fails
            if not location_name and len(ocr_text) > 3:
                # Basic cleaning for common patterns
                clean_text = ocr_text.replace("BREAKING NEWS", "").replace("MAJOR", "").replace("FLOODING", "").replace("IN", "").strip()
                if "," in clean_text:
                    parts = clean_text.split(",")
                    location_name = parts[0] + ", " + parts[1]
                else:
                    location_name = clean_text
                
            # Step 3: Geocode the extracted location
            if location_name:
                location = self.geolocator.geocode(location_name)
                if location:
                    detected_location = {"lat": location.latitude, "lon": location.longitude, "name": location.address}
                    print(f"   📍 Geotagged: {location.address}")
                else:
                    print(f"   ⚠️ Could not geocode: {location_name}")
        except Exception as e:
            print(f"   ⚠️ OCR/Geo Error: {e}")
            
        # -------------------------
        
        # ----------------------------------------

        # Trigger Alerts logic
        alerts = []
        if primary_disaster not in ["None", "person", "car"]:
            alerts.append(f"DETECTED {primary_disaster.upper()}")
            if max_conf > 0.6:
                alerts.append("High Confidence Alert")
        
        if detected_location:
            alerts.append(f"BREAKING NEWS: {primary_disaster.upper()} IN {detected_location['name'].upper()}")

        # Prepare Payload
        payload = {
            "source": os.path.basename(image_path),
            "type": "image",
            "detected_disaster": primary_disaster,
            "confidence": {"visual": float(max_conf), "source": detection_source},
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "detections": detections,
            "triggered_alerts": alerts,
            "ocr_text": ocr_text,
            "location": detected_location if detected_location else {"lat": 28.7041, "lon": 77.1025, "name": "Global Monitoring Area"}
        }

        # --- LLM ENRICHED SUMMARY ---
        enriched_summary = self.generate_enriched_summary(
            primary_disaster, 
            detected_location['name'] if detected_location else "Unknown Location",
            ocr_text
        )
        print(f"   🤖 Enriched Summary: {enriched_summary}")
        payload["content"] = enriched_summary
        
        # 1. Upsert to Visual Memory (CLIP - 512d)
        point_id = int(time.time() * 1000)
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payload
                )
            ]
        )
        
        # 2. Upsert to Tactical Memory (BGE - 384d) for high-score text search
        # This solves the score issue for queries like "Kochi bridge"
        try:
            bge_vector = list(self.bge_model.embed([enriched_summary]))[0].tolist()
            self.client.upsert(
                collection_name="tactical_memory",
                points=[
                    models.PointStruct(
                        id=point_id + 1, # Unique ID
                        vector=bge_vector,
                        payload={
                            **payload,
                            "data_type": "text",
                            "is_visual_metadata": True # Flag to distinguish
                        }
                    )
                ]
            )
        except Exception as e:
            print(f"   ⚠️ Tactical indexing failed: {e}")

        print(f"✅ Indexed Image & Summary: {primary_disaster}")

def main():
    if not os.path.exists(IMAGE_INBOX):
        os.makedirs(IMAGE_INBOX)
        print(f"📂 Created {IMAGE_INBOX}")

    client = get_qdrant_client()
    ensure_collection(client, COLLECTION_NAME, 512)
    
    print("⏳ Loading Models for Image Agent...")
    embed_model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
    # Using CLIP text model to match the vision model space
    text_model = TextEmbedding(model_name="Qdrant/clip-ViT-B-32-text")
    
    yolo_model = YOLOWorld(MODEL_NAME)
    yolo_model.set_classes(DISASTER_CLASSES)
    
    observer = Observer()
    handler = ImageHandler(client, embed_model, yolo_model, text_model)
    observer.schedule(handler, IMAGE_INBOX, recursive=False)
    observer.start()
    
    print(f"👀 Image Agent active. Monitoring '{IMAGE_INBOX}'...")

    # Process existing files on startup
    print("🔄 Checking for existing images...")
    for filename in os.listdir(IMAGE_INBOX):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            file_path = os.path.join(IMAGE_INBOX, filename)
            handler.process_image(file_path)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
