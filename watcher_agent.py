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
from groq import Groq
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
from config import get_qdrant_client, ensure_collection, DISASTER_CLASSES
from llm_manager import get_llm_manager

load_dotenv()

# Configuration
VIDEO_INBOX = "video_inbox"
COLLECTION_NAME = "visual_memory"
MODEL_NAME = "yolov8s-worldv2.pt"

class VideoHandler(FileSystemEventHandler):
    # ... (init unchanged) ...
    def __init__(self, client, embed_model, yolo_model, text_model):
        self.client = client
        self.embed_model = embed_model
        self.yolo_model = yolo_model
        
        # Pre-compute text embeddings for Hybrid Detection
        print("🧠 Pre-computing disaster class embeddings for Hybrid Detection...")
        self.labels = DISASTER_CLASSES
        self.label_embeddings = list(text_model.embed(self.labels))
        print("✅ Hybrid Memory Ready.")
        
        # Initialize OCR and Geocoder
        print("🧠 Loading OCR Model (English)...")
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.geolocator = Nominatim(user_agent="aegis_crisis_center")
        print("✅ OCR & Geolocation Ready.")
        
        # Initialize LLM
        # Initialize LLM
        self.llm_manager = get_llm_manager()
        
        # Initialize Memory Manager
        from memory_manager import get_memory_manager
        self.memory_manager = get_memory_manager()
        self.memory_manager.ensure_collections() # Ensure all exist

    def extract_location_with_llm(self, text):
        """Use LLM to extract location from text."""
        if not self.llm_manager or not text or len(text) < 10:
            return None
        
        try:
            prompt = f"""Extract a SPECIFIC geographic location (City, District, or State) from this OCR text.
            CRITICAL RULES:
            1. REJECT BROAD COUNTRY NAMES like "India", "USA", "America", "UK". Return "NONE" if only country is found.
            2. REJECT noise words: "Ila", "The", "Express", "EXPRES", "Usually", "News".
            3. Return ONLY precise locations (e.g., "Mumbai", "Kerala", "Visakhapatnam").
            4. If unsure or no clear specific location found, return "NONE".
            
            Text: "{text[:300]}"
            
            Location (or NONE):"""
            
            location = self.llm_manager.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,
                temperature=0.0
            )
            
            # Extra validation - reject common noise AND countries
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

    def on_created(self, event):
        # ... logic unchanged ...
        if not event.is_directory and event.src_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"🎬 New video detected: {event.src_path}")
            self.process_video(event.src_path)

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

    def process_video(self, video_path):
        # Wait for file to be ready
        if not self.wait_for_file_ready(video_path):
            print(f"❌ File not ready or empty: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            return

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"⚡ Processing Video: {video_path} ({total_frames} frames)...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break


            frame_count += 1
            frame_count += 1
            if frame_count % 30 != 0:
                continue

            if frame_count % 300 == 0:  # Log every ~10 seconds of video
                print(f"   Processed {frame_count}/{total_frames} frames...")

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
                    
                    # Update primary disaster
                    if name not in ["person", "car"]:
                        if conf > max_conf:
                            max_conf = conf
                            primary_disaster = name
                    elif primary_disaster == "None" and conf > max_conf:
                        max_conf = conf
                        primary_disaster = name
            
            # Save Latest Frame
            annotated_frame = result.plot()
            cv2.imwrite("latest_frame.jpg", annotated_frame)
            
            # Embedding (Generate BEFORE Hybrid Logic)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            embeddings = list(self.embed_model.embed([pil_image]))
            vector = embeddings[0]

            # --- HYBRID DETECTION (CLIP FALLBACK) ---
            detection_source = "YOLO"
            if primary_disaster in ["None", "person", "car"]:
                # Compute Cosine Similarity between Image and all Labels
                scores = []
                for i, label_emb in enumerate(self.label_embeddings):
                    score = sum(v * l for v, l in zip(vector, label_emb))
                    scores.append((self.labels[i], score))
                
                # Sort by score
                scores.sort(key=lambda x: x[1], reverse=True)
                top_label, top_score = scores[0]
                
                # Threshold Check - Raised to 0.30 to reduce false positives
                if top_score > 0.30 and top_label not in ["person", "car", "truck"]:
                    primary_disaster = top_label
                    max_conf = float(top_score)
                    detection_source = "CLIP (Hybrid)"
                    print(f"   🧠 Hybrid Correction: Reclassified as '{primary_disaster}' (Score: {top_score:.2f})")
            
            # ----------------------------------------

            # Alerts
            alerts = []
            if primary_disaster not in ["None", "person", "car"]:
                alerts.append(f"DETECTED {primary_disaster.upper()}")

            # OCR & GEOLOCATION (Every 5th processed frame = every 150th raw frame)
            detected_location = None
            if frame_count % 150 == 0:
                try:
                    results_ocr = self.reader.readtext(frame, detail=0)
                    ocr_text = " ".join(results_ocr)
                    if ocr_text:
                        print(f"   📝 OCR: {ocr_text[:50]}...")
                        location_name = self.extract_location_with_llm(ocr_text)
                        if location_name:
                            location = self.geolocator.geocode(location_name)
                            if location:
                                detected_location = {"lat": location.latitude, "lon": location.longitude, "name": location.address}
                                print(f"   📍 Geotagged (OCR): {location.address}")
                except Exception as e:
                    pass  # Silent fail for OCR on video frames
            
            # SKIP non-disaster frames (unless they have location)
            if primary_disaster in ["None", "person", "car", "truck"] and not detected_location:
                # Demo log every 300 frames
                if frame_count % 300 == 0:
                    print(f"   ⛔ Frame {frame_count}: No disaster detected - Skipping")
                continue  # Don't index this frame

            payload = {
                "source": os.path.basename(video_path),
                "type": "visual",
                "detected_disaster": primary_disaster,
                "confidence": {"visual": float(max_conf), "source": detection_source},
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "detections": detections,
                "triggered_alerts": alerts,
                "location": detected_location if detected_location else {"lat": 28.7041, "lon": 77.1025}
            }
            
            # Upsert with retry logic for network timeouts
            for attempt in range(3):
                success = self.memory_manager.upsert_point(
                    collection=COLLECTION_NAME,
                    vector=vector.tolist(),
                    payload=payload,
                    point_id=int(time.time() * 1000)
                )
                if success:
                    break
                else:
                    if attempt < 2:
                        time.sleep(1 * (attempt + 1))  # Backoff: 1s, 2s
                    else:
                        print(f"   ⚠️ Upsert failed after retries.")
            
        cap.release()
        print(f"✅ Finished Video.")

def main():
    print("Initializing Watcher Agent (Video Only)...")
    
    if not os.path.exists(VIDEO_INBOX):
        os.makedirs(VIDEO_INBOX)
        print(f"📂 Created {VIDEO_INBOX}")

    client = get_qdrant_client()
    # ensure_collection(client, COLLECTION_NAME, 512) <--- managed by MemoryManager now
    
    print("⏳ Loading Models for Watcher Agent...")
    embed_model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
    # Using CLIP text model
    text_model = TextEmbedding(model_name="Qdrant/clip-ViT-B-32-text")
    
    yolo_model = YOLOWorld(MODEL_NAME)
    yolo_model.set_classes(DISASTER_CLASSES)
    
    observer = Observer()
    handler = VideoHandler(client, embed_model, yolo_model, text_model)
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
