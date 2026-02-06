import time
import json
import datetime
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Import Strategic Retrieval Modules
from llm_manager import get_llm_manager
from retrieval_tools import RetrievalToolkit
from config import get_qdrant_client
from fastembed import TextEmbedding

# Configuration
VISUAL_COLLECTION = "visual_memory"
ALERT_FILE = "alerts.json"
CHECK_INTERVAL_SEC = 10 # Increase interval slightly

def save_alert(alert_data):
    alerts = []
    if os.path.exists(ALERT_FILE):
        try:
            with open(ALERT_FILE, "r") as f:
                alerts = json.load(f)
        except json.JSONDecodeError:
            pass
    
    # Deduplicate (Simple check of last 10 alerts)
    for existing in alerts[:10]:
        if existing['msg'] == alert_data['msg'] and existing['source'] == alert_data['source']:
            print(f"   Duplicate alert skipped: {alert_data['msg']}")
            return

    # Prepend new alert
    alerts.insert(0, alert_data)
    alerts = alerts[:50]
    
    # ATOMIC WRITE: Write to temp file then rename
    temp_file = f"{ALERT_FILE}.tmp"
    try:
        with open(temp_file, "w") as f:
            json.dump(alerts, f, indent=2)
        os.replace(temp_file, ALERT_FILE)
        print(f"🚨 ALERT SAVED: {alert_data['msg']}")
    except Exception as e:
        print(f"❌ Error saving alert: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)

def verify_hazard(llm, toolkit, hazard_type, location, source_description):
    """
    Use LLM + Audio Retrieval to verify a visual hazard.
    """
    print(f"   🔍 Verifying {hazard_type} at {location}...")
    
    # Cross-reference with Audio Memory
    audio_results = toolkit.search_audio_memory(f"{hazard_type} in {location}")
    audio_context = "No audio reports found."
    if audio_results.get("result"):
        items = audio_results["result"]
        audio_context = "\n".join([f"- {item['transcript']}" for item in items[:2]])
    
    # LLM Verification
    system_prompt = """You are the Aegis Safety Officer.
    Verify if a situation warrants a public alert.
    
    INPUT DATA:
    - Visual Report: What the camera saw.
    - Audio Intel: Related radio calls.
    
    DECISION RULES:
    1. If Visual + Audio confirm danger -> CONFIRM
    2. If Visual is high confidence (Fire/Flood) even without audio -> CONFIRM
    3. If Visual seems trivial (e.g., 'car') -> DISMISS
    
    OUTPUT:
    Return ONLY 'ALERT_CONFIRMED' or 'ALERT_DISMISSED'.
    """
    
    user_prompt = f"""
    Visual Report: {hazard_type} detected at {location}. Source: {source_description}
    Audio Intel:
    {audio_context}
    
    Verdict:
    """
    
    decision = llm.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=10
    )
    
    return "ALERT_CONFIRMED" in (decision or "")

def main():
    print("Initializing Analyst Agent (Autonomous Verification Mode)...")
    
    # Init Resources
    try:
        client = get_qdrant_client()
        llm = get_llm_manager()
        # We need embeddings for toolkit. 
        # For background agent, we can re-init or pass None if tools handle it?
        # RetrievalToolkit requires models.
        text_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        toolkit = RetrievalToolkit(client, text_model, None) # No CLIP needed for audio search
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return

    processed_ids = set()

    while True:
        try:
            # 1. Scan Latest Visual Events
            if client.collection_exists(VISUAL_COLLECTION):
                # Fetch latest 5 items
                search_result = client.scroll(
                    collection_name=VISUAL_COLLECTION,
                    limit=5,
                    with_payload=True,
                    with_vectors=False
                )
                points, _ = search_result
                
                if not points:
                    print("   (No visual data found)")
                
                for point in points:
                    payload = point.payload
                    pid = point.id
                    
                    if pid in processed_ids:
                        continue
                        
                    hazard_type = payload.get("detected_disaster")
                    if not hazard_type or hazard_type == "None":
                        continue
                        
                    location = payload.get("location", "Unknown Location")
                    description = payload.get("ocr_text", "")
                    
                    # 2. Verify Hazard
                    is_confirmed = verify_hazard(llm, toolkit, hazard_type, location, description)
                    
                    if is_confirmed:
                        alert_msg = f"CONFIRMED THREAT: {hazard_type} at {location}."
                        save_alert({
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "msg": alert_msg,
                            "source": "Analyst Agent (Verified)",
                            "location": location
                        })
                    else:
                        print(f"   ℹ️ Threat Dismissed: {hazard_type}")
                    
                    processed_ids.add(pid)
                    # Keep set size manageable
                    if len(processed_ids) > 1000:
                        processed_ids.clear()
            
        except Exception as e:
            print(f"Analyst Loop Error: {e}") 
        
        time.sleep(CHECK_INTERVAL_SEC)

if __name__ == "__main__":
    main()
