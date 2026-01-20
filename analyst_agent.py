import time
import json
import datetime
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configuration
VISUAL_COLLECTION = "visual_memory"
CIVILIAN_COLLECTION = "civilian_memory"
QDRANT_URL = "http://localhost:6333"
ALERT_FILE = "alerts.json"
CHECK_INTERVAL_SEC = 5
DANGER_RADIUS_METERS = 5000

def init_qdrant():
    return QdrantClient(url=QDRANT_URL)

def save_alert(alert_data):
    alerts = []
    if os.path.exists(ALERT_FILE):
        try:
            with open(ALERT_FILE, "r") as f:
                alerts = json.load(f)
        except json.JSONDecodeError:
            pass
    
    # Prepend new alert
    alerts.insert(0, alert_data)
    # Keep last 50 alerts
    alerts = alerts[:50]
    
    with open(ALERT_FILE, "w") as f:
        json.dump(alerts, f, indent=2)
    print(f"ALERT SAVED: {alert_data['msg']}")

def main():
    print("Initializing Analyst Agent (Geospatial & Conflict Resolution)...")
    client = init_qdrant()

    while True:
        try:
            # 1. Fetch latest visual memory with a hazard_type
            # We sort by timestamp descending to get the "latest"
            # Note: In a real system we'd filter by timestamp > last_checked
            search_result = client.scroll(
                collection_name=VISUAL_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="detected_disaster",
                            match=models.MatchExcept(except_integers=[], except_keywords=["None", "person", "car"]) 
                        )
                    ]
                ),
                limit=10, # Fetch top 10 latest events to capture recent updates
                with_payload=True,
                with_vectors=False
            )
            
            points, _ = search_result
            
            if not points:
                print("Analyst Agent: No active hazards found yet. Scanning...")
            else:
                for hazard_point in points:
                    hazard_payload = hazard_point.payload
                    hazard_loc = hazard_payload.get("location")
                    hazard_type = hazard_payload.get("detected_disaster")
                    
                    # ALERT FORWARDING LOGIC
                    # Check for 'triggered_alerts' in payload (set by Image Agent)
                    triggered_alerts = hazard_payload.get("triggered_alerts", [])
                    if triggered_alerts:
                        for alert_msg in triggered_alerts:
                            # Create a unique ID for the alert to prevent spamming
                            alert_id = f"{hazard_point.id}_{hash(alert_msg)}"
                            
                            # Determine if we should save this (simple check: is it already in logs?)
                            # For now, we save it. In a real system, we'd check against a memory cache of sent alerts.
                            
                            # Filter for "BREAKING NEWS" specifically if requested, or all high priority
                            if "BREAKING NEWS" in alert_msg or "DETECTED" in alert_msg:
                                alert_entry = {
                                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                    "msg": alert_msg,
                                    "source": hazard_payload.get("source", "Unknown"),
                                    "location": hazard_loc
                                }
                                # Simple deduplication by reading current alerts (inefficient but works for demo)
                                current_alerts = []
                                if os.path.exists(ALERT_FILE):
                                    with open(ALERT_FILE, 'r') as f:
                                        try:
                                            current_alerts = json.load(f)
                                        except: pass
                                
                                # Check if identical message exists in last 5 alerts
                                is_duplicate = False
                                for existing in current_alerts[:5]:
                                    if existing['msg'] == alert_msg and existing['source'] == alert_entry['source']:
                                        is_duplicate = True
                                        break
                                
                                if not is_duplicate:
                                    save_alert(alert_entry)
                
                if hazard_loc:
                    print(f"Analyst Agent: Detected {hazard_type} at {hazard_loc}. Correlating with Civilian data...")
                    
                    # 2. Geospatial Search for Civilians
                    # (In a real scenario, we'd use Qdrant's Geo radius filter)
                    # For simulation, we check if civilian collection exists
                    if client.collection_exists(CIVILIAN_COLLECTION):
                         civilian_hits = client.scroll(collection_name=CIVILIAN_COLLECTION, limit=100, with_payload=True)[0]
                         if civilian_hits:
                             print(f"   Analyst Agent found {len(civilian_hits)} civilians in the sector. Calculating risks...")
                    
                    # (Alert logic sim - replaced by forwarding logic above)
                    # ...
                    
        except Exception as e:
            # print(f"Analyst Loop Error: {e}") 
            # Suppress noise for now
            pass
        
        time.sleep(CHECK_INTERVAL_SEC)

if __name__ == "__main__":
    main()
