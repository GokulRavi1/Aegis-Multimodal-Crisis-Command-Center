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
    print("Initializing Safety Agent...")
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
                            key="hazard_type",
                            match=models.MatchValue(value="flood") # Check specifically for flood or any hazard
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            
            points, _ = search_result
            
            if not points:
                print("No active hazards found yet. Scanning...")
            else:
                hazard_point = points[0]
                hazard_payload = hazard_point.payload
                hazard_loc = hazard_payload.get("location")
                hazard_type = hazard_payload.get("hazard_type")
                
                if hazard_loc:
                    print(f"Detected {hazard_type} at {hazard_loc}")
                    
                    # 2. Geospatial Search for Civilians
                    civilian_hits = client.scroll(
                        collection_name=CIVILIAN_COLLECTION,
                        scroll_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="location",
                                    geo_bounding_box=None, # Not used
                                    geo_radius=models.GeoRadius(
                                        center=models.GeoPoint(
                                            lat=hazard_loc["lat"],
                                            lon=hazard_loc["lon"]
                                        ),
                                        radius=DANGER_RADIUS_METERS
                                    )
                                )
                            ]
                        ),
                        limit=100,
                        with_payload=True
                    )
                    
                    civilians_found, _ = civilian_hits
                    
                    if civilians_found:
                        print(f"Found {len(civilians_found)} civilians in danger zone!")
                        for civ in civilians_found:
                            civ_data = civ.payload
                            
                            # Create Alert
                            alert = {
                                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                "msg": f"URGENT: {civ_data['name']} is within {DANGER_RADIUS_METERS/1000}km of {hazard_type}!",
                                "civilian_id": civ.id,
                                "hazard_location": hazard_loc,
                                "civilian_location": civ_data['location']
                            }
                            save_alert(alert)
                    else:
                        print("No civilians in immediate danger zone.")

        except Exception as e:
            print(f"Error in Safety Loop: {e}")
        
        time.sleep(CHECK_INTERVAL_SEC)

if __name__ == "__main__":
    main()
