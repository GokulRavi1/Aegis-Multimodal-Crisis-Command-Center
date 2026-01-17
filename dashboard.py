import streamlit as st
import pandas as pd
import json
import os
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

# Configuration
QDRANT_URL = "http://localhost:6333"
VISUAL_COLLECTION = "visual_memory"
CIVILIAN_COLLECTION = "civilian_memory"
AUDIO_COLLECTION = "audio_memory"
TACTICAL_COLLECTION = "tactical_memory"
ALERT_FILE = "alerts.json"

st.set_page_config(page_title="Aegis Command Center", layout="wide")

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL)

@st.cache_resource
def get_text_embedding_model():
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

@st.cache_resource
def get_clip_text_embedding_model():
    return TextEmbedding(model_name="Qdrant/clip-ViT-B-32-text")

client = get_qdrant_client()
text_embedding_model = get_text_embedding_model()
clip_text_model = get_clip_text_embedding_model()

st.title("🛡️ Aegis: Multimodal Crisis Command Center")

# Sidebar for Alerts
st.sidebar.header("⚠️ Live Alert Feed")
if os.path.exists(ALERT_FILE):
    with open(ALERT_FILE, "r") as f:
        try:
            alerts = json.load(f)
            for alert in alerts[:10]:
                st.sidebar.error(f"{alert['timestamp']}\n\n{alert['msg']}")
        except:
            st.sidebar.write("No valid alerts log found.")
else:
    st.sidebar.write("No alerts yet.")

# Radio Chatter Section in Sidebar
st.sidebar.header("📻 Radio Chatter")
try:
    if client.collection_exists(AUDIO_COLLECTION):
        audio_res = client.scroll(collection_name=AUDIO_COLLECTION, limit=5, with_payload=True)
        audio_points = audio_res[0]
        
        if audio_points:
            for p in audio_points:
                payload = p.payload
                urgency = payload.get("urgency", "unknown")
                urgency_color = "🔴" if urgency == "high" else "🟡" if urgency == "medium" else "🟢"
                st.sidebar.markdown(f"**{urgency_color} {payload.get('unit_id', 'Unknown')}**: {payload.get('transcript', '')[:50]}...")
        else:
            st.sidebar.info("No radio chatter yet.")
    else:
        st.sidebar.info("Audio memory not initialized.")
except Exception as e:
    st.sidebar.warning(f"Could not load radio chatter: {e}")

# Main Layout with Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📍 Warzone Map", "🔍 Semantic Search", "📻 Audio Logs", "🐦 Social & Sensors"])

with tab1:
    st.header("📍 Warzone Map (Real-time)")

    if os.path.exists("latest_frame.jpg"):
        pass 

    
    try:
        visual_res = client.scroll(collection_name=VISUAL_COLLECTION, limit=100, with_payload=True)
        visual_points = visual_res[0]
        
        civ_res = client.scroll(collection_name=CIVILIAN_COLLECTION, limit=100, with_payload=True)
        civ_points = civ_res[0]
        
        map_data = []
        
        for p in visual_points:
            loc = p.payload.get("location")
            if loc:
                map_data.append({"lat": loc["lat"], "lon": loc["lon"], "type": "hazard"})
                
        for p in civ_points:
            loc = p.payload.get("location")
            if loc:
                map_data.append({"lat": loc["lat"], "lon": loc["lon"], "type": "civilian"})

        if map_data:
            df = pd.DataFrame(map_data)
            st.map(df)
            st.caption(f"Showing {len(visual_points)} hazard points and {len(civ_points)} civilian locations")
        else:
            st.info("No data points to display on map.")
            
    except Exception as e:
        st.error(f"Error fetching map data: {e}")

with tab2:
    st.header("🔍 Semantic Search")
    query = st.text_input("Search the Warzone", placeholder="e.g., 'Show me fire' or 'flooded streets'")
    
    if query:
        st.subheader("Results")
    if query:
        st.subheader("Results")
        
        # --- 1. Visual Search (CLIP) ---
        visual_q = list(clip_text_model.embed([query]))[0]
        vis_res = client.query_points(collection_name=VISUAL_COLLECTION, query=visual_q.tolist(), limit=3)
        vis_hits = vis_res.points
        
        # --- 2. Text/Audio Search (BGE) ---
        text_q = list(text_embedding_model.embed([query]))[0]
        
        audio_hits = []
        if client.collection_exists(AUDIO_COLLECTION):
            aud_res = client.query_points(collection_name=AUDIO_COLLECTION, query=text_q.tolist(), limit=3)
            audio_hits = aud_res.points
            
        text_hits = []
        if client.collection_exists(TACTICAL_COLLECTION):
            tac_res = client.query_points(collection_name=TACTICAL_COLLECTION, query=text_q.tolist(), limit=3)
            text_hits = tac_res.points

        # --- Display Results ---
        
        if vis_hits:
            st.markdown("### 🖼️ Visual Matches")
            for hit in vis_hits:
                payload = hit.payload
                score = hit.score
                source = payload.get("source", "Unknown")
                msg_type = payload.get("type", "visual")
                
                with st.expander(f"{source} (Score: {score:.2f})", expanded=True):
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if msg_type == "image":
                             path = os.path.join("image_inbox", source)
                             if os.path.exists(path):
                                 st.image(path, caption=source, width="stretch")
                        else: # visual/video
                             path = os.path.join("video_inbox", source)
                             if os.path.exists(path):
                                 st.video(path)
                    
                    with col2:
                        st.caption(f"Timestamp: {payload.get('timestamp')}")
                        if "detections" in payload:
                            st.json(payload["detections"])

        if audio_hits:
            st.markdown("### 📻 Audio Matches")
            for hit in audio_hits:
                payload = hit.payload
                score = hit.score
                source = payload.get("source", "Unknown")
                transcript = payload.get("transcript", "No transcript")
                
                with st.expander(f"{source} (Score: {score:.2f})", expanded=True):
                    path = os.path.join("audio_inbox", source)
                    if os.path.exists(path):
                        st.audio(path)
                    st.markdown(f"**Transcript:** {transcript}")

        if text_hits:
            st.markdown("### 📄 Intel Matches")
            for hit in text_hits:
                payload = hit.payload
                score = hit.score
                source = payload.get("source", "Unknown")
                content = payload.get("content", "")
                
                with st.expander(f"{source} (Score: {score:.2f})", expanded=True):
                    st.markdown(content)
                    
        if not (vis_hits or audio_hits or text_hits):
            st.info("No matches found across any channel.")

with tab3:
    st.header("📻 Audio Logs (Radio Transcripts)")
    
    try:
        if client.collection_exists(AUDIO_COLLECTION):
            audio_res = client.scroll(collection_name=AUDIO_COLLECTION, limit=20, with_payload=True)
            audio_points = audio_res[0]
            
            if audio_points:
                audio_data = []
                for p in audio_points:
                    payload = p.payload
                    audio_data.append({
                        "Timestamp": payload.get("timestamp", "")[:19],
                        "Unit ID": payload.get("unit_id", "Unknown"),
                        "Urgency": payload.get("urgency", "unknown").upper(),
                        "Transcript": payload.get("transcript", "")
                    })
                
                df_audio = pd.DataFrame(audio_data)
                
                def highlight_urgency(val):
                    if val == "HIGH":
                        return 'background-color: #ffcccc'
                    elif val == "MEDIUM":
                        return 'background-color: #fff3cd'
                    else:
                        return 'background-color: #d4edda'
                
                styled_df = df_audio.style.map(highlight_urgency, subset=['Urgency'])
                st.dataframe(styled_df, width='stretch')
            else:
                st.info("No audio logs. Run `python listener_agent.py` to generate data.")
        else:
            st.warning("Audio memory collection not found.")
    except Exception as e:
        st.error(f"Error loading audio logs: {e}")

with tab4:
    st.header("🐦 Social & Sensors (Tactical Intel)")
    
    try:
        if client.collection_exists(TACTICAL_COLLECTION):
            tac_res = client.scroll(collection_name=TACTICAL_COLLECTION, limit=20, with_payload=True)
            tac_points = tac_res[0]
            
            if tac_points:
                tac_data = []
                for p in tac_points:
                    payload = p.payload
                    reliability = payload.get("reliability_score", 0.5)
                    tac_data.append({
                        "Timestamp": payload.get("timestamp", "")[:19],
                        "Source": payload.get("source", "Unknown"),
                        "Location": payload.get("location_ref", "Unknown"),
                        "Reliability": f"{'🔴 HIGH' if reliability >= 0.9 else '🟡 MED'}",
                        "Content": payload.get("content", "")
                    })
                
                df_tac = pd.DataFrame(tac_data)
                
                def highlight_reliability(row):
                    if "HIGH" in row["Reliability"]:
                        return ['background-color: #ffcccc'] * len(row)
                    else:
                        return ['background-color: #fff3cd'] * len(row)
                
                styled_tac = df_tac.style.apply(highlight_reliability, axis=1)
                st.dataframe(styled_tac, width='stretch')
            else:
                st.info("No tactical data. Run `python text_agent.py` to generate data.")
        else:
            st.warning("Tactical memory collection not found. Run `python text_agent.py` first.")
    except Exception as e:
        st.error(f"Error loading tactical data: {e}")

# Refresh button
if st.button("🔄 Refresh Monitor"):
    st.rerun()
