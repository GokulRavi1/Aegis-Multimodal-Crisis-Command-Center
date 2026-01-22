import streamlit as st
import pandas as pd
import json
import os
import time
from groq import Groq
from fastembed import TextEmbedding
from config import get_qdrant_client
from llm_manager import get_llm_manager

# Auto-refresh setup (DISABLED - was disrupting user workflow)
# try:
#     from streamlit_autorefresh import st_autorefresh
#     AUTO_REFRESH_AVAILABLE = True
# except ImportError:
#     AUTO_REFRESH_AVAILABLE = False

# Constants
VISUAL_COLLECTION = "visual_memory"
AUDIO_COLLECTION = "audio_memory"
TACTICAL_COLLECTION = "tactical_memory"
CIVILIAN_COLLECTION = "civilian_memory"
ALERT_FILE = "alerts.json"

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Aegis Command Center", page_icon="🛡️")

# Auto-refresh disabled (was refreshing entire app)

# --- CSS Styling (for Chat) ---
st.markdown("""
<style>
    .report-card {
        border: 1px solid #444;
        border-radius: 10px;
        padding: 15px;
        background-color: #1e1e1e;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .report-title {
        color: #ff4b4b;
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 5px;
    }
    .source-tag {
        font-size: 0.8em;
        color: #aaa;
        background-color: #333;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialization ---
@st.cache_resource
def init_resources():
    client = get_qdrant_client()
    # BGE for general text/metadata search
    text_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    # CLIP for visual search
    clip_text_model = TextEmbedding(model_name="Qdrant/clip-ViT-B-32-text")
    return client, text_model, clip_text_model

client, text_model, clip_text_model = init_resources()

# --- Sidebar: Alerts & Radio ---
st.title("🛡️ Aegis: Crisis Command Center")

st.sidebar.header("⚠️ Live Alert Feed")

# Manual refresh button for sidebar
if st.sidebar.button("🔄 Refresh Alerts"):
    st.rerun()

# Dynamic Alert Feed - Fetch from all Qdrant collections
def generate_alert_text(payload, data_type):
    """Use LLM to generate a concise alert message."""
    # FAST PATH: Use pre-computed enriched content if available
    content = payload.get("content", "")
    if content and len(content) < 150: # Use pre-computed summary if short enough
        return content
        
    llm_manager = get_llm_manager()
    if not llm_manager:
        # Fallback to simple format
        disaster = payload.get("detected_disaster", "Unknown")
        location = payload.get("location", {})
        loc_name = location.get("name", "") if isinstance(location, dict) else ""
        return f"{disaster.upper()}: {loc_name[:30]}" if loc_name else disaster.upper()
    
    try:
        # Build context based on type
        if data_type == "visual":
            ctx = f"Disaster: {payload.get('detected_disaster')}, Location: {payload.get('location', {}).get('name', 'Unknown')}, OCR: {payload.get('ocr_text', '')[:100]}"
        elif data_type == "audio":
            ctx = f"Audio Transcript: {payload.get('transcript', '')[:150]}"
        else:
            ctx = f"Text Report: {payload.get('content', '')[:150]}"
        
        return llm_manager.chat_completion(
            messages=[
                {"role": "system", "content": "Generate a 10-15 word emergency alert headline. Be direct, tactical. No quotes."},
                {"role": "user", "content": ctx}
            ],
            temperature=0.3,
            max_tokens=30
        )
    except:
        disaster = payload.get("detected_disaster", "Unknown")
        return f"ALERT: {disaster.upper()}"

try:
    all_alerts = []
    
    # 1. High-Priority Analyst Alerts (from alerts.json)
    # Only show recent alerts (within last 24 hours)
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    
    if os.path.exists(ALERT_FILE):
        try:
            with open(ALERT_FILE, "r") as f:
                analyst_alerts = json.load(f)
                for a in analyst_alerts[:10]:
                    # Check if alert is recent (within 24 hours)
                    alert_ts = a.get("timestamp", "")
                    if alert_ts:
                        try:
                            alert_time = datetime.datetime.fromisoformat(alert_ts.replace("Z", "+00:00"))
                            age = (now - alert_time).total_seconds()
                            if age > 86400:  # Skip if older than 24 hours
                                continue
                        except:
                            pass  # If can't parse, include it
                    
                    all_alerts.append({
                        "timestamp": alert_ts,
                        "type": "🛡️ ANALYST",
                        "msg": a.get("msg", ""),
                        "payload": a,
                        "data_type": "analyst"
                    })
        except: pass

    # 2. Raw Ingestion Feed (Fallbacks from Qdrant)
    # Fetch more and sort by timestamp to get 'newest'
    if client.collection_exists(VISUAL_COLLECTION):
        vis_res = client.scroll(collection_name=VISUAL_COLLECTION, limit=50, with_payload=True)[0]
        for p in vis_res:
            if p.payload.get("detected_disaster") not in ["None", "person", "car"]:
                all_alerts.append({
                    "timestamp": p.payload.get("timestamp", ""),
                    "type": "🖼️ RAW VISUAL",
                    "payload": p.payload,
                    "data_type": "visual"
                })
    
    if client.collection_exists(AUDIO_COLLECTION):
        aud_res = client.scroll(collection_name=AUDIO_COLLECTION, limit=30, with_payload=True)[0]
        for p in aud_res:
            if p.payload.get("detected_disaster") not in ["None"]:
                all_alerts.append({
                    "timestamp": p.payload.get("timestamp", ""),
                    "type": "🎙️ RAW AUDIO",
                    "payload": p.payload,
                    "data_type": "audio"
                })
    
    # Sort by timestamp (newest first)
    all_alerts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    if all_alerts:
        for alert in all_alerts[:15]:  # Show top 15
            ts = alert["timestamp"][:19].replace("T", " ") if alert["timestamp"] else "Just now"
            
            if alert["data_type"] == "analyst":
                alert_text = alert["msg"]
            else:
                alert_text = generate_alert_text(alert["payload"], alert["data_type"])
            
            # Location info - only show if we have a real name
            loc = alert["payload"].get("location", {})
            loc_name = ""
            if isinstance(loc, dict):
                loc_name = loc.get("name", "")
                # Skip generic fallback names
                if loc_name in ["Global Monitoring Area", "Unknown Location", ""]:
                    loc_name = ""
            
            location_str = f"\n\n📍 **Location:** {loc_name}" if loc_name else ""
            st.sidebar.error(f"**{alert['type']}** | {ts}\n\n{alert_text}{location_str}")
    else:
        st.sidebar.info("No active alerts. System monitoring...")
        
except Exception as e:
    st.sidebar.warning(f"Alert feed error: {e}")

# --- RAG Logic for Chat ---
def get_rag_response(query):
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return "⚠️ Error: Groq API Key not found. Please check your .env configuration.", []

    try:
        # Import retrieval logger
        try:
            from retrieval_logger import get_retrieval_logger, format_evidence_for_llm, create_grounded_prompt
            logger = get_retrieval_logger()
        except:
            logger = None
        
        # 1. Embed Query
        text_q = list(text_model.embed([query]))[0]
        visual_q = list(clip_text_model.embed([query]))[0]
        
        results = []
        results = []
        score_threshold = 0.55  # Lowered to capture specific queries (relies on Strict Keyword Filter)

        # Search Visual
        if client.collection_exists(VISUAL_COLLECTION):
            vis_res = client.query_points(collection_name=VISUAL_COLLECTION, query=visual_q.tolist(), limit=5, with_payload=True)
            for hit in vis_res.points:
                if hit.score > score_threshold:
                    results.append({"type": "visual", "score": hit.score, "payload": hit.payload, "id": hit.id})

        # Search Audio
        if client.collection_exists(AUDIO_COLLECTION):
            aud_res = client.query_points(collection_name=AUDIO_COLLECTION, query=text_q.tolist(), limit=5, with_payload=True)
            for hit in aud_res.points:
                if hit.score > score_threshold:
                    results.append({"type": "audio", "score": hit.score, "payload": hit.payload, "id": hit.id})

        # Search Tactical
        if client.collection_exists(TACTICAL_COLLECTION):
            tac_res = client.query_points(collection_name=TACTICAL_COLLECTION, query=text_q.tolist(), limit=5, with_payload=True)
            for hit in tac_res.points:
                if hit.score > score_threshold:
                    results.append({"type": "text", "score": hit.score, "payload": hit.payload, "id": hit.id})

        # Sort and limit to top-k
        results.sort(key=lambda x: x['score'], reverse=True)
        top_results = results[:5]

        # Format evidence for logging
        retrieved_points = [
            {
                "id": r["id"],
                "score": round(r["score"], 3),
                "source": r["payload"].get("source", "Unknown"),
                "disaster_type": r["payload"].get("detected_disaster", "Unknown"),
                "content_preview": (r["payload"].get("ocr_text", "") or r["payload"].get("transcript", "") or r["payload"].get("content", ""))[:100]
            }
            for r in top_results
        ]

        # Generate context with citations
        context_str = ""
        for i, r in enumerate(top_results, 1):
            p = r['payload']
            src = p.get('source', 'Unknown')
            if r['type'] == 'visual':
                context_str += f"[Evidence {i}: {src}] Type: VIDEO/IMAGE | Disaster: {p.get('detected_disaster')} | Text: {p.get('ocr_text', 'N/A')[:100]}\n"
            elif r['type'] == 'audio':
                context_str += f"[Evidence {i}: {src}] Type: AUDIO | Transcript: {p.get('transcript', 'N/A')[:100]}\n"
            else:
                context_str += f"[Evidence {i}: {src}] Type: TEXT | Content: {p.get('content', 'N/A')[:100]}\n"

        if not context_str:
            return "No recent data found matching your query.", []

        # Grounded system prompt with Safety AI Persona
        sys_prompt = """You are the Aegis Safety AI Assistant. 
        
        YOUR MISSION:
        1. Analyze the provided evidence for safety threats.
        2. IF DANGER EXISTS: You MUST start with "⚠️ WARNING:" and advise caution.
        3. IF UNCERTAIN: Advise "Proceed with caution, data is limited."
        4. CITE SOURCES: Always reference the evidence files (e.g., [video.mp4]).
        5. Be concise, direct, and protective of the user."""

        llm_manager = get_llm_manager()
        if not llm_manager:
            return "⚠️ LLM Service Unavailable.", []

        full_prompt = f"QUERY: {query}\n\n=== EVIDENCE ===\n{context_str}\n\n=== SAFETY ANALYSIS (cite sources) ==="

        response_text = llm_manager.chat_completion(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3, max_tokens=150
        )
        
        # Log retrieval provenance
        if logger:
            logger.log_retrieval(
                query=query,
                retrieved_points=retrieved_points,
                llm_prompt=full_prompt,
                llm_response=response_text
            )
        
        return response_text, top_results

    except Exception as e:
        return f"⚠️ System Error: {str(e)}", []


# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 Crisis Map", 
    "🔍 Semantic Search", 
    "📻 Audio Logs", 
    "🐦 Social & Sensors",
    "🛡️ Safety AI Chat"
])

# --- Tab 1: Map ---
with tab1:
    st.header("📍 Crisis Operational Map (Real-time)")
    try:
        visual_points = []
        if client.collection_exists(VISUAL_COLLECTION):
            visual_res = client.scroll(collection_name=VISUAL_COLLECTION, limit=100, with_payload=True)
            visual_points = visual_res[0]

        civ_points = []
        if client.collection_exists(CIVILIAN_COLLECTION):
            civ_res = client.scroll(collection_name=CIVILIAN_COLLECTION, limit=100, with_payload=True)
            civ_points = civ_res[0]
        
        map_data = []
        
        # Countries to skip (country-only locations without city)
        COUNTRY_ONLY_NAMES = ['india', 'usa', 'china', 'japan', 'germany', 'france', 'uk', 'united kingdom', 
                               'united states', 'australia', 'russia', 'brazil', 'canada', 'mexico', 'indonesia',
                               'pakistan', 'bangladesh', 'philippines', 'vietnam', 'thailand', 'south korea',
                               'spain', 'italy', 'netherlands', 'turkey', 'saudi arabia', 'iran', 'egypt']
        
        def is_country_only(loc_name):
            """Check if location is just a country name without city."""
            if not loc_name:
                return False
            loc_lower = loc_name.lower().strip()
            # Check if the entire loc_name is ONLY a country name
            for country in COUNTRY_ONLY_NAMES:
                if loc_lower == country or loc_lower == f"{country}.":
                    return True
            return False
        
        for p in visual_points:
            loc = p.payload.get("location")
            if loc: 
                # SKIP static/default Delhi location
                if isinstance(loc, dict) and abs(loc.get("lat", 0) - 28.7041) < 0.01:
                    continue
                
                location_name = loc.get("name", "") if isinstance(loc, dict) else ""
                
                # SKIP country-only locations
                if is_country_only(location_name.split(",")[0] if location_name else ""):
                    continue
                
                # Add 'disaster' field strictly for map tooltip
                disaster_name = p.payload.get("detected_disaster", "Unknown")
                ocr_text = p.payload.get("ocr_text", "")
                location_name = loc.get("name", "") if isinstance(loc, dict) else ""
                label = f"{disaster_name} @ {location_name[:30]}" if location_name else disaster_name
                
                map_data.append({
                    "lat": loc["lat"], 
                    "lon": loc["lon"], 
                    "type": "hazard", 
                    "disaster": label
                })
        
        # Add Text/Tactical Memory to Map
        if client.collection_exists(TACTICAL_COLLECTION):
            tac_res = client.scroll(collection_name=TACTICAL_COLLECTION, limit=50, with_payload=True)[0]
            for p in tac_res:
                loc = p.payload.get("location")
                if loc and isinstance(loc, dict):
                    if abs(loc.get("lat", 0) - 28.7041) < 0.01:
                        continue  # Skip default location
                    location_name = loc.get("name", "")[:30] if isinstance(loc, dict) else ""
                    # Skip country-only locations
                    if is_country_only(location_name.split(",")[0] if location_name else ""):
                        continue
                    disaster_name = p.payload.get("detected_disaster", "Report")
                    map_data.append({
                        "lat": loc["lat"], 
                        "lon": loc["lon"], 
                        "type": "text", 
                        "disaster": f"📄 {disaster_name} @ {location_name}"
                    })
        
        # Add Audio Memory to Map
        if client.collection_exists(AUDIO_COLLECTION):
            aud_res = client.scroll(collection_name=AUDIO_COLLECTION, limit=50, with_payload=True)[0]
            for p in aud_res:
                loc = p.payload.get("location")
                if loc and isinstance(loc, dict):
                    if abs(loc.get("lat", 0) - 28.7041) < 0.01:
                        continue  # Skip default location
                    location_name = loc.get("name", "")[:30] if isinstance(loc, dict) else ""
                    # Skip country-only locations
                    if is_country_only(location_name.split(",")[0] if location_name else ""):
                        continue
                    disaster_name = p.payload.get("detected_disaster", "Audio")
                    map_data.append({
                        "lat": loc["lat"], 
                        "lon": loc["lon"], 
                        "type": "audio", 
                        "disaster": f"🎙️ {disaster_name} @ {location_name}"
                    })
        
        for p in civ_points:
            loc = p.payload.get("location")
            if loc: map_data.append({"lat": loc["lat"], "lon": loc["lon"], "type": "civilian", "disaster": "Civilian"})

        import pydeck as pdk

        if map_data:
            # Convert to DF
            df = pd.DataFrame(map_data)
            
            # Define colors
            # Hazard = Red, Text = Orange, Audio = Blue, Civilian = Green
            def get_color(t):
                if t == 'hazard': return [220, 50, 50, 220]
                elif t == 'text': return [255, 165, 0, 220]  # Orange
                elif t == 'audio': return [50, 150, 255, 220]  # Blue
                else: return [50, 200, 50, 200]  # Green
            df['color'] = df['type'].apply(get_color)

            st.pydeck_chart(pdk.Deck(
                map_style='https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
                initial_view_state=pdk.ViewState(
                    latitude=df['lat'].mean(),
                    longitude=df['lon'].mean(),
                    zoom=10,  # Zoomed in to see the specific location
                    pitch=0,
                ),
                layers=[
                    pdk.Layer(
                        'ScatterplotLayer',
                        data=df,
                        get_position='[lon, lat]',
                        get_color='color',
                        get_radius=50000,  # 50km radius in meters
                        radius_min_pixels=8,  # MINIMUM 8 pixels regardless of zoom
                        radius_max_pixels=30,
                        pickable=True,
                        auto_highlight=True
                    ),
                ],
                tooltip={"html": "<b>{type}</b><br/>{disaster}", "style": {"color": "white"}}
            ))
            
            st.caption(f"Showing {len(map_data)} geotagged hazards and {len(civ_points)} civilians")
        else:
            st.info("No geotagged data points for map. Upload images with location names or wait for events.")
    except Exception as e:
        st.error(f"Error fetching map data: {e}")

# --- Tab 2: Semantic Search ---
with tab2:
    st.header("🔍 Semantic Search")
    query = st.text_input("Search the Warzone", placeholder="e.g., 'Bengaluru flood warning'")
    
    # CLIP (Visual) scores are often lower (0.25-0.45) while BGE (Text/Audio) are higher (0.60-0.85)
    search_threshold = 0.70
    
    def generate_warning_summary(payload, data_type):
        """Generate 10-20 word warning from LLM."""
        llm_manager = get_llm_manager()
        if not llm_manager:
            return f"⚠️ {payload.get('detected_disaster', 'Alert').upper()}"
        
        try:
            if data_type == "visual":
                ctx = f"Disaster: {payload.get('detected_disaster')}, Location: {payload.get('location', {}).get('name', 'Unknown')}, OCR Text: {payload.get('ocr_text', '')[:150]}"
            elif data_type == "audio":
                ctx = f"Audio Transcript: {payload.get('transcript', '')[:200]}"
            else:
                ctx = f"Report: {payload.get('content', '')[:200]}"
            
            generated = llm_manager.chat_completion(
                messages=[
                    {"role": "system", "content": "Generate a 10-20 word emergency warning based on this crisis data. Be direct (example: 'Major flooding in Saidapet area - evacuation recommended')."},
                    {"role": "user", "content": ctx}
                ],
                temperature=0.3, max_tokens=40
            )
            return generated if generated else f"⚠️ {payload.get('detected_disaster', 'Alert').upper()}"
        except:
            return f"⚠️ {payload.get('detected_disaster', 'Alert').upper()}"
    
    if query:
        st.subheader("Results")
        all_results = []
        
        # Extract keywords from query for filtering
        # Improved: Look for "in [location]" pattern or known city names
        import re
        
        query_lower = query.lower()
        required_keywords = []
        
        # 1. Check for "in [place]" pattern
        in_pattern = re.search(r'\bin\s+([a-zA-Z]+)', query_lower)
        if in_pattern:
            required_keywords.append(in_pattern.group(1))
            
        # 2. Check for known major locations if no "in" pattern found
        # (Add common operational areas here)
        known_locations = ["chennai", "bengaluru", "mumbai", "delhi", "kerala", "visakhapatnam", "andhra", "tamil nadu"]
        if not required_keywords:
            for loc in known_locations:
                if loc in query_lower:
                    required_keywords.append(loc)
        
        # 3. Fallback: Check for capitalized words (if user typed "Chennai Flood")
        if not required_keywords:
            original_words = [w for w in query.split() if len(w) > 2]
            required_keywords = [w.lower() for w in original_words if w[0].isupper()]

        def matches_query_keywords(payload):
            """Check if payload matches required location keywords."""
            if not required_keywords:
                return True # No specific location requested
                
            # Get all searchable text from payload
            searchable = ""
            loc = payload.get("location", {})
            if isinstance(loc, dict):
                searchable += loc.get("name", "").lower() + " "
            searchable += payload.get("ocr_text", "").lower() + " "
            searchable += payload.get("transcript", "").lower() + " "
            searchable += payload.get("content", "").lower() + " "
            
            # Strict And-Logic: Must contain at least one of the required location keywords
            # (If we extracted multiple, typically we want to match the specific one)
            return any(kw in searchable for kw in required_keywords)
        
        # Search Visual Memory (CLIP embeddings)
        visual_q = list(clip_text_model.embed([query]))[0]
        if client.collection_exists(VISUAL_COLLECTION):
            vis_res = client.query_points(collection_name=VISUAL_COLLECTION, query=visual_q.tolist(), limit=10, with_payload=True)
            for hit in vis_res.points:
                if hit.score > search_threshold and matches_query_keywords(hit.payload):
                    all_results.append({"type": "🖼️ VISUAL", "hit": hit, "data_type": "visual", "score": hit.score})
        
        # Search Audio Memory (Text embeddings)
        text_q = list(text_model.embed([query]))[0]
        if client.collection_exists(AUDIO_COLLECTION):
            aud_res = client.query_points(collection_name=AUDIO_COLLECTION, query=text_q.tolist(), limit=10, with_payload=True)
            for hit in aud_res.points:
                if hit.score > search_threshold and matches_query_keywords(hit.payload):
                    all_results.append({"type": "🎙️ AUDIO", "hit": hit, "data_type": "audio", "score": hit.score})
        
        # Search Tactical/Text Memory (Text embeddings)
        if client.collection_exists(TACTICAL_COLLECTION):
            tac_res = client.query_points(collection_name=TACTICAL_COLLECTION, query=text_q.tolist(), limit=10, with_payload=True)
            for hit in tac_res.points:
                if hit.score > search_threshold and matches_query_keywords(hit.payload):
                    all_results.append({"type": "📄 TEXT", "hit": hit, "data_type": "text", "score": hit.score})
        
        # Sort by score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        if not all_results:
            st.warning("No results found. Try different keywords.")
        else:
            st.success(f"Found {len(all_results)} matches across all sources")
            
            for result in all_results[:10]:  # Top 10
                hit = result["hit"]
                p = hit.payload
                src = p.get("source", "Unknown")
                loc = p.get("location", {})
                loc_name = loc.get("name", "")[:40] if isinstance(loc, dict) else ""
                
                # Generate warning
                warning = generate_warning_summary(p, result["data_type"])
                
                with st.expander(f"{result['type']} | {src} | Score: {result['score']:.2f}", expanded=False):
                    # Warning header
                    st.error(f"**{warning}**")
                    
                    # Location
                    if loc_name:
                        st.caption(f"📍 {loc_name}")
                    
                    # Media display based on type
                    if result["data_type"] == "visual":
                        if p.get("type") == "image":
                            img_path = os.path.join("image_inbox", src)
                            if os.path.exists(img_path):
                                st.image(img_path, width=400)
                        else:
                            vid_path = os.path.join("video_inbox", src)
                            if os.path.exists(vid_path):
                                st.video(vid_path)
                        
                        # OCR Text
                        if p.get("ocr_text"):
                            st.info(f"📝 OCR Text: {p.get('ocr_text')}")
                    
                    elif result["data_type"] == "audio":
                        aud_path = os.path.join("audio_inbox", src)
                        if os.path.exists(aud_path):
                            st.audio(aud_path)
                        st.info(f"📝 Transcript: {p.get('transcript', 'No transcript')}")
                    
                    elif result["data_type"] == "text":
                        content = p.get("content", "")
                        st.info(f"📝 Content: {content[:300]}...")
                    
                    # Metadata
                    st.caption(f"🕐 {p.get('timestamp', '')[:19]} | Disaster: {p.get('detected_disaster', 'Unknown')}")

# --- Tab 3: Audio Logs ---
with tab3:
    st.header("📻 Audio Logs")
    if client.collection_exists(AUDIO_COLLECTION):
        res = client.scroll(collection_name=AUDIO_COLLECTION, limit=20, with_payload=True)[0]
        if res:
            data = [{"Time": p.payload.get("timestamp")[:19], "Unit": p.payload.get("unit_id"), "Transcript": p.payload.get("transcript")} for p in res]
            st.dataframe(pd.DataFrame(data), width=1000)
        else:
            st.info("No audio logs.")

# --- Tab 4: Social ---
with tab4:
    st.header("🐦 Social & Sensors")
    if client.collection_exists(TACTICAL_COLLECTION):
        res = client.scroll(collection_name=TACTICAL_COLLECTION, limit=20, with_payload=True)[0]
        if res:
            data = [{"Time": p.payload.get("timestamp")[:19], "Source": p.payload.get("source"), "Content": p.payload.get("content")} for p in res]
            st.dataframe(pd.DataFrame(data), width=1000)
        else:
            st.info("No tactical data.")

# --- Tab 5: Analyst Agent ---
with tab5:
    st.header("🤖 Analyst Agent (Reasoning & Conflict Resolution)")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "I am the Aegis Analyst Agent. I am monitoring cross-modal streams to provide actionable tactical insights."}
        ]

    # Display History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "data" in msg:
                for item in msg["data"]:
                    p = item['payload']
                    m_type = item['type']
                    src = p.get('source', 'Unknown')
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="report-card">
                            <div class="report-title">{p.get('detected_disaster', 'General').upper()}</div>
                            <div><span class="source-tag">{m_type.upper()}</span></div>
                        </div>""", unsafe_allow_html=True)
                        
                        if m_type == "visual":
                            if p.get("type") == "image":
                                if os.path.exists(os.path.join("image_inbox", src)):
                                    st.image(os.path.join("image_inbox", src), width=400)
                            else:
                                if os.path.exists(os.path.join("video_inbox", src)):
                                    st.video(os.path.join("video_inbox", src))
                        elif m_type == "audio":
                             if os.path.exists(os.path.join("audio_inbox", src)):
                                 st.audio(os.path.join("audio_inbox", src))
                             st.markdown(f"*{p.get('transcript')}*")

    # Chat Input
    if prompt := st.chat_input("Ask for tactical analysis..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyst is correlating data..."):
                # Update System Prompt for Analyst Persona
                # Note: get_rag_response uses the internal prompt, we should update it there or pass it in.
                # For now, we rely on the internal prompt being updated or we update the function below.
                response_text, media_items = get_rag_response(prompt)
                st.write(response_text)
                
                # Render content
                for item in media_items:
                    p = item['payload']
                    m_type = item['type']
                    src = p.get('source', 'Unknown')
                    st.markdown("---")
                    st.caption(f"EVIDENCE: {src}")
                    if m_type == "visual":
                        if p.get("type") == "image":
                            img_path = os.path.join("image_inbox", src)
                            if os.path.exists(img_path):
                                st.image(img_path)
                            else:
                                st.warning(f"⚠️ Image not found: {src}")
                        else:
                            vid_path = os.path.join("video_inbox", src)
                            if os.path.exists(vid_path):
                                st.video(vid_path)
                            else:
                                st.warning(f"⚠️ Video not found: {src}")
                        
                        if p.get('ocr_text'):
                            st.info(f"📝 Read Text: {p.get('ocr_text')}")
                    elif m_type == "audio":
                        aud_path = os.path.join("audio_inbox", src)
                        if os.path.exists(aud_path):
                            st.audio(aud_path)
                        else:
                            st.warning(f"⚠️ Audio not found: {src}")
                        st.write(f" Transcript: {p.get('transcript')}")
        
        st.session_state.messages.append({"role": "assistant", "content": response_text, "data": media_items})
