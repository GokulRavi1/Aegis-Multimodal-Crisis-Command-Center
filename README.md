# 🛡️ Aegis – Multimodal Crisis Command Center

A fully offline, edge-first disaster response system  Aegis uses Qdrant as its vector database to process and correlate multimodal data (video, audio, text) for real-time situational awareness during crisis events.

![System Architecture](system_architecture.png)

---

## 🧠 AI Models (Hackathon "Speed" Stack)

We explicitly selected the following models to prioritize real-time performance on edge hardware for the hackathon:

| Modality | Selected Model (Speed) | Alternative (Accuracy) | Rationale |
|----------|------------------------|------------------------|-----------|
| **Text** | `BAAI/bge-small-en-v1.5` | `nomic-embed-text-v1.5` | BGE is faster (optimized for <512 tokens). Nomic supports 8192 tokens for long reports. |
| **Image** | `Qdrant/clip-ViT-B-32-vision` | `Qdrant/Unicom-ViT-B-32` | CLIP is the standard for speed. Unicom offers better understanding of complex scenes (rubble), but is heavier. |
| **Video** | `Same as Image` (Frame Indexing) | `Same as Image` | We index video frames using the Image model. |
| **Audio** | `Whisper` (Speech-to-Text) + `BGE` | `CLAP` | Whisper is robust and safer to implement quickly than CLAP (which allows searching raw sounds like "siren"). |

## 🏗️ System Architecture

| Agent | Collection | Model | Vector Dims |
|-------|------------|-------|-------------|
| `watcher_agent.py` | `visual_memory` | CLIP-ViT-B-32 | 512 |
| `image_agent.py` | `visual_memory` | CLIP-ViT-B-32 | 512 |
| `listener_agent.py` | `audio_memory` | BGE-small-en | 384 |
| `text_agent.py` | `tactical_memory` | BGE-small-en | 384 |
| `generate_civilians.py` | `civilian_memory` | (geo only) | 1 |
| `safety_agent.py` | (reads all) | N/A | N/A |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker Desktop

### Step 1: Start Qdrant
```bash
docker-compose up -d
```
Wait for the container to be ready. Verify at: http://localhost:6333/dashboard

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Generate Test Video
```bash
python create_test_video.py
```

### Step 4: Seed Civilian Data
```bash
python generate_civilians.py
```

### Step 5: Run Video Ingestion
```bash
python watcher_agent.py
```

### Step 6: Run Audio Ingestion
```bash
python listener_agent.py
```

### Step 7: Run Text Ingestion
```bash
python text_agent.py
```

### Step 8: Start Safety Agent (Background)
```bash
python safety_agent.py
```
> Keep this running in a separate terminal. It monitors for civilians in danger.

### Step 9: Launch Dashboard
```bash
streamlit run dashboard.py
```
Open http://localhost:8501 in your browser.

---

## 📁 Project Structure

```
Aegis/
├── docker-compose.yml      # Qdrant container config
├── requirements.txt        # Python dependencies
├── create_test_video.py    # Generates dummy flood footage
├── generate_civilians.py   # Seeds civilian data
├── watcher_agent.py        # Video → visual_memory
├── listener_agent.py       # Audio → audio_memory  
├── text_agent.py           # Text → tactical_memory
├── safety_agent.py         # Geofencing & alerts
├── dashboard.py            # Streamlit UI
├── alerts.json             # Generated alert logs
└── qdrant_storage/         # Qdrant data (auto-created)
```

---

## 🔧 Technical Stack

| Component | Technology |
|-----------|------------|
| Vector DB | Qdrant (Docker) |
| Embeddings | FastEmbed (CLIP, BGE) |
| Frontend | Streamlit |
| Video Processing | OpenCV |
| Geospatial | Qdrant GeoRadius |

---


