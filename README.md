# 🛡️ Aegis – Multimodal Crisis Command Center

**Aegis** is an AI-powered Situation Awareness system designed for crisis management. It fuses data from **Video, Image, Audio, and Text** sources into a unified vector database to provide real-time intelligence and semantic search capabilities.

![System Architecture](system_architecture.png)

## 🚀 Features

### 1. Multimodal Ingestion Agents
*   **🎥 Watcher Agent (`watcher_agent.py`)**: Monitors video feeds (drones/CCTV), detects hazards (YOLO-World), and indexes frames (CLIP) into Qdrant.
*   **🖼️ Image Agent (`image_agent.py`)**: Processes static imagery from field operatives, detecting objects and embedding semantic content.
*   **📻 Listener Agent (`listener_agent.py`)**: Transcribes radio/audio logs (SpeechRecognition) and embeds transcripts (BGE) for search.
*   **📄 Text Agent (`text_agent.py`)**: Ingests social media, SITREPs, and logs, embedding them for tactical retrieval.

### 2. Intelligent Core
*   **Vector Database**: **Qdrant Cloud** stores embeddings for high-speed similarity search.
*   **AI Models**:
    *   **Contextual Understanding**: `BAAI/bge-small-en-v1.5` (Text/Audio).
    *   **Visual Understanding**: `Qdrant/clip-ViT-B-32-vision` (Video/Image).
    *   **Object Detection**: `YOLO-World` (Real-time custom vocabulary detection).

### 3. Command Dashboard
*   **📍 Crisis Operational Map**: Real-time geospatial view of hazards and civilians.
*   **🔍 Semantic Search**: Natural language search across ALL modalities (e.g., "show me collapsed bridges" finds video frames and radio logs).
*   **Relevance Filtering**: Automatically filters low-confidence results to reduce noise.

---

## 🛠️ Setup & Installation

### 1. Prerequisites
*   Python 3.10+
*   [Qdrant Cloud Account](https://cloud.qdrant.io/) (Free Tier is sufficient)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory (copy from `.env.example` if available) and add your Qdrant Cloud credentials:

```ini
QDRANT_URL=https://your-cluster-url.qdrant.io:6333
QDRANT_API_KEY=your-api-key
```

### 4. Running the System
Open separate terminals for each agent to simulate parallel ingestion:

**Terminal 1 (Visual Intelligence):**
```bash
python watcher_agent.py
```
*(Monitors `video_inbox/`)*

**Terminal 2 (Image Intelligence):**
```bash
python image_agent.py
```
*(Monitors `image_inbox/`)*

**Terminal 3 (Audio Intelligence):**
```bash
python listener_agent.py
```
*(Monitors `audio_inbox/`)*

**Terminal 4 (Text Intelligence):**
```bash
python text_agent.py
```
*(Monitors `text_inbox/`)*

**Terminal 5 (Dashboard):**
```bash
streamlit run dashboard.py
```

---

## 📂 Project Structure

*   `_inbox/` folders: Drop files here to simulate incoming data streams.
*   `config.py`: Centralized configuration for cloud connections.
*   `dashboard.py`: Streamlit-based user interface.
*   `generate_civilians.py`: Utility to generate synthetic civilian location data.

## 🧠 AI Stack Rationale

| Component | Model/Tool | Reason |
| :--- | :--- | :--- |
| **Embeddings** | FastEmbed (BGE & CLIP) | Lightweight, fast CPU inference, no external API costs (runs locally). |
| **Object Detection** | YOLO-World | "Open Vocabulary" detection allows searching for new hazards (e.g., "cyclone") without retraining. |
| **Vector DB** | Qdrant | Efficient storage and retrieval of high-dimensional vectors with metadata filtering. |
