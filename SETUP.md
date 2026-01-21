# Aegis – Setup Guide

Complete step-by-step setup instructions for running Aegis with Qdrant Cloud and Groq LLM.

---

## Prerequisites

### 1. Install Python 3.10+
Download from: [python.org](https://www.python.org/downloads/)

Verify:
```bash
python --version
```

### 2. API Credentials
*   **Qdrant Cloud**: Create a cluster and get your **URL** and **API Key** from [cloud.qdrant.io](https://cloud.qdrant.io/).
*   **Groq Cloud**: Get your **API Key** from [console.groq.com](https://console.groq.com/) (required for Analyst Agent and Warning Summaries).

---

## Installation Steps

### Step 1: Clone/Navigate to Project
```bash
cd "c:\workspace\Aegis – Multimodal Crisis Command Center"
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```
> **Note**: On first run, the system will download embedding models (BGE and CLIP) which are ~600MB total.

### Step 3: Configure Environment
Create a `.env` file in the root directory:
```ini
QDRANT_URL=https://your-cluster-url.qdrant.io
QDRANT_API_KEY=your-api-key
GROQ_API_KEY=your-groq-key
```

---

## Running the System

Execute these commands in order, each in a separate terminal:

### Phase 1: Ingestion Agents
Run all agents to start "listening" to their respective inboxes:
*   **Terminal 1**: `python watcher_agent.py` (Video)
*   **Terminal 2**: `python image_agent.py` (Images)
*   **Terminal 3**: `python listener_agent.py` (Audio)
*   **Terminal 4**: `python text_agent.py` (Text/Social)

### Phase 2: Intelligence & Cleanup
*   **Terminal 5**: `python analyst_agent.py` (Geospatial correlation & alert logging)
*   **Terminal 6**: `python memory_manager.py` (Initializes collections and runs cleanup policies)

### Phase 3: Launch Dashboard
*   **Terminal 7**: 
```bash
streamlit run dashboard.py
```
**Dashboard URL:** http://localhost:8501

---

## Verification & Testing

### 1. Run Evaluation Benchmark
Verify retrieval accuracy and system latency:
```bash
python tests/benchmark.py
```
Check `benchmark_results.json` for performance metrics.

### 2. Verify RAG Traceability
After performing a search or chat in the dashboard, check `retrieval_logs.json` to see the full evidence trail and citations.

### 3. Seed Test Data
To populate the map with simulated civilian data:
```bash
python generate_civilians.py
```

---

## Troubleshooting

### `getaddrinfo failed`
**Error**: `qdrant_client.http.exceptions.ResponseHandlingException: [Errno 11001]`
**Fix**: This is usually a DNS or network issue reaching Qdrant Cloud. Check your internet connection or `.env` URL format.

### `ModuleNotFoundError: No module named 'config'`
**Fix**: Ensure you are running scripts from the root directory. If running benchmarks, ensure `sys.path` is correctly set (already fixed in `benchmark.py`).

### `Vector dimension error`
**Error**: `expected dim: 512, got 384`
**Fix**: You are likely searching a visual collection with text embeddings. Ensure the dashboard is using the correct model for each collection.

---

## Data Reset
To clear memory and start fresh:
1. Delete collections in Qdrant Cloud UI.
2. Run `python memory_manager.py` to re-initialize empty collections.
3. Empty the `_inbox` folders.
