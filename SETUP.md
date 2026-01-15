# Aegis – Setup Guide

Complete step-by-step setup instructions for running Aegis locally.

---

## Prerequisites

### 1. Install Python 3.10+
Download from: https://www.python.org/downloads/

Verify installation:
```bash
python --version
```

### 2. Install Docker Desktop
Download from: https://www.docker.com/products/docker-desktop/

After installation:
1. Open Docker Desktop
2. Wait for the engine to start (whale icon stops animating)
3. Keep Docker Desktop running

---

## Installation Steps

### Step 1: Clone/Navigate to Project
```bash
cd "c:\workspace\Aegis – Multimodal Crisis Command Center"
```

### Step 2: Start Qdrant Vector Database
```bash
docker-compose up -d
```

**Expected Output:**
```
[+] Running 1/1
 ✔ Container aegismultimodalcrisiscommandcenter-qdrant-1  Started
```

**Verify:** Open http://localhost:6333/dashboard in browser.

### Step 3: Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Note:** First run will download embedding models (~400MB).

---

## Running the System

Execute these commands **in order**, each in a separate terminal:

### Terminal 1: Generate Test Data
```bash
python create_test_video.py
python generate_civilians.py
```

### Terminal 2: Run All Agents
```bash
python watcher_agent.py
python listener_agent.py
python text_agent.py
```

### Terminal 3: Start Safety Monitor
```bash
python safety_agent.py
```
> This runs continuously. Press `Ctrl+C` to stop.

### Terminal 4: Launch Dashboard
```bash
streamlit run dashboard.py
```

**Dashboard URL:** http://localhost:8501

---

## Verification Checklist

| Check | How to Verify |
|-------|---------------|
| Qdrant Running | http://localhost:6333/dashboard shows UI |
| Civilians Seeded | Dashboard shows green dots on map |
| Video Processed | Dashboard shows red dots on map |
| Alerts Generated | `alerts.json` file exists and has content |
| Dashboard Works | All 4 tabs load without errors |

---

## Troubleshooting

### Docker not running
**Error:** `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`

**Fix:** Start Docker Desktop and wait for it to fully initialize.

### Connection refused to Qdrant
**Error:** `Connection refused` or timeout errors

**Fix:** 
```bash
docker-compose down
docker-compose up -d
```

### Model download fails
**Error:** Network errors during embedding model download

**Fix:** Ensure you have internet connectivity for the first run.

---

## Stopping the System

```bash
# Stop Safety Agent: Ctrl+C in its terminal

# Stop Dashboard: Ctrl+C in its terminal

# Stop Qdrant:
docker-compose down
```

---

## Data Reset

To clear all data and start fresh:
```bash
docker-compose down -v
rm -rf qdrant_storage
rm alerts.json
docker-compose up -d
```

Then re-run all agent scripts.
