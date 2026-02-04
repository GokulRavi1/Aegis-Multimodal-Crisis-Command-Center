
import os
from retrieval_tools import RetrievalToolkit
from config import get_qdrant_client
from fastembed import TextEmbedding

# Mock setup
try:
    print("Initializing resources...")
    client = get_qdrant_client()
    text_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    clip_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5") # Using text model as mock for clip to avoid download

    toolkit = RetrievalToolkit(client, text_model, clip_model)

    print("\n--- Testing Visual Search ---")
    res = toolkit.search_visual_memory("fire")
    print(res)

    print("\n--- Testing Tool Definitions ---")
    print(toolkit.get_tool_definitions()[0]['function']['name'])
    
    print("\n✅ Retrieval Toolkit Test Passed")
except Exception as e:
    print(f"\n❌ Test Failed: {e}")
