
import time
from episodic_memory import get_episodic_memory

def test_episodic():
    print("Initializing Episodic Memory...")
    mem = get_episodic_memory()
    
    print("\n--- Adding Episode ---")
    query = "Show me floods in Chennai"
    response = "I found 3 flood images in Chennai."
    summary = "User investigated flood levels in Chennai."
    
    mem.add_episode(query, response, summary)
    
    print("Waiting for indexing...")
    time.sleep(2) 
    
    print("\n--- Retrieving Recent Context ---")
    # Should get the one we just added
    context = mem.retrieve_context("What is the status?")
    print("Context retrieved:", context)
    
    expected = "User investigated flood levels in Chennai"
    if any(expected in c for c in context):
        print("\n✅ Episodic Memory Test Passed")
    else:
        print("\n❌ context mismatch")

if __name__ == "__main__":
    test_episodic()
