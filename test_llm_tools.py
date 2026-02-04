
import os
from llm_manager import get_llm_manager
from retrieval_tools import RetrievalToolkit

# Mock client/models just to get tool definitions
class MockThing:
    pass

def test_llm_tools():
    print("Initializing LLM Manager...")
    llm = get_llm_manager()
    
    # Get tool definitions (we don't need real Qdrant client for this)
    toolkit = RetrievalToolkit(None, None, None) 
    tools = toolkit.get_tool_definitions()
    
    print("\n--- Tool Definitions ---")
    print(tools)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools if needed."},
        {"role": "user", "content": "Show me evidence of fire."}
    ]
    
    print("\n--- Sending Request to Groq ---")
    response = llm.tool_chat_completion(messages, tools=tools)
    
    if response:
        print("\n✅ Success!")
        print(f"Content: {response.content}")
        print(f"Tool Calls: {response.tool_calls}")
    else:
        print("\n❌ Failed to get response")

if __name__ == "__main__":
    test_llm_tools()
