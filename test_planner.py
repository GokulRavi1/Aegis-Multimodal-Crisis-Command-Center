
import os
from dotenv import load_dotenv
load_dotenv()
print(f"DEBUG: ENV KEY Present: {bool(os.getenv('GROQ_API_KEY'))}")

from planner_agent import get_planner_agent

def test_planner():
    print("Initializing Planner...")
    planner = get_planner_agent()
    
    queries = [
        "Hello", # Simple
        "Show me flood images in Chennai", # Single step
        "Assess visual damage in Chennai and check for radio distress calls" # Multi step
    ]
    
    for q in queries:
        print(f"\nQUERY: {q}")
        steps = planner.create_plan(q)
        if not steps:
            print("   -> Plan: [Empty/Simple Interaction]")
        else:
            for step in steps:
                print(f"   -> Step {step.id}: [{step.tool}] {step.query} ({step.reasoning})")

if __name__ == "__main__":
    test_planner()
