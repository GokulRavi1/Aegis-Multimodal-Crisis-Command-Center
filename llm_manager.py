import os
import time
import logging
from groq import Groq, RateLimitError, APIError

class LLMManager:
    """
    Centralized Manager for LLM interactions with robust fallback strategies.
    Cycles through multiple models to handle Rate Limit (429) errors.
    """
    
    # Priority list of models to try
    # 1. Llama 3.3 70B (Best quality)
    # 2. Mixtral 8x7B (High quality, different quota)
    # 3. Llama 3.1 8B (Fast, different quota)
    # 4. Gemma 2 9B (Google's model, separate quota)
    # 5. Llama 3 70B (Legacy, fallback)
    MODELS = [
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
        "llama3-70b-8192"
    ]
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            print("⚠️ GROQ_API_KEY not found. LLM features will be disabled.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
            print(f"🧠 LLMManager initialized with {len(self.MODELS)} fallback models.")

    def chat_completion(self, messages, max_tokens=100, temperature=0.7):
        """
        Attempts to get a chat completion, trying models in sequence if Rate Limits are hit.
        """
        if not self.client:
            return None
            
        for model in self.MODELS:
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return completion.choices[0].message.content.strip()
                
            except RateLimitError as e:
                print(f"   ⚠️ Rate Limit hit for {model}. Switching to backup...")
                continue # Try next model
                
            except APIError as e:
                print(f"   ⚠️ API Error with {model}: {e}")
                continue # Try next model
                
            except Exception as e:
                print(f"   ❌ Unexpected LLM Error ({model}): {e}")
                break # Stop on non-API errors (like network/auth)
                
        print("   ❌ All LLM models exhausted or failed.")
        return None

# Singleton Pattern
_llm_manager = None

def get_llm_manager():
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager
