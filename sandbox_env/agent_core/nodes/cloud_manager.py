import os
import json
import litellm
from litellm import completion
from langchain_core.messages import AIMessage
from agent_core.graph_state import AgentState

os.makedirs("/sandbox_env/memory_state", exist_ok=True)
STATE_FILE = "/sandbox_env/memory_state/current_state.md"

def cloud_worker_node(state: AgentState) -> dict:
    """Delegates heavy HTML parsing and reasoning to Cloud APIs with fallback limits."""
    print("[Cloud Worker] Initializing heavy DOM parsing...")
    
    # Grab the HTML and the latest instruction from the Local Manager
    html_context = state.get("html_context", "")
    latest_task = state["messages"][-1].content
    
    messages = [
        {
            "role": "system", 
            "content": "You are a data extraction specialist. Read the HTML context and extract the requested data in clear text or JSON."
        },
        {
            "role": "user", 
            "content": f"Task: {latest_task}\n\nHTML Context:\n{html_context[:200000]}"
        }
    ]

    try:
        print("[Cloud Worker] Sending payload to Primary API (Gemini 3.5 Flash)...")
        response = completion(
            model="gemini/gemini-3.5-flash",
            messages=messages,
            temperature=0.1
        )
        result_text = response.choices[0].message.content
        print("[Cloud Worker] Gemini extraction successful.")

    # --- RATE LIMIT CAUGHT: CONTEXT HANDOFF & FALLBACK ---
    except litellm.exceptions.RateLimitError as e:
        print(f"[Cloud Worker] Gemini Rate Limit Hit (429)! Saving state to disk...")
        
        # Save the execution state to disk so we don't lose the DOM data
        state_dump = {
            "failed_model": "gemini-1.5-flash",
            "pending_task": latest_task,
            "html_snapshot": html_context[:100000] # Compressed for the fallback model
        }
        with open(STATE_FILE, "w") as f:
            f.write(json.dumps(state_dump, indent=2))
            
        print(f"[Cloud Worker] State saved to {STATE_FILE}. Switching to Fallback API...")
        
        # Attempt the Fallback API
        try:
            response = completion(
                model="github/gpt-4o-mini",
                messages=messages,
                temperature=0.1
            )
            result_text = response.choices[0].message.content
            print("[Cloud Worker] Fallback (GPT-4o-mini) extraction successful.")
            
        except Exception as fallback_error:
             print(f"[Cloud Worker] Fallback also failed: {fallback_error}")
             result_text = "System Error: All cloud models failed due to rate limits."
             
    except Exception as e:
         print(f"[Cloud Worker] Unexpected general error: {e}")
         result_text = f"System Error: {e}"

    # --- ROUTE BACK TO LOCAL MANAGER ---
    # We update the state with the extracted data, and set next_node back to 
    # "local_manager" so Qwen can decide what to do with the information.
    return {
        "messages": [AIMessage(content=f"Cloud Extraction Result:\n{result_text}")],
        "next_node": "local_manager" 
    }