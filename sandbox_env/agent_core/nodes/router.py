import json
from litellm import completion

# Point to your host's Ollama instance via the Podman gateway
LOCAL_MODEL = "ollama/qwen2.5-coder:7b"
OLLAMA_API_BASE = "http://host.containers.internal:11434"

ROUTER_PROMPT = """
You are the routing engine for a multi-agent system.
Analyze the user's task and determine its complexity to route it to the correct execution environment.

Criteria for 'cloud_worker' (High Complexity):
- Requires parsing massive data (like raw HTML scraping).
- Requires writing, testing, or debugging complex code.
- Requires deep, multi-step logical reasoning.

Criteria for 'local_worker' (Low Complexity):
- Routine text processing (summarization, extraction, translation).
- Generating short templates (emails, cover letters).
- Involves highly sensitive personal data.

Respond ONLY with a JSON object in this exact format:
{
  "reasoning": "Brief explanation of why it requires this environment",
  "route": "cloud_worker" or "local_worker"
}
"""

def triage_node(state: dict):
    """Evaluates the prompt complexity and decides the execution route."""
    print("[Router] Evaluating task complexity using Local LLM...")
    
    user_prompt = state.get("main_prompt", "")
    
    try:
        response = completion(
            model=LOCAL_MODEL,
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": f"Task: {user_prompt}"}
            ],
            api_base=OLLAMA_API_BASE,
            response_format={"type": "json_object"}
        )
        
        decision = json.loads(response.choices[0].message.content)
        print(f"[Router] Decision: {decision['route']} | Reason: {decision['reasoning']}")
        
        # Update the state with the decided route
        return {"next_node": decision["route"]}
        
    except Exception as e:
        print(f"[Router] Error during classification: {e}")
        # Failsafe: Default to local to prevent unintended API usage
        return {"next_node": "local_worker"}