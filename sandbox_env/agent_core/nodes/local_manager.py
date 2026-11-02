import json
from litellm import completion
from langchain_core.messages import AIMessage, HumanMessage
from agent_core.graph_state import AgentState

OLLAMA_API_BASE = "http://host.containers.internal:11434"
LOCAL_MODEL = "ollama/qwen2.5-coder:7b"

# We use a system prompt to force the local model to output structured JSON routing commands
SYSTEM_PROMPT = """You are the Local Manager of a multi-agent system.
Your job is to read the user request and decide the next step.
You must respond ONLY with a JSON object. Do not include markdown formatting.

Available next_nodes:
- "browser_tool": If you need to navigate to a URL or scrape a page.
- "cloud_worker": If you need to parse massive HTML or do complex reasoning.
- "rpc_bridge": If you need to save a file to the host PC.
- "end": If the task is completely finished.

Example Response:
{"next_node": "browser_tool", "command": "Go to linkedin.com/jobs", "target_url": "https://linkedin.com/jobs"}
"""

def local_manager_node(state: AgentState) -> dict:
    """Evaluates the state and decides the next action."""
    print("[Local Manager] Evaluating next steps...")
    
    # Convert LangChain messages to LiteLLM format
    formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in state["messages"]:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        formatted_messages.append({"role": role, "content": msg.content})

    # Call the local Qwen model
    response = completion(
        model=LOCAL_MODEL,
        messages=formatted_messages,
        api_base=OLLAMA_API_BASE,
        response_format={"type": "json_object"} # Forces valid JSON output
    )
    
    output = response.choices[0].message.content
    
    try:
        parsed_output = json.loads(output)
        next_node = parsed_output.get("next_node", "end")
        command_text = parsed_output.get("command", "")
        url = parsed_output.get("target_url", state.get("current_url", ""))
        
        # Return the dictionary to update the LangGraph state
        return {
            "messages": [AIMessage(content=f"Manager Plan: {command_text}")],
            "next_node": next_node,
            "current_url": url
        }
    except Exception as e:
        print(f"[Local Manager] JSON Parse Error: {e}")
        # Failsafe: End execution if output is malformed
        return {"next_node": "end"}