import json
import os
from fastmcp import FastMCP

mcp = FastMCP("DualVaultMemoryServer")

PROFILE_FILE = "user_profile.json"
TOPIC_FILE = "topic_knowledge.json"

def _load_json(file_path: str) -> dict:
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_json(file_path: str, data: dict):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# ----------------- TIER 1: PERSONALIZATION VAULT -----------------
@mcp.tool()
def update_user_profile(category: str, detail: str) -> str:
    """Saves user personal traits, habits, hobbies, communication style, or preferences."""
    profile = _load_json(PROFILE_FILE)
    if category not in profile:
        profile[category] = []
    
    if detail not in profile[category]:
        profile[category].append(detail)
        _save_json(PROFILE_FILE, profile)
        return f"User profile updated under '{category}': {detail}"
    return f"Detail already exists in user profile under '{category}'."

@mcp.tool()
def get_user_profile() -> str:
    """Retrieves the complete personalization profile of the user."""
    profile = _load_json(PROFILE_FILE)
    if not profile:
        return "User profile is currently empty."
    return json.dumps(profile, indent=2)

# ----------------- TIER 2: DENSE TOPIC KNOWLEDGE VAULT -----------------
@mcp.tool()
def save_dense_topic_knowledge(topic: str, key_excerpts: list[str]) -> str:
    """
    Saves 4-6 high-density, factual lines/excerpts about a specific topic 
    (e.g., NASA, technical specs, rules) to preserve exact information.
    """
    vault = _load_json(TOPIC_FILE)
    topic_key = topic.strip().title()
    
    if topic_key not in vault:
        vault[topic_key] = []
        
    for item in key_excerpts:
        if item not in vault[topic_key]:
            vault[topic_key].append(item)
            
    _save_json(TOPIC_FILE, vault)
    return f"Saved {len(key_excerpts)} dense factual lines under topic '{topic_key}'."

@mcp.tool()
def search_topic_knowledge(query: str) -> str:
    """Searches the dense topic knowledge base for exact preserved factual excerpts."""
    vault = _load_json(TOPIC_FILE)
    if not vault:
        return "Dense topic knowledge vault is currently empty."
        
    query_lower = query.lower()
    matches = {}
    
    for topic, lines in vault.items():
        if query_lower in topic.lower():
            matches[topic] = lines
        else:
            matching_lines = [l for l in lines if query_lower in l.lower()]
            if matching_lines:
                matches[topic] = matching_lines
                
    if matches:
        output = "FOUND IN DENSE TOPIC VAULT:\n"
        for top, lines in matches.items():
            output += f"\n### Topic: {top}\n"
            for line in lines:
                output += f"• {line}\n"
        return output
    return "No exact factual matches found in the dense topic vault."

if __name__ == "__main__":
    mcp.run(transport="stdio")