# Cognitive-Dual-Vault-Memory
A dual-tier cognitive memory architecture for LangGraph agents. Automatically extracts dense technical facts and user personality traits into local FastMCP vaults while aggressively pruning context windows to save tokens.

# 🧠 Neuro-Vault: Dual-Tier Cognitive Agent

Ever noticed how LLMs forget exactly what you said 10 minutes ago, or worse, they summarize a highly technical prompt into "User talked about NASA"? 

I got tired of context rot and massive API bills, so I built Neuro-Vault. This isn't just a basic RAG pipeline. It’s a stateful LangGraph architecture that uses a dual-vault memory system via the Model Context Protocol (FastMCP). 

Instead of holding everything in RAM, the agent actively extracts dense technical facts (preserving exact numbers and specs) into one local JSON vault, and user personality traits into another. Every 3 turns, it wipes its own short-term memory to save tokens. If you ask a question it forgot, it autonomously searches the vaults and "rehydrates" its context.

## 🏗️ How It Works (The Architecture)

* **Tier 1 (Working Context):** Holds the active conversation for exactly 3 turns.
* **Tier 2 (Dense Topic Vault):** `topic_knowledge.json`. A background Extractor Agent scans your prompts. If you drop a 30-line technical spec, it extracts 4-6 high-density bullet points and saves them to disk.
* **Tier 3 (User Profile Vault):** `user_profile.json`. Silently builds a profile on your habits, tone preferences, and background.
* **The Pruner:** Every 3 turns, the agent compresses the chat into a rolling summary and completely dumps the raw messages from the context window.
* **Rehydration:** When asked a question, a Router Agent decides if it needs to query the local disk (via FastMCP) to pull facts back into the active context before answering.

## 🛠️ Tech Stack
* **Orchestration:** LangGraph 
* **Local Tooling:** FastMCP 2.0 (Model Context Protocol)
* **LLM:** Gemini (for reasoning/extraction & for fast routing)
* **Evals:** LLM-as-a-Judge for hallucination checks


