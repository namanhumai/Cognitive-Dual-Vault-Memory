import os
import json
from typing import TypedDict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

load_dotenv()

def _to_clean_str(val: Any) -> str:
    """Safely converts strings, dicts, lists of content blocks into a clean plain text string."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        extracted = []
        for item in val:
            if isinstance(item, str):
                extracted.append(item)
            elif isinstance(item, dict) and "text" in item:
                extracted.append(str(item["text"]))
            elif hasattr(item, "text"):
                extracted.append(str(item.text))
            else:
                extracted.append(str(item))
        return "\n".join(extracted).strip()
    if isinstance(val, dict):
        if "text" in val:
            return str(val["text"]).strip()
        return json.dumps(val, indent=2)
    return str(val).strip()

class GraphState(TypedDict):
    user_input: str
    chat_history: str
    rolling_summary: str
    turn_count: int
    retrieved_topic_data: str
    retrieved_profile_data: str
    ai_response: str
    extractor_logs: list[str]

class RehydrationPlan(BaseModel):
    needs_topic_search: bool = Field(description="True if query asks about previously discussed technical/factual topics.")
    topic_search_query: str = Field(description="Search term for the dense topic knowledge file, or 'NONE'.")
    needs_user_profile: bool = Field(description="True if query depends on user personality, habits, or preferences.")

class DualExtraction(BaseModel):
    has_personal_info: bool = Field(description="True if user mentioned personal habits, traits, preferences, or lifestyle.")
    personal_category: str = Field(description="Category e.g. 'Preferences', 'Career', 'Hobbies', or 'NONE'.")
    personal_detail: str = Field(description="Specific detail about the user, or 'NONE'.")
    
    has_dense_topic_info: bool = Field(description="True if user or AI shared detailed, substantive, or technical information.")
    topic_name: str = Field(description="Subject name (e.g., 'NASA Artemis Missions'), or 'NONE'.")
    dense_excerpts: list[str] = Field(
        description="4 to 6 high-density, informative, factual lines extracted directly from the conversation."
    )

async def create_agent_workflow(mcp_tools: dict):
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("CRITICAL ERROR: GOOGLE_API_KEY is missing in your .env file!")

    llm = ChatGoogleGenerativeAI(model="models/gemini-3.6-flash", temperature=0.2, api_key=api_key)

    # --- NODE 1: Smart Retrieval & Rehydration ---
    async def retriever_node(state: GraphState):
        prompt = ChatPromptTemplate.from_template(
            "You are a Memory Retrieval Router.\n"
            "Active Rolling Dialogue Summary: {summary}\n"
            "User's Latest Query: '{input}'\n\n"
            "Determine if answering this requires:\n"
            "1. Searching the dense factual topic files on disk.\n"
            "2. Reading the user's personal profile file.\n"
            "Provide your retrieval plan."
        )
        chain = prompt | llm.with_structured_output(RehydrationPlan)
        plan = await chain.ainvoke({
            "summary": state["rolling_summary"],
            "input": state["user_input"]
        })

        topic_data = ""
        profile_data = ""

        if plan.needs_topic_search and plan.topic_search_query != "NONE":
            raw_res = await mcp_tools["search_topic_knowledge"].ainvoke({"query": plan.topic_search_query})
            topic_data = _to_clean_str(raw_res)
            
        if plan.needs_user_profile:
            raw_prof = await mcp_tools["get_user_profile"].ainvoke({})
            profile_data = _to_clean_str(raw_prof)

        return {
            "retrieved_topic_data": topic_data,
            "retrieved_profile_data": profile_data
        }

    # NODE 2: Context-Augmented Responder ---
    async def chat_node(state: GraphState):
        prompt = ChatPromptTemplate.from_template(
            "You are an intelligent assistant with access to permanent cognitive storage.\n\n"
            "--- BACKGROUND ROLLING SUMMARY ---\n{summary}\n\n"
            "--- RECENT MESSAGES ---\n{history}\n\n"
            "--- DEEP RETRIEVED TOPIC FACTS (From topic_knowledge.json) ---\n{topic_data}\n\n"
            "--- USER PERSONAL PROFILE (From user_profile.json) ---\n{profile_data}\n\n"
            "User Query: {input}\n\n"
            "Instruction: If deep topic facts or user profile data are provided above, prioritize them over vague summaries. Provide a direct, natural response."
        )
        chain = prompt | llm
        response = await chain.ainvoke({
            "summary": state["rolling_summary"],
            "history": state["chat_history"],
            "topic_data": state["retrieved_topic_data"],
            "profile_data": state["retrieved_profile_data"],
            "input": state["user_input"]
        })

        clean_resp = _to_clean_str(response.content)
        updated_history = state["chat_history"] + f"\nUser: {state['user_input']}\nAssistant: {clean_resp}\n"
        return {"ai_response": clean_resp, "chat_history": updated_history}

    # NODE 3: Dual-Vault Extractor ---
    async def dual_extractor_node(state: GraphState):
        prompt = ChatPromptTemplate.from_template(
            "Analyze this conversational turn:\n"
            "User said: '{user_input}'\n"
            "Assistant said: '{ai_response}'\n\n"
            "Task 1: Did the user share personal facts (traits, likes, job, preferences)?\n"
            "Task 2: Did this conversation contain substantive knowledge, explanations, or data about a topic (e.g. NASA, technical specs, rules)?\n"
            "If yes to Task 2, extract 4 to 6 dense, highly informative sentences capturing the exact core data so fine details are never lost."
        )
        chain = prompt | llm.with_structured_output(DualExtraction)
        extraction = await chain.ainvoke({
            "user_input": state["user_input"],
            "ai_response": state["ai_response"]
        })

        logs = []
        if extraction.has_personal_info and extraction.personal_detail != "NONE":
            await mcp_tools["update_user_profile"].ainvoke({
                "category": extraction.personal_category,
                "detail": extraction.personal_detail
            })
            logs.append(f"👤 Saved Personal Profile: [{extraction.personal_category}] {extraction.personal_detail}")

        if extraction.has_dense_topic_info and extraction.dense_excerpts:
            await mcp_tools["save_dense_topic_knowledge"].ainvoke({
                "topic": extraction.topic_name,
                "key_excerpts": extraction.dense_excerpts
            })
            logs.append(f"📚 Saved {len(extraction.dense_excerpts)} Dense Excerpts under topic: '{extraction.topic_name}'")

        return {
            "extractor_logs": logs,
            "turn_count": state["turn_count"] + 1
        }

    # NODE 4: Context Pruner & Rolling Summarizer ---
    async def summarize_node(state: GraphState):
        prompt = ChatPromptTemplate.from_template(
            "You are a Context Pruning Agent.\n"
            "Existing Summary: {summary}\n\n"
            "Recent Chat turns to compress:\n{history}\n\n"
            "Synthesize both into a concise rolling summary focusing on conversational flow.\n"
            "Exact technical details are already safely backed up in the file vault."
        )
        chain = prompt | llm
        response = await chain.ainvoke({
            "summary": state["rolling_summary"],
            "history": state["chat_history"]
        })

        return {
            "rolling_summary": _to_clean_str(response.content),
            "chat_history": "",
            "turn_count": 0,
            "extractor_logs": state.get("extractor_logs", []) + ["🧹 Buffer Limit Reached: Compressed history into rolling summary & wiped raw turns."]
        }

    def route_pruning(state: GraphState):
        if state["turn_count"] >= 3:
            return "summarize_node"
        return END

    workflow = StateGraph(GraphState)
    workflow.add_node("retriever_node", retriever_node)
    workflow.add_node("chat_node", chat_node)
    workflow.add_node("dual_extractor_node", dual_extractor_node)
    workflow.add_node("summarize_node", summarize_node)

    workflow.set_entry_point("retriever_node")
    workflow.add_edge("retriever_node", "chat_node")
    workflow.add_edge("chat_node", "dual_extractor_node")
    workflow.add_conditional_edges("dual_extractor_node", route_pruning)
    workflow.add_edge("summarize_node", END)

    return workflow.compile()