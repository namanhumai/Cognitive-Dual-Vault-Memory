import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

class EvalMetrics(BaseModel):
    score: int = Field(description="Fidelity score from 1 to 5 (5 = perfectly grounded in retrieved vault lines)")
    utilized_dense_vault: bool = Field(description="True if the response contained the preserved dense facts from the file.")
    hallucination_detected: bool = Field(description="True if the response invented details contrary to the vault.")
    audit_reasoning: str = Field(description="Brief explanation of the score.")

async def evaluate_response(user_query: str, retrieved_facts: str, response: str) -> EvalMetrics:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(model="models/gemini-3.6-flash", temperature=0, api_key=api_key)

    prompt = ChatPromptTemplate.from_template(
        "You are a strict QA Compliance Evaluator.\n\n"
        "User Query: {query}\n"
        "Ground Truth Vault Facts (Retrieved from disk): {facts}\n"
        "Assistant's Response: {response}\n\n"
        "Score the response on whether it accurately rehydrated the specific facts without hallucinating."
    )
    chain = prompt | llm.with_structured_output(EvalMetrics)
    return await chain.ainvoke({
        "query": user_query,
        "facts": retrieved_facts if retrieved_facts else "None retrieved.",
        "response": response
    })
