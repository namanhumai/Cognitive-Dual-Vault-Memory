import asyncio
import sys
import warnings
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

warnings.filterwarnings("ignore")

from graph import create_agent_workflow
from evals import evaluate_response

load_dotenv()
console = Console()

async def main():
    console.print(Panel.fit(
        "[bold cyan]🧠 NEURO-VAULT: DUAL-TIER COGNITIVE MEMORY[/bold cyan]\n"
        "[dim]Dynamic Topic Extraction • Personal Profile Vault • Auto-Pruning[/dim]",
        border_style="cyan"
    ))

    console.print("[yellow]Connecting to MCP Memory Server...[/yellow]")
    client = MultiServerMCPClient({
        "dual_vault": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["mcp_server.py"]
        }
    })

    tools = await client.get_tools()
    mcp_tools = {tool.name: tool for tool in tools}
    console.print(f"[green]✔ Connected. Tools ready:[/green] {list(mcp_tools.keys())}\n")

    app = await create_agent_workflow(mcp_tools)

    state = {
        "user_input": "",
        "chat_history": "",
        "rolling_summary": "Conversation freshly initialized.",
        "turn_count": 0,
        "retrieved_topic_data": "",
        "retrieved_profile_data": "",
        "ai_response": "",
        "extractor_logs": []
    }

    console.print("[bold green]System active. Enter your queries below (type 'exit' to finish).[/bold green]\n")

    last_query = ""
    last_retrieved = ""
    last_response = ""

    while True:
        user_msg = input("\nYou: ").strip()
        if not user_msg:
            continue
        if user_msg.lower() in ["exit", "quit"]:
            break

        state["user_input"] = user_msg
        state["extractor_logs"] = []

        # ---------------------------------------------------------
        # ASYNC UI OPTIMIZATION: Read while computing
        # ---------------------------------------------------------
        async for event in app.astream(state):
            for node_name, node_state in event.items():
                state.update(node_state)

                if node_name == "retriever_node":
                    topic_data = str(state.get("retrieved_topic_data") or "")
                    profile_data = str(state.get("retrieved_profile_data") or "")

                    if "FOUND IN DENSE TOPIC VAULT" in topic_data:
                        console.print("  [cyan]🔍 [Rehydration] Retrieved dense facts from topic_knowledge.json[/cyan]")
                    if profile_data and "empty" not in profile_data.lower():
                        console.print("  [cyan]👤 [Rehydration] Retrieved identity data from user_profile.json[/cyan]")

                # FIX: Print the Assistant's reply immediately after the Chat node finishes!
                # Do not wait for the Extractor or Summarizer to run.
                elif node_name == "chat_node":
                    clean_reply = str(node_state.get("ai_response", ""))
                    console.print(f"\n[bold blue]Assistant:[/bold blue] {clean_reply}")
                    console.print("  [dim italic]...processing background memories...[/dim italic]")

                # Extractor and Summarizer logs now print silently below the answer
                elif node_name == "dual_extractor_node":
                    for log_entry in node_state.get("extractor_logs", []):
                        console.print(f"  [dim magenta]{log_entry}[/dim magenta]")

                elif node_name == "summarize_node":
                    console.print("  [dim red]🗑️ [Buffer Full] Context pruned. Short-term turns wiped to save tokens.[/dim red]")

        last_query = user_msg
        last_retrieved = f"{str(state.get('retrieved_topic_data', ''))} {str(state.get('retrieved_profile_data', ''))}".strip()
        last_response = str(state.get("ai_response", ""))

    if last_response:
        console.print("\n[yellow]Running Final Evaluation Audit on last exchange...[/yellow]")
        metrics = await evaluate_response(last_query, last_retrieved, last_response)

        table = Table(title="Memory Integrity & Retrieval Audit", border_style="blue")
        table.add_column("Metric", style="bold white")
        table.add_column("Score / Result", style="cyan")

        score_style = "green" if metrics.score >= 4 else "red"
        table.add_row("Fidelity Score", f"[{score_style}]{metrics.score} / 5[/{score_style}]")
        table.add_row("Vault Lines Utilized", str(metrics.utilized_dense_vault))
        table.add_row("Hallucination Detected", str(metrics.hallucination_detected))
        table.add_row("Reasoning", metrics.audit_reasoning)
        console.print(table)

if __name__ == "__main__":
    asyncio.run(main())