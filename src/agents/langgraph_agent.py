"""
LangGraph Stateful Agent — Multi-step reasoning agent with memory.
Option B (Recommended) from the project guide: maintains state across turns,
loops until sufficient context is retrieved, and produces cited legal analysis.
"""

import sys
from pathlib import Path
from typing import TypedDict, Annotated, Sequence, List
import operator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from src.llm.providers import get_llm
from src.agents.tools import get_tools
from src.prompts.legal_prompts import AGENT_SYSTEM_PROMPT


# ── State Definition ────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """State maintained across the agent's reasoning steps."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    retrieved_docs: List[str]
    retrieval_count: int
    is_sufficient: bool


# ── Node Functions ──────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """Decide whether to continue retrieving or generate the final answer."""
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM made tool calls, route to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise, the agent is done
    return "end"


def agent_node(state: AgentState, llm_with_tools):
    """The main agent reasoning node. Decides what to do next."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def check_sufficiency(state: AgentState) -> str:
    """
    After tool execution, check if we have enough context.
    If retrieval count exceeds max, force generation.
    """
    retrieval_count = state.get("retrieval_count", 0) + 1

    # Max 3 retrieval rounds to prevent infinite loops
    if retrieval_count >= 3:
        return "end"

    return "agent"


# ── Graph Builder ───────────────────────────────────────────────────────────

def create_langgraph_agent(llm_provider: str = None):
    """
    Create a LangGraph stateful agent for legal research.

    The agent follows this flow:
    1. Receive user question
    2. Decide which tools to call (search case law, contracts, or both)
    3. Execute tools and receive results
    4. Decide if more context is needed
    5. Generate final cited answer

    Args:
        llm_provider: LLM provider to use

    Returns:
        A compiled LangGraph application
    """
    llm = get_llm(provider=llm_provider)
    tools = get_tools()

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # Create the tool execution node
    tool_node = ToolNode(tools)

    # Build the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node(
        "agent",
        lambda state: agent_node(state, llm_with_tools)
    )
    workflow.add_node("tools", tool_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # After tools, go back to agent for reasoning
    workflow.add_edge("tools", "agent")

    # Compile
    app = workflow.compile()

    return app


def run_langgraph_agent(question: str, llm_provider: str = None, chat_history: list = None) -> dict:
    """
    Run the LangGraph agent on a legal question.

    Args:
        question: The legal question to answer
        llm_provider: LLM provider to use
        chat_history: Optional list of previous messages for multi-turn conversation

    Returns:
        Dict with answer, messages, and metadata
    """
    app = create_langgraph_agent(llm_provider)

    # Build initial messages
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]

    # Add chat history if provided
    if chat_history:
        messages.extend(chat_history)

    messages.append(HumanMessage(content=question))

    # Run the agent
    initial_state = {
        "messages": messages,
        "retrieved_docs": [],
        "retrieval_count": 0,
        "is_sufficient": False,
    }

    result = app.invoke(initial_state)

    # Extract the final answer
    final_message = result["messages"][-1]
    answer = final_message.content if hasattr(final_message, "content") else str(final_message)

    # Count tool calls for metrics
    tool_calls = sum(
        1 for msg in result["messages"]
        if hasattr(msg, "tool_calls") and msg.tool_calls
    )

    return {
        "question": question,
        "answer": answer,
        "messages": result["messages"],
        "tool_calls": tool_calls,
        "total_messages": len(result["messages"]),
    }


if __name__ == "__main__":
    question = "Can the defendant claim force majeure due to COVID-19 under a supply contract that lists natural disasters and governmental action?"
    print(f"Question: {question}\n")
    result = run_langgraph_agent(question)
    print(f"\nFinal Answer:\n{result['answer']}")
    print(f"\nTool calls made: {result['tool_calls']}")
    print(f"Total messages: {result['total_messages']}")
