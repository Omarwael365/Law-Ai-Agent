"""
ReAct Agent — LangChain ReAct agent with legal research tools.
Option A from the project guide: wraps RAG retriever as tools and lets
the agent decide when and how to use them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from src.llm.providers import get_llm
from src.agents.tools import get_tools
from src.prompts.legal_prompts import AGENT_SYSTEM_PROMPT


REACT_PROMPT_TEMPLATE = """{system_prompt}

You have access to the following tools:
{tools}

Use the following format:

Question: the input legal question you must answer
Thought: think about what information you need and which tool to use
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat as needed)
Thought: I now have enough information to provide a comprehensive answer
Final Answer: the final legal analysis with citations

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

REACT_PROMPT = PromptTemplate(
    template=REACT_PROMPT_TEMPLATE,
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
    partial_variables={"system_prompt": AGENT_SYSTEM_PROMPT},
)


def create_react_legal_agent(llm_provider: str = None):
    """
    Create a ReAct agent for legal research.

    Args:
        llm_provider: LLM provider to use (openai, google, deepseek)

    Returns:
        An AgentExecutor instance
    """
    llm = get_llm(provider=llm_provider)
    tools = get_tools()

    agent = create_react_agent(llm, tools, REACT_PROMPT)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    return agent_executor


def run_react_agent(question: str, llm_provider: str = None) -> dict:
    """
    Run the ReAct agent on a legal question.

    Args:
        question: The legal question to answer
        llm_provider: LLM provider to use

    Returns:
        Dict with answer and intermediate steps
    """
    agent = create_react_legal_agent(llm_provider)
    result = agent.invoke({"input": question})

    return {
        "question": question,
        "answer": result.get("output", ""),
        "steps": result.get("intermediate_steps", []),
    }


if __name__ == "__main__":
    question = "Does COVID-19 qualify as force majeure under a supply contract clause listing natural disasters and governmental action?"
    print(f"Question: {question}\n")
    result = run_react_agent(question)
    print(f"\nFinal Answer:\n{result['answer']}")
