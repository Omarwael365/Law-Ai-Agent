"""
Evaluator — Measures Faithfulness, Task Success Rate, Latency, and Cost.
These are the four key metrics specified in the project guide.
"""

import time
import json
from typing import List, Dict, Any, Callable
from src.llm.providers import get_llm


def measure_latency(func: Callable, *args, **kwargs) -> tuple:
    """
    Measure the execution time of a function.
    Returns (result, latency_seconds).
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    latency = end_time - start_time
    return result, latency


def evaluate_faithfulness(answer: str, context: str, llm_provider: str = None) -> dict:
    """
    Evaluate whether the answer is faithful to the retrieved context.
    Uses LLM-as-judge to check if the answer only contains information
    from the retrieved documents.

    Returns:
        Dict with faithfulness_score (0.0-1.0) and explanation
    """
    llm = get_llm(provider=llm_provider, temperature=0.0)

    evaluation_prompt = f"""You are an evaluation judge. Your task is to assess whether the ANSWER is faithful to the CONTEXT — meaning the answer only contains information that can be directly supported by the provided context.

CONTEXT:
{context}

ANSWER:
{answer}

Evaluate on a scale of 0.0 to 1.0:
- 1.0 = Every claim in the answer is directly supported by the context
- 0.5 = Some claims are supported but others are not in the context
- 0.0 = The answer contains mostly information not in the context

Respond in this exact JSON format:
{{"faithfulness_score": <float>, "explanation": "<brief explanation>", "unsupported_claims": ["<claim1>", "<claim2>"]}}"""

    response = llm.invoke(evaluation_prompt)
    content = response.content.strip()

    # Parse JSON response
    try:
        # Try to extract JSON from the response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        result = json.loads(content)
    except (json.JSONDecodeError, IndexError):
        # Fallback if JSON parsing fails
        result = {
            "faithfulness_score": 0.5,
            "explanation": "Could not parse evaluation response",
            "unsupported_claims": [],
        }

    return result


def evaluate_task_success(answer: str, expected_topics: List[str], llm_provider: str = None) -> dict:
    """
    Evaluate whether the agent successfully completed the task.
    Checks if the answer addresses all expected topics/issues.

    Args:
        answer: The agent's answer
        expected_topics: List of topics/issues the answer should address

    Returns:
        Dict with success_rate (0.0-1.0) and per-topic results
    """
    llm = get_llm(provider=llm_provider, temperature=0.0)

    topics_str = "\n".join(f"- {topic}" for topic in expected_topics)

    evaluation_prompt = f"""You are an evaluation judge. Determine whether the ANSWER addresses each of the following EXPECTED TOPICS.

ANSWER:
{answer}

EXPECTED TOPICS:
{topics_str}

For each topic, respond with "YES" if the answer addresses it or "NO" if it does not.
Respond in this exact JSON format:
{{"topics": [{{"topic": "<topic>", "addressed": true/false, "evidence": "<brief quote or explanation>"}}], "overall_success": true/false}}"""

    response = llm.invoke(evaluation_prompt)
    content = response.content.strip()

    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        result = json.loads(content)
    except (json.JSONDecodeError, IndexError):
        result = {
            "topics": [{"topic": t, "addressed": False, "evidence": "Parse error"} for t in expected_topics],
            "overall_success": False,
        }

    # Calculate success rate
    if result.get("topics"):
        addressed = sum(1 for t in result["topics"] if t.get("addressed", False))
        result["success_rate"] = addressed / len(result["topics"])
    else:
        result["success_rate"] = 0.0

    return result


def estimate_cost(input_tokens: int, output_tokens: int, provider: str = "openai") -> dict:
    """
    Estimate the API cost based on token usage.

    Pricing (approximate, per 1M tokens):
    - GPT-4o: $2.50 input / $10.00 output
    - Gemini 1.5 Pro: $3.50 input / $10.50 output
    - DeepSeek-V3: $0.27 input / $1.10 output
    """
    pricing = {
        "openai": {"input": 2.50, "output": 10.00},
        "google": {"input": 3.50, "output": 10.50},
        "deepseek": {"input": 0.27, "output": 1.10},
    }

    rates = pricing.get(provider, pricing["openai"])
    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (output_tokens / 1_000_000) * rates["output"]

    return {
        "provider": provider,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
    }


def run_full_evaluation(
    question: str,
    answer: str,
    context: str,
    expected_topics: List[str],
    latency: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    provider: str = "openai",
    llm_provider: str = None,
) -> dict:
    """
    Run all four evaluation metrics on a single query result.

    Returns:
        Comprehensive evaluation dict
    """
    # Faithfulness
    faithfulness = evaluate_faithfulness(answer, context, llm_provider)

    # Task Success Rate
    task_success = evaluate_task_success(answer, expected_topics, llm_provider)

    # Cost
    cost = estimate_cost(input_tokens, output_tokens, provider)

    return {
        "question": question,
        "faithfulness": faithfulness,
        "task_success": task_success,
        "latency_seconds": round(latency, 3),
        "cost": cost,
        "summary": {
            "faithfulness_score": faithfulness.get("faithfulness_score", 0),
            "success_rate": task_success.get("success_rate", 0),
            "latency_seconds": round(latency, 3),
            "total_cost_usd": cost.get("total_cost_usd", 0),
        },
    }
