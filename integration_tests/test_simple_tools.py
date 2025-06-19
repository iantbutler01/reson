"""
Simple example showing the tool type pattern.

Demonstrates how callables are converted to typed tool classes.
"""

import asyncio
from pydantic import BaseModel
from typing import Union, Callable

from reson.reson import agentic, Runtime


class CalculationResult(BaseModel):
    """Result of a calculation."""
    expression: str
    result: float


def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    # In real usage, use a safe expression evaluator
    try:
        # Simple safe evaluation for basic operations
        allowed_names = {
            "__builtins__": {}
        }
        return float(eval(expression, allowed_names))
    except:
        return 0.0


def get_fact(topic: str) -> str:
    """Get a fact about a topic."""
    facts = {
        "python": "Python was created by Guido van Rossum in 1991.",
        "math": "Pi is approximately 3.14159.",
        "space": "The universe is about 13.8 billion years old.",
    }
    return facts.get(topic.lower(), f"I don't have a fact about {topic}.")


@agentic(model="openrouter:openai/gpt-4o")
async def simple_agent(
    question: str,
    calculate: Callable,
    get_fact: Callable,
    runtime: Runtime
) -> str:
    """
    Answer a question using calculation or fact retrieval tools.
    
    Available tools:
    - calculate(expression: str) -> float: Evaluate math expressions
    - get_fact(topic: str) -> str: Get facts about topics
    
    Return a string answer to the question.
    """
    # First call - LLM decides which tool to use or returns final answer
    result = await runtime.run()
    
    # If it's a tool call, execute it
    if runtime.is_tool_call(result):
        tool_name = runtime.get_tool_name(result)
        print(f"Using tool: {tool_name}")
        
        tool_output = await runtime.execute_tool(result)
        print(f"Tool output: {tool_output}")
        
        # Get final answer with tool result
        return await runtime.run(
            prompt=f"The {tool_name} tool returned: {tool_output}\n\nNow provide the final answer."
        )
    
    # Otherwise, we got the answer directly
    return result


async def main():
    """Run examples."""
    print("Example 1: Math question")
    answer1 = await simple_agent(
        question="What is 15 * 24 + 7?",
        calculate=calculate,
        get_fact=get_fact
    )
    print(f"Answer: {answer1}\n")
    
    print("Example 2: Fact question")
    answer2 = await simple_agent(
        question="Tell me about Python programming language.",
        calculate=calculate,
        get_fact=get_fact
    )
    print(f"Answer: {answer2}\n")
    
    print("Example 3: Direct answer (no tools needed)")
    answer3 = await simple_agent(
        question="What color is the sky?",
        calculate=calculate,
        get_fact=get_fact
    )
    print(f"Answer: {answer3}")


if __name__ == "__main__":
    asyncio.run(main())
