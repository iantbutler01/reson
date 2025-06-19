"""
Example showing the @agentic decorator with Google GenAI.
"""

import asyncio
from pydantic import BaseModel
from typing import Callable

from reson.reson import agentic, Runtime


def get_fact(topic: str) -> str:
    """Get a fact about a topic."""
    facts = {
        "python": "Python was created by Guido van Rossum in 1991.",
        "math": "Pi is approximately 3.14159.",
        "space": "The universe is about 13.8 billion years old.",
    }
    return facts.get(topic.lower(), f"I don't have a fact about {topic}.")


@agentic(model="google/gemini-flash-1.5")
async def simple_agent(
    question: str,
    get_fact: Callable,
    runtime: Runtime
) -> str:
    """
    Answer a question using the get_fact tool.
    
    Available tools:
    - get_fact(topic: str) -> str: Get facts about topics
    
    Return a string answer to the question.
    """
    result = await runtime.run()
    
    if runtime.is_tool_call(result):
        tool_name = runtime.get_tool_name(result)
        print(f"Using tool: {tool_name}")
        
        tool_output = await runtime.execute_tool(result)
        print(f"Tool output: {tool_output}")
        
        return await runtime.run(
            prompt=f"The {tool_name} tool returned: {tool_output}\n\nNow provide the final answer."
        )
    
    return result


async def main():
    """Run examples."""
    print("Example: Fact question")
    answer = await simple_agent(
        question="Tell me about Python programming language.",
        get_fact=get_fact
    )
    print(f"Answer: {answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
