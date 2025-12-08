"""
Example: Tool Call Hydration with Real LLM

Demonstrates how tool handlers receive typed instances (Pydantic, dataclass)
instead of raw dictionaries when tool_type is registered.

Run with:
    OPENROUTER_API_KEY=xxx python examples/tool_hydration_example.py
"""

import asyncio
from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel

from reson.reson import agentic, Runtime


# =============================================================================
# Tool Input Types - Pydantic for validation
# =============================================================================

class WeatherQuery(BaseModel):
    """Query for weather information."""
    location: str
    units: Optional[str] = "celsius"


class MathOperation(BaseModel):
    """A math operation with two numbers."""
    a: int
    b: int


# =============================================================================
# Tool Handlers - receive TYPED instances, not dicts!
# =============================================================================

async def get_weather(query: WeatherQuery) -> str:
    """Handler receives WeatherQuery instance with typed attributes."""
    print(f"  📍 get_weather called with: location={query.location}, units={query.units}")

    weather_data = {
        "tokyo": {"temp": 25, "condition": "cloudy"},
        "new york": {"temp": 22, "condition": "sunny"},
        "london": {"temp": 15, "condition": "rainy"},
        "paris": {"temp": 18, "condition": "partly cloudy"},
    }

    city = query.location.lower()
    data = weather_data.get(city, {"temp": 20, "condition": "unknown"})

    temp = data["temp"]
    if query.units == "fahrenheit":
        temp = int(temp * 9/5 + 32)

    unit_symbol = "°F" if query.units == "fahrenheit" else "°C"
    return f"Weather in {query.location}: {temp}{unit_symbol}, {data['condition']}"


def add_numbers(op: MathOperation) -> str:
    """Handler receives MathOperation instance."""
    print(f"  🔢 add_numbers called with: a={op.a}, b={op.b}")
    return f"{op.a} + {op.b} = {op.a + op.b}"


def multiply_numbers(op: MathOperation) -> str:
    """Handler receives MathOperation instance."""
    print(f"  ✖️  multiply_numbers called with: a={op.a}, b={op.b}")
    return f"{op.a} × {op.b} = {op.a * op.b}"


# =============================================================================
# Agent with typed tools
# =============================================================================

@agentic(model="openrouter:anthropic/claude-sonnet-4")
async def assistant(request: str, runtime: Runtime) -> str:
    """
    A helpful assistant with weather and math tools.

    Tools available:
    - get_weather(location, units): Get weather for a city
    - add_numbers(a, b): Add two numbers
    - multiply_numbers(a, b): Multiply two numbers

    Help the user with their request: {{request}}
    """
    # Register tools WITH their types for automatic hydration
    runtime.tool(get_weather, name="get_weather", tool_type=WeatherQuery)
    runtime.tool(add_numbers, name="add_numbers", tool_type=MathOperation)
    runtime.tool(multiply_numbers, name="multiply_numbers", tool_type=MathOperation)

    result = await runtime.run()

    # Tool loop
    while runtime.is_tool_call(result):
        tool_name = runtime.get_tool_name(result)
        print(f"🔧 LLM requested tool: {tool_name}")

        # execute_tool hydrates JSON args into typed instance automatically
        tool_output = await runtime.execute_tool(result)
        print(f"✅ Tool result: {tool_output}")

        result = await runtime.run(prompt=f"Tool returned: {tool_output}. Respond to the user.")

    return result


async def main():
    print("=" * 60)
    print("Tool Hydration Example - Real LLM Calls")
    print("=" * 60)

    requests = [
        "What's the weather like in Tokyo?",
        "What is 42 plus 17?",
        "Multiply 15 by 8",
    ]

    for request in requests:
        print(f"\n👤 User: {request}")
        print("-" * 40)
        response = await assistant(request=request)
        print(f"\n🤖 Assistant: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
