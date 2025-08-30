#!/usr/bin/env python3
"""
ToolResult Examples - Proper usage patterns with signature preservation.

Shows the three main usage patterns:
1. @agentic with runtime.run (non-streaming)
2. @agentic with runtime.run_stream (streaming)
3. @agentic_generator with runtime.run_stream (streaming generator)
"""

import asyncio
from typing import List, AsyncGenerator
from reson import agentic, agentic_generator, Runtime
from reson.types import Deserializable


class WeatherQuery(Deserializable):
    location: str
    units: str = "celsius"


class CalculationRequest(Deserializable):
    operation: str
    a: int
    b: int


def get_weather(query: WeatherQuery) -> str:
    """Get weather information for a location."""
    return f"Weather in {query.location}: 22°{query.units[0].upper()}, partly cloudy"


def calculate_numbers(calc: CalculationRequest) -> str:
    """Perform basic calculations."""
    if calc.operation == "add":
        result = calc.a + calc.b
    elif calc.operation == "multiply":
        result = calc.a * calc.b
    else:
        result = 0
    return f"{calc.a} {calc.operation} {calc.b} = {result}"


# Example 1: @agentic with runtime.run (non-streaming)
@agentic(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
async def simple_agent(query: str, runtime: Runtime) -> str:
    """Simple agent using runtime.run - demonstrates ToolResult creation and usage."""
    runtime.tool(get_weather)
    runtime.tool(calculate_numbers)

    # Single call that may return tool calls or final result
    result = await runtime.run(prompt=query)

    # Handle tool execution loop
    history = []
    max_iterations = 3
    iteration = 0

    while runtime.is_tool_call(result) and iteration < max_iterations:
        iteration += 1
        print(f"🔧 Tool call {iteration}: {result._tool_name}")

        # Execute tool
        tool_result = await runtime.execute_tool(result)
        print(f"✅ Tool executed: {tool_result}")

        # Create ToolResult - preserves signatures and tool_obj!
        tool_result_msg = runtime.create_tool_result_message(result, str(tool_result))

        # Handle different return types
        if hasattr(tool_result_msg, "tool_use_id"):  # Single ToolResult
            history.append(tool_result_msg)
            print(f"📦 ToolResult created: tool_use_id={tool_result_msg.tool_use_id}")
            print(f"   signature: {tool_result_msg.signature}")
        elif isinstance(tool_result_msg, list):  # Multiple ToolResults
            history.extend(tool_result_msg)
            print(f"📦 {len(tool_result_msg)} ToolResults created")
        else:  # ChatMessage (XML mode)
            history.append(tool_result_msg)
            print(f"📦 ChatMessage created (XML mode)")

        # Continue conversation with ToolResult in history
        result = await runtime.run(
            prompt="Continue processing and provide final answer", history=history
        )

    return result


# Example 2: @agentic with runtime.run_stream (streaming)
@agentic(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
async def streaming_agent(query: str, runtime: Runtime) -> str:
    """Streaming agent using runtime.run_stream - shows real-time ToolResult handling."""
    runtime.tool(get_weather)
    runtime.tool(calculate_numbers)

    history = []
    final_result = ""

    print(f"🚀 Streaming query: {query}")

    async for chunk_type, chunk_value in runtime.run_stream(
        prompt=query, max_tokens=2042
    ):
        print(chunk_type)
        if chunk_type == "reasoning":
            print(f"🧠 Reasoning: {chunk_value[-30:]}...")
        elif chunk_type == "content":
            print(f"💬 {chunk_value}")
            final_result += chunk_value
        elif chunk_type == "tool_call_partial":
            pass
        elif chunk_type == "tool_call_complete":
            tool_call = chunk_value
            print(f"🔧 Tool call: {tool_call._tool_name}")

            # Execute and create ToolResult
            tool_execution = await runtime.execute_tool(tool_call)
            tool_result = runtime.create_tool_result_message(
                tool_call, str(tool_execution)
            )

            # Handle different return types
            if hasattr(tool_result, "tool_use_id"):  # Single ToolResult
                print(
                    f"📦 ToolResult: {tool_result.tool_use_id}, signature={tool_result.signature}"
                )
                history.append(tool_result)
            elif isinstance(tool_result, list):  # Multiple ToolResults
                print(f"📦 {len(tool_result)} ToolResults created")
                history.extend(tool_result)
            else:  # ChatMessage (XML mode)
                print(f"📦 ChatMessage created")
                history.append(tool_result)

            # Continue with ToolResult in history
            print(f"🔄 Continuing with ToolResult in history...")
            async for chunk_type, chunk_value in runtime.run_stream(
                prompt="Provide summary", history=history
            ):
                if chunk_type == "content":
                    print(f"📝 {chunk_value}")
                    final_result += chunk_value
            break

    return final_result


# Example 3: @agentic_generator with runtime.run_stream (streaming generator)
@agentic_generator(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
async def streaming_generator_agent(
    query: str, runtime: Runtime
) -> AsyncGenerator[str, None]:
    """Generator agent using runtime.run_stream - yields ToolResult updates as they happen."""
    runtime.tool(get_weather)
    runtime.tool(calculate_numbers)

    history = []

    yield f"🚀 Starting: {query}\n"

    async for chunk_type, chunk_value in runtime.run_stream(prompt=query):
        if chunk_type == "reasoning":
            yield f"🧠 Reasoning: {chunk_value[-20:]}...\n"
        elif chunk_type == "content":
            yield f"💬 {chunk_value}\n"
        elif chunk_type == "tool_call_complete":
            tool_call = chunk_value
            yield f"🔧 Tool call: {tool_call._tool_name}\n"

            # Execute and create ToolResult
            tool_execution = await runtime.execute_tool(tool_call)
            tool_result = runtime.create_tool_result_message(
                tool_call, str(tool_execution)
            )

            yield f"📦 ToolResult created: {tool_result.tool_use_id}\n"
            yield f"   signature: {tool_result.signature}\n"
            yield f"   tool_obj preserved: {tool_result.tool_obj is not None}\n"

            history.append(tool_result)

            # Continue with ToolResult in history
            yield f"🔄 Continuing with ToolResult in history...\n"
            async for chunk_type, chunk_value in runtime.run_stream(
                prompt="Summarize results", history=history
            ):
                if chunk_type == "content":
                    yield f"📝 {chunk_value}\n"
            break


async def main():
    """Run all three example patterns."""
    # print("🚀 ToolResult Examples - All Three Patterns")
    # print("=" * 60)

    # print("\n📋 Example 1: @agentic with runtime.run")
    # result1 = await simple_agent(
    #     "Use the get_weather tool to get weather for Tokyo in fahrenheit"
    # )
    # print(f"Result: {result1[:100]}...\n")

    print("\n📋 Example 2: @agentic with runtime.run_stream")
    result2 = await streaming_agent(
        "IMMEDIATELY call the calculate_numbers tool with operation='add', a=25, b=17. Do NOT explain, just call the tool. Call get weather too, I want to see multiple tool calls."
    )
    print(f"Final result: {result2}...\n")

    # print("\n📋 Example 3: @agentic_generator with runtime.run_stream")
    # async for output in streaming_generator_agent(
    #     "IMMEDIATELY call the get_weather tool with location='London', units='celsius'. Do NOT explain first."
    # ):
    #     print(output.strip())

    # print("\n🎉 All three ToolResult patterns demonstrated!")


if __name__ == "__main__":
    asyncio.run(main())
