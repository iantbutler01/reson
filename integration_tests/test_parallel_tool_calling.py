#!/usr/bin/env python3
"""Test parallel tool calling across all providers - initially failing tests to drive implementation."""

import asyncio
import os
from typing import AsyncGenerator, List, Dict
from dataclasses import dataclass
from reson import agentic, agentic_generator, Runtime
from reson.types import Deserializable
from reson.stores import MemoryStore
from pydantic import BaseModel


class WeatherQuery(Deserializable):
    """A weather query for a location."""

    location: str
    units: str = "celsius"


class SearchQuery(Deserializable):
    """A search query with parameters."""

    text: str
    max_results: int = 5


class CalculationQuery(Deserializable):
    """A calculation with two numbers."""

    a: int
    b: int
    operation: str = "add"


def get_weather(query: WeatherQuery) -> str:
    """Get weather for a location."""
    return f"Weather in {query.location}: 22°{query.units[0].upper()}, sunny"


def search_database(query: SearchQuery) -> str:
    """Search a database."""
    return f"Found {query.max_results} results for '{query.text}'"


def calculate(query: CalculationQuery) -> int:
    """Perform a calculation."""
    if query.operation == "add":
        return query.a + query.b
    elif query.operation == "multiply":
        return query.a * query.b
    return 0


def get_current_time() -> str:
    """Get current time."""
    return "2024-08-24 18:30:00 PST"


def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    """Convert currency (mock)."""
    # Mock conversion rates
    rates = {"USD": 1.0, "EUR": 0.85, "GBP": 0.75}
    usd_amount = amount / rates.get(from_currency, 1.0)
    return usd_amount * rates.get(to_currency, 1.0)


async def test_openai_parallel_tool_calling():
    """Test OpenAI parallel tool calling - multiple tools in single response."""
    print("🧪 Testing OpenAI Parallel Tool Calling")

    @agentic_generator(model="openrouter:openai/gpt-4o", native_tools=True)
    async def parallel_agent(query: str, runtime: Runtime) -> AsyncGenerator[str, None]:
        runtime.tool(get_weather, tool_type=WeatherQuery)
        runtime.tool(search_database, tool_type=SearchQuery)
        runtime.tool(calculate, tool_type=CalculationQuery)
        runtime.tool(get_current_time)

        tools_completed = []

        async for chunk_type, content in runtime.run_stream(prompt=query):
            if chunk_type == "tool_call_complete":
                yield f"🔧 Tool completed: {content._tool_name if hasattr(content, '_tool_name') else 'unknown'}"
                tools_completed.append(
                    content._tool_name if hasattr(content, "_tool_name") else "unknown"
                )

                # Execute tool
                try:
                    result = await runtime.execute_tool(content)
                    yield f"🎯 Tool result: {result}"
                except Exception as e:
                    yield f"❌ Tool error: {e}"
            elif chunk_type == "content":
                if content:
                    yield f"📝 Content: {content}"
            elif chunk_type == "reasoning":
                yield f"🧠 Reasoning: {content}"

        yield f"📊 Total tools completed: {len(tools_completed)}"
        yield f"🔧 Tools: {', '.join(tools_completed)}"

    try:
        print("Starting OpenAI parallel test...")
        chunks = []
        tools_found = []

        async for chunk in parallel_agent(
            "I need you to: 1) Get weather for 'New York', 2) Search for 'python tutorials', 3) Calculate 15 + 27, and 4) Get current time. Use the appropriate tools for each task."
        ):
            print(chunk)
            chunks.append(chunk)

            if "Tool completed:" in chunk:
                tool_name = chunk.split("Tool completed: ")[1]
                tools_found.append(tool_name)

        print(f"\n✅ OpenAI parallel test completed with {len(chunks)} chunks")
        print(f"🎯 Tools found: {len(tools_found)} - {tools_found}")

        # This should find multiple tools (currently will only find 1 due to early return)
        if len(tools_found) >= 3:
            print("🎉 OpenAI parallel tool calling working!")
            return True
        else:
            print(f"❌ Expected multiple tools, only found {len(tools_found)}")
            return False

    except Exception as e:
        print(f"❌ OpenAI parallel test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_anthropic_parallel_tool_calling():
    """Test Anthropic parallel tool calling - multiple tool_use blocks."""
    print("\n🧪 Testing Anthropic Parallel Tool Calling")

    @agentic_generator(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def anthropic_parallel_agent(
        query: str, runtime: Runtime
    ) -> AsyncGenerator[str, None]:
        runtime.tool(get_weather, tool_type=WeatherQuery)
        runtime.tool(search_database, tool_type=SearchQuery)
        runtime.tool(get_current_time)

        tools_completed = []

        async for chunk_type, content in runtime.run_stream(prompt=query):
            if chunk_type == "tool_call_complete":
                tool_name = (
                    content._tool_name if hasattr(content, "_tool_name") else "unknown"
                )
                yield f"🔧 Anthropic tool completed: {tool_name}"
                tools_completed.append(tool_name)

                # Execute tool
                try:
                    result = await runtime.execute_tool(content)
                    yield f"🎯 Anthropic result: {result}"
                except Exception as e:
                    yield f"❌ Anthropic error: {e}"
            elif chunk_type == "content":
                if content:
                    yield f"📝 Anthropic content: {content}"

        yield f"📊 Anthropic total tools: {len(tools_completed)}"

    try:
        print("Starting Anthropic parallel test...")
        chunks = []
        tools_found = []

        async for chunk in anthropic_parallel_agent(
            "Please: 1) Get weather for 'London', 2) Search for 'machine learning', and 3) Get current time. Use the tools available."
        ):
            print(chunk)
            chunks.append(chunk)

            if "Anthropic tool completed:" in chunk:
                tool_name = chunk.split("Anthropic tool completed: ")[1]
                tools_found.append(tool_name)

        print(f"\n✅ Anthropic parallel test completed with {len(chunks)} chunks")
        print(f"🎯 Anthropic tools found: {len(tools_found)} - {tools_found}")

        # This should find multiple tools (currently will only find 1)
        if len(tools_found) >= 2:
            print("🎉 Anthropic parallel tool calling working!")
            return True
        else:
            print(f"❌ Expected multiple tools, only found {len(tools_found)}")
            return False

    except Exception as e:
        print(f"❌ Anthropic parallel test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_google_parallel_tool_calling():
    """Test Google parallel/compositional tool calling - buffered tools."""
    print("\n🧪 Testing Google Parallel Tool Calling")

    @agentic_generator(model="vertex-gemini:gemini-2.5-flash", native_tools=True)
    async def google_parallel_agent(
        query: str, runtime: Runtime
    ) -> AsyncGenerator[str, None]:
        runtime.tool(calculate, tool_type=CalculationQuery)
        runtime.tool(convert_currency)
        runtime.tool(get_weather, tool_type=WeatherQuery)

        tools_completed = []

        async for chunk_type, content in runtime.run_stream(prompt=query):
            if chunk_type == "tool_call_complete":
                tool_name = (
                    content._tool_name if hasattr(content, "_tool_name") else "unknown"
                )
                yield f"🔧 Google tool completed: {tool_name}"
                tools_completed.append(tool_name)

                # Execute tool
                try:
                    result = await runtime.execute_tool(content)
                    yield f"🎯 Google result: {result}"
                except Exception as e:
                    yield f"❌ Google error: {e}"
            elif chunk_type == "content":
                if content:
                    yield f"📝 Google content: {content}"
            elif chunk_type == "reasoning":
                yield f"🧠 Google thinking: {content}"

        yield f"📊 Google total tools: {len(tools_completed)}"

    try:
        print("Starting Google parallel test...")
        chunks = []
        tools_found = []

        async for chunk in google_parallel_agent(
            "Please: 1) Calculate 25 + 17, 2) Convert 100 USD to EUR, 3) Get weather for 'Tokyo'. Use the appropriate tools."
        ):
            print(chunk)
            chunks.append(chunk)

            if "Google tool completed:" in chunk:
                tool_name = chunk.split("Google tool completed: ")[1]
                tools_found.append(tool_name)

        print(f"\n✅ Google parallel test completed with {len(chunks)} chunks")
        print(f"🎯 Google tools found: {len(tools_found)} - {tools_found}")

        # This should find multiple tools
        if len(tools_found) >= 2:
            print("🎉 Google parallel tool calling working!")
            return True
        else:
            print(f"❌ Expected multiple tools, only found {len(tools_found)}")
            return False

    except Exception as e:
        print(f"❌ Google parallel test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_backwards_compatibility():
    """Test that existing single tool patterns still work."""
    print("\n🧪 Testing Backwards Compatibility")

    @agentic(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def single_tool_agent(query: str, runtime: Runtime) -> str:
        runtime.tool(calculate, tool_type=CalculationQuery)
        return await runtime.run(prompt=query)

    try:
        result = await single_tool_agent(
            "Calculate 10 + 5 using the calculate function"
        )
        print(f"✅ Backwards compatibility result: {result}")

        if hasattr(result, "_tool_name"):
            print(f"🔧 Single tool detected: {result._tool_name}")

            # Test existing execute_tool still works
            rt = Runtime(
                model="openrouter:anthropic/claude-sonnet-4",
                store=MemoryStore(),
                native_tools=True,
            )
            rt.tool(calculate, tool_type=CalculationQuery)

            if rt.is_tool_call(result):
                tool_result = await rt.execute_tool(result)
                print(f"🎯 Single tool result: {tool_result}")
            return True
        else:
            print(f"📝 Text response: {result}")
            return True

    except Exception as e:
        print(f"❌ Backwards compatibility test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_parallel_execution_pattern():
    """Test user pattern for handling parallel tools with async tasks."""
    print("\n🧪 Testing Parallel Execution Pattern")

    @agentic_generator(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def execution_pattern_agent(
        query: str, runtime: Runtime
    ) -> AsyncGenerator[str, None]:
        runtime.tool(get_weather, tool_type=WeatherQuery)
        runtime.tool(calculate, tool_type=CalculationQuery)
        runtime.tool(get_current_time)

        pending_tasks = []
        completed_tools = []

        async for chunk_type, content in runtime.run_stream(prompt=query):
            if chunk_type == "tool_call_complete":
                tool_name = (
                    content._tool_name if hasattr(content, "_tool_name") else "unknown"
                )
                yield f"🚀 Starting tool: {tool_name}"

                # Create async task for parallel execution
                task = asyncio.create_task(runtime.execute_tool(content))
                pending_tasks.append((tool_name, task))

            elif chunk_type == "content":
                if content:
                    yield f"📝 Content: {content}"

        # Wait for all parallel tool executions to complete
        for tool_name, task in pending_tasks:
            try:
                result = await task
                completed_tools.append(tool_name)
                yield f"✅ {tool_name}: {result}"
            except Exception as e:
                yield f"❌ {tool_name}: {e}"

        yield f"🏁 Completed {len(completed_tools)} tools in parallel"

    try:
        print("Starting parallel execution pattern test...")
        chunks = []
        parallel_tools = []

        async for chunk in execution_pattern_agent(
            "Get weather for 'Paris', calculate 20 + 22, and get current time"
        ):
            print(chunk)
            chunks.append(chunk)

            if "Starting tool:" in chunk:
                tool_name = chunk.split("Starting tool: ")[1]
                parallel_tools.append(tool_name)

        print(f"\n✅ Parallel execution test completed with {len(chunks)} chunks")
        print(f"🎯 Parallel tools handled: {len(parallel_tools)} - {parallel_tools}")

        # This demonstrates the user pattern for parallel execution
        if len(parallel_tools) >= 2:
            print("🎉 Parallel execution pattern working!")
            return True
        else:
            print(f"❌ Expected multiple tools, only found {len(parallel_tools)}")
            return False

    except Exception as e:
        print(f"❌ Parallel execution pattern test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_google_compositional_chaining():
    """Test Google compositional tool calling - tool results feed into next tool."""
    print("\n🧪 Testing Google Compositional Tool Chaining")

    @agentic_generator(model="vertex-gemini:gemini-2.5-flash", native_tools=True)
    async def compositional_agent(
        query: str, runtime: Runtime
    ) -> AsyncGenerator[str, None]:
        runtime.tool(calculate, tool_type=CalculationQuery)
        runtime.tool(convert_currency)
        runtime.tool(get_weather, tool_type=WeatherQuery)

        tools_completed = []
        tool_results = {}

        async for chunk_type, content in runtime.run_stream(prompt=query):
            if chunk_type == "tool_call_complete":
                tool_name = (
                    content._tool_name if hasattr(content, "_tool_name") else "unknown"
                )
                yield f"🔗 Compositional tool: {tool_name}"
                tools_completed.append(tool_name)

                # Execute tool and store result for potential chaining
                try:
                    result = await runtime.execute_tool(content)
                    tool_results[tool_name] = result
                    yield f"⚡ Chained result: {tool_name} -> {result}"
                except Exception as e:
                    yield f"❌ Chain error: {e}"
            elif chunk_type == "content":
                if content:
                    yield f"📝 Google content: {content}"

        yield f"🔗 Compositional chain: {len(tools_completed)} tools"
        yield f"📊 Chain results: {tool_results}"

    try:
        print("Starting Google compositional test...")
        chunks = []
        chain_tools = []

        async for chunk in compositional_agent(
            "Calculate 50 + 30, then convert that result from USD to EUR, then get weather for 'Berlin'"
        ):
            print(chunk)
            chunks.append(chunk)

            if "Compositional tool:" in chunk:
                tool_name = chunk.split("Compositional tool: ")[1]
                chain_tools.append(tool_name)

        print(f"\n✅ Google compositional test completed with {len(chunks)} chunks")
        print(f"🔗 Compositional tools: {len(chain_tools)} - {chain_tools}")

        # This tests tool chaining capability
        if len(chain_tools) >= 2:
            print("🎉 Google compositional chaining working!")
            return True
        else:
            print(f"❌ Expected multiple chained tools, only found {len(chain_tools)}")
            return False

    except Exception as e:
        print(f"❌ Google compositional test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_mixed_parallel_tool_types():
    """Test parallel tools with mixed registration types."""
    print("\n🧪 Testing Mixed Parallel Tool Types")

    @agentic_generator(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def mixed_parallel_agent(
        query: str, runtime: Runtime
    ) -> AsyncGenerator[str, None]:
        # Mix of tools with and without tool_type
        runtime.tool(calculate, tool_type=CalculationQuery)  # With type
        runtime.tool(get_current_time)  # Without type
        runtime.tool(get_weather, tool_type=WeatherQuery)  # With type

        tools_completed = []

        async for chunk_type, content in runtime.run_stream(prompt=query):
            if chunk_type == "tool_call_complete":
                tool_name = (
                    content._tool_name if hasattr(content, "_tool_name") else "unknown"
                )
                yield f"🔀 Mixed tool: {tool_name}"
                tools_completed.append(tool_name)

                # Execute tool
                try:
                    result = await runtime.execute_tool(content)
                    yield f"🎭 Mixed result: {result}"
                except Exception as e:
                    yield f"❌ Mixed error: {e}"
            elif chunk_type == "content":
                if content:
                    yield f"📝 Mixed content: {content}"

        yield f"🔀 Mixed tools total: {len(tools_completed)}"

    try:
        print("Starting mixed parallel test...")
        chunks = []
        mixed_tools = []

        async for chunk in mixed_parallel_agent(
            "Calculate 8 * 7, get current time, and get weather for 'San Francisco'"
        ):
            print(chunk)
            chunks.append(chunk)

            if "Mixed tool:" in chunk:
                tool_name = chunk.split("Mixed tool: ")[1]
                mixed_tools.append(tool_name)

        print(f"\n✅ Mixed parallel test completed with {len(chunks)} chunks")
        print(f"🔀 Mixed tools: {len(mixed_tools)} - {mixed_tools}")

        # Should handle mixed tool registration types
        if len(mixed_tools) >= 2:
            print("🎉 Mixed parallel tool types working!")
            return True
        else:
            print(f"❌ Expected multiple mixed tools, only found {len(mixed_tools)}")
            return False

    except Exception as e:
        print(f"❌ Mixed parallel test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all parallel tool calling tests - initially expected to fail."""
    print("🚀 Testing Parallel Tool Calling (Initially Expected to Fail)")
    print("=" * 80)

    # Check environment
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not set")
        return

    print("✅ API key configured")
    print(
        "⚠️  These tests are expected to FAIL initially due to early return limitations"
    )
    print("⚠️  After implementation, they should all PASS")

    # Run parallel tool calling tests
    tests = [
        ("OpenAI Parallel Tools", test_openai_parallel_tool_calling()),
        ("Anthropic Parallel Tools", test_anthropic_parallel_tool_calling()),
        ("Google Parallel/Compositional", test_google_parallel_tool_calling()),
        ("Backwards Compatibility", test_backwards_compatibility()),
        ("Mixed Parallel Types", test_mixed_parallel_tool_types()),
        ("Google Compositional Chaining", test_google_compositional_chaining()),
    ]

    results = []
    for test_name, test_coro in tests:
        try:
            result = await test_coro
            results.append(result)
            status = "PASSED" if result else "FAILED"
            print(f"{'✅' if result else '❌'} {test_name}: {status}")
        except Exception as e:
            print(f"❌ {test_name}: EXCEPTION - {e}")
            results.append(False)

    success_count = sum(1 for r in results if r)
    print(f"\n📊 Parallel Tool Test Results: {success_count}/{len(results)} passed")

    if success_count == len(results):
        print("🎉 All parallel tool tests passed! Implementation complete!")
    else:
        print("⚠️  Some parallel tool tests failed - implementation needed.")
        print(
            "💡 Expected behavior: Tests should fail initially, then pass after fixing early returns"
        )

    return success_count == len(results)


if __name__ == "__main__":
    asyncio.run(main())
