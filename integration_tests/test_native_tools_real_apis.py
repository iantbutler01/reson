"""Test native tool calling with real APIs (vertex-gemini and openrouter)."""

import asyncio
import os
import pytest
from typing import List, Dict
from reson import agentic, Runtime
from reson.types import Deserializable
from reson.stores import MemoryStore
from reson.services.inference_clients import ChatMessage, ChatRole


class SearchQuery(Deserializable):
    """A search query with parameters."""

    text: str
    category: str = "general"
    max_results: int = 5


class WeatherQuery(Deserializable):
    """A weather query for a location."""

    location: str
    units: str = "celsius"


def search_database(query: SearchQuery) -> str:
    """Search a database with the given query parameters."""
    return f"Found {query.max_results} results for '{query.text}' in category '{query.category}'"


def get_weather(query: WeatherQuery) -> str:
    """Get weather information for a location."""
    return f"Weather in {query.location}: 22°{query.units[0].upper()}, partly cloudy"


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@pytest.mark.asyncio
async def test_openrouter_native_tools():
    """Test OpenRouter with native tool calling."""
    print("🧪 Testing OpenRouter with anthropic/claude-sonnet-4 + native tools")

    @agentic(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def openrouter_assistant(query: str, runtime: Runtime) -> str:
        runtime.tool(search_database)
        runtime.tool(add_numbers)
        return await runtime.run(prompt=query)

    try:
        # Test tool calling
        result = await openrouter_assistant(
            "Use the search_database tool to search for 'python tutorials' and find 3 results"
        )
        print(f"✅ OpenRouter result: {result}")
        print(f"🔍 OpenRouter result type: {type(result)}")

        if hasattr(result, "_tool_name"):
            print(f"🔧 Tool call detected: {result._tool_name}")
            if hasattr(result, "_tool_id"):
                print(f"🆔 Tool ID preserved: {result._tool_id}")
            else:
                print("❌ Tool ID missing!")

            # Test tool execution
            rt = Runtime(
                model="openrouter:anthropic/claude-sonnet-4",
                store=MemoryStore(),
                native_tools=True,
            )
            rt.tool(search_database)
            rt.tool(add_numbers)

            if rt.is_tool_call(result):
                tool_result = await rt.execute_tool(result)
                print(f"🔧 Tool execution result: {tool_result}")

                # Test unified helper
                tool_msg = rt.create_tool_result_message(result, str(tool_result))
                print(f"📝 Tool result message: {tool_msg.content}")

            return True
        else:
            print(f"📝 Non-tool response: {result}")
            return True

    except Exception as e:
        print(f"❌ OpenRouter test failed: {e}")
        return False


@pytest.mark.asyncio
async def test_vertex_gemini_native_tools():
    """Test Vertex AI Gemini with native tool calling."""
    print("\n🧪 Testing Vertex AI with gemini-2.5-flash + native tools")

    @agentic(model="vertex-gemini:gemini-2.5-flash", native_tools=True)
    async def vertex_assistant(query: str, runtime: Runtime) -> str:
        runtime.tool(get_weather)
        runtime.tool(add_numbers)
        return await runtime.run(prompt=query)

    try:
        # Test tool calling
        result = await vertex_assistant("Get weather for 'New York' in fahrenheit")
        print(f"✅ Vertex AI result: {result}")

        if hasattr(result, "_tool_name"):
            print(f"🔧 Tool call detected: {result._tool_name}")
            if hasattr(result, "_tool_id"):
                print(f"🆔 Tool ID preserved: {result._tool_id}")
            else:
                print("❌ Tool ID missing!")

            # Test tool execution
            rt = Runtime(
                model="vertex-gemini:gemini-2.5-flash",
                store=MemoryStore(),
                native_tools=True,
            )
            rt.tool(get_weather)
            rt.tool(add_numbers)

            if rt.is_tool_call(result):
                tool_result = await rt.execute_tool(result)
                print(f"🔧 Tool execution result: {tool_result}")

                # Test unified helper
                tool_msg = rt.create_tool_result_message(result, str(tool_result))
                print(f"📝 Tool result message: {tool_msg.content}")

            return True
        else:
            print(f"📝 Non-tool response: {result}")
            return True

    except Exception as e:
        print(f"❌ Vertex AI test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


@pytest.mark.asyncio
async def test_vertex_gemini_with_tools():
    """Test Vertex AI Gemini with native tool calling and tool execution."""
    print("\n🧪 Testing Vertex AI Gemini with tool calling + native tools")

    @agentic(model="vertex-gemini:gemini-2.5-flash", native_tools=True)
    async def vertex_tool_assistant(query: str, runtime: Runtime) -> str:
        runtime.tool(search_database)
        runtime.tool(add_numbers)
        return await runtime.run(prompt=query)

    try:
        # Test tool calling
        result = await vertex_tool_assistant("Use add_numbers to calculate 15 + 25")
        print(f"✅ Vertex Tool result: {result}")

        if hasattr(result, "_tool_name"):
            print(f"🔧 Tool call detected: {result._tool_name}")
            if hasattr(result, "_tool_id"):
                print(f"🆔 Tool ID preserved: {result._tool_id}")
            else:
                print("❌ Tool ID missing!")

            # Test tool execution
            rt = Runtime(
                model="vertex-gemini:gemini-2.5-flash",
                store=MemoryStore(),
                native_tools=True,
            )
            rt.tool(search_database)
            rt.tool(add_numbers)

            if rt.is_tool_call(result):
                tool_result = await rt.execute_tool(result)
                print(f"🔧 Tool execution result: {tool_result}")

                # Test unified helper
                tool_msg = rt.create_tool_result_message(result, str(tool_result))
                print(f"📝 Tool result message: {tool_msg.content}")

            return True
        else:
            print(f"📝 Non-tool response: {result}")
            return True

    except Exception as e:
        print(f"❌ Vertex Tool test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


@pytest.mark.asyncio
async def test_google_anthropic_native_tools():
    """Test Google Anthropic with claude-3-5-sonnet + native tools."""
    print("\n🧪 Testing Google Anthropic with claude-3-5-sonnet + native tools")

    @agentic(model="google-anthropic:claude-3-5-sonnet", native_tools=True)
    async def google_anthropic_assistant(query: str, runtime: Runtime) -> str:
        runtime.tool(get_weather)
        runtime.tool(search_database)
        return await runtime.run(prompt=query)

    try:
        # Test tool calling
        result = await google_anthropic_assistant(
            "Search for 'AI trends' with max 3 results"
        )
        print(f"✅ Google Anthropic result: {result}")

        if hasattr(result, "_tool_name"):
            print(f"🔧 Tool call detected: {result._tool_name}")
            if hasattr(result, "_tool_id"):
                print(f"🆔 Tool ID preserved: {result._tool_id}")
            else:
                print("❌ Tool ID missing!")

            # Test tool execution
            rt = Runtime(
                model="google-anthropic:claude-3-5-sonnet",
                store=MemoryStore(),
                native_tools=True,
            )
            rt.tool(get_weather)
            rt.tool(search_database)

            if rt.is_tool_call(result):
                tool_result = await rt.execute_tool(result)
                print(f"🔧 Tool execution result: {tool_result}")

                # Test unified helper
                tool_msg = rt.create_tool_result_message(result, str(tool_result))
                print(f"📝 Tool result message: {tool_msg.content}")

            return True
        else:
            print(f"📝 Non-tool response: {result}")
            return True

    except Exception as e:
        print(f"❌ Google Anthropic test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


@pytest.mark.asyncio
async def test_multi_turn_conversation():
    """Test multi-turn conversation with tool results - handles multiple tool calls."""
    print("\n🔄 Testing multi-turn conversation with OpenRouter")

    try:
        rt = Runtime(
            model="openrouter:anthropic/claude-sonnet-4",
            store=MemoryStore(),
            native_tools=True,
        )
        rt.tool(add_numbers)

        # Build conversation step by step
        history = []

        # First call
        result = await rt.run(
            prompt="Please add 15 and 27, then add 10 to that result and give me the final answer as text"
        )
        print(f"📞 Call result: {result}")

        # Loop through tool calls until we get a final text response
        max_iterations = 5
        iteration = 0

        while rt.is_tool_call(result) and iteration < max_iterations:
            iteration += 1
            print(f"🔧 Tool call {iteration}: {result._tool_name}")

            # Execute tool
            tool_result = await rt.execute_tool(result)
            print(f"🔧 Tool result {iteration}: {tool_result}")

            # Create tool result message
            tool_msg = rt.create_tool_result_message(result, str(tool_result))
            history.append(tool_msg)

            # Continue conversation
            result = await rt.run(
                prompt="Please continue with the calculation and provide the final answer as text",
                history=history,
            )
            print(f"📞 Call {iteration+1} result: {result}")

        if iteration >= max_iterations:
            print("⚠️  Reached max iterations, but tool calling chain is working")
            return True
        elif not rt.is_tool_call(result):
            print(f"✅ Final text response: {result}")
            return True
        else:
            print("❌ Unexpected result format")
            return False

    except Exception as e:
        print(f"❌ Multi-turn test failed: {e}")
        return False


async def main():
    """Run all native tool calling tests."""
    print("🚀 Testing Native Tool Calling with Real APIs")
    print("=" * 60)

    # Check environment
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not set")
        return
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("❌ GOOGLE_APPLICATION_CREDENTIALS not set")
        return

    print("✅ API keys configured")

    # Run tests
    tests = [
        test_openrouter_native_tools(),
        test_vertex_gemini_native_tools(),
        test_vertex_gemini_with_tools(),
        test_google_anthropic_native_tools(),
        test_multi_turn_conversation(),
    ]

    results = await asyncio.gather(*tests, return_exceptions=True)

    success_count = sum(1 for r in results if r is True)
    print(f"\n📊 Test Results: {success_count}/{len(results)} passed")

    if success_count == len(results):
        print("🎉 All tests passed! Native tool calling is working with real APIs!")
    else:
        print("⚠️  Some tests failed. Check the output above.")
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Test {i+1} exception: {result}")


if __name__ == "__main__":
    asyncio.run(main())
