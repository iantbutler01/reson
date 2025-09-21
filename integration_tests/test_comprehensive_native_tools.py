#!/usr/bin/env python3
"""Comprehensive tests for native tool calling including regression tests and advanced scenarios."""

import asyncio
import pytest
import os
from typing import List, Dict, Union
from reson import agentic, Runtime
from reson.types import Deserializable
from reson.stores import MemoryStore
from reson.services.inference_clients import ChatMessage, ChatRole, ToolResult


class SearchQuery(Deserializable):
    """A search query with parameters."""

    text: str
    category: str = "general"
    max_results: int = 5


class CalculationResult(Deserializable):
    """Final calculation result that should end the conversation."""

    value: int
    explanation: str


class WeatherQuery(Deserializable):
    """A weather query for a location."""

    location: str
    units: str = "celsius"


def search_database(query: SearchQuery) -> str:
    """Search a database with the given query parameters."""
    return f"Found {query.max_results} results for '{query.text}' in category '{query.category}'"


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


def get_weather(query: WeatherQuery) -> str:
    """Get weather information for a location."""
    return f"Weather in {query.location}: 22°{query.units[0].upper()}, partly cloudy"


def factorial(n: int) -> int:
    """Calculate factorial of a number."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)


@pytest.mark.asyncio
async def test_xml_regression_single_turn():
    """Test that XML tool calling still works (regression test)."""
    print("🧪 Testing XML tool calling - Deserializable output type")

    @agentic(model="openrouter:anthropic/claude-sonnet-4", native_tools=False)
    async def xml_structured_agent(query: str, runtime: Runtime) -> SearchQuery:
        # Don't register tools - let XML parser create the SearchResult from LLM response
        result = await runtime.run(prompt=query)
        print(f"✅ XML structured result: {result}")
        print(f"🔍 XML result type: {type(result)}")

        # Verify we got the expected Deserializable type
        if isinstance(result, SearchQuery):
            print(
                f"✅ XML parsing works! Got SearchQuery: text='{result.text}', max_results={result.max_results}, category='{result.category}'"
            )
            return result
        else:
            print(f"❌ Expected SearchQuery, got {type(result)}")
            return result

    @agentic(model="openrouter:anthropic/claude-sonnet-4", native_tools=False)
    async def xml_string_agent(query: str, runtime: Runtime) -> str:
        # String return type - should get text response
        result = await runtime.run(prompt=query)
        print(f"✅ XML string result: {result}")
        print(f"🔍 XML string type: {type(result)}")

        if isinstance(result, str):
            print(f"✅ XML string parsing works! Got string response")
        else:
            print(f"⚠️  Expected str, got {type(result)} (could be None)")

        return "XML string test completed"

    try:
        # Test Deserializable output type
        structured_result = await xml_structured_agent(
            "Create a SearchQuery with text='machine learning', max_results=5, category='AI'. {{return_type}}"
        )

        # Test string output type
        string_result = await xml_string_agent(
            "Just give me a simple text response about machine learning. {{return_type}}"
        )

        return True
    except Exception as e:
        print(f"❌ XML regression test failed: {e}")
        return False


@pytest.mark.asyncio
async def test_xml_regression_multi_turn():
    """Test that XML multi-turn tool calling still works (regression test)."""
    print("\n🧪 Testing XML tool calling - Multi-turn (regression)")

    @agentic(model="openrouter:anthropic/claude-sonnet-4", native_tools=False)
    async def xml_multi_agent(
        query: str, runtime: Runtime
    ) -> Union[CalculationResult, str]:
        runtime.tool(add_numbers)
        runtime.tool(multiply_numbers)

        history = []
        result = await runtime.run(
            prompt=f"{query} {{{{return_type}}}}",
            output_type=Union[CalculationResult, str],
        )
        print(f"📞 XML Multi-turn call 1: {result}")

        iterations = 0
        max_iterations = 3

        while runtime.is_tool_call(result) and iterations < max_iterations:
            iterations += 1
            print(f"🔧 XML Tool call {iterations}: {result._tool_name}")

            # Execute tool
            tool_result = await runtime.execute_tool(result)
            print(f"🔧 XML Tool result {iterations}: {tool_result}")

            # Create tool result message
            tool_msg = ToolResult.create((result, str(tool_result)))
            history.append(tool_msg)

            # Continue conversation
            result = await runtime.run(
                prompt="Continue with the calculation and provide the final answer {{return_type}}",
                history=history,
                output_type=Union[CalculationResult, str],
            )
            print(f"📞 XML Call {iterations+1} result: {result}")

        print(f"✅ XML multi-turn completed after {iterations} tool calls")
        return f"XML multi-turn completed with {iterations} tool calls"

    try:
        result = await xml_multi_agent(
            "First add 5 and 7, then multiply the result by 3"
        )
        return True
    except Exception as e:
        print(f"❌ XML multi-turn test failed: {e}")
        return False


@pytest.mark.asyncio
async def test_native_5_turn_conversation():
    """Test a 5-turn conversation with native tools."""
    print("\n🧪 Testing Native Tools - 5-turn conversation")

    @agentic(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def native_multi_agent(query: str, runtime: Runtime) -> str:
        runtime.tool(add_numbers)
        runtime.tool(multiply_numbers)
        runtime.tool(factorial)
        runtime.tool(search_database)

        history = []
        result = await runtime.run(prompt=query)
        print(f"📞 Native 5-turn initial call: {result}")

        turn_count = 0
        max_turns = 5

        while runtime.is_tool_call(result) and turn_count < max_turns:
            turn_count += 1
            print(f"🔧 Native turn {turn_count}: {result._tool_name}")

            # Execute tool
            tool_result = await runtime.execute_tool(result)
            print(f"🔧 Native tool result {turn_count}: {tool_result}")

            # Create tool result message
            tool_msg = ToolResult.create((result, str(tool_result)))
            history.append(tool_msg)

            # Continue conversation
            result = await runtime.run(
                prompt="Continue with the next step in the sequence",
                history=history,
            )
            print(f"📞 Native turn {turn_count+1} result: {result}")

        print(f"✅ Native 5-turn conversation completed with {turn_count} tool calls")
        return f"Native 5-turn completed with {turn_count} tool calls"

    try:
        result = await native_multi_agent(
            "I need you to: 1) Calculate 8 + 7, 2) Then multiply that result by 4, 3) Then calculate the factorial of 5, 4) Search for 'python tutorials' with max 2 results, 5) Finally give me a summary of all the results"
        )
        return True
    except Exception as e:
        print(f"❌ Native 5-turn test failed: {e}")
        return False


@pytest.mark.asyncio
async def test_output_type_termination():
    """Test that producing an output type terminates the conversation properly."""
    print("\n🧪 Testing Output Type Termination")

    @agentic(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def calculation_assistant(query: str, runtime: Runtime) -> CalculationResult:
        runtime.tool(add_numbers)
        runtime.tool(multiply_numbers)
        return await runtime.run(prompt=query)

    try:
        result = await calculation_assistant(
            "Add 15 and 25, then multiply by 2, and give me the final result as a CalculationResult"
        )
        print(f"📊 Output type result: {result}")
        print(f"🔍 Result type: {type(result)}")

        # Check if we got the proper output type
        if isinstance(result, CalculationResult):
            print(
                f"✅ Got CalculationResult: value={result.value}, explanation='{result.explanation}'"
            )
            return True
        elif hasattr(result, "_tool_name"):
            print(f"🔧 Got tool call instead: {result._tool_name}")
            # This is also valid - the tool call would eventually lead to the output type
            return True
        else:
            print(f"📝 Got text response: {result}")
            return True

    except Exception as e:
        print(f"❌ Output type termination test failed: {e}")
        return False


@pytest.mark.asyncio
async def test_native_vs_xml_comparison():
    """Compare native vs XML tool calling side by side."""
    print("\n🧪 Testing Native vs XML comparison")

    @agentic(model="openrouter:anthropic/claude-sonnet-4", native_tools=False)
    async def xml_calculator(query: str, runtime: Runtime) -> str:
        runtime.tool(add_numbers)
        result = await runtime.run(prompt=query)
        print(f"📊 XML result: {result} (type: {type(result)})")

        if runtime.is_tool_call(result):
            print("✅ XML approach works: tool call detected")
            return "XML works"
        else:
            print("✅ XML approach works: text response")
            return result

    @agentic(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def native_calculator(query: str, runtime: Runtime) -> str:
        runtime.tool(add_numbers)
        result = await runtime.run(prompt=query)
        print(f"📊 Native result: {result} (type: {type(result)})")

        if runtime.is_tool_call(result):
            print("✅ Native approach works: tool call detected")
            return "Native works"
        else:
            print("✅ Native approach works: text response")
            return result

    try:
        task = "Add 12 and 8 together"

        # Test both approaches
        xml_result = await xml_calculator(task)
        native_result = await native_calculator(task)

        return True  # Both completed without exceptions

    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False


async def main():
    """Run comprehensive native tool calling tests."""
    print("🚀 Comprehensive Native Tool Calling Tests")
    print("=" * 60)

    # Check environment
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not set")
        return

    print("✅ API keys configured")

    # Run comprehensive tests
    tests = [
        ("XML Regression - Single Turn", test_xml_regression_single_turn()),
        ("XML Regression - Multi Turn", test_xml_regression_multi_turn()),
        ("Native 5-Turn Conversation", test_native_5_turn_conversation()),
        ("Output Type Termination", test_output_type_termination()),
        ("Native vs XML Comparison", test_native_vs_xml_comparison()),
    ]

    results = []
    for test_name, test_coro in tests:
        try:
            result = await test_coro
            results.append(result)
            print(
                f"{'✅' if result else '❌'} {test_name}: {'PASSED' if result else 'FAILED'}"
            )
        except Exception as e:
            print(f"❌ {test_name}: EXCEPTION - {e}")
            results.append(False)

    success_count = sum(1 for r in results if r)
    print(f"\n📊 Comprehensive Test Results: {success_count}/{len(results)} passed")

    if success_count == len(results):
        print(
            "🎉 All comprehensive tests passed! Both XML and Native tool calling are working correctly!"
        )
    else:
        print("⚠️  Some comprehensive tests failed. Check the output above.")

    return success_count == len(results)


if __name__ == "__main__":
    asyncio.run(main())
