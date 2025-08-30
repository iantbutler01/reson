#!/usr/bin/env python3
"""Test tool call format validation - ensuring correct formats are returned based on registration."""

import asyncio
import pytest
import os
from typing import AsyncGenerator, List, Dict, Any
from reson import agentic_generator, Runtime
from reson.types import Deserializable
from reson.stores import MemoryStore


class SearchQuery(Deserializable):
    """A search query with parameters."""

    text: str
    max_results: int = 5


class CalculationQuery(Deserializable):
    """A calculation with two numbers."""

    a: int
    b: int
    operation: str = "add"


def search_function(text: str, max_results: int = 5) -> str:
    """Search for something."""
    return f"Found {max_results} results for '{text}'"


def calculate_function(a: int, b: int, operation: str = "add") -> int:
    """Perform a calculation."""
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    return 0


def untyped_function(message: str) -> str:
    """A function without tool_type registration."""
    return f"Processed: {message}"


@pytest.mark.asyncio
async def test_unregistered_tool_returns_dict():
    """Test that tools without tool_type return OpenAI dict format."""
    print("🧪 Testing Unregistered Tool → Dict Format")

    @agentic_generator(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def unregistered_agent(
        query: str, runtime: Runtime
    ) -> AsyncGenerator[str, None]:
        # Register tool WITHOUT tool_type
        runtime.tool(untyped_function)  # No tool_type parameter

        async for chunk_type, content in runtime.run_stream(prompt=query):
            if chunk_type == "tool_call_complete":
                yield f"🔍 Tool call type: {type(content).__name__}"
                yield f"🔍 Tool call content: {content}"

                # CRITICAL VALIDATION: Should be dict, not Deserializable
                assert isinstance(content, dict), f"Expected dict, got {type(content)}"
                assert "id" in content, f"Missing 'id' in tool call: {content}"
                assert (
                    "function" in content
                ), f"Missing 'function' in tool call: {content}"
                assert (
                    "name" in content["function"]
                ), f"Missing function.name: {content}"
                assert (
                    "arguments" in content["function"]
                ), f"Missing function.arguments: {content}"
                assert isinstance(
                    content["function"]["arguments"], str
                ), f"Arguments should be JSON string: {content}"

                yield f"✅ Dict format validated: {content['function']['name']}"
                break
            elif chunk_type == "content":
                if content:
                    yield f"📝 Content: {content}"

    try:
        results = []
        async for result in unregistered_agent(
            "Process the message 'hello world' using untyped_function"
        ):
            print(result)
            results.append(result)

        # Ensure we got the dict format validation
        dict_validations = [r for r in results if "Dict format validated" in r]
        assert len(dict_validations) > 0, "No dict format validation found"

        print("✅ Unregistered tool dict format test passed")
        return True

    except Exception as e:
        print(f"❌ Unregistered tool test failed: {e}")
        return False


@pytest.mark.asyncio
async def test_partial_parsing_returns_none():
    """Test that failed partial parsing returns None during streaming."""
    print("\n🧪 Testing Failed Partial Parsing → None")

    @agentic_generator(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def partial_parsing_agent(
        query: str, runtime: Runtime
    ) -> AsyncGenerator[str, None]:
        # Register tool WITH tool_type
        runtime.tool(search_function, tool_type=SearchQuery)

        none_count = 0
        valid_count = 0

        async for chunk_type, content in runtime.run_stream(prompt=query):
            if chunk_type == "tool_call_partial":
                yield f"🔍 Partial type: {type(content).__name__ if content else 'None'}"

                if content is None:
                    none_count += 1
                    yield f"❌ Partial parsing failed (expected for invalid JSON)"
                elif isinstance(content, SearchQuery):
                    valid_count += 1
                    yield f"✅ Partial parsing succeeded: {content._tool_name}"
                else:
                    yield f"🔍 Unexpected partial type: {type(content)}"

            elif chunk_type == "tool_call_complete":
                yield f"🏁 Complete tool: {type(content).__name__}"
                assert isinstance(
                    content, SearchQuery
                ), f"Expected SearchQuery, got {type(content)}"
                break
            elif chunk_type == "content":
                if content:
                    yield f"📝 Content: {content}"

        yield f"📊 None count: {none_count}, Valid count: {valid_count}"

        # We should get some Nones during early parsing attempts
        if none_count > 0:
            yield "✅ Found None returns during partial parsing as expected"
        else:
            yield "⚠️ No None returns found - may indicate different behavior"

    try:
        results = []
        async for result in partial_parsing_agent(
            "Search for 'machine learning' with max_results 10"
        ):
            print(result)
            results.append(result)

        print("✅ Partial parsing None test completed")
        return True

    except Exception as e:
        print(f"❌ Partial parsing test failed: {e}")
        return False


@pytest.mark.asyncio
async def test_registered_tool_returns_type_instance():
    """Test that tools with tool_type return Deserializable instances."""
    print("\n🧪 Testing Registered Tool Type → Deserializable Instance")

    @agentic_generator(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def typed_agent(query: str, runtime: Runtime) -> AsyncGenerator[str, None]:
        # Register tool WITH tool_type
        runtime.tool(calculate_function, tool_type=CalculationQuery)

        async for chunk_type, content in runtime.run_stream(prompt=query):
            if chunk_type == "tool_call_complete":
                yield f"🔍 Tool call type: {type(content).__name__}"
                yield f"🔍 Tool call content: {content}"

                # CRITICAL VALIDATION: Should be CalculationQuery instance
                assert isinstance(
                    content, CalculationQuery
                ), f"Expected CalculationQuery, got {type(content)}"
                assert hasattr(content, "_tool_name"), f"Missing _tool_name attribute"
                assert (
                    content._tool_name == "calculate_function"
                ), f"Wrong tool name: {content._tool_name}"

                # Validate Deserializable attributes
                assert hasattr(content, "a"), "Missing 'a' attribute"
                assert hasattr(content, "b"), "Missing 'b' attribute"
                assert hasattr(content, "operation"), "Missing 'operation' attribute"

                yield f"✅ Deserializable instance validated: {content._tool_name}"
                break
            elif chunk_type == "content":
                if content:
                    yield f"📝 Content: {content}"

    try:
        results = []
        async for result in typed_agent("Use calculate_function to add 15 and 27"):
            print(result)
            results.append(result)

        # Ensure we got the Deserializable validation
        type_validations = [
            r for r in results if "Deserializable instance validated" in r
        ]
        assert len(type_validations) > 0, "No Deserializable validation found"

        print("✅ Registered tool type test passed")
        return True

    except Exception as e:
        print(f"❌ Registered tool type test failed: {e}")
        return False


@pytest.mark.asyncio
async def test_mixed_tool_registration():
    """Test mixed tool types in same runtime return appropriate formats."""
    print("\n🧪 Testing Mixed Tool Registration → Different Formats")

    @agentic_generator(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def mixed_agent(query: str, runtime: Runtime) -> AsyncGenerator[str, None]:
        # Register one tool WITH tool_type, one WITHOUT
        runtime.tool(
            calculate_function, tool_type=CalculationQuery
        )  # Should return CalculationQuery
        runtime.tool(untyped_function)  # Should return dict

        tool_calls_found = []

        async for chunk_type, content in runtime.run_stream(prompt=query):
            if chunk_type == "tool_call_complete":
                yield f"🔍 Mixed tool type: {type(content).__name__}"
                yield f"🔍 Mixed tool content: {content}"

                if isinstance(content, dict):
                    # Untyped tool - should be OpenAI dict format
                    assert "id" in content, f"Dict tool missing 'id': {content}"
                    assert (
                        "function" in content
                    ), f"Dict tool missing 'function': {content}"
                    assert content["function"]["name"] in [
                        "untyped_function"
                    ], f"Unexpected dict tool: {content}"
                    yield f"✅ Dict tool validated: {content['function']['name']}"
                    tool_calls_found.append(("dict", content["function"]["name"]))

                elif isinstance(content, CalculationQuery):
                    # Typed tool - should be Deserializable instance
                    assert hasattr(
                        content, "_tool_name"
                    ), "Typed tool missing _tool_name"
                    assert (
                        content._tool_name == "calculate_function"
                    ), f"Wrong typed tool name: {content._tool_name}"
                    yield f"✅ Typed tool validated: {content._tool_name}"
                    tool_calls_found.append(("typed", content._tool_name))

                else:
                    yield f"❌ Unexpected tool type: {type(content)}"
                    assert False, f"Unexpected tool type: {type(content)}"

            elif chunk_type == "content":
                if content:
                    yield f"📝 Content: {content}"

        yield f"📊 Tool calls found: {tool_calls_found}"

        # Validate we got both types
        dict_tools = [call for call in tool_calls_found if call[0] == "dict"]
        typed_tools = [call for call in tool_calls_found if call[0] == "typed"]

        if len(dict_tools) > 0 and len(typed_tools) > 0:
            yield f"✅ Mixed registration working: {len(dict_tools)} dict, {len(typed_tools)} typed"
        elif len(dict_tools) > 0:
            yield f"⚠️ Only found dict tools: {dict_tools}"
        elif len(typed_tools) > 0:
            yield f"⚠️ Only found typed tools: {typed_tools}"
        else:
            yield "❌ No tool calls found"

    try:
        results = []
        async for result in mixed_agent(
            "Calculate 8 + 12 using calculate_function, then process the result message using untyped_function"
        ):
            print(result)
            results.append(result)

        print("✅ Mixed tool registration test completed")
        return True

    except Exception as e:
        print(f"❌ Mixed tool registration test failed: {e}")
        return False


@pytest.mark.asyncio
async def test_cross_provider_format_consistency():
    """Test that all providers yield consistent OpenAI format."""
    print("\n🧪 Testing Cross-Provider Format Consistency")

    providers = [
        "openrouter:anthropic/claude-sonnet-4",
        "vertex-gemini:gemini-2.5-flash",
        "openrouter:openai/gpt-4o",
    ]

    for provider in providers:
        print(f"\n🔄 Testing provider: {provider}")

        @agentic_generator(model=provider, native_tools=True)
        async def format_consistency_agent(
            query: str, runtime: Runtime
        ) -> AsyncGenerator[str, None]:
            # Test both registration types
            runtime.tool(calculate_function, tool_type=CalculationQuery)
            runtime.tool(untyped_function)

            async for chunk_type, content in runtime.run_stream(prompt=query):
                if chunk_type == "tool_call_complete":
                    yield f"🔍 {provider} tool type: {type(content).__name__}"

                    if isinstance(content, dict):
                        # Validate OpenAI dict format consistency
                        assert "id" in content, f"{provider}: Missing id in dict tool"
                        assert (
                            "function" in content
                        ), f"{provider}: Missing function in dict tool"
                        assert (
                            "name" in content["function"]
                        ), f"{provider}: Missing function.name"
                        assert (
                            "arguments" in content["function"]
                        ), f"{provider}: Missing function.arguments"
                        assert isinstance(
                            content["function"]["arguments"], str
                        ), f"{provider}: Arguments not JSON string"
                        yield f"✅ {provider} dict format consistent"

                    elif isinstance(content, CalculationQuery):
                        # Validate Deserializable format consistency
                        assert hasattr(
                            content, "_tool_name"
                        ), f"{provider}: Missing _tool_name"
                        assert (
                            content._tool_name == "calculate_function"
                        ), f"{provider}: Wrong _tool_name"
                        yield f"✅ {provider} typed format consistent"

                    break
                elif chunk_type == "content":
                    if content:
                        yield f"📝 {provider} content: {content}"

        try:
            results = []
            async for result in format_consistency_agent(
                f"Use calculate_function to add 5 and 3"
            ):
                print(f"  {result}")
                results.append(result)

            # Check if format validation passed
            consistency_checks = [r for r in results if "format consistent" in r]
            if len(consistency_checks) > 0:
                print(f"  ✅ {provider} format consistency validated")
            else:
                print(f"  ⚠️ {provider} format consistency not validated")

        except Exception as e:
            print(f"  ❌ {provider} format test failed: {e}")
            return False

    print("✅ All providers tested for format consistency")
    return True


@pytest.mark.asyncio
async def test_streaming_partial_marshalling_progression():
    """Test progressive marshalling from None → valid type during streaming."""
    print("\n🧪 Testing Streaming Partial Marshalling Progression")

    @agentic_generator(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def progression_agent(
        query: str, runtime: Runtime
    ) -> AsyncGenerator[str, None]:
        # Register tool with type for marshalling
        runtime.tool(search_function, tool_type=SearchQuery)

        partial_progression = []

        async for chunk_type, content in runtime.run_stream(prompt=query):
            if chunk_type == "tool_call_partial":
                partial_progression.append(content)

                if content is None:
                    yield "❌ Partial: None (parsing failed - expected early in stream)"
                elif isinstance(content, SearchQuery):
                    yield f"✅ Partial: SearchQuery({content.text}, max_results={content.max_results})"
                elif isinstance(content, dict):
                    yield f"🔍 Partial: Dict (fallback format)"
                else:
                    yield f"🔍 Partial: {type(content).__name__}"

            elif chunk_type == "tool_call_complete":
                yield f"🏁 Complete: {type(content).__name__}"
                assert isinstance(
                    content, SearchQuery
                ), f"Expected SearchQuery complete, got {type(content)}"
                break
            elif chunk_type == "content":
                if content:
                    yield f"📝 Content: {content}"

        # Analyze progression
        none_count = sum(1 for p in partial_progression if p is None)
        typed_count = sum(1 for p in partial_progression if isinstance(p, SearchQuery))
        dict_count = sum(1 for p in partial_progression if isinstance(p, dict))

        yield f"📊 Progression: {none_count} None, {typed_count} typed, {dict_count} dict"

        # Should show progression from None/failed parsing to successful typed instances
        if none_count > 0 or typed_count > 0:
            yield "✅ Streaming progression validated"
        else:
            yield "⚠️ No streaming progression detected"

    try:
        results = []
        async for result in progression_agent(
            "Search for 'artificial intelligence' with max_results 8"
        ):
            print(result)
            results.append(result)

        print("✅ Streaming partial marshalling progression test completed")
        return True

    except Exception as e:
        print(f"❌ Streaming progression test failed: {e}")
        return False


@pytest.mark.asyncio
async def test_tool_call_format_edge_cases():
    """Test edge cases in tool call format validation."""
    print("\n🧪 Testing Tool Call Format Edge Cases")

    @agentic_generator(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def edge_case_agent(
        query: str, runtime: Runtime
    ) -> AsyncGenerator[str, None]:
        # Multiple tools with different registrations
        runtime.tool(calculate_function, tool_type=CalculationQuery)
        runtime.tool(search_function, tool_type=SearchQuery)
        runtime.tool(untyped_function)  # No type

        tools_processed = {}

        async for chunk_type, content in runtime.run_stream(prompt=query):
            if chunk_type == "tool_call_complete":
                tool_name = None

                if isinstance(content, dict):
                    tool_name = content["function"]["name"]
                    tools_processed[tool_name] = "dict"

                    # Validate dict structure
                    assert "id" in content
                    assert "function" in content
                    assert "name" in content["function"]
                    assert "arguments" in content["function"]

                elif hasattr(content, "_tool_name"):
                    tool_name = content._tool_name
                    tools_processed[tool_name] = type(content).__name__

                    # Validate Deserializable structure
                    assert hasattr(content, "_tool_name")

                yield f"✅ Processed {tool_name}: {tools_processed[tool_name]}"

            elif chunk_type == "content":
                if content:
                    yield f"📝 Content: {content}"

        yield f"📊 Tools processed: {tools_processed}"

    try:
        results = []
        async for result in edge_case_agent(
            "Calculate 10 + 5, search for 'python', and process message 'done'"
        ):
            print(result)
            results.append(result)

        print("✅ Edge cases test completed")
        return True

    except Exception as e:
        print(f"❌ Edge cases test failed: {e}")
        return False


@pytest.mark.asyncio
async def test_multi_turn_toolresult_conversation():
    """Test 4-5 turn streaming conversation with ToolResult objects in history."""
    print("\n🧪 Testing Multi-Turn Streaming ToolResult Conversation")

    # Define tool types for realistic usage
    class WeatherQuery(Deserializable):
        location: str

    class TipCalculation(Deserializable):
        bill_amount: float
        tip_percent: float

    class TableBooking(Deserializable):
        restaurant: str
        time: str
        people: int

    class ConfirmationMessage(Deserializable):
        message: str

    # Tools for multi-turn conversation
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"Weather in {location}: 22°C, sunny"

    def calculate_tip(bill_amount: float, tip_percent: int) -> float:
        """Calculate tip amount."""
        return round(bill_amount * (tip_percent / 100), 2)

    def book_table(people: int, restaurant: str, time: str) -> str:
        """Book a restaurant table."""
        return f"Booked table for {people} at {restaurant} for {time}"

    def send_confirmation(message: str) -> str:
        """Send a confirmation message."""
        return f"Confirmation sent: {message}"

    @agentic_generator(native_tools=True)
    async def multi_turn_agent(
        model: str, runtime: Runtime
    ) -> AsyncGenerator[str, None]:
        # Register ALL tools WITH tool_types (realistic usage)
        runtime.tool(get_weather, tool_type=WeatherQuery)
        runtime.tool(calculate_tip, tool_type=TipCalculation)
        runtime.tool(book_table, tool_type=TableBooking)
        runtime.tool(send_confirmation, tool_type=ConfirmationMessage)

        conversation_history = []
        tool_results_created = []

        # Define the conversation turns
        prompts = [
            "Get weather for New York",
            "Calculate tip for a $45.50 bill at 20%",
            "Book a table for 4 people at 'Italian Bistro' for 7:30 PM",
            "Send a confirmation message with all our details using the send_confirmation tool. This is a test so you can't ask for details just fill out the tool call please.",
            "Give me a summary of everything we accomplished",
        ]

        for turn_num, prompt in enumerate(prompts, 1):
            yield f"\n🔄 Turn {turn_num}: {prompt}"

            # Stream this turn
            tool_call = None
            async for chunk_type, content in runtime.run_stream(
                model=model, prompt=prompt, history=conversation_history
            ):
                if chunk_type == "tool_call_complete":
                    yield f"🔧 Turn {turn_num} tool call: {type(content).__name__}"

                    # Validate tool_type instances (realistic usage)
                    assert hasattr(
                        content, "_tool_name"
                    ), f"Turn {turn_num}: Missing _tool_name"

                    if isinstance(content, WeatherQuery):
                        assert (
                            content._tool_name == "get_weather"  # type: ignore
                        ), f"Turn {turn_num}: Wrong weather tool name"
                        yield f"✅ Turn {turn_num} WeatherQuery: {content.location}"
                    elif isinstance(content, TipCalculation):
                        assert (
                            content._tool_name == "calculate_tip"  # type: ignore
                        ), f"Turn {turn_num}: Wrong tip tool name"
                        yield f"✅ Turn {turn_num} TipCalculation: ${content.bill_amount} at {content.tip_percent}%"
                    elif isinstance(content, TableBooking):
                        assert (
                            content._tool_name == "book_table"  # type: ignore
                        ), f"Turn {turn_num}: Wrong booking tool name"
                        print(content)
                        yield f"✅ Turn {turn_num} TableBooking: {content.people} people at {content.restaurant}"
                    elif isinstance(content, ConfirmationMessage):
                        assert (
                            content._tool_name == "send_confirmation"  # type: ignore
                        ), f"Turn {turn_num}: Wrong confirmation tool name"
                        yield f"✅ Turn {turn_num} ConfirmationMessage: {content.message[:50]}..."
                    else:
                        yield f"❌ Turn {turn_num}: Unexpected tool type {type(content)}"
                        raise ValueError("Unexpected tool type.")

                    # Execute tool
                    tool_output = await runtime.execute_tool(content)
                    yield f"🎯 Turn {turn_num} tool result: {tool_output}"

                    from reson.types import ToolResult

                    tool_result_obj = ToolResult.create((content, str(tool_output)))
                    tool_results_created.append(tool_result_obj)
                    conversation_history.append(tool_result_obj)
                    yield f"✅ Turn {turn_num} ToolResult added to history"

                    tool_call = content
                    break
                else:
                    print(content)

            # If this is the last turn (summary), don't expect a tool call
            if turn_num == len(prompts) and not tool_call:
                yield f"📝 Turn {turn_num}: Final summary completed"

        # Final validation
        yield f"\n📊 Multi-turn conversation completed:"
        yield f"   - Total turns: {len(prompts)}"
        yield f"   - ToolResults created: {len(tool_results_created)}"
        yield f"   - History length: {len(conversation_history)}"

        # Validate all ToolResults
        from reson.services.inference_clients import ToolResult

        for i, tr in enumerate(tool_results_created):
            yield f"   - Turn {i+1} ToolResult: {type(tr).__name__}"
            assert isinstance(tr, ToolResult), f"Expected ToolResult, got {type(tr)}"
            assert tr.tool_use_id, f"Missing tool_use_id in ToolResult {i+1}"
            assert tr.content, f"Missing content in ToolResult {i+1}"

        yield "✅ All ToolResults validated for conversation history usage"

    try:
        results = []
        models = [
            # "vertex-gemini:gemini-2.5-pro",
            # "openrouter:anthropic/claude-sonnet-4",
            "anthropic:claude-sonnet-4-20250514",
        ]

        for model in models:
            async for result in multi_turn_agent(model):
                print(result)
                results.append(result)

        # Check for successful completion
        success_indicators = [r for r in results if "All ToolResults validated" in r]
        assert len(success_indicators) > 0, "Multi-turn validation not completed"

        print("✅ Multi-turn streaming ToolResult conversation test passed")
        return True

    except Exception as e:
        print(f"❌ Multi-turn test failed: {e}")
        print("🔍 Full stack trace for debugging:")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all tool call format validation tests."""
    print("🚀 Tool Call Format Validation Tests")
    print("=" * 60)

    # Check environment
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not set")
        return

    print("✅ API key configured")

    tests = [
        # ("Unregistered Tool → Dict Format", test_unregistered_tool_returns_dict()),
        # ("Partial Parsing → None", test_partial_parsing_returns_none()),
        # (
        #     "Registered Type → Deserializable",
        #     test_registered_tool_returns_type_instance(),
        # ),
        # ("Mixed Registration → Different Formats", test_mixed_tool_registration()),
        # ("Cross-Provider Format Consistency", test_cross_provider_format_consistency()),
        # ("Edge Cases", test_tool_call_format_edge_cases()),
        (
            "Multi-Turn ToolResult Conversation",
            test_multi_turn_toolresult_conversation(),
        ),
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
    print(f"\n📊 Format Validation Test Results: {success_count}/{len(results)} passed")

    if success_count == len(results):
        print("🎉 All format validation tests passed!")
    else:
        print("⚠️ Some format validation tests failed.")

    return success_count == len(results)


if __name__ == "__main__":
    asyncio.run(main())
