#!/usr/bin/env python3
"""
Integration tests for ToolCall class hydration and provider conversion.

Tests the ability to create ToolCall objects from various provider formats
and convert them back to provider-specific assistant messages.
"""

import asyncio
import pytest
import os
from typing import List, Dict, Union
from reson import agentic, Runtime
from reson.types import Deserializable, ToolCall, ToolResult, ChatMessage
from reson.services.inference_clients import InferenceProvider, ChatRole


class WeatherQuery(Deserializable):
    """A weather query for a location."""

    location: str
    units: str = "celsius"


class CalculationResult(Deserializable):
    """Final calculation result."""

    value: int
    explanation: str


def get_weather(query: WeatherQuery) -> str:
    """Get weather information for a location."""
    return f"Weather in {query.location}: 22°{query.units[0].upper()}, partly cloudy"


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


class TestToolCallCreation:
    """Test ToolCall creation from various provider formats."""

    def test_toolcall_create_from_openai_format(self):
        """Test creating ToolCall from OpenAI tool_calls format."""
        # OpenAI format: {"id": "call_abc", "function": {"name": "get_weather", "arguments": '{"location":"SF"}'}}
        openai_tool_call = {
            "id": "call_abc123",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco", "units": "celsius"}',
            },
        }

        tool_call = ToolCall.create(openai_tool_call)

        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_use_id == "call_abc123"
        assert tool_call.tool_name == "get_weather"
        assert tool_call.args == {"location": "San Francisco", "units": "celsius"}
        assert (
            tool_call.raw_arguments
            == '{"location": "San Francisco", "units": "celsius"}'
        )
        assert tool_call.tool_obj == openai_tool_call

    def test_toolcall_create_from_openai_format_list(self):
        """Test creating multiple ToolCalls from OpenAI tool_calls format."""
        openai_tool_calls = [
            {
                "id": "call_1",
                "function": {"name": "get_weather", "arguments": '{"location": "SF"}'},
            },
            {
                "id": "call_2",
                "function": {"name": "add_numbers", "arguments": '{"a": 5, "b": 3}'},
            },
        ]

        tool_calls = ToolCall.create(openai_tool_calls)

        assert isinstance(tool_calls, list)
        assert len(tool_calls) == 2

        assert tool_calls[0].tool_use_id == "call_1"
        assert tool_calls[0].tool_name == "get_weather"
        assert tool_calls[0].args == {"location": "SF"}

        assert tool_calls[1].tool_use_id == "call_2"
        assert tool_calls[1].tool_name == "add_numbers"
        assert tool_calls[1].args == {"a": 5, "b": 3}

    def test_toolcall_create_from_anthropic_format(self):
        """Test creating ToolCall from Anthropic tool_use format."""
        # Anthropic format: {"id": "toolu_01", "name": "get_weather", "input": {"location": "SF"}}
        anthropic_tool_call = {
            "id": "toolu_abc123",
            "name": "get_weather",
            "input": {"location": "San Francisco", "units": "celsius"},
        }

        tool_call = ToolCall.create(anthropic_tool_call)

        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_use_id == "toolu_abc123"
        assert tool_call.tool_name == "get_weather"
        assert tool_call.args == {"location": "San Francisco", "units": "celsius"}
        assert tool_call.raw_arguments is None  # Anthropic doesn't use raw JSON strings
        assert tool_call.tool_obj == anthropic_tool_call

    def test_toolcall_create_from_google_format(self):
        """Test creating ToolCall from Google functionCall format."""
        # Google format: {"functionCall": {"name": "get_weather", "args": {"location": "SF"}}}
        google_tool_call = {
            "functionCall": {
                "name": "get_weather",
                "args": {"location": "San Francisco", "units": "celsius"},
            }
        }

        tool_call = ToolCall.create(google_tool_call)

        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_use_id.startswith("google_get_weather_")  # Generated ID
        assert tool_call.tool_name == "get_weather"
        assert tool_call.args == {"location": "San Francisco", "units": "celsius"}
        assert tool_call.tool_obj == google_tool_call

    def test_toolcall_create_from_deserializable(self):
        """Test creating ToolCall from NativeToolParser Deserializable objects."""

        # Mock a Deserializable object from NativeToolParser
        class MockDeserializable:
            def __init__(self):
                self._tool_name = "get_weather"
                self._tool_use_id = "mock_tool_123"
                self.location = "San Francisco"
                self.units = "celsius"
                self.signature = "mock_signature"
                self._private_attr = "should_be_ignored"

        deserializable_obj = MockDeserializable()

        tool_call = ToolCall.create(deserializable_obj)

        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_use_id == "mock_tool_123"
        assert tool_call.tool_name == "get_weather"
        assert tool_call.args == {"location": "San Francisco", "units": "celsius"}
        assert tool_call.signature == "mock_signature"
        assert tool_call.tool_obj == deserializable_obj
        assert (
            tool_call.args is not None and "_private_attr" not in tool_call.args
        )  # Private attrs excluded

    def test_toolcall_create_from_deserializable_model_dump(self):
        """Test creating ToolCall from Deserializable model_dump() with include_underscore=True."""
        # Simulate a model dump with include_underscore=True
        model_dump_data = {
            "location": "San Francisco",
            "units": "celsius",
            "_tool_name": "get_weather",
            "_tool_use_id": "toolu_dump_123",
            "signature": "dump_signature",
        }

        tool_call = ToolCall.create(model_dump_data)

        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_use_id == "toolu_dump_123"
        assert tool_call.tool_name == "get_weather"
        assert tool_call.args == {"location": "San Francisco", "units": "celsius"}
        assert tool_call.signature == "dump_signature"
        assert tool_call.tool_obj == model_dump_data
        # Verify underscore fields are excluded from args
        assert tool_call.args is not None
        assert "_tool_name" not in tool_call.args
        assert "_tool_use_id" not in tool_call.args

    def test_toolcall_create_invalid_format(self):
        """Test error handling for unsupported tool call formats."""
        invalid_tool_call = {"invalid": "format"}

        with pytest.raises(ValueError, match="Unsupported tool call format"):
            ToolCall.create(invalid_tool_call)

    def test_toolcall_create_empty_list(self):
        """Test error handling for empty tool call list."""
        with pytest.raises(ValueError, match="No tool calls provided"):
            ToolCall.create([])


class TestToolCallProviderConversion:
    """Test ToolCall conversion to provider-specific assistant message formats."""

    def test_toolcall_provider_conversion_openai(self):
        """Test ToolCall.to_provider_assistant_message() for OpenAI."""
        tool_call = ToolCall(
            tool_use_id="call_123",
            tool_name="get_weather",
            args={"location": "SF", "units": "celsius"},
            raw_arguments='{"location": "SF", "units": "celsius"}',
        )

        message = tool_call.to_provider_assistant_message(InferenceProvider.OPENAI)

        expected = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "SF", "units": "celsius"}',
                    },
                }
            ],
        }

        assert message == expected

    def test_toolcall_provider_conversion_openai_no_raw_args(self):
        """Test OpenAI conversion when raw_arguments is None."""
        tool_call = ToolCall(
            tool_use_id="call_123",
            tool_name="get_weather",
            args={"location": "SF"},
            raw_arguments=None,
        )

        message = tool_call.to_provider_assistant_message(InferenceProvider.OPENAI)

        assert message["tool_calls"][0]["function"]["arguments"] == '{"location": "SF"}'

    def test_toolcall_provider_conversion_anthropic(self):
        """Test ToolCall.to_provider_assistant_message() for Anthropic."""
        tool_call = ToolCall(
            tool_use_id="toolu_123",
            tool_name="get_weather",
            args={"location": "SF", "units": "celsius"},
        )

        message = tool_call.to_provider_assistant_message(InferenceProvider.ANTHROPIC)

        expected = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"location": "SF", "units": "celsius"},
                }
            ],
        }

        assert message == expected

    def test_toolcall_provider_conversion_bedrock(self):
        """Test ToolCall.to_provider_assistant_message() for Bedrock (same as Anthropic)."""
        tool_call = ToolCall(
            tool_use_id="toolu_123", tool_name="get_weather", args={"location": "SF"}
        )

        message = tool_call.to_provider_assistant_message(InferenceProvider.BEDROCK)

        # Should use Anthropic format
        assert message["role"] == "assistant"
        assert message["content"][0]["type"] == "tool_use"
        assert message["content"][0]["id"] == "toolu_123"

    def test_toolcall_provider_conversion_google_genai(self):
        """Test ToolCall.to_provider_assistant_message() for Google GenAI."""
        tool_call = ToolCall(
            tool_use_id="google_123",
            tool_name="get_weather",
            args={"location": "SF", "units": "celsius"},
        )

        message = tool_call.to_provider_assistant_message(
            InferenceProvider.GOOGLE_GENAI
        )

        expected = {
            "role": "model",
            "parts": [
                {
                    "functionCall": {
                        "name": "get_weather",
                        "args": {"location": "SF", "units": "celsius"},
                    }
                }
            ],
        }

        assert message == expected

    def test_toolcall_provider_conversion_fallback(self):
        """Test fallback to OpenAI format for unknown providers."""
        tool_call = ToolCall(
            tool_use_id="call_123", tool_name="get_weather", args={"location": "SF"}
        )

        # Use OpenRouter which should use OpenAI format
        message = tool_call.to_provider_assistant_message(InferenceProvider.OPENROUTER)

        assert message["role"] == "assistant"
        assert "tool_calls" in message
        assert message["tool_calls"][0]["type"] == "function"

    def test_roundtrip_tool_call_creation_and_conversion(self):
        """Test creating ToolCall from provider format and converting back."""
        # Test OpenAI roundtrip
        original_openai = {
            "id": "call_test123",
            "function": {"name": "calculate", "arguments": '{"a": 2, "b": 2}'},
        }

        tool_call = ToolCall.create(original_openai)
        converted = tool_call.to_provider_assistant_message(InferenceProvider.OPENAI)

        assert converted["tool_calls"][0]["id"] == "call_test123"
        assert converted["tool_calls"][0]["function"]["name"] == "calculate"
        assert converted["tool_calls"][0]["function"]["arguments"] == '{"a": 2, "b": 2}'

        # Test Anthropic roundtrip
        original_anthropic = {
            "id": "toolu_test123",
            "name": "calculate",
            "input": {"a": 2, "b": 2},
        }

        tool_call = ToolCall.create(original_anthropic)
        converted = tool_call.to_provider_assistant_message(InferenceProvider.ANTHROPIC)

        assert converted["content"][0]["id"] == "toolu_test123"
        assert converted["content"][0]["name"] == "calculate"
        assert converted["content"][0]["input"] == {"a": 2, "b": 2}


@pytest.mark.asyncio
async def test_toolcall_in_conversation_history():
    """Test ToolCall objects can be used in conversation history."""
    print("\n🧪 Testing ToolCall in conversation history")

    # Skip if no API key
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    @agentic(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def weather_assistant(query: str, runtime: Runtime) -> str:
        runtime.tool(get_weather)

        # Simulate a previous assistant tool call that we're hydrating from storage
        previous_tool_call = ToolCall(
            tool_use_id="toolu_previous_123",
            tool_name="get_weather",
            args={"location": "New York", "units": "celsius"},
        )

        # Simulate the corresponding tool result
        previous_tool_result = ToolResult(
            tool_use_id="toolu_previous_123", content="Weather in New York: 18°C, rainy"
        )

        # Use the hydrated ToolCall and ToolResult in history
        history = [
            ChatMessage(
                role=ChatRole.USER, content="What's the weather like in New York?"
            ),
            previous_tool_call,  # Assistant's previous tool call
            previous_tool_result,  # Previous tool result
            ChatMessage(
                role=ChatRole.USER, content="Thanks! Now what about San Francisco?"
            ),
        ]

        # Continue the conversation with this history
        result = await runtime.run(prompt=query, history=history)
        print(f"📊 Weather assistant result: {result}")

        return str(result)

    try:
        result = await weather_assistant(
            "Please get the weather for San Francisco and compare it to the previous New York weather"
        )
        print(f"✅ ToolCall in history test completed: {result}")
        return True
    except Exception as e:
        print(f"❌ ToolCall in history test failed: {e}")
        return False


@pytest.mark.asyncio
async def test_toolcall_hydration_workflow():
    """Test complete ToolCall hydration workflow - create from provider format, use in conversation."""
    print("\n🧪 Testing ToolCall hydration workflow")

    # Skip if no API key
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    @agentic(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def calculator_assistant(query: str, runtime: Runtime) -> str:
        runtime.tool(add_numbers)

        # Simulate hydrating a tool call from an OpenAI provider format
        openai_tool_call_data = {
            "id": "call_previous_calculation",
            "function": {"name": "add_numbers", "arguments": '{"a": 10, "b": 5}'},
        }

        # Create ToolCall from provider format
        hydrated_tool_call = ToolCall.create(openai_tool_call_data)

        # Create corresponding tool result
        tool_result = ToolResult(tool_use_id="call_previous_calculation", content="15")

        # Use in conversation history
        history = [
            ChatMessage(role=ChatRole.USER, content="Please add 10 and 5"),
            hydrated_tool_call,  # Hydrated from storage
            tool_result,
            ChatMessage(
                role=ChatRole.USER, content="Great! Now multiply that result by 2"
            ),
        ]

        result = await runtime.run(prompt=query, history=history)
        print(f"📊 Calculator result: {result}")

        return str(result)

    try:
        result = await calculator_assistant(
            "Please continue with the calculation and tell me the final answer"
        )
        print(f"✅ ToolCall hydration workflow completed: {result}")
        return True
    except Exception as e:
        print(f"❌ ToolCall hydration workflow failed: {e}")
        return False


async def main():
    """Run ToolCall hydration tests."""
    print("🚀 ToolCall Hydration Tests")
    print("=" * 50)

    # Check environment
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  OPENROUTER_API_KEY not set - skipping integration tests")
        print("✅ Unit tests will still run")
    else:
        print("✅ API keys configured")

    # Run integration tests if API key is available
    if os.getenv("OPENROUTER_API_KEY"):
        tests = [
            (
                "ToolCall in Conversation History",
                test_toolcall_in_conversation_history(),
            ),
            ("ToolCall Hydration Workflow", test_toolcall_hydration_workflow()),
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
        print(f"\n📊 Integration Test Results: {success_count}/{len(results)} passed")

        if success_count == len(results):
            print("🎉 All ToolCall hydration tests passed!")
        else:
            print("⚠️  Some ToolCall hydration tests failed. Check the output above.")

        return success_count == len(results)
    else:
        print(
            "📝 Unit tests available in TestToolCallCreation and TestToolCallProviderConversion classes"
        )
        return True


def test_toolcall_message_conversion_integration():
    """Test that ToolCall objects integrate properly with message conversion."""
    from reson.services.inference_clients import InferenceClient

    class TestClient(InferenceClient):
        def __init__(self, provider):
            super().__init__()
            self.provider = provider

        async def get_generation(
            self, messages, max_tokens=4096, top_p=0.9, temperature=0.5, tools=None
        ):
            return "test"

        async def connect_and_listen(
            self, messages, max_tokens=4096, top_p=0.9, temperature=0.5, tools=None
        ):
            yield ("content", "test")

    # Test Anthropic message conversion
    client = TestClient(InferenceProvider.ANTHROPIC)

    tool_call = ToolCall(
        tool_use_id="toolu_123", tool_name="get_weather", args={"location": "SF"}
    )

    tool_result = ToolResult(tool_use_id="toolu_123", content="72°F sunny")

    messages = [
        ChatMessage(role=ChatRole.USER, content="Weather?"),
        tool_call,
        tool_result,
        ChatMessage(role=ChatRole.USER, content="Thanks!"),
    ]

    converted = client._convert_messages_to_provider_format(messages, client.provider)

    # Should have user, assistant tool call, user (merged tool result + thanks)
    assert len(converted) == 3
    assert converted[1]["role"] == "assistant"
    assert converted[1]["content"][0]["type"] == "tool_use"
    assert converted[1]["content"][0]["id"] == "toolu_123"


if __name__ == "__main__":
    asyncio.run(main())
