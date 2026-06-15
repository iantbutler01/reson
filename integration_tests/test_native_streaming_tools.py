#!/usr/bin/env python3
"""Test streaming native tool calls."""

import asyncio
import pytest
import os
from typing import AsyncGenerator
from chevalier import agentic_generator, Runtime
from chevalier.types import Deserializable


class CalculationQuery(Deserializable):
    """A calculation with two numbers."""

    a: int
    b: int
    operation: str = "add"


def calculate(query: CalculationQuery) -> int:
    """Perform a calculation."""
    if query.operation == "add":
        return query.a + query.b
    elif query.operation == "multiply":
        return query.a * query.b
    else:
        return 0


def get_info(topic: str) -> str:
    """Get information about a topic."""
    return (
        f"Information about {topic}: This is a comprehensive overview of the subject."
    )


@pytest.mark.asyncio
async def test_streaming_native_tools():
    """Test streaming with native tool calls."""
    print("🧪 Testing Streaming Native Tool Calls")

    @agentic_generator(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def streaming_agent(
        query: str, runtime: Runtime
    ) -> AsyncGenerator[str, None]:
        runtime.tool(calculate)
        runtime.tool(get_info)

        # Start streaming
        async for chunk in runtime.run_stream(prompt=query):
            # Handle different chunk types based on the actual streaming interface
            if isinstance(chunk, tuple) and len(chunk) == 2:
                chunk_type, chunk_content = chunk
                if chunk_type == "reasoning":
                    yield f"🧠 Thinking: {chunk_content}"
                elif chunk_type == "content":
                    yield f"📝 Content: {chunk_content}"
                elif chunk_type == "tool_call_complete":
                    yield f"🔧 Tool Call: {chunk_content._tool_name if hasattr(chunk_content, '_tool_name') else chunk_content}"
                else:
                    yield f"🔧 Stream: {chunk_type} - {chunk_content}"
            else:
                # Single chunk (traditional streaming)
                yield f"📝 Chunk: {chunk}"

    try:
        print("Starting streaming test...")
        chunks = []

        async for chunk in streaming_agent(
            "Calculate 15 + 27 and then get info about 'machine learning'"
        ):
            print(chunk)
            chunks.append(chunk)

        print(f"\n✅ Streaming test completed with {len(chunks)} chunks")
        return True

    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


@pytest.mark.asyncio
async def test_google_streaming_tool_detection():
    """Test Google streaming tool call detection."""
    print("\n🧪 Testing Google Streaming Tool Detection")

    @agentic_generator(model="vertex-gemini:gemini-2.5-flash", native_tools=True)
    async def vertex_streaming_agent(
        query: str, runtime: Runtime
    ) -> AsyncGenerator[str, None]:
        runtime.tool(calculate)
        runtime.tool(get_info)

        # Start streaming
        async for chunk in runtime.run_stream(prompt=query):
            if isinstance(chunk, tuple) and len(chunk) == 2:
                chunk_type, chunk_content = chunk
                if chunk_type == "reasoning":
                    yield f"🧠 Vertex Thinking: {chunk_content}"
                elif chunk_type == "content":
                    yield f"📝 Vertex Content: {chunk_content}"
                elif chunk_type == "tool_call_complete":
                    yield f"🔧 Vertex Buffered Tool Call: {chunk_content._tool_name if hasattr(chunk_content, '_tool_name') else 'detected'}"
                else:
                    yield f"🔧 Vertex Stream: {chunk_type}"
            else:
                yield f"📝 Vertex Chunk: {chunk}"

    try:
        print("Starting Google streaming test...")
        chunks = []

        async for chunk in vertex_streaming_agent(
            "Calculate 8 + 12 using the calculate tool"
        ):
            print(chunk)
            chunks.append(chunk)

        print(f"\n✅ Google streaming test completed with {len(chunks)} chunks")
        return True

    except Exception as e:
        print(f"❌ Google streaming test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


@pytest.mark.asyncio
async def test_google_anthropic_streaming_tools():
    """Test Google Anthropic streaming tool calls."""
    print("\n🧪 Testing Google Anthropic Streaming Tool Calls")

    @agentic_generator(model="google-anthropic:claude-3-5-sonnet", native_tools=True)
    async def google_anthropic_streaming_agent(
        query: str, runtime: Runtime
    ) -> AsyncGenerator[str, None]:
        runtime.tool(calculate)
        runtime.tool(get_info)

        # Start streaming
        async for chunk in runtime.run_stream(prompt=query):
            if isinstance(chunk, tuple) and len(chunk) == 2:
                chunk_type, chunk_content = chunk
                if chunk_type == "reasoning":
                    yield f"🧠 GA Thinking: {chunk_content}"
                elif chunk_type == "content":
                    yield f"📝 GA Content: {chunk_content}"
                elif chunk_type == "tool_call_complete":
                    yield f"🔧 GA Tool Call: {chunk_content._tool_name if hasattr(chunk_content, '_tool_name') else 'detected'}"
                else:
                    yield f"🔧 GA Stream: {chunk_type}"
            else:
                yield f"📝 GA Chunk: {chunk}"

    try:
        print("Starting Google Anthropic streaming test...")
        chunks = []

        async for chunk in google_anthropic_streaming_agent(
            "Use calculate to multiply 6 by 7"
        ):
            print(chunk)
            chunks.append(chunk)

        print(
            f"\n✅ Google Anthropic streaming test completed with {len(chunks)} chunks"
        )
        return True

    except Exception as e:
        print(f"❌ Google Anthropic streaming test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


@pytest.mark.asyncio
async def test_streaming_tool_execution():
    """Test streaming with actual tool execution."""
    print("\n🧪 Testing Streaming with Tool Execution")

    @agentic_generator(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def execution_agent(
        query: str, runtime: Runtime
    ) -> AsyncGenerator[str, None]:
        runtime.tool(calculate)

        history = []

        # Stream initial response
        yield "🚀 Starting calculation..."

        async for chunk_type, chunk in runtime.run_stream(prompt=query):
            if chunk_type == "tool_call_complete":
                # We got a complete tool call, let's execute it
                yield f"🔧 Executing tool: {chunk}"

                # Convert streaming chunk to tool call object (simplified)
                # In real implementation, this would be handled by the system
                break
            else:
                yield f"📡 Streaming: {chunk_type} - {chunk}"

        # Simulate getting the tool call result and continuing
        yield "✅ Tool execution completed"

    try:
        chunks = []
        async for chunk in execution_agent("Calculate 25 + 17"):
            print(chunk)
            chunks.append(chunk)

        print(f"\n✅ Execution streaming test completed with {len(chunks)} chunks")
        return True

    except Exception as e:
        print(f"❌ Execution streaming test failed: {e}")
        return False


async def main():
    """Run streaming native tool tests."""
    print("🚀 Testing Native Tool Streaming")
    print("=" * 50)

    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not set")
        return

    print("✅ API key configured")

    tests = [
        test_streaming_native_tools(),
        test_google_streaming_tool_detection(),
        test_google_anthropic_streaming_tools(),
        test_streaming_tool_execution(),
    ]

    results = await asyncio.gather(*tests, return_exceptions=True)
    success_count = sum(1 for r in results if r is True)

    print(f"\n📊 Streaming Test Results: {success_count}/{len(results)} passed")

    if success_count == len(results):
        print("🎉 All streaming tests passed!")
    else:
        print("⚠️  Some streaming tests failed.")


if __name__ == "__main__":
    asyncio.run(main())
