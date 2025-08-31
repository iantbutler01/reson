import asyncio
import pytest
import os
from reson import agentic_generator, Runtime
from reson.stores import MemoryStore


@pytest.mark.asyncio
async def test_reasoning_stream():
    """Test reasoning tokens with streaming using modern API patterns."""
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    runtime_instance = None

    @agentic_generator(model="openrouter:openai/gpt-5-mini@reasoning=high")
    async def reasoning_stream_agent(query: str, runtime: Runtime):
        nonlocal runtime_instance
        runtime_instance = runtime
        # Use modern streaming API without forcing output_type
        async for chunk in runtime.run_stream(prompt=query, max_tokens=5000):
            yield chunk

    print("Testing streaming with reasoning...")
    print("Content chunks:")

    # Collect all chunks
    chunks = []
    content_chunks = []
    reasoning_chunks = []

    async for chunk in reasoning_stream_agent(
        "How would you build the world's tallest skyscraper?"
    ):
        chunks.append(chunk)
        print(f"  {repr(chunk)}")

        # Parse chunk types for modern API
        if isinstance(chunk, tuple) and len(chunk) == 2:
            chunk_type, chunk_content = chunk
            if chunk_type == "content":
                content_chunks.append(chunk_content)
            elif chunk_type == "reasoning":
                reasoning_chunks.append(chunk_content)

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Content chunks: {len(content_chunks)}")
    print(f"Reasoning chunks: {len(reasoning_chunks)}")
    print(f"Full content: {''.join(str(c) for c in content_chunks)}")
    if runtime_instance:
        print(f"Reasoning: {runtime_instance.reasoning}")
    print("-" * 80)

    # Modern test - ensure we get some output
    assert len(chunks) > 0
    if runtime_instance:
        assert len(runtime_instance.reasoning) > 0
    else:
        assert len(reasoning_chunks) > 0


@pytest.mark.asyncio
async def test_anthropic_reasoning_stream():
    """Test reasoning tokens with Anthropic model using modern streaming API."""
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    runtime_instance = None

    @agentic_generator(model="openrouter:anthropic/claude-sonnet-4@reasoning=2000")
    async def anthropic_stream_agent(query: str, runtime: Runtime):
        nonlocal runtime_instance
        runtime_instance = runtime
        # Use modern streaming API
        async for chunk in runtime.run_stream(prompt=query, max_tokens=5000):
            yield chunk

    print("\nTesting Anthropic streaming with max_tokens reasoning...")
    print("Content chunks:")

    chunks = []
    content_chunks = []
    reasoning_chunks = []

    async for chunk in anthropic_stream_agent(
        "Explain quantum computing in simple terms."
    ):
        chunks.append(chunk)
        print(f"  {repr(chunk)}")

        # Parse chunk types for modern API
        if isinstance(chunk, tuple) and len(chunk) == 2:
            chunk_type, chunk_content = chunk
            if chunk_type == "content":
                content_chunks.append(chunk_content)
            elif chunk_type == "reasoning":
                reasoning_chunks.append(chunk_content)

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Content chunks: {len(content_chunks)}")
    print(f"Reasoning chunks: {len(reasoning_chunks)}")
    print(f"Full content: {''.join(str(c) for c in content_chunks)}")
    print(f"Reasoning: {runtime_instance.reasoning}")

    # Modern test - ensure we get some output
    assert len(chunks) > 0
    assert len(runtime_instance.reasoning) > 0
