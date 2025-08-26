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

    @agentic_generator(model="openrouter:openai/o3-mini@reasoning=high")
    async def reasoning_stream_agent(query: str, runtime: Runtime):
        # Use modern streaming API without forcing output_type
        async for chunk in runtime.run_stream(prompt=query):
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
    print("-" * 80)

    # Modern test - ensure we get some output
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_anthropic_reasoning_stream():
    """Test reasoning tokens with Anthropic model using modern streaming API."""
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    @agentic_generator(model="openrouter:anthropic/claude-3-5-sonnet@reasoning=2000")
    async def anthropic_stream_agent(query: str, runtime: Runtime):
        # Use modern streaming API
        async for chunk in runtime.run_stream(prompt=query):
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
    print(f"Reasoning: Available in runtime.reasoning after streaming")

    # Modern test - ensure we get some output
    assert len(chunks) > 0
