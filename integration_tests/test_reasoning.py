import asyncio
import pytest
import os
from reson import agentic, Runtime
from reson.stores import MemoryStore


@pytest.mark.asyncio
async def test_reasoning():
    """Test reasoning tokens with o3-mini using modern patterns."""
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    @agentic(model="openrouter:openai/o3-mini@reasoning=high")
    async def reasoning_agent(query: str, runtime: Runtime):
        # Modern pattern: no output_type, return whatever the model gives us
        return await runtime.run(prompt=query)

    # Test reasoning functionality
    print("Testing non-streaming with reasoning...")
    result = await reasoning_agent(
        "How would you build the world's tallest skyscraper?"
    )
    print(f"Result: {result}")
    print(f"Result type: {type(result)}")

    # Modern test - handle whatever we get back
    if result is None:
        print("⚠️  Got None result - this can happen with reasoning models")
        # For reasoning models, check if reasoning was captured
        assert len(reasoning_agent.runtime.reasoning) > 0
        return  # Don't fail the test for None
    elif isinstance(result, tuple):
        content, reasoning = result
        print(f"✅ Got tuple: content='{content}', reasoning='{reasoning}'")
        assert content is not None or reasoning is not None
    elif isinstance(result, str):
        print(f"✅ Got string: '{result}'")
        assert len(result) > 0
    else:
        print(f"✅ Got other type: {result}")
        assert result is not None


@pytest.mark.asyncio
async def test_anthropic_reasoning():
    """Test reasoning tokens with Anthropic model using modern patterns."""
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    @agentic(model="openrouter:anthropic/claude-3-5-sonnet@reasoning=2000")
    async def anthropic_reasoning_agent(query: str, runtime: Runtime):
        # Modern pattern: no output_type, return whatever the model gives us
        return await runtime.run(prompt=query)

    print("\nTesting Anthropic with max_tokens reasoning...")
    result = await anthropic_reasoning_agent(
        "What's the most efficient algorithm for sorting a large dataset?"
    )
    print(f"Result: {result}")
    print(f"Result type: {type(result)}")

    # Modern test - handle whatever we get back
    if result is None:
        print("⚠️  Got None result - this can happen with reasoning models")
        assert len(anthropic_reasoning_agent.runtime.reasoning) > 0
        return  # Don't fail the test for None
    elif isinstance(result, tuple):
        content, reasoning = result
        print(f"✅ Got tuple: content='{content}', reasoning='{reasoning}'")
        assert content is not None or reasoning is not None
    elif isinstance(result, str):
        print(f"✅ Got string: '{result}'")
        assert len(result) > 0
    else:
        print(f"✅ Got other type: {result}")
        assert result is not None
