import asyncio
import pytest
import os
from reson import agentic_generator, Runtime
from reson.types import ReasoningSegment


models_to_test = [
    "openrouter:anthropic/claude-3.5-sonnet@reasoning=2000",
    "openrouter:openai/gpt-4o@reasoning=high",
    "google-gemini:gemini-flash-latest@reasoning=1024",
]


@pytest.mark.parametrize("model", models_to_test)
@pytest.mark.asyncio
async def test_reasoning_segments_across_providers(model: str):
    """Test basic ReasoningSegment functionality across multiple providers."""
    if "google" in model and not os.getenv("GOOGLE_GEMINI_API_KEY"):
        pytest.skip("GOOGLE_GEMINI_API_KEY not set")
    if "openrouter" in model and not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    print(f"\nTesting model: {model}")

    @agentic_generator()
    async def reasoning_segments_agent(query: str, runtime: Runtime):
        async for chunk_type, content in runtime.run_stream(prompt=query, model=model):
            if chunk_type == "reasoning":
                yield f"📝 Reasoning chunk: {content[:50]}..."
            elif chunk_type == "content":
                yield f"💬 Content: {content}"

        # Check reasoning segments after streaming
        yield f"🔍 Reasoning segments count: {len(runtime.reasoning_segments)}"
        for i, segment in enumerate(runtime.reasoning_segments):
            yield f"Segment {i}: content_length={len(segment.content)}, has_signature={segment.signature is not None}"

    try:
        results = []
        async for result in reasoning_segments_agent(
            "Explain the process of photosynthesis in detail"
        ):
            print(result)
            results.append(result)

        # Check if we got reasoning segments
        segment_info = [r for r in results if "Reasoning segments count:" in r]
        assert len(segment_info) > 0, "No reasoning segments info found"

        print(f"✅ Reasoning segments test for {model} completed")

    except Exception as e:
        print(f"❌ Reasoning segments test for {model} failed: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Test failed for model {model} with exception {e}")


@pytest.mark.asyncio
async def test_reasoning_segment_provider_formats():
    """Test ReasoningSegment provider format conversion."""
    from reson.services.inference_clients import InferenceProvider

    # Test basic ReasoningSegment creation and format conversion
    segment = ReasoningSegment(
        content="This is test reasoning content",
        signature="test_signature_123",
        provider_metadata={"test": "metadata"},
        segment_index=1,
    )

    # Test Anthropic format
    anthropic_format = segment.to_provider_format(InferenceProvider.ANTHROPIC)
    print(f"Anthropic format: {anthropic_format}")
    assert anthropic_format["type"] == "thinking"
    assert anthropic_format["thinking"] == "This is test reasoning content"
    assert anthropic_format["signature"] == "test_signature_123"
    assert anthropic_format["test"] == "metadata"

    # Test OpenAI format
    openai_format = segment.to_provider_format(InferenceProvider.OPENAI)
    print(f"OpenAI format: {openai_format}")
    assert openai_format["type"] == "reasoning"
    assert openai_format["content"] == "This is test reasoning content"
    assert openai_format["signature"] == "test_signature_123"

    # Test Google format
    google_format = segment.to_provider_format(InferenceProvider.GOOGLE_GENAI)
    print(f"Google format: {google_format}")
    assert google_format["thought"] == True
    assert google_format["text"] == "This is test reasoning content"
    assert google_format["thought_signature"] == "test_signature_123"

    print("✅ ReasoningSegment provider format conversion test passed")
    return True


if __name__ == "__main__":
    asyncio.run(test_reasoning_segment_provider_formats())
    # To run the provider tests individually:
    # asyncio.run(test_reasoning_segments_across_providers(models_to_test[0]))
