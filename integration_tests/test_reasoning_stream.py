import asyncio
from reson import agentic, Runtime

@agentic(model="openrouter:openai/o3-mini@reasoning=high")
async def test_reasoning_stream(runtime: Runtime) -> str:
    """Test reasoning tokens with streaming
    
    How would you build the world's tallest skyscraper?
    
    {{return_type}}
    """
    
    print("Testing streaming with reasoning...")
    print("Content chunks:")
    
    # Collect all chunks
    chunks = []
    async for chunk in runtime.run_stream():
        chunks.append(chunk)
        print(f"  {repr(chunk)}")
    
    print(f"\nFull result: {''.join(str(c) for c in chunks)}")
    print(f"Raw response: {runtime.raw_response}")
    print(f"Reasoning: {runtime.reasoning}")
    print("-" * 80)
    
    return ''.join(str(c) for c in chunks)

@agentic(model="openrouter:anthropic/claude-3.7-sonnet@reasoning=2000")
async def test_anthropic_reasoning_stream(runtime: Runtime) -> str:
    """Test reasoning tokens with Anthropic model using streaming
    
    Explain quantum computing in simple terms.
    
    {{return_type}}
    """
    
    print("\nTesting Anthropic streaming with max_tokens reasoning...")
    print("Content chunks:")
    
    chunks = []
    async for chunk in runtime.run_stream():
        chunks.append(chunk)
        print(f"  {repr(chunk)}")
    
    print(f"\nFull result: {''.join(str(c) for c in chunks)}")
    print(f"Raw response: {runtime.raw_response}")
    print(f"Reasoning: {runtime.reasoning}")
    
    return ''.join(str(c) for c in chunks)

async def main():
    # Test OpenAI o3-mini with reasoning in streaming mode
    await test_reasoning_stream()
    
    # Test Anthropic with reasoning in streaming mode
    await test_anthropic_reasoning_stream()

if __name__ == "__main__":
    asyncio.run(main())
