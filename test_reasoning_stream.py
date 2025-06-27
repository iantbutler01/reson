import asyncio
from reson import agentic, Runtime

@agentic(model="openrouter:openai/o3-mini@reasoning=high")
async def test_reasoning_stream(runtime: Runtime):
    """Test reasoning tokens with streaming"""
    
    print("Testing streaming with reasoning...")
    print("Content chunks:")
    
    # Collect all chunks
    chunks = []
    async for chunk in runtime.run_stream(prompt="How would you build the world's tallest skyscraper?"):
        chunks.append(chunk)
        print(f"  {repr(chunk)}")
    
    print(f"\nFull result: {''.join(str(c) for c in chunks)}")
    print(f"Raw response: {runtime.raw_response}")
    print(f"Reasoning: {runtime.reasoning}")
    print("-" * 80)

@agentic(model="openrouter:anthropic/claude-3.7-sonnet@reasoning=2000")
async def test_anthropic_reasoning_stream(runtime: Runtime):
    """Test reasoning tokens with Anthropic model using streaming"""
    
    print("\nTesting Anthropic streaming with max_tokens reasoning...")
    print("Content chunks:")
    
    chunks = []
    async for chunk in runtime.run_stream(prompt="Explain quantum computing in simple terms.", output_type=str):
        chunks.append(chunk)
        print(f"  {repr(chunk)}")
    
    print(f"\nFull result: {''.join(str(c) for c in chunks)}")
    print(f"Raw response: {runtime.raw_response}")
    print(f"Reasoning: {runtime.reasoning}")

async def main():
    # Test OpenAI o3-mini with reasoning in streaming mode
    await test_reasoning_stream()
    
    # Test Anthropic with reasoning in streaming mode
    # await test_anthropic_reasoning_stream()

if __name__ == "__main__":
    asyncio.run(main())
