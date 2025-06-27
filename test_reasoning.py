import asyncio
from reson import agentic, Runtime

@agentic(model="openrouter:anthropic/claude-opus-4@reasoning=high")
async def test_reasoning(runtime: Runtime):
    """Test reasoning tokens with o3-mini"""
    
    # Test non-streaming
    print("Testing non-streaming with reasoning...")
    result = await runtime.run(prompt="How would you build the world's tallest skyscraper? {{return_type}}", output_type=str)
    print(f"Result: {result}")
    print(f"Raw response: {runtime.raw_response}")
    print(f"Reasoning: {runtime.reasoning}")
    print("-" * 80)
    
    # Test with a different reasoning level
    print("\nTesting with medium reasoning effort...")
    result2 = await runtime.run(prompt="What's bigger, 9.11 or 9.9? {{return_type}}", output_type=str)
    print(f"Result: {result2}")
    print(f"Reasoning: {runtime.reasoning}")
    print("-" * 80)

@agentic(model="openrouter:anthropic/claude-3.7-sonnet@reasoning=2000")
async def test_anthropic_reasoning(runtime: Runtime):
    """Test reasoning tokens with Anthropic model using max_tokens"""
    
    print("\nTesting Anthropic with max_tokens reasoning...")
    result = await runtime.run(prompt="What's the most efficient algorithm for sorting a large dataset? {{return_type}}", output_type=str)
    print(f"Result: {result}")
    print(f"Raw response: {runtime.raw_response}")
    print(f"Reasoning: {runtime.reasoning}")

async def main():
    # Test OpenAI o3-mini with reasoning
    # await test_reasoning()
    
    # Test Anthropic with reasoning
    await test_anthropic_reasoning()

if __name__ == "__main__":
    asyncio.run(main())
