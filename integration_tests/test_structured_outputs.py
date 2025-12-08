"""Integration tests for structured outputs with real LLM APIs.

Tests structured output (output_type) functionality across:
- OpenRouter (OpenAI proxy)
- Google Gemini
- Anthropic (via OpenRouter)

Run with:
    OPENROUTER_API_KEY=xxx GOOGLE_GEMINI_API_KEY=xxx pytest integration_tests/test_structured_outputs.py -v --capture=no
"""

import asyncio
import os
import pytest
from typing import List, Optional
from pydantic import BaseModel, Field
from reson import Runtime
from reson.types import Deserializable


# ============================================================================
# Test Output Types
# ============================================================================

class Person(BaseModel):
    """A person with basic information."""
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")
    occupation: Optional[str] = Field(default=None, description="The person's job or profession")


class MovieReview(BaseModel):
    """A structured movie review."""
    title: str = Field(description="The movie title")
    rating: int = Field(description="Rating from 1-10")
    summary: str = Field(description="Brief summary of the review")
    pros: List[str] = Field(description="List of positive aspects")
    cons: List[str] = Field(description="List of negative aspects")


class MathResult(BaseModel):
    """Result of a mathematical calculation."""
    expression: str = Field(description="The mathematical expression")
    result: float = Field(description="The numerical result")
    explanation: str = Field(description="Step-by-step explanation")


class WeatherInfo(Deserializable):
    """Weather information using Deserializable."""
    location: str
    temperature: float
    conditions: str
    humidity: Optional[int] = None


class CodeAnalysis(BaseModel):
    """Analysis of a code snippet."""
    language: str = Field(description="Programming language")
    purpose: str = Field(description="What the code does")
    complexity: str = Field(description="Complexity level: simple, moderate, complex")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")


# ============================================================================
# OpenRouter (OpenAI Proxy) Tests
# ============================================================================

@pytest.mark.asyncio
async def test_openrouter_structured_output_pydantic():
    """Test OpenRouter structured output with Pydantic model."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")

    print("\n" + "=" * 60)
    print("Testing OpenRouter Structured Output (Pydantic)")
    print("=" * 60)

    runtime = Runtime(model="openrouter:openai/gpt-4o-mini", api_key=api_key)

    result = await runtime.run(
        prompt="Create a person named Alice who is 28 years old and works as a software engineer.",
        output_type=Person
    )

    print(f"Result type: {type(result)}")
    print(f"Result: {result}")

    # Verify it's hydrated into a Person instance
    assert isinstance(result, Person), f"Expected Person, got {type(result)}"
    assert result.name == "Alice" or "alice" in result.name.lower()
    assert isinstance(result.age, int)
    assert result.age == 28
    print(f"Name: {result.name}")
    print(f"Age: {result.age}")
    print(f"Occupation: {result.occupation}")
    print("OpenRouter Pydantic test PASSED")


@pytest.mark.asyncio
async def test_openrouter_structured_output_complex():
    """Test OpenRouter with complex nested Pydantic model."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")

    print("\n" + "=" * 60)
    print("Testing OpenRouter Structured Output (Complex Model)")
    print("=" * 60)

    runtime = Runtime(model="openrouter:openai/gpt-4o-mini", api_key=api_key)

    result = await runtime.run(
        prompt="Write a review of the movie 'Inception'. Give it a rating of 9, mention the dream sequences as a pro, and the complex plot as a con.",
        output_type=MovieReview
    )

    print(f"Result type: {type(result)}")
    print(f"Result: {result}")

    assert isinstance(result, MovieReview), f"Expected MovieReview, got {type(result)}"
    assert "inception" in result.title.lower()
    assert isinstance(result.rating, int)
    assert 1 <= result.rating <= 10
    assert isinstance(result.pros, list)
    assert isinstance(result.cons, list)
    print(f"Title: {result.title}")
    print(f"Rating: {result.rating}")
    print(f"Summary: {result.summary}")
    print(f"Pros: {result.pros}")
    print(f"Cons: {result.cons}")
    print("OpenRouter Complex Model test PASSED")


@pytest.mark.asyncio
async def test_openrouter_structured_output_math():
    """Test OpenRouter with math calculation output."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")

    print("\n" + "=" * 60)
    print("Testing OpenRouter Structured Output (Math)")
    print("=" * 60)

    runtime = Runtime(model="openrouter:openai/gpt-4o-mini", api_key=api_key)

    result = await runtime.run(
        prompt="Calculate 15 * 7 + 23. Show your work.",
        output_type=MathResult
    )

    print(f"Result type: {type(result)}")
    print(f"Result: {result}")

    assert isinstance(result, MathResult), f"Expected MathResult, got {type(result)}"
    assert isinstance(result.result, (int, float))
    # 15 * 7 + 23 = 105 + 23 = 128
    assert abs(result.result - 128) < 0.01, f"Expected 128, got {result.result}"
    print(f"Expression: {result.expression}")
    print(f"Result: {result.result}")
    print(f"Explanation: {result.explanation}")
    print("OpenRouter Math test PASSED")


# ============================================================================
# Google Gemini Tests
# ============================================================================

@pytest.mark.asyncio
async def test_google_gemini_structured_output_pydantic():
    """Test Google Gemini structured output with Pydantic model."""
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_GEMINI_API_KEY not set")

    print("\n" + "=" * 60)
    print("Testing Google Gemini Structured Output (Pydantic)")
    print("=" * 60)

    runtime = Runtime(model="google-genai:gemini-2.0-flash", api_key=api_key)

    result = await runtime.run(
        prompt="Create a person named Bob who is 35 years old and works as a doctor.",
        output_type=Person
    )

    print(f"Result type: {type(result)}")
    print(f"Result: {result}")

    assert isinstance(result, Person), f"Expected Person, got {type(result)}"
    assert "bob" in result.name.lower()
    assert isinstance(result.age, int)
    assert result.age == 35
    print(f"Name: {result.name}")
    print(f"Age: {result.age}")
    print(f"Occupation: {result.occupation}")
    print("Google Gemini Pydantic test PASSED")


@pytest.mark.asyncio
async def test_google_gemini_structured_output_deserializable():
    """Test Google Gemini structured output with Deserializable class."""
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_GEMINI_API_KEY not set")

    print("\n" + "=" * 60)
    print("Testing Google Gemini Structured Output (Deserializable)")
    print("=" * 60)

    runtime = Runtime(model="google-genai:gemini-2.0-flash", api_key=api_key)

    result = await runtime.run(
        prompt="Give me weather info for Tokyo with temperature 22.5 degrees, sunny conditions, and 65% humidity.",
        output_type=WeatherInfo
    )

    print(f"Result type: {type(result)}")
    print(f"Result: {result}")

    assert isinstance(result, WeatherInfo), f"Expected WeatherInfo, got {type(result)}"
    assert "tokyo" in result.location.lower()
    assert isinstance(result.temperature, (int, float))
    print(f"Location: {result.location}")
    print(f"Temperature: {result.temperature}")
    print(f"Conditions: {result.conditions}")
    print(f"Humidity: {result.humidity}")
    print("Google Gemini Deserializable test PASSED")


@pytest.mark.asyncio
async def test_google_gemini_structured_output_complex():
    """Test Google Gemini with complex nested model."""
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_GEMINI_API_KEY not set")

    print("\n" + "=" * 60)
    print("Testing Google Gemini Structured Output (Complex)")
    print("=" * 60)

    runtime = Runtime(model="google-genai:gemini-2.0-flash", api_key=api_key)

    result = await runtime.run(
        prompt="Analyze this Python code: `def add(a, b): return a + b`. It's a simple function.",
        output_type=CodeAnalysis
    )

    print(f"Result type: {type(result)}")
    print(f"Result: {result}")

    assert isinstance(result, CodeAnalysis), f"Expected CodeAnalysis, got {type(result)}"
    assert "python" in result.language.lower()
    assert result.complexity.lower() in ["simple", "moderate", "complex"]
    print(f"Language: {result.language}")
    print(f"Purpose: {result.purpose}")
    print(f"Complexity: {result.complexity}")
    print(f"Suggestions: {result.suggestions}")
    print("Google Gemini Complex test PASSED")


# ============================================================================
# Anthropic (via OpenRouter) Tests
# ============================================================================
# NOTE: OpenRouter does NOT support structured outputs for Anthropic models.
# OpenRouter uses the OpenAI API format, and Anthropic's structured output
# feature (response_format with json_schema) is not available through OpenRouter.
# To use structured outputs with Anthropic models, use the native Anthropic API.
# These tests are skipped for now.

@pytest.mark.asyncio
@pytest.mark.skip(reason="OpenRouter does not support structured outputs for Anthropic models")
async def test_anthropic_structured_output_pydantic():
    """Test Anthropic (via OpenRouter) structured output with Pydantic model."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")

    print("\n" + "=" * 60)
    print("Testing Anthropic Structured Output (Pydantic)")
    print("=" * 60)

    # Use Claude via OpenRouter
    runtime = Runtime(model="openrouter:anthropic/claude-sonnet-4", api_key=api_key)

    result = await runtime.run(
        prompt="Create a person named Carol who is 42 years old and works as an architect.",
        output_type=Person
    )

    print(f"Result type: {type(result)}")
    print(f"Result: {result}")

    assert isinstance(result, Person), f"Expected Person, got {type(result)}"
    assert "carol" in result.name.lower()
    assert isinstance(result.age, int)
    assert result.age == 42
    print(f"Name: {result.name}")
    print(f"Age: {result.age}")
    print(f"Occupation: {result.occupation}")
    print("Anthropic Pydantic test PASSED")


@pytest.mark.asyncio
@pytest.mark.skip(reason="OpenRouter does not support structured outputs for Anthropic models")
async def test_anthropic_structured_output_complex():
    """Test Anthropic with complex movie review model."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")

    print("\n" + "=" * 60)
    print("Testing Anthropic Structured Output (Complex)")
    print("=" * 60)

    runtime = Runtime(model="openrouter:anthropic/claude-sonnet-4", api_key=api_key)

    result = await runtime.run(
        prompt="Write a review of 'The Matrix'. Rating 8. Pros: groundbreaking visuals, philosophical depth. Cons: some dated effects.",
        output_type=MovieReview
    )

    print(f"Result type: {type(result)}")
    print(f"Result: {result}")

    assert isinstance(result, MovieReview), f"Expected MovieReview, got {type(result)}"
    assert "matrix" in result.title.lower()
    assert isinstance(result.rating, int)
    assert isinstance(result.pros, list)
    assert isinstance(result.cons, list)
    print(f"Title: {result.title}")
    print(f"Rating: {result.rating}")
    print(f"Summary: {result.summary}")
    print(f"Pros: {result.pros}")
    print(f"Cons: {result.cons}")
    print("Anthropic Complex test PASSED")


@pytest.mark.asyncio
@pytest.mark.skip(reason="OpenRouter does not support structured outputs for Anthropic models")
async def test_anthropic_structured_output_math():
    """Test Anthropic with math calculation."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")

    print("\n" + "=" * 60)
    print("Testing Anthropic Structured Output (Math)")
    print("=" * 60)

    runtime = Runtime(model="openrouter:anthropic/claude-sonnet-4", api_key=api_key)

    result = await runtime.run(
        prompt="Calculate (25 + 15) * 2. Show your work step by step.",
        output_type=MathResult
    )

    print(f"Result type: {type(result)}")
    print(f"Result: {result}")

    assert isinstance(result, MathResult), f"Expected MathResult, got {type(result)}"
    assert isinstance(result.result, (int, float))
    # (25 + 15) * 2 = 40 * 2 = 80
    assert abs(result.result - 80) < 0.01, f"Expected 80, got {result.result}"
    print(f"Expression: {result.expression}")
    print(f"Result: {result.result}")
    print(f"Explanation: {result.explanation}")
    print("Anthropic Math test PASSED")


# ============================================================================
# Main runner
# ============================================================================

async def main():
    """Run all structured output tests."""
    print("=" * 60)
    print("STRUCTURED OUTPUT INTEGRATION TESTS")
    print("=" * 60)

    # Check environment
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    gemini_key = os.getenv("GOOGLE_GEMINI_API_KEY")

    print(f"OpenRouter API Key: {'set' if openrouter_key else 'MISSING'}")
    print(f"Google Gemini API Key: {'set' if gemini_key else 'MISSING'}")

    tests = []

    if openrouter_key:
        tests.extend([
            ("OpenRouter Pydantic", test_openrouter_structured_output_pydantic()),
            ("OpenRouter Complex", test_openrouter_structured_output_complex()),
            ("OpenRouter Math", test_openrouter_structured_output_math()),
            ("Anthropic Pydantic", test_anthropic_structured_output_pydantic()),
            ("Anthropic Complex", test_anthropic_structured_output_complex()),
            ("Anthropic Math", test_anthropic_structured_output_math()),
        ])

    if gemini_key:
        tests.extend([
            ("Google Gemini Pydantic", test_google_gemini_structured_output_pydantic()),
            ("Google Gemini Deserializable", test_google_gemini_structured_output_deserializable()),
            ("Google Gemini Complex", test_google_gemini_structured_output_complex()),
        ])

    if not tests:
        print("No API keys set. Set OPENROUTER_API_KEY and/or GOOGLE_GEMINI_API_KEY")
        return

    results = []
    for name, test_coro in tests:
        try:
            await test_coro
            results.append((name, True, None))
        except Exception as e:
            results.append((name, False, str(e)))
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "PASSED" if success else f"FAILED: {error}"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll structured output tests PASSED!")
    else:
        print("\nSome tests FAILED - check output above")


if __name__ == "__main__":
    asyncio.run(main())
