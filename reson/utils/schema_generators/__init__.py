"""Schema generators for native tool calling support."""

from .base import SchemaGenerator
from .openai import OpenAISchemaGenerator
from .anthropic import AnthropicSchemaGenerator
from .google import GoogleSchemaGenerator
from .openrouter import OpenRouterSchemaGenerator

# Provider mapping
SCHEMA_GENERATORS = {
    "openai": OpenAISchemaGenerator,
    "anthropic": AnthropicSchemaGenerator,
    "google-gemini": GoogleSchemaGenerator,
    "vertex-gemini": GoogleSchemaGenerator,
    "openrouter": OpenRouterSchemaGenerator,
    "bedrock": AnthropicSchemaGenerator,  # Uses Anthropic format
    "custom-openai": OpenAISchemaGenerator,
}


def get_schema_generator(model_string: str) -> SchemaGenerator:
    """Get appropriate schema generator for model string."""
    provider = model_string.split(":", 1)[0] if ":" in model_string else model_string

    generator_class = SCHEMA_GENERATORS.get(provider)
    if not generator_class:
        raise ValueError(f"Native tools not supported for provider: {provider}")

    return generator_class()


def supports_native_tools(model_string: str) -> bool:
    """Check if model supports native tool calling."""
    provider = model_string.split(":", 1)[0] if ":" in model_string else model_string
    return provider in SCHEMA_GENERATORS


__all__ = [
    "SchemaGenerator",
    "get_schema_generator",
    "supports_native_tools",
    "SCHEMA_GENERATORS",
]
