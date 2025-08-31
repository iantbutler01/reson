"""Schema generators for native tool calling support."""

from .base import SchemaGenerator
from .openai import OpenAISchemaGenerator
from .anthropic import AnthropicSchemaGenerator
from .google import GoogleSchemaGenerator
from .openrouter import OpenRouterSchemaGenerator
from reson.services.inference_clients import InferenceProvider

# Provider mapping
SCHEMA_GENERATORS = {
    InferenceProvider.OPENAI.value: OpenAISchemaGenerator,
    InferenceProvider.ANTHROPIC.value: AnthropicSchemaGenerator,
    InferenceProvider.GOOGLE_GENAI.value: GoogleSchemaGenerator,
    InferenceProvider.GOOGLE_GEMINI.value: GoogleSchemaGenerator,
    InferenceProvider.VERTEX_GEMINI.value: GoogleSchemaGenerator,
    InferenceProvider.GOOGLE_ANTHROPIC.value: AnthropicSchemaGenerator,  # Anthropic models on Google Vertex
    InferenceProvider.OPENROUTER.value: OpenRouterSchemaGenerator,
    InferenceProvider.BEDROCK.value: AnthropicSchemaGenerator,  # Uses Anthropic format
    InferenceProvider.CUSTOM_OPENAI.value: OpenAISchemaGenerator,
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
