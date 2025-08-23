"""OpenRouter-format tool schema generator."""

from .openai import OpenAISchemaGenerator


class OpenRouterSchemaGenerator(OpenAISchemaGenerator):
    """Generate OpenRouter-compatible schemas (inherits from OpenAI format)."""

    def get_provider_name(self) -> str:
        return "openrouter"
