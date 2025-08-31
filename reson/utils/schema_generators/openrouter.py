"""OpenRouter-format tool schema generator."""

from .openai import OpenAISchemaGenerator
from reson.services.inference_clients import InferenceProvider


class OpenRouterSchemaGenerator(OpenAISchemaGenerator):
    """Generate OpenRouter-compatible schemas (inherits from OpenAI format)."""

    def get_provider_name(self) -> str:
        return InferenceProvider.OPENROUTER.value
