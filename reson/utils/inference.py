import re
from typing import Optional, Protocol
from reson.services.inference_clients import (
    InferenceClient,
    BedrockInferenceClient,
    AnthropicInferenceClient,
    OpenRouterInferenceClient,
    GoogleGenAIInferenceClient,
    GoogleAnthropicInferenceClient,
)
import os


class CreateInferenceClientProtocol(Protocol):
    def __call__(self, model: str) -> InferenceClient: ...


def create_google_gemini_api_client(
    model: str, api_key: Optional[str] = None
) -> GoogleGenAIInferenceClient:
    return GoogleGenAIInferenceClient(
        model,
        api_key or os.environ["GOOGLE_GEMINI_API_KEY"],
    )


if os.environ.get("ART_ENABLED"):
    from reson.services.inference_clients import ARTInferenceClient

    def create_art_inference_client(
        model: str, name, project, backend
    ) -> ARTInferenceClient:
        return ARTInferenceClient(name, model, project, backend)


def create_openrouter_inference_client(
    model: str, api_key: Optional[str] = None, reasoning: str = ""
) -> OpenRouterInferenceClient:
    return OpenRouterInferenceClient(
        model=model,
        api_key=api_key or os.environ["OPENROUTER_KEY"],
        reasoning=reasoning,
    )


def create_anthropic_inference_client(
    model: str, api_key: Optional[str] = None, thinking: Optional[int] = None
) -> AnthropicInferenceClient:
    return AnthropicInferenceClient(
        model=model,
        api_key=api_key or os.environ["ANTHROPIC_KEY"],
        thinking=thinking,
    )


def create_bedrock_inference_client(
    model: str,
) -> BedrockInferenceClient:
    region_name = os.environ.get("AWS_REGION", "us-west-2")
    return BedrockInferenceClient(model=model, region_name=region_name)


def create_google_anthropic_inference_client(
    model: str,
) -> GoogleAnthropicInferenceClient:
    return GoogleAnthropicInferenceClient(
        model=model,
    )


def create_vertex_gemini_api_client(
    model: str,
    reasoning: Optional[str] = None,
) -> GoogleGenAIInferenceClient:
    return GoogleGenAIInferenceClient(
        model=model,
        vertexai=True,
        location="global",
        reasoning=reasoning,
    )
