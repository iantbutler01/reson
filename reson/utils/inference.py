import re
from typing import Optional, Protocol
from asimov.services.inference_clients import (
    InferenceClient,
    BedrockInferenceClient,
    AnthropicInferenceClient,
    OpenRouterInferenceClient,
    GoogleGenAIInferenceClient,
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


def create_openrouter_inference_client(
    model: str, api_key: Optional[str] = None
) -> OpenRouterInferenceClient:
    return OpenRouterInferenceClient(
        model=model,
        api_key=api_key or os.environ["OPENROUTER_KEY"],
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
    return BedrockInferenceClient(
        model=model, 
        region_name=region_name
    )
