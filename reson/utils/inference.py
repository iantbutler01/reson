import re
from typing import Optional, Protocol
from asimov.services.inference_clients import (
    InferenceClient,
    BedrockInferenceClient,
    AnthropicInferenceClient,
    OpenRouterInferenceClient,
    GoogleGenAIInferenceClient,
    GoogleAnthropicInferenceClient,
)
import os

from reson.constants import BIG_MODEL, MODEL_CONFIGURATION


class CreateInferenceClientProtocol(Protocol):
    def __call__(self, model: str, suite: str = "default") -> InferenceClient: ...


def create_google_gemini_api_client(
    model: str, suite: str = "default", api_key: Optional[str] = None
) -> GoogleGenAIInferenceClient:
    return GoogleGenAIInferenceClient(
        MODEL_CONFIGURATION["google-gemini"][suite][model],
        api_key or os.environ["GOOGLE_GEMINI_API_KEY"],
    )


def create_openrouter_inference_client(
    model: str, suite: str = "default", api_key: Optional[str] = None
) -> OpenRouterInferenceClient:
    return OpenRouterInferenceClient(
        model=MODEL_CONFIGURATION["openrouter"][suite][model],
        api_key=api_key or os.environ["OPENROUTER_KEY"],
    )


def create_anthropic_inference_client(
    model: str, suite: str = "default", api_key: Optional[str] = None
) -> AnthropicInferenceClient:
    if suite.startswith("thinking"):
        return AnthropicInferenceClient(
            model=MODEL_CONFIGURATION["anthropic"]["thinking"][model],
            api_key=api_key or os.environ["ANTHROPIC_KEY"],
            thinking=(int(suite[len("thinking:") :]) if model == BIG_MODEL else None),
        )
    return AnthropicInferenceClient(
        model=MODEL_CONFIGURATION["anthropic"][suite][model],
        api_key=api_key or os.environ["ANTHROPIC_KEY"],
    )


def create_bedrock_inference_client(
    model: str,
    suite: str = "default",
) -> BedrockInferenceClient:
    region_name = os.environ.get("AWS_REGION", "us-west-2")
    return BedrockInferenceClient(
        model=MODEL_CONFIGURATION["bedrock"][suite][model], region_name=region_name
    )


def create_google_anthropic_inference_client(
    model: str,
    suite: str = "default",
) -> GoogleAnthropicInferenceClient:
    return GoogleAnthropicInferenceClient(
        MODEL_CONFIGURATION["google-anthropic"][suite][model],
    )