import pytest
from reson.reson import _create_inference_client
from reson.services.inference_clients import (
    OpenRouterInferenceClient,
    AnthropicInferenceClient,
    GoogleGenAIInferenceClient,
    OAIInferenceClient,
)


def test_create_inference_client_openrouter():
    # Test case 1: No reasoning parameter
    model_str = "openrouter:anthropic/claude-sonnet-4"
    client = _create_inference_client(model_str, api_key="test").client
    assert isinstance(client, OpenRouterInferenceClient)
    assert client.model == "anthropic/claude-sonnet-4"
    assert client.api_key == "test"
    assert client.reasoning == ""

    # Test case 2: String reasoning parameter
    model_str_reasoning_str = "openrouter:anthropic/claude-sonnet-4@reasoning=auto"
    client_reasoning_str = _create_inference_client(
        model_str_reasoning_str, api_key="test"
    ).client
    assert isinstance(client_reasoning_str, OpenRouterInferenceClient)
    assert client_reasoning_str.model == "anthropic/claude-sonnet-4"
    assert client_reasoning_str.api_key == "test"
    assert client_reasoning_str.reasoning == "auto"

    # Test case 3: Numeric reasoning parameter
    model_str_reasoning_num = "openrouter:anthropic/claude-sonnet-4@reasoning=1"
    client_reasoning_num = _create_inference_client(
        model_str_reasoning_num, api_key="test"
    ).client
    assert isinstance(client_reasoning_num, OpenRouterInferenceClient)
    assert client_reasoning_num.model == "anthropic/claude-sonnet-4"
    assert client_reasoning_num.api_key == "test"
    assert client_reasoning_num.reasoning == "1"


def test_create_inference_client_anthropic():
    # Test case 1: No thinking parameter
    model_str = "anthropic:claude-sonnet-4-20250514"
    client = _create_inference_client(model_str, api_key="test").client
    assert isinstance(client, AnthropicInferenceClient)
    assert client.model == "claude-sonnet-4-20250514"
    assert client.api_key == "test"
    assert client.thinking is None

    # Test case 2: With thinking parameter
    model_str_thinking = "anthropic:claude-sonnet-4-20250514@thinking=1"
    client_thinking = _create_inference_client(
        model_str_thinking, api_key="test"
    ).client
    assert isinstance(client_thinking, AnthropicInferenceClient)
    assert client_thinking.model == "claude-sonnet-4-20250514"
    assert client_thinking.api_key == "test"
    assert client_thinking.thinking == 1


def test_create_inference_client_vertex_gemini():
    # Test case 1: No reasoning parameter
    model_str = "vertex-gemini:gemini-2.5-pro"
    client = _create_inference_client(model_str).client
    assert isinstance(client, GoogleGenAIInferenceClient)
    assert client.model == "gemini-2.5-pro"
    assert client.reasoning is None

    # Test case 2: String reasoning parameter
    model_str_reasoning_str = "vertex-gemini:gemini-2.5-pro@reasoning=auto"
    client_reasoning_str = _create_inference_client(model_str_reasoning_str).client
    assert isinstance(client_reasoning_str, GoogleGenAIInferenceClient)
    assert client_reasoning_str.model == "gemini-2.5-pro"
    assert client_reasoning_str.reasoning == "auto"

    # Test case 3: Numeric reasoning parameter
    model_str_reasoning_num = "vertex-gemini:gemini-2.5-pro@reasoning=1"
    client_reasoning_num = _create_inference_client(model_str_reasoning_num).client
    assert isinstance(client_reasoning_num, GoogleGenAIInferenceClient)
    assert client_reasoning_num.model == "gemini-2.5-pro"
    assert client_reasoning_num.reasoning == "1"


def test_create_inference_client_openai():
    # Test case 1: No reasoning parameter
    model_str = "openai:gpt-4o"
    client = _create_inference_client(model_str).client
    assert isinstance(client, OAIInferenceClient)
    assert client.model == "gpt-4o"
    assert client.reasoning is None

    # Test case 2: Reasoning parameter is stripped
    model_str_reasoning = "openai:gpt-4o@reasoning=auto"
    client_reasoning = _create_inference_client(model_str_reasoning).client
    assert isinstance(client_reasoning, OAIInferenceClient)
    assert client_reasoning.model == "gpt-4o"
    assert client_reasoning.reasoning is None


def test_create_inference_client_custom_openai():
    # Test case 1: No reasoning parameter
    model_str = "custom-openai:kimi-k2@server_url=http://localhost:8080"
    client = _create_inference_client(model_str).client
    assert isinstance(client, OAIInferenceClient)
    assert client.model == "kimi-k2"
    assert client.api_url == "http://localhost:8080"
    assert client.reasoning is None

    # Test case 2: With reasoning parameter
    model_str_reasoning = (
        "custom-openai:kimi-k2@server_url=http://localhost:8080@reasoning=auto"
    )
    client_reasoning = _create_inference_client(model_str_reasoning).client
    assert isinstance(client_reasoning, OAIInferenceClient)
    assert client_reasoning.model == "kimi-k2"
    assert client_reasoning.api_url == "http://localhost:8080"
    assert client_reasoning.reasoning == "auto"

    # Test case 3: Missing server_url
    with pytest.raises(
        ValueError, match="Custom OpenAI model must include @server_url=<url> parameter"
    ):
        _create_inference_client("custom-openai:kimi-k2")


def test_unsupported_provider():
    with pytest.raises(ValueError, match="Unsupported provider: fake-provider"):
        _create_inference_client("fake-provider:some-model")
