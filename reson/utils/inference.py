from __future__ import annotations
from typing import Optional, Dict, Callable, List, Type, Any, Protocol
from reson.services.inference_clients import (
    InferenceClient,
    BedrockInferenceClient,
    AnthropicInferenceClient,
    OpenRouterInferenceClient,
    GoogleGenAIInferenceClient,
    GoogleAnthropicInferenceClient,
    OAIInferenceClient,
)
import os
from reson.types import (
    ChatMessage,
    ToolResult,
    ReasoningSegment,
    Deserializable,
    ChatRole,
)
from reson.stores import Store
from reson.services.inference_clients import InferenceProvider
from reson.tracing_inference_client import TracingInferenceClient
import re
from reson.services.inference_clients import (
    ChatMessage,
    ChatRole,
    ToolResult,
    ReasoningSegment,
    InferenceProvider,
)
import os
from typing import (
    Union,
    List,
    Dict,
    Any,
    Optional,
    Type,
    get_origin,
    get_args,
    Callable,
    Union,
)

from typing import Union


from pydantic import BaseModel

import inspect
from typing import Callable, Any, Union
import re

from reson.services.inference_clients import ChatMessage, ChatRole


from reson.utils.parsers import OutputParser, get_default_parser, NativeToolParser

from reson.types import Deserializable

from reson.utils.schema_generators import get_schema_generator
from reson.tracing_inference_client import TracingInferenceClient

from reson.utils.parsers import OutputParser, get_default_parser, NativeToolParser


class CreateInferenceClientProtocol(Protocol):
    def __call__(self, model: str) -> InferenceClient: ...


def create_google_gemini_api_client(
    model: str, api_key: Optional[str] = None, reasoning: Optional[int] = None
) -> GoogleGenAIInferenceClient:
    return GoogleGenAIInferenceClient(
        model,
        api_key or os.environ["GOOGLE_GEMINI_API_KEY"],
        reasoning=reasoning,
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
        api_key=api_key
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ["OPENROUTER_KEY"],
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
    model: str, thinking: Optional[int] = None
) -> GoogleAnthropicInferenceClient:
    return GoogleAnthropicInferenceClient(
        model=model,
        thinking=thinking,
    )


def create_vertex_gemini_api_client(
    model: str,
    reasoning: Optional[int] = None,
) -> GoogleGenAIInferenceClient:
    return GoogleGenAIInferenceClient(
        model=model,
        vertexai=True,
        location="global",
        reasoning=reasoning,
    )


def create_openai_inference_client(
    model: str,
    api_url: str = "https://api.openai.com/v1/chat/completions",
    api_key: Optional[str] = None,
    reasoning: Optional[str] = None,
) -> OAIInferenceClient:
    return OAIInferenceClient(
        model=model,
        api_url=api_url,
        api_key=api_key or "",
        reasoning=reasoning,
    )


def _create_inference_client(model_str, store=None, api_key=None, art_backend=None):
    """Create an appropriate inference client based on the model string."""
    # Parse model string to get provider and model
    parts = model_str.split(":", 1)

    provider, model_name = parts

    if provider == InferenceProvider.ART.value and os.environ.get("ART_ENABLED"):
        from reson.utils.inference import create_art_inference_client

        name_project_match = re.match(
            r"(.+)@name=([a-z].*)project=([a-z].*)", model_name
        )

        if not name_project_match:
            raise AttributeError("Name and Project must be included for ART runs.")

        model, name, project = name_project_match.groups()

        return create_art_inference_client(model, name, project, art_backend)

    if provider == InferenceProvider.OPENROUTER.value:
        reasoning_match = re.match(r"(.+)@reasoning=([a-z].*)", model_name)
        if not reasoning_match:
            # Try numeric pattern
            reasoning_match = re.match(r"(.+)@reasoning=(\d+)", model_name)
        if reasoning_match:
            model_name, reasoning = reasoning_match.groups()
            client = create_openrouter_inference_client(
                model_name, reasoning=reasoning, api_key=api_key
            )
        else:
            client = create_openrouter_inference_client(model_name, api_key=api_key)
    elif provider == InferenceProvider.ANTHROPIC.value:
        # Parse out reasoning parameter if provided
        reasoning_match = re.match(r"(.+)@reasoning=(\d+)", model_name)
        if reasoning_match:
            model_name, reasoning = reasoning_match.groups()
            client = create_anthropic_inference_client(
                model_name, thinking=int(reasoning), api_key=api_key
            )
        else:
            client = create_anthropic_inference_client(model_name, api_key=api_key)
    elif provider == InferenceProvider.BEDROCK.value:
        client = create_bedrock_inference_client(model_name)
    elif provider == InferenceProvider.GOOGLE_GEMINI.value:
        reasoning_match = re.match(r"(.+)@reasoning=([a-z].*)", model_name)
        if not reasoning_match:
            # Try numeric pattern
            reasoning_match = re.match(r"(.+)@reasoning=(\d+)", model_name)
        if reasoning_match:
            model_name, reasoning = reasoning_match.groups()
            client = create_google_gemini_api_client(
                model_name, api_key=api_key, reasoning=int(reasoning)
            )
        else:
            client = create_google_gemini_api_client(model_name, api_key=api_key)
    elif provider == InferenceProvider.VERTEX_GEMINI.value:
        reasoning_match = re.match(r"(.+)@reasoning=([a-z].*)", model_name)
        if not reasoning_match:
            # Try numeric pattern
            reasoning_match = re.match(r"(.+)@reasoning=(\d+)", model_name)
        if reasoning_match:
            model_name, reasoning = reasoning_match.groups()
            client = create_vertex_gemini_api_client(
                model_name, reasoning=int(reasoning)
            )
        else:
            client = create_vertex_gemini_api_client(model_name)
    elif provider == InferenceProvider.GOOGLE_ANTHROPIC.value:
        # Parse out reasoning parameter if provided for Google Anthropic
        reasoning_match = re.match(r"(.+)@reasoning=(\d+)", model_name)
        if reasoning_match:
            model_name, reasoning = reasoning_match.groups()
            # Google Anthropic client needs to be created differently - it inherits from Anthropic
            from reson.utils.inference import create_google_anthropic_inference_client

            client = create_google_anthropic_inference_client(
                model_name, thinking=int(reasoning)
            )
        else:
            from reson.utils.inference import create_google_anthropic_inference_client

            client = create_google_anthropic_inference_client(model_name)
    elif provider == InferenceProvider.OPENAI.value:
        # Strip reasoning= from model name if present
        model_name = re.sub(r"@reasoning=.*$", "", model_name)
        client = create_openai_inference_client(model_name, api_key=api_key)
    elif provider == InferenceProvider.CUSTOM_OPENAI.value:
        if "@server_url=" not in model_name:
            raise ValueError(
                "Custom OpenAI model must include @server_url=<url> parameter"
            )
        server_url_match = re.match(r".+@server_url=([^@]+)", model_name)
        if not server_url_match:
            raise ValueError("Could not parse server_url from model name")
        server_url = server_url_match.group(1)
        reasoning_match = re.match(r".+@reasoning=([^@]+)", model_name)
        model_name = re.sub(r"@.*$", "", model_name)
        client = create_openai_inference_client(
            model_name,
            api_url=server_url,
            api_key=api_key,
            reasoning=reasoning_match.group(1) if reasoning_match else None,
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Wrap in TracingInferenceClient
    return TracingInferenceClient(client, store)


def _validate_callable_params(func: Callable, name: str) -> None:
    """Ensure all parameters are properly typed."""
    sig = inspect.signature(func)
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param.annotation == inspect.Parameter.empty:
            raise ValueError(
                f"Parameter '{param_name}' in tool '{name}' must have a type annotation"
            )


def _generate_native_tool_schemas(
    tools: Dict[str, Callable], model: str
) -> List[Dict[str, Any]]:
    """Generate native tool schemas for the given model/provider."""
    if not tools:
        return []

    schema_generator = get_schema_generator(model)
    return schema_generator.generate_tool_schemas(tools)


def _is_pydantic_type(type_annotation) -> bool:
    """Check if a type is Pydantic-based."""
    origin = get_origin(type_annotation)
    if origin is not None:
        # For generics like List[PydanticModel], check the args
        args = get_args(type_annotation)
        return any(_is_pydantic_type(arg) for arg in args if arg != type(None))
    return hasattr(type_annotation, "__mro__") and BaseModel in type_annotation.__mro__


def _is_deserializable_type(type_annotation) -> bool:
    """Check if a type is Deserializable-based."""
    origin = get_origin(type_annotation)
    if origin is not None:
        args = get_args(type_annotation)
        return any(_is_deserializable_type(arg) for arg in args if arg != type(None))
    try:
        return (
            hasattr(type_annotation, "__mro__")
            and Deserializable in type_annotation.__mro__
        )
    except:
        return False


def _get_parser_for_type(output_type=None) -> OutputParser:
    """Get the appropriate parser for the given output type."""
    # For now, just use the default parser
    # In the future, we might select based on type or user configuration
    return get_default_parser()


def _create_pydantic_tool_model(func: Callable, name: str) -> Type:
    """Create a Pydantic model from a callable's signature."""
    from typing import ClassVar

    sig = inspect.signature(func)

    # Build the class dynamically
    attrs = {
        "__annotations__": {},
        "__doc__": func.__doc__ or f"Tool for {name}",
        # Only store the tool name, not the function
        "_tool_name": name,
    }

    # Add annotations for ClassVar
    attrs["__annotations__"]["_tool_name"] = ClassVar[str]

    # Add parameter fields
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        attrs["__annotations__"][param_name] = param.annotation

        # Handle defaults
        if param.default != inspect.Parameter.empty:
            attrs[param_name] = param.default

    # Create the class
    tool_class = type(f"{name.capitalize().replace('_', '')}Tool", (BaseModel,), attrs)

    return tool_class


def _create_deserializable_tool_class(func: Callable, name: str) -> Type:
    """Create a Deserializable class from a callable's signature using type()."""

    sig = inspect.signature(func)

    annotations = {}

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        annotations[param_name] = param.annotation

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        # Store tool name on instance
        self._tool_name = name

    class_attrs = {
        "__annotations__": annotations,
        "__doc__": func.__doc__ or f"Tool for {name}",
        "__init__": __init__,
    }

    for param_name, param in sig.parameters.items():
        if param_name != "self" and param.default != inspect.Parameter.empty:
            class_attrs[param_name] = param.default

    tool_class = type(
        f"{name.capitalize()}Tool", (Deserializable,), class_attrs  # Base class
    )

    return tool_class


def _extract_content_from_response(response: Any, provider: str) -> Optional[str]:
    """Extract text content from provider response when no tool calls made."""
    if (
        provider.startswith(InferenceProvider.OPENAI.value)
        or provider == InferenceProvider.OPENROUTER.value
        or provider == InferenceProvider.CUSTOM_OPENAI.value
    ):
        # Handle both dict and object responses
        if isinstance(response, dict) and "choices" in response:
            choices = response["choices"]
            if choices and len(choices) > 0:
                choice = choices[0]
                if isinstance(choice, dict) and "message" in choice:
                    message = choice["message"]
                    if isinstance(message, dict) and "content" in message:
                        return message["content"]
        elif hasattr(response, "choices") and response.choices:  # type: ignore
            choice = response.choices[0]  # type: ignore
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                return choice.message.content

    elif (
        provider == InferenceProvider.ANTHROPIC.value
        or provider == InferenceProvider.BEDROCK.value
    ):
        if hasattr(response, "content"):
            for block in response.content:
                if hasattr(block, "type") and block.type == "text":
                    return block.text

    elif (
        provider == InferenceProvider.GOOGLE_GENAI.value
        or provider == InferenceProvider.GOOGLE_ANTHROPIC.value
        or provider.startswith("google")
    ):
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                for part in candidate.content.parts:
                    if hasattr(part, "text"):
                        return part.text

    return None


def _parse_list_of_complete_tools(
    tool_calls: List[dict], parser: NativeToolParser, registry: dict[str, Any]
) -> List[Any]:
    tools = []

    for call in tool_calls:
        tool_name = parser.extract_tool_name(call)
        args_json = parser.extract_arguments(call)
        tool_id = parser.extract_tool_id(call) or ""
        delta_result = parser.parse_tool(tool_name, args_json, tool_id)
        if delta_result.success:
            tools.append(delta_result.value)
        else:
            tools.append(call)

    return tools


async def _call_llm_stream(
    prompt: Optional[str],
    model: str,
    tools: Dict[str, Callable],
    output_type: Optional[Type[Any]],
    store: Optional[Store] = None,
    api_key: Optional[str] = None,
    system: Optional[str] = None,
    history: Optional[List[ChatMessage | ToolResult | ReasoningSegment]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    call_context: Optional[Dict[str, Any]] = None,
    art_backend=None,
    native_tools: bool = False,
    tool_types: Dict[str, Type[Deserializable]] = {},
):
    """Execute a streaming LLM call with optional native tool support."""
    client = _create_inference_client(
        model, store, api_key=api_key, art_backend=art_backend
    )

    native_tool_schemas = None
    if native_tools and tools:
        native_tool_schemas = _generate_native_tool_schemas(tools, model)

    effective_output_type = output_type

    if not native_tools and tools and output_type:
        # Validate all callables have typed parameters
        if isinstance(tools, dict):
            _iter_tools = tools.items()
        else:
            _iter_tools = []
        for name, func in _iter_tools:
            _validate_callable_params(func, name)

        tool_models = []
        if _is_pydantic_type(output_type):
            tool_models = [
                _create_pydantic_tool_model(func, name) for name, func in _iter_tools
            ]
        elif _is_deserializable_type(output_type):
            tool_models = [
                _create_deserializable_tool_class(func, name)
                for name, func in _iter_tools
            ]

        if tool_models:
            effective_output_type = Union[*(tool_models + [output_type])]

    parser = _get_parser_for_type(effective_output_type)

    enhanced_prompt_content = prompt
    if prompt is not None and effective_output_type:
        enhanced_prompt_content = parser.enhance_prompt(
            prompt, effective_output_type, call_context=call_context
        )  # MODIFIED

    stream_parser = None
    if effective_output_type:
        stream_parser = parser.create_stream_parser(effective_output_type)

    messages: List[ChatMessage | ToolResult | ReasoningSegment] = []
    if system:
        messages.append(ChatMessage(role=ChatRole.SYSTEM, content=system))
    if history:
        messages.extend(history)
    if enhanced_prompt_content is not None:  # Add the current prompt as a user message
        messages.append(
            ChatMessage(role=ChatRole.USER, content=enhanced_prompt_content)
        )

    if not messages:
        raise ValueError(
            "LLM call attempted with no system message, history, or current prompt."
        )

    call_kwargs = {
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else 8192,
        "top_p": top_p if top_p is not None else 0.9,
        "temperature": temperature if temperature is not None else 0.5,
    }

    if native_tools and native_tool_schemas:
        call_kwargs["tools"] = native_tool_schemas

    if native_tools and tools:
        native_tool_parser = None
        if tool_types:
            native_tool_parser = NativeToolParser(tool_types)

        async for chunk in client.connect_and_listen(**call_kwargs):
            chunk_type, chunk_content = chunk

            if chunk_type == "reasoning":
                yield None, chunk_content, "reasoning"
            elif chunk_type == "content":
                yield chunk_content, chunk_content, "content"
            elif chunk_type == "signature":
                # TODO: Ian -- I wanted to handle signatures properly but got sidetracked. I'll come back to this later
                pass
            elif (
                chunk_type == "tool_call_partial" or chunk_type == "tool_call_complete"
            ):
                if (
                    native_tool_parser
                    and (
                        tool_name := native_tool_parser.extract_tool_name(chunk_content)
                    )
                    in tool_types
                ):
                    args_json = native_tool_parser.extract_arguments(chunk_content)
                    tool_id = native_tool_parser.extract_tool_id(chunk_content)

                    if not tool_id:
                        yield None, chunk_content, chunk_type
                    else:
                        result = native_tool_parser.parse_tool(
                            tool_name, args_json, tool_id
                        )

                        if result.success:
                            yield result.value, chunk_content, chunk_type
                        else:
                            yield None, chunk_content, chunk_type
                else:
                    yield chunk_content, chunk_content, chunk_type
            else:
                yield chunk_content, chunk_content, "content"

    # Handle XML tool approach (non-native tools)
    else:
        async for ctype, chunk in client.connect_and_listen(**call_kwargs):
            if ctype == "reasoning":
                yield None, chunk, "reasoning"

            if effective_output_type and stream_parser:
                result = parser.feed_chunk(stream_parser, chunk)
                if result.success and result.value is not None:
                    yield result.value, chunk, ctype
                else:
                    yield None, chunk, ctype
            else:
                yield chunk, chunk, ctype

        if effective_output_type and stream_parser:
            final_result = parser.validate_final(stream_parser)
            if final_result.success and final_result.value is not None:
                # Yield the final validated object, with an empty string for raw part
                yield final_result.value, "", ""


async def _call_llm(
    prompt: Optional[str],
    model: str,
    tools: Dict[str, Callable],
    output_type: Optional[R],
    store: Optional[Store] = None,
    api_key: Optional[str] = None,
    system: Optional[str] = None,
    history: Optional[List[ChatMessage | ReasoningSegment | ToolResult]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    call_context: Optional[Dict[str, Any]] = None,  # ADDED
    art_backend=None,
    native_tools: bool = False,
    tool_types: Dict[str, Type[Deserializable]] = {},
):
    """Execute a non-streaming LLM call with optional native tool support."""
    client = _create_inference_client(model, store, api_key, art_backend=art_backend)

    # Handle native tools
    native_tool_schemas = None
    if native_tools and tools:
        native_tool_schemas = _generate_native_tool_schemas(tools, model)

    # Standard approach for non-native tools
    effective_output_type = output_type
    if not native_tools and tools and output_type:
        # Validate all callables have typed parameters
        if isinstance(tools, dict):
            _iter_tools = tools.items()
        else:
            _iter_tools = []
        for name, func in _iter_tools:
            _validate_callable_params(func, name)

        # Determine which type system to use based on output_type
        tool_models = []
        if _is_pydantic_type(output_type):
            tool_models = [
                _create_pydantic_tool_model(func, name) for name, func in _iter_tools
            ]

        elif _is_deserializable_type(output_type):
            tool_models = [
                _create_deserializable_tool_class(func, name)
                for name, func in _iter_tools
            ]

        if tool_models:
            # Create Union type including tools and output
            effective_output_type = Union[*(tool_models + [output_type])]

    # Get parser (skip if using native tools)
    parser = None if native_tools else _get_parser_for_type(effective_output_type)

    # Enhance prompt (skip if using native tools)
    enhanced_prompt_content = prompt
    if prompt is not None and not native_tools and effective_output_type and parser:
        enhanced_prompt_content = parser.enhance_prompt(
            prompt, effective_output_type, call_context=call_context  # type: ignore
        )  # MODIFIED

    # Construct messages list
    messages: List[Union[ChatMessage, ToolResult, ReasoningSegment, dict]] = []
    if system:
        messages.append(ChatMessage(role=ChatRole.SYSTEM, content=system))
    if history:
        messages.extend(history)

    if enhanced_prompt_content is not None:  # Add the current prompt as a user message
        messages.append(
            ChatMessage(role=ChatRole.USER, content=enhanced_prompt_content)
        )

    if not messages:
        raise ValueError(
            "LLM call attempted with no system message, history, or current prompt."
        )

    # Make API call with native tools if enabled
    call_kwargs = {
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else 8192,
        "top_p": top_p if top_p is not None else 0.9,
        "temperature": temperature if temperature is not None else 0.01,
    }

    if native_tools and native_tool_schemas:
        call_kwargs["tools"] = native_tool_schemas

    result = await client.get_generation(**call_kwargs)

    # Handle response based on native tools usage
    if native_tools and tools:
        provider = model.split(":", 1)[0] if ":" in model else model
        native_tool_parser = NativeToolParser(tool_types)

        tool_calls = _parse_list_of_complete_tools(
            result[1]["tool_calls"], native_tool_parser, tool_types
        )

        if tool_calls:
            return tool_calls, None, "tool_calls"
        else:
            # No tool calls, return text response and honor output_type
            content = _extract_content_from_response(result, provider)
            if output_type and content:
                # Honor output_type for non-tool responses in native mode
                parser = _get_parser_for_type(output_type)
                parsed_result = parser.parse(content, output_type)  # type: ignore
                if parsed_result.success:
                    return parsed_result.value, content, None
            return content, content, None
    else:
        # Handle traditional XML parsing approach
        if isinstance(result, tuple) and len(result) == 2:
            content, reasoning = result
            # Parse the content if output_type is provided
            if effective_output_type and content is not None:
                parsed_result = parser.parse(content, effective_output_type)  # type: ignore
                print(parsed_result)
                if parsed_result.success:
                    return parsed_result.value, content, reasoning
                else:
                    # Parsing failed, return the content as-is
                    return None, content, reasoning
            else:
                # No output_type specified, return content as-is
                return content, content, reasoning
        else:
            # Legacy format - just content
            if effective_output_type and result is not None:
                parsed_result = parser.parse(result, effective_output_type)  # type: ignore
                print(parsed_result)
                if parsed_result.success:
                    return parsed_result.value, result, None
                else:
                    # Parsing failed, return the content as-is
                    return result, result, None
            else:
                # No output_type specified, return result as-is
                return result, result, None
