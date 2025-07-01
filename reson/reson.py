# reson/agentic.py
from __future__ import annotations

from reson.services.inference_clients import InferenceClient, ChatMessage, ChatRole
from typing import (
    List,
    Dict,
    Any,
    AsyncGenerator,
    Optional,
    Type,
    TypeVar,
    get_origin,
    get_args,
    Callable,
    ParamSpec,
    Union,
    Awaitable,
    cast,
    Concatenate,
    Coroutine,
)
from reson.stores import (
    StoreConfigBase,
    MemoryStore,
    MemoryStoreConfig,
    RedisStore,
    RedisStoreConfig,
    PostgresStore,
    PostgresStoreConfig,
    Store,
)
from reson.reson_base import ResonBase
from pydantic import PrivateAttr, Field, BaseModel

import inspect, functools
from typing import Callable, ParamSpec, TypeVar, Awaitable, Any, AsyncIterator, Union
import re
import json
from reson.services.inference_clients import ChatMessage, ChatRole
from gasp.jinja_helpers import create_type_environment  # Added import

from reson.utils.parsers import OutputParser, get_default_parser

try:
    from reson.utils.parsers.baml_parser import BamlParser  # type: ignore[import-untyped]
except ImportError:
    pass

from reson.types import Deserializable

from reson.utils.inference import (
    create_google_gemini_api_client,
    create_openrouter_inference_client,
    create_anthropic_inference_client,
    create_bedrock_inference_client,
    create_vertex_gemini_api_client,
)
from reson.tracing_inference_client import TracingInferenceClient

P = ParamSpec("P")
R = TypeVar("R")


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


# TODO: This isn't properly creating a Deserializable class but because our parser can handle generic python types now it still works properly.
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


def _build_store(cfg: StoreConfigBase | None) -> Store:
    if cfg is None or isinstance(cfg, MemoryStoreConfig):
        return MemoryStore()

    if isinstance(cfg, RedisStoreConfig):
        return RedisStore(
            host=cfg.host,
            port=cfg.port,
            db=cfg.db,
            password=cfg.password,
        )

    if isinstance(cfg, PostgresStoreConfig):
        return PostgresStore(
            dsn=cfg.dsn,
            table=cfg.table,
            column=cfg.column,  # Pass the required column parameter
        )

    raise ValueError(f"Unsupported store config: {cfg}")


# ───────── context store (simple in-mem; swap for redis/pg) ─────────
class _Ctx:
    """Async wrapper around a Store instance."""

    def __init__(self, store: Store):
        self._store = store

    async def get(self, k: str, d=None):
        return await self._store.get(k, d)

    async def set(self, k: str, v: Any):
        await self._store.set(k, v)


# ───────── runtime object ─────────
class Runtime(ResonBase):
    """
    Runtime object that wraps calls to the underlying LLM and exposes
    dynamically bound tools.

    Inherits from `ResonBase` (a thin shim over `pydantic.BaseModel`)
    so public attributes are declared as model fields. Private runtime
    state lives in `PrivateAttr` fields.
    """

    # ───── public model fields ─────
    model: Optional[str] = Field(default=None)
    store: Store
    used: bool = Field(default=False)
    api_key: Optional[str] = Field(default=None, exclude=True)

    # ───── private runtime fields ─────
    _tools: dict[str, Callable] = PrivateAttr(default_factory=dict)
    _default_prompt: str = PrivateAttr(default="")
    _context: Optional[_Ctx] = PrivateAttr(default=None)
    _return_type: Optional[Type[Any]] = PrivateAttr(default=None)
    _raw_response_accumulator: List[str] = PrivateAttr(default_factory=list)
    _reasoning_accumulator: List[str] = PrivateAttr(default_factory=list)
    _current_call_args: Optional[Dict[str, Any]] = PrivateAttr(default=None)  # ADDED

    def model_post_init(self, __context):
        """Initialize private attributes after model fields are set."""
        self._context = _Ctx(self.store)

    # ───── public API ─────
    @property
    def raw_response(self) -> str:
        """
        Returns the accumulated raw string response from the LLM
        for the last run() or run_stream() call.
        For run(), this is the complete raw response.
        For run_stream(), this accumulates token by token and can be
        inspected mid-stream. It represents the full raw response
        once the stream is complete.
        """
        return "".join(self._raw_response_accumulator)

    def clear_raw_response(self) -> None:
        """Clears the accumulated raw LLM response."""
        self._raw_response_accumulator = []

    @property
    def reasoning(self) -> str:
        """
        Returns the accumulated reasoning tokens from the LLM
        for the last run() or run_stream() call.
        """
        return "".join(self._reasoning_accumulator)

    def clear_reasoning(self) -> None:
        """Clears the accumulated reasoning tokens."""
        self._reasoning_accumulator = []

    def tool(self, fn: Callable, *, name: str | None = None):
        """Register a callable so the LLM can invoke it as a tool."""
        self._tools[name or fn.__name__] = fn

    async def run(
        self,
        *,
        prompt: str | None = None,
        system: Optional[str] = None,
        history: Optional[List[ChatMessage]] = None,
        output_type: type | None = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        # agent_call_args: Optional[Dict[str, Any]] = None # REMOVED from signature
    ):
        """Execute a single, non-streaming LLM call."""
        self.used = True
        self.clear_raw_response()  # Clear accumulator for new call
        self.clear_reasoning()  # Clear reasoning for new call
        prompt = prompt or self._default_prompt
        # Use _return_type if output_type is not provided
        effective_output_type = (
            output_type if output_type is not None else self._return_type
        )

        # Determine which model to use
        effective_model = model if model is not None else self.model
        if effective_model is None:
            raise ValueError(
                "No model specified. Provide model either in decorator or at runtime."
            )

        # Determine which API key to use
        effective_api_key = api_key if api_key is not None else self.api_key

        # _call_llm will be modified to return (parsed_value, raw_response_str, reasoning_str)
        result = await _call_llm(
            prompt,
            effective_model,
            self._tools,
            effective_output_type,
            self.store,
            effective_api_key,
            system=system,
            history=history,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            call_context=self._current_call_args,  # MODIFIED: Use instance attr
        )

        # Handle tuple response with reasoning
        if isinstance(result, tuple) and len(result) == 3:
            parsed_value, raw_response_str, reasoning_str = result
            if reasoning_str:
                self._reasoning_accumulator.append(reasoning_str)
        else:
            # Legacy format
            parsed_value, raw_response_str = result

        if raw_response_str is not None:
            self._raw_response_accumulator.append(raw_response_str)
        return parsed_value

    async def run_stream(
        self,
        *,
        prompt: str | None = None,
        system: Optional[str] = None,
        history: Optional[List[ChatMessage]] = None,
        output_type: type | None = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        # agent_call_args: Optional[Dict[str, Any]] = None # REMOVED from signature
    ) -> AsyncIterator[tuple[str, Any]]:
        """Execute a streaming LLM call yielding chunks as they arrive.

        Yields:
            Tuples of (chunk_type, value) where:
            - ("reasoning", accumulated_reasoning_str) during reasoning phase
            - ("content", parsed_chunk) during content generation phase
        """
        self.used = True
        self.clear_raw_response()  # Clear accumulator for new call
        self.clear_reasoning()  # Clear reasoning for new call
        prompt = prompt or self._default_prompt
        # Use _return_type if output_type is not provided
        effective_output_type = (
            output_type if output_type is not None else self._return_type
        )

        # Determine which model to use
        effective_model = model if model is not None else self.model
        if effective_model is None:
            raise ValueError(
                "No model specified. Provide model either in decorator or at runtime."
            )

        # Determine which API key to use
        effective_api_key = api_key if api_key is not None else self.api_key

        # _call_llm_stream will be modified to yield (parsed_chunk, raw_chunk_str, chunk_type)
        async for chunk_data in _call_llm_stream(
            prompt,
            effective_model,
            self._tools,
            effective_output_type,
            self.store,
            effective_api_key,
            system=system,
            history=history,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            call_context=self._current_call_args,  # MODIFIED: Use instance attr
        ):
            # Handle tuple format with chunk type
            if isinstance(chunk_data, tuple) and len(chunk_data) == 3:
                parsed_chunk, raw_chunk_str, chunk_type = chunk_data
                if chunk_type == "reasoning" and raw_chunk_str:
                    self._reasoning_accumulator.append(raw_chunk_str)
                    # Yield reasoning progress
                    yield ("reasoning", self.reasoning)
                elif raw_chunk_str is not None:
                    self._raw_response_accumulator.append(raw_chunk_str)
                    if parsed_chunk is not None:
                        # Yield content with type indicator
                        yield ("content", parsed_chunk)
            else:
                parsed_chunk, raw_chunk_str = chunk_data
                if raw_chunk_str is not None:
                    self._raw_response_accumulator.append(raw_chunk_str)
                if parsed_chunk is not None:
                    # Always yield as tuple for consistency
                    yield ("content", parsed_chunk)

    async def run_with_baml(
        self,
        *,
        baml_request: Any,
        output_type: type | None = None,
    ):
        """
        Execute an LLM call using a BAML request.

        Args:
            baml_request: A BAML request object
            output_type: The type to parse into
        """
        self.used = True
        try:
            # Get the parser
            from reson.utils.parsers import get_default_parser

            parser = get_default_parser()

            # Attempt to access BAML parser methods if this is the BAML parser
            if hasattr(parser, "extract_prompt_from_baml_request"):
                typed_parser = cast(BamlParser, parser)
                # Extract the prompt from the BAML request
                prompt = typed_parser.extract_prompt_from_baml_request(baml_request)

                # Use _return_type if output_type is not provided
                effective_output_type = (
                    output_type if output_type is not None else self._return_type
                )

                # Determine which model to use
                if self.model is None:
                    raise ValueError(
                        "No model specified. Provide model either in decorator or at runtime."
                    )

                # Call the LLM using the extracted prompt
                return await _call_llm(
                    prompt, self.model, self._tools, effective_output_type, self.store
                )
            else:
                # If not using BAML parser, use the BAML request directly
                return await baml_request
        except Exception as e:
            raise RuntimeError(f"Error executing BAML request: {e}")

    async def run_stream_with_baml(
        self,
        *,
        baml_request: Any,
        output_type: type | None = None,
    ) -> AsyncIterator[Any]:
        """
        Execute a streaming LLM call using a BAML request.

        Args:
            baml_request: A BAML request object
            output_type: The type to parse into
        """
        self.used = True
        try:
            # Get the parser
            from reson.utils.parsers import get_default_parser

            parser = get_default_parser()

            # Attempt to access BAML parser methods if this is the BAML parser
            if hasattr(parser, "extract_prompt_from_baml_request"):
                typed_parser = cast(BamlParser, parser)
                # Extract the prompt from the BAML request
                prompt = typed_parser.extract_prompt_from_baml_request(baml_request)

                # Use _return_type if output_type is not provided
                effective_output_type = (
                    output_type if output_type is not None else self._return_type
                )

                # Determine which model to use
                if self.model is None:
                    raise ValueError(
                        "No model specified. Provide model either in decorator or at runtime."
                    )

                # Call the LLM using the extracted prompt
                async for chunk in _call_llm_stream(
                    prompt, self.model, self._tools, effective_output_type, self.store
                ):
                    yield chunk
            else:
                # If not using BAML parser, use the BAML request directly
                async for chunk in baml_request.stream():
                    yield chunk
        except Exception as e:
            raise RuntimeError(f"Error executing BAML streaming request: {e}")

    @property
    def context(self):
        """Legacy accessor for context."""
        return self._context

    def is_tool_call(self, result: Any) -> bool:
        """Check if a result is a tool call."""
        return (
            hasattr(result, "_tool_name")
            and getattr(result, "_tool_name", None) in self._tools
        )

    def get_tool_name(self, result: Any) -> Optional[str]:
        """Get the tool name from a tool call result."""
        return getattr(result, "_tool_name", None)

    async def execute_tool(self, tool_result: Any) -> Any:
        """Execute a tool call result."""
        if not self.is_tool_call(tool_result):
            raise ValueError("Not a tool call result")

        # Look up the function from the tools registry
        tool_name = self.get_tool_name(tool_result)

        if not tool_name:
            raise ValueError("Tool result does not have a valid tool name")

        func = self._tools.get(tool_name)

        if func is None:
            raise ValueError(f"Tool '{tool_name}' not found in runtime tools")

        # Convert the tool model instance to kwargs
        if hasattr(tool_result, "model_dump"):  # Pydantic
            # Get all fields excluding private attributes
            all_fields = tool_result.model_dump()
            # Remove any fields that start with underscore or are class-level attributes
            kwargs = {
                k: v
                for k, v in all_fields.items()
                if not k.startswith("_") and k not in ["_tool_func", "_tool_name"]
            }
        else:  # Deserializable
            kwargs = {
                k: v for k, v in tool_result.__dict__.items() if not k.startswith("_")
            }

        if inspect.iscoroutinefunction(func):
            return await func(**kwargs)
        else:
            return func(**kwargs)


def _create_empty_value(output_type):
    """
    Create an appropriate empty value for the given type.
    This properly handles generics like List, Dict, etc.

    Args:
        output_type: The type to create an empty value for

    Returns:
        An empty value appropriate for the type
    """
    # Handle None type
    if output_type is None:
        return None

    # Get origin and args for generic types
    origin = get_origin(output_type)
    args = get_args(output_type)

    # Handle list types (List[T])
    if origin == list:
        return []

    # Handle dict types (Dict[K, V])
    if origin == dict:
        return {}

    # Handle Union types (including Optional[T])
    if origin == Union:
        if type(None) in args:  # Handle Optional[T]
            # Return None for Optional types
            return None
        else:
            # For Union[A, B, ...], try to create empty A
            return _create_empty_value(args[0])

    # Handle primitive types
    if output_type == str:
        return ""
    if output_type == int:
        return 0
    if output_type == float:
        return 0.0
    if output_type == bool:
        return False

    # Handle Pydantic models (v1 or v2)
    if hasattr(output_type, "model_construct") or hasattr(output_type, "construct"):
        try:
            # Try to construct an empty instance with default values
            if hasattr(output_type, "model_construct"):  # Pydantic v2
                return output_type.model_construct()
            else:  # Pydantic v1
                return output_type.construct()
        except Exception:
            pass

    # Try direct instantiation with no args as fallback
    try:
        return output_type()
    except Exception:
        # If all else fails, return None
        return None


async def _create_inference_client(model_str, store=None, api_key=None):
    """Create an appropriate inference client based on the model string."""
    # Parse model string to get provider and model
    parts = model_str.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid model string format: {model_str}")

    provider, model_name = parts

    if provider == "openrouter":
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
    elif provider == "anthropic":
        # Parse out thinking parameter if provided
        thinking_match = re.match(r"(.+)@thinking=(\d+)", model_name)
        if thinking_match:
            model_name, thinking = thinking_match.groups()
            client = create_anthropic_inference_client(
                model_name, thinking=int(thinking), api_key=api_key
            )
        else:
            client = create_anthropic_inference_client(model_name, api_key=api_key)
    elif provider == "bedrock":
        client = create_bedrock_inference_client(model_name)
    elif provider == "google-gemini":
        client = create_google_gemini_api_client(model_name, api_key=api_key)
    elif provider == "vertex-gemini":
        reasoning_match = re.match(r"(.+)@reasoning=([a-z].*)", model_name)
        if not reasoning_match:
            # Try numeric pattern
            reasoning_match = re.match(r"(.+)@reasoning=(\d+)", model_name)
        if reasoning_match:
            model_name, reasoning = reasoning_match.groups()
            client = create_vertex_gemini_api_client(model_name, reasoning=reasoning)
        else:
            client = create_vertex_gemini_api_client(model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Wrap in TracingInferenceClient
    return TracingInferenceClient(client, store)


def _get_parser_for_type(output_type=None) -> OutputParser:
    """Get the appropriate parser for the given output type."""
    # For now, just use the default parser
    # In the future, we might select based on type or user configuration
    return get_default_parser()


async def _call_llm(
    prompt: Optional[str],
    model: str,
    tools: Dict[str, Callable],
    output_type: Optional[Type[Any]],
    store: Optional[Store] = None,
    api_key: Optional[str] = None,
    system: Optional[str] = None,
    history: Optional[List[ChatMessage]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    call_context: Optional[Dict[str, Any]] = None,  # ADDED
):
    """Execute a non-streaming LLM call, possibly with tool use."""
    client = await _create_inference_client(model, store, api_key)

    # Determine effective output type (with tools if applicable)
    effective_output_type = output_type

    if tools and output_type:
        # Validate all callables have typed parameters
        for name, func in tools.items():
            _validate_callable_params(func, name)

        # Determine which type system to use based on output_type
        tool_models = []
        if _is_pydantic_type(output_type):
            tool_models = [
                _create_pydantic_tool_model(func, name) for name, func in tools.items()
            ]

        elif _is_deserializable_type(output_type):
            tool_models = [
                _create_deserializable_tool_class(func, name)
                for name, func in tools.items()
            ]

        if tool_models:
            # Create Union type including tools and output
            effective_output_type = Union[*(tool_models + [output_type])]

    # Get the appropriate parser
    parser = _get_parser_for_type(effective_output_type)

    # Enhance the prompt string if a prompt is provided
    enhanced_prompt_content = prompt
    if prompt is not None and effective_output_type:
        enhanced_prompt_content = parser.enhance_prompt(
            prompt, effective_output_type, call_context=call_context
        )  # MODIFIED

    # Construct messages list
    messages: List[ChatMessage] = []
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

    # Standard LLM call (now includes tool types in effective_output_type)
    call_kwargs = {
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else 8192,
        "top_p": top_p if top_p is not None else 0.9,
        "temperature": temperature if temperature is not None else 0.01,
    }
    result = await client.get_generation(**call_kwargs)

    if isinstance(result, tuple) and len(result) == 2:
        content, reasoning = result
        # Parse the content if output_type is provided
        if effective_output_type and content is not None:
            parsed_result = parser.parse(content, effective_output_type)
            print(parsed_result)
            if parsed_result.success:
                return parsed_result.value, content, reasoning
            else:
                # Parsing failed, return None for parsed value
                return None, content, reasoning
        else:
            # No output_type specified, return content as-is
            return content, content, reasoning
    else:
        # Legacy format - just content
        if effective_output_type and result is not None:
            parsed_result = parser.parse(result, effective_output_type)
            print(parsed_result)
            if parsed_result.success:
                return parsed_result.value, result, None
            else:
                # Parsing failed, return None for parsed value
                return None, result, None
        else:
            # No output_type specified, return result as-is
            return result, result, None


async def _call_llm_stream(
    prompt: Optional[str],
    model: str,
    tools: Dict[str, Callable],
    output_type: Optional[Type[Any]],
    store: Optional[Store] = None,
    api_key: Optional[str] = None,
    system: Optional[str] = None,
    history: Optional[List[ChatMessage]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    call_context: Optional[Dict[str, Any]] = None,  # ADDED
):
    """Execute a streaming LLM call, possibly with tool use."""
    client = await _create_inference_client(model, store, api_key=api_key)

    # Determine effective output type (with tools if applicable)
    effective_output_type = output_type

    if tools and output_type:
        # Validate all callables have typed parameters
        for name, func in tools.items():
            _validate_callable_params(func, name)

        # Determine which type system to use based on output_type
        tool_models = []
        if _is_pydantic_type(output_type):
            tool_models = [
                _create_pydantic_tool_model(func, name) for name, func in tools.items()
            ]
        elif _is_deserializable_type(output_type):
            tool_models = [
                _create_deserializable_tool_class(func, name)
                for name, func in tools.items()
            ]

        if tool_models:
            # Create Union type including tools and output
            effective_output_type = Union[*(tool_models + [output_type])]

    # Get the appropriate parser
    parser = _get_parser_for_type(effective_output_type)

    # Enhance the prompt string if a prompt is provided
    enhanced_prompt_content = prompt
    if (
        prompt is not None and output_type
    ):  # Note: using output_type for prompt enhancement, not effective_output_type
        enhanced_prompt_content = parser.enhance_prompt(
            prompt, output_type, call_context=call_context
        )  # MODIFIED

    # Create a streaming parser if output_type is provided
    stream_parser = None
    if effective_output_type:
        stream_parser = parser.create_stream_parser(effective_output_type)

    # Construct messages list
    messages: List[ChatMessage] = []
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

    if (
        tools
        and not _is_pydantic_type(output_type)
        and not _is_deserializable_type(output_type)
    ):
        call_kwargs = {
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else 8192,
            "top_p": top_p if top_p is not None else 0.9,
            "temperature": temperature if temperature is not None else 0.5,
        }
        async for chunk in client.connect_and_listen(**call_kwargs):
            print(f"CHUNK: {chunk}")
            if effective_output_type and stream_parser:
                # Feed the chunk to the parser
                result = parser.feed_chunk(stream_parser, chunk)
                # Yield tuple: (parsed_chunk, raw_chunk)
                yield result.value, chunk
            else:
                # Yield tuple: (None or raw_chunk as parsed, raw_chunk)
                yield chunk, chunk

        # Validate the final result if using a parser
        if effective_output_type and stream_parser:
            final_result = parser.validate_final(stream_parser)
            if final_result.success and final_result.value is not None:
                yield final_result.value, ""  # Yield empty string for raw part of final validation
    else:
        call_kwargs = {
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else 8192,
            "top_p": top_p if top_p is not None else 0.9,
            "temperature": temperature if temperature is not None else 0.5,
        }
        async for chunk in client.connect_and_listen(**call_kwargs):
            if isinstance(chunk, tuple) and len(chunk) == 2:
                chunk_type, chunk_content = chunk
                if chunk_type == "reasoning":
                    # Yield reasoning chunks with type indicator
                    yield None, chunk_content, "reasoning"
                else:  # content
                    parsed_chunk_value = None
                    if effective_output_type and stream_parser:
                        result = parser.feed_chunk(stream_parser, chunk_content)
                        if result.value is not None:
                            parsed_chunk_value = result.value
                    else:
                        parsed_chunk_value = chunk_content
                    yield parsed_chunk_value, chunk_content, "content"
            else:
                # Legacy format - just content
                parsed_chunk_value = None
                if effective_output_type and stream_parser:
                    result = parser.feed_chunk(stream_parser, chunk)
                    if result.value is not None:
                        parsed_chunk_value = result.value
                else:
                    parsed_chunk_value = chunk
                yield parsed_chunk_value, chunk, "content"

        # Validate the final result if using a parser
        if effective_output_type and stream_parser:
            final_result = parser.validate_final(stream_parser)
            if final_result.success and final_result.value is not None:
                # Yield the final validated object, with an empty string for raw part
                yield final_result.value, "", ""


def agentic(
    *,
    model: Optional[str] = None,
    api_key: str | None = None,
    store_cfg: StoreConfigBase = MemoryStoreConfig(),
    autobind: bool = True,
) -> Any:  # Temporarily cast to Any to isolate Pylance issue
    """Decorator for *awaitable* agentic functions."""
    return cast(
        Any,  # Temporarily cast to Any
        _agentic_core(
            model=model,
            api_key=api_key,
            store_cfg=store_cfg,
            autobind=autobind,
        ),
    )


def agentic_generator(
    *,
    model: Optional[str] = None,
    api_key: str | None = None,
    store_cfg: StoreConfigBase = MemoryStoreConfig(),
    autobind: bool = True,
) -> Any:  # Temporarily cast to Any to isolate Pylance issue
    """Decorator for *async-generator* agentic functions."""
    return cast(
        Any,  # Temporarily cast to Any
        _agentic_core(
            model=model,
            api_key=api_key,
            store_cfg=store_cfg,
            autobind=autobind,
        ),
    )


# ───────── decorator ─────────
def _agentic_core(
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    store_cfg: StoreConfigBase = MemoryStoreConfig(),
    autobind: bool = True,
):
    """
    Decorator factory for agentic functions that interact with LLMs.

    There are two ways to use this decorator:

    1. With regular async functions - the function will be executed and its
       return value will be awaited and returned.

    2. With async generator functions - the function will be treated as a generator
       and values yielded from the function will be yielded from the decorator.
       This allows for streaming results or providing intermediate status updates.

    Parameters:
      • model (str, optional) – model identifier (e.g., "openrouter:openai/gpt-4o"). If not provided, must be specified at runtime.
      • api_key (str, optional) – API key for the model provider
      • store_cfg (StoreConfigBase) – configuration for storage ("memory", "redis", "postgres")
      • autobind (bool) – automatically expose callable params as tools

    In generator mode, you can yield values as they are processed:

    ```python
    @agentic(model="openrouter:openai/gpt-4o")
    async def process_items(items: List[str], runtime: Runtime) -> AsyncGenerator[Dict, None]:
        for item in items:
            result = await runtime.run(prompt=f"Process: {item}")
            yield {"item": item, "result": result}

    # Usage
    async for result in process_items(my_items):
        print(result)
    ```
    """

    store_obj = _build_store(store_cfg)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fn)
        if "runtime" not in sig.parameters:
            raise TypeError("`runtime` parameter missing in agentic function")

        default_prompt = inspect.getdoc(fn) or ""
        ret_type = sig.return_annotation

        # Check if the function is an async generator
        is_async_gen = inspect.isasyncgenfunction(fn)

        # Check return type annotation if it's not directly detected as an async gen function
        if not is_async_gen and ret_type != inspect.Signature.empty:
            origin = get_origin(ret_type)
            # Check if the return type is AsyncGenerator or AsyncIterator
            if origin is not None:
                try:
                    is_async_gen = (
                        origin is AsyncGenerator
                        or origin is AsyncIterator
                        or (
                            isinstance(origin, type)
                            and issubclass(origin, (AsyncGenerator, AsyncIterator))
                        )
                    )
                except TypeError:
                    # In case origin is not a class, issubclass would fail
                    pass

        if is_async_gen:
            # Async generator wrapper
            async def gen_wrapper(*args, **kwargs):
                rt = Runtime(
                    model=model,
                    store=store_obj,
                    api_key=api_key,
                )
                rt._default_prompt = default_prompt  # Set raw docstring
                # Set the return type from the function's signature
                if ret_type != inspect.Signature.empty and ret_type is not None:
                    # For generators, extract the yield type
                    origin = get_origin(ret_type)
                    if origin in (AsyncGenerator, AsyncIterator):
                        args = get_args(ret_type)
                        if args:
                            rt._return_type = args[0]  # Get the yield type
                        else:
                            rt._return_type = None
                    else:
                        rt._return_type = ret_type

                ba = sig.bind_partial(*args, **kwargs, runtime=rt)

                rt._current_call_args = ba.arguments  # Store args before calling fn

                if autobind:
                    for name, param in sig.parameters.items():
                        val = ba.arguments.get(name)
                        if callable(val):
                            rt.tool(val, name=name)

                try:
                    # Iterate and yield values from the generator
                    async for value in fn(*ba.args, **ba.kwargs):
                        yield value
                finally:
                    rt._current_call_args = None  # Clear args after fn completes

                # Check runtime was used after generator is exhausted
                if not rt.used:
                    raise RuntimeError(
                        "agentic generator function completed without calling runtime.run() "
                        "or runtime.run_stream()"
                    )

            return functools.wraps(fn)(gen_wrapper)
        else:
            # Original async function wrapper
            async def wrapper(*args, **kwargs):
                rt = Runtime(
                    model=model,
                    store=store_obj,
                    api_key=api_key,
                )
                rt._default_prompt = default_prompt  # Set raw docstring
                # Set the return type from the function's signature
                if ret_type != inspect.Signature.empty and ret_type is not None:
                    rt._return_type = ret_type

                ba = sig.bind_partial(*args, **kwargs, runtime=rt)

                rt._current_call_args = ba.arguments  # Store args before calling fn

                if autobind:
                    for name, param in sig.parameters.items():
                        val = ba.arguments.get(name)
                        if callable(val):
                            rt.tool(val, name=name)

                try:
                    result = await fn(*ba.args, **ba.kwargs)
                finally:
                    rt._current_call_args = None  # Clear args after fn completes

                if not rt.used:
                    raise RuntimeError(
                        "agentic function completed without calling runtime.run() "
                        "or runtime.run_stream()"
                    )
                return result

        return functools.wraps(fn)(wrapper)

    return decorator


__all__ = ["agentic", "agentic_generator", "Runtime"]
