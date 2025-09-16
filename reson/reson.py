# reson/agentic.py
from __future__ import annotations
from types import UnionType
import uuid

from reson.training import TrainingManager
from reson.services.inference_clients import (
    ChatMessage,
    ChatRole,
    ToolResult,
    ReasoningSegment,
    InferenceProvider,
)
import os
from typing import (
    TYPE_CHECKING,
    Union,
    List,
    Dict,
    Any,
    AsyncGenerator,
    Optional,
    Type,
    TypeVar,
    get_origin,
    get_args,
    get_type_hints,
    Callable,
    ParamSpec,
    Union,
    Tuple,
    cast,
)
from pathlib import Path
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
from typing import Union
import types
import sys

from reson.reson_base import ResonBase
from pydantic import ConfigDict, PrivateAttr, Field, BaseModel

import inspect, functools
from typing import Callable, ParamSpec, TypeVar, Awaitable, Any, AsyncIterator, Union
import re
import json
from reson.services.inference_clients import ChatMessage, ChatRole
from gasp.jinja_helpers import create_type_environment  # Added import

from reson.utils.parsers import OutputParser, get_default_parser, NativeToolParser

from reson.types import Deserializable

from reson.utils.inference import (
    create_google_gemini_api_client,
    create_openrouter_inference_client,
    create_anthropic_inference_client,
    create_bedrock_inference_client,
    create_vertex_gemini_api_client,
    create_openai_inference_client,
)
from reson.utils.schema_generators import get_schema_generator, supports_native_tools
from reson.tracing_inference_client import TracingInferenceClient

if TYPE_CHECKING:
    from reson.training import TrainingManager

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
    native_tools: bool = Field(default=False)

    # ───── private runtime fields ─────
    _tools: dict[str, Callable] = PrivateAttr(default_factory=dict)
    _tool_types: dict[str, Any] = PrivateAttr(
        default_factory=dict
    )  # NEW: tool type mappings
    _default_prompt: str = PrivateAttr(default="")
    _context: Optional[_Ctx] = PrivateAttr(default=None)
    _return_type: Optional[Type[Any]] = PrivateAttr(default=None)
    _raw_response_accumulator: List[str] = PrivateAttr(default_factory=list)
    _reasoning_accumulator: List[str] = PrivateAttr(default_factory=list)
    _reasoning_segments: List["ReasoningSegment"] = PrivateAttr(default_factory=list)
    _current_reasoning_segment: Optional["ReasoningSegment"] = PrivateAttr(default=None)
    _current_call_args: Optional[Dict[str, Any]] = PrivateAttr(default=None)  # ADDED
    _training_manager: Optional["TrainingManager"] = PrivateAttr(default=None)

    def model_post_init(self, __context):
        """Initialize private attributes after model fields are set."""
        self._context = _Ctx(self.store)

        # Validate native tools support if enabled
        if self.native_tools and self.model:
            if not supports_native_tools(self.model):
                raise ValueError(
                    f"Native tools not supported for model: {self.model}. "
                    f"Set native_tools=False or use a supported provider."
                )

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

    @property
    def reasoning_segments(self) -> List[ReasoningSegment]:
        """
        Returns the list of reasoning segments from the LLM
        for the last run() or run_stream() call.
        """
        return self._reasoning_segments

    def clear_reasoning_segments(self) -> None:
        """Clears the accumulated reasoning segments."""
        self._reasoning_segments = []

    def tool(
        self,
        fn: Callable,
        *,
        name: str | None = None,
        tool_type: Type[Deserializable] | None = None,
    ):
        """Register a callable so the LLM can invoke it as a tool.

        Args:
            fn: The callable function to register
            name: Optional name override for the tool
            tool_type: Optional Deserializable type for automatic marshalling of tool call deltas/complete events
        """
        tool_name = name or fn.__name__
        self._tools[tool_name] = fn

        # Store tool type mapping if provided
        if tool_type is not None:
            self._tool_types[tool_name] = tool_type

        # If a Deserializable tool_type is provided and it has a class docstring,
        # wrap the function and set the wrapper's docstring so schema generators pick it up
        # without mutating the original function (safe for multiple registrations)
        if tool_type is not None and getattr(tool_type, "__doc__", None):
            try:
                if inspect.iscoroutinefunction(fn):

                    async def wrapper_async(*args, **kwargs):
                        return await fn(*args, **kwargs)

                    base_wrapper = wrapper_async
                else:

                    def wrapper_sync(*args, **kwargs):
                        return fn(*args, **kwargs)

                    base_wrapper = wrapper_sync

                wrapped = functools.wraps(fn)(base_wrapper)
                wrapped.__doc__ = tool_type.__doc__
                # Override the stored function with the wrapped version
                self._tools[tool_name] = wrapped
            except Exception:
                # If wrapping fails, leave original function unchanged
                pass

    def load_training_manager(self, path: Path | str):
        from reson.training import TrainingManager

        self._training_manager = TrainingManager.load(path)

    def init_training_manager(self, name: str):
        from reson.training import TrainingManager

        self._training_manager = TrainingManager(name=name)

        return self._training_manager

    @property
    def training_manager(self):
        if not self._training_manager:
            self.init_training_manager(str(uuid.uuid4()))

        return self._training_manager

    async def run(
        self,
        *,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        history: Optional[List[ChatMessage | ToolResult | ReasoningSegment]] = None,
        output_type: Optional[Type[R] | UnionType] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        art_backend=None,
    ) -> R:
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
            art_backend=art_backend,
            native_tools=self.native_tools,  # Pass native tools flag
        )

        parsed_value, raw_response_str, reasoning_str = result
        if reasoning_str:
            self._reasoning_accumulator.append(reasoning_str)

        if raw_response_str is not None:
            self._raw_response_accumulator.append(raw_response_str)  # type: ignore

        return parsed_value  # type: ignore

    async def run_stream(
        self,
        *,
        prompt: str | None = None,
        system: Optional[str] = None,
        history: Optional[List[ChatMessage | ToolResult | ReasoningSegment]] = None,
        output_type: Optional[Type[R] | UnionType] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        art_backend=None,
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
        self.clear_reasoning_segments()  # Clear segments for new call
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
        last_chunk_type = None
        async for chunk_data in _call_llm_stream(
            prompt,
            effective_model,
            self._tools,
            effective_output_type,  # type: ignore
            self.store,
            effective_api_key,
            system=system,
            history=history,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            call_context=self._current_call_args,  # MODIFIED: Use instance attr
            art_backend=art_backend,
            native_tools=self.native_tools,  # Pass native tools flag
            tool_types=self._tool_types,  # Pass tool types for Deserializable parsing
        ):
            # Handle tuple format with chunk type
            if isinstance(chunk_data, tuple) and len(chunk_data) == 3:
                parsed_chunk, raw_chunk_str, chunk_type = chunk_data

                if last_chunk_type == "reasoning" and chunk_type not in (
                    "reasoning",
                    "signature",
                ):
                    if self._current_reasoning_segment:
                        self._reasoning_segments.append(self._current_reasoning_segment)
                        self._current_reasoning_segment = None

                if chunk_type == "reasoning" and raw_chunk_str:
                    if not self._current_reasoning_segment:
                        self._current_reasoning_segment = ReasoningSegment(
                            content="", segment_index=len(self._reasoning_segments)
                        )
                    self._current_reasoning_segment.content += raw_chunk_str
                    self._reasoning_accumulator.append(raw_chunk_str)
                    # Yield reasoning progress
                    yield ("reasoning", self.reasoning)
                elif chunk_type == "signature":
                    if self._current_reasoning_segment:
                        self._current_reasoning_segment.signature = parsed_chunk
                elif chunk_type == "reasoning" and raw_chunk_str:
                    if not self._current_reasoning_segment:
                        self._current_reasoning_segment = ReasoningSegment(
                            content="", segment_index=len(self._reasoning_segments)
                        )
                    self._current_reasoning_segment.content += raw_chunk_str
                    self._reasoning_accumulator.append(raw_chunk_str)
                    # Yield reasoning progress
                    yield ("reasoning", self.reasoning)
                elif chunk_type == "tool_call_complete":
                    # Yield tool call directly for processing
                    yield ("tool_call_complete", parsed_chunk)
                elif chunk_type == "tool_call_partial":
                    # Yield tool call delta for OpenAI incremental streaming
                    yield ("tool_call_partial", parsed_chunk)
                elif raw_chunk_str is not None:
                    self._raw_response_accumulator.append(raw_chunk_str)
                    if parsed_chunk is not None:
                        # Yield content with type indicator
                        yield ("content", parsed_chunk)
                last_chunk_type = chunk_type
            else:
                parsed_chunk, raw_chunk_str = chunk_data
                if raw_chunk_str is not None:
                    self._raw_response_accumulator.append(raw_chunk_str)  # type: ignore
                if parsed_chunk is not None:
                    # Always yield as tuple for consistency
                    yield ("content", parsed_chunk)

        if self._current_reasoning_segment:
            self._reasoning_segments.append(self._current_reasoning_segment)
            self._current_reasoning_segment = None

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

        # Smart marshalling: convert tool result to function's expected parameter types
        kwargs = self._marshall_arguments_to_function_signature(func, tool_result)

        if inspect.iscoroutinefunction(func):
            return await func(**kwargs)
        else:
            return func(**kwargs)

    def _marshall_arguments_to_function_signature(
        self, func: Callable, tool_result: Any
    ) -> Dict[str, Any]:
        """Convert tool result to function's expected parameter types with Deserializable priority."""
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        marshalled_kwargs = {}

        # Get raw arguments from tool result
        if hasattr(tool_result, "model_dump"):  # Pydantic
            raw_args = tool_result.model_dump()
        else:  # Deserializable or ToolCallInstance
            raw_args = {
                k: v for k, v in tool_result.__dict__.items() if not k.startswith("_")
            }

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = type_hints.get(param_name, str)
            raw_value = raw_args.get(param_name)

            if raw_value is None:
                # Parameter not provided - use default if available
                if param.default != inspect.Parameter.empty:
                    marshalled_kwargs[param_name] = param.default
                continue

            if isinstance(raw_value, dict):
                # Priority 1: Deserializable classes (your core type system)
                try:
                    if hasattr(param_type, "__mro__") and any(
                        "Deserializable" in cls.__name__ for cls in param_type.__mro__
                    ):
                        marshalled_kwargs[param_name] = param_type(**raw_value)
                        continue
                except Exception:
                    pass

                # Priority 2: Pydantic models
                try:
                    if hasattr(param_type, "model_validate"):
                        marshalled_kwargs[param_name] = param_type.model_validate(
                            raw_value
                        )
                        continue
                except Exception:
                    pass

                # Priority 3: Dataclasses
                try:
                    if hasattr(param_type, "__dataclass_fields__"):
                        marshalled_kwargs[param_name] = param_type(**raw_value)
                        continue
                except Exception:
                    pass

                # Priority 4: Try direct instantiation
                try:
                    marshalled_kwargs[param_name] = param_type(**raw_value)
                except Exception:
                    marshalled_kwargs[param_name] = raw_value
            else:
                # Primitive value - use directly
                marshalled_kwargs[param_name] = raw_value
        return marshalled_kwargs


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


def _get_parser_for_type(output_type=None) -> OutputParser:
    """Get the appropriate parser for the given output type."""
    # For now, just use the default parser
    # In the future, we might select based on type or user configuration
    return get_default_parser()


def _generate_native_tool_schemas(
    tools: Dict[str, Callable], model: str
) -> List[Dict[str, Any]]:
    """Generate native tool schemas for the given model/provider."""
    if not tools:
        return []

    schema_generator = get_schema_generator(model)
    return schema_generator.generate_tool_schemas(tools)


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

    # Handle native tools
    native_tool_schemas = None
    if native_tools and tools:
        native_tool_schemas = _generate_native_tool_schemas(tools, model)

    # Determine effective output type (with tools if applicable)
    effective_output_type = output_type

    # Only use XML tool approach if NOT using native tools
    if not native_tools and tools and output_type:
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

    # Create a streaming parser if effective_output_type is provided
    stream_parser = None
    if effective_output_type:
        stream_parser = parser.create_stream_parser(effective_output_type)

    # Construct messages list
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

    # Prepare call arguments
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
    elif (
        tools
        and not _is_pydantic_type(output_type)
        and not _is_deserializable_type(output_type)
    ):
        async for ctype, chunk in client.connect_and_listen(**call_kwargs):
            if ctype == "reasoning":
                yield None, chunk, "reasoning"

            if effective_output_type and stream_parser:
                result = parser.feed_chunk(stream_parser, chunk)
                yield result.value, chunk, ctype
            else:
                yield chunk, chunk, ctype

        if effective_output_type and stream_parser:
            final_result = parser.validate_final(stream_parser)
            if final_result.success and final_result.value is not None:
                yield final_result.value, ""

    # Handle regular streaming (no tools)
    else:
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
    native_tools: bool = False,
) -> Any:  # Temporarily cast to Any to isolate Pylance issue
    """Decorator for *awaitable* agentic functions."""
    return cast(
        Any,  # Temporarily cast to Any
        _agentic_core(
            model=model,
            api_key=api_key,
            store_cfg=store_cfg,
            autobind=autobind,
            native_tools=native_tools,
        ),
    )


def agentic_generator(
    *,
    model: Optional[str] = None,
    api_key: str | None = None,
    store_cfg: StoreConfigBase = MemoryStoreConfig(),
    autobind: bool = True,
    native_tools: bool = False,
) -> Any:  # Temporarily cast to Any to isolate Pylance issue
    """Decorator for *async-generator* agentic functions."""
    return cast(
        Any,  # Temporarily cast to Any
        _agentic_core(
            model=model,
            api_key=api_key,
            store_cfg=store_cfg,
            autobind=autobind,
            native_tools=native_tools,
        ),
    )


# ───────── decorator ─────────
def _agentic_core(
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    store_cfg: StoreConfigBase = MemoryStoreConfig(),
    autobind: bool = True,
    native_tools: bool = False,
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
                    native_tools=native_tools,
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
                    native_tools=native_tools,
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
