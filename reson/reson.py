# reson/agentic.py
from __future__ import annotations

from reson.services.inference_clients import InferenceClient, ChatMessage, ChatRole
from typing import List, Dict, Any, AsyncGenerator, Optional, Type, TypeVar, get_origin, get_args, Callable, ParamSpec, Union, Awaitable, cast, Concatenate
from reson.stores import StoreConfigBase, MemoryStore, MemoryStoreConfig, RedisStore, RedisStoreConfig, PostgresStore, PostgresStoreConfig, Store
from reson.reson_base import ResonBase
from pydantic import PrivateAttr, Field, BaseModel

import inspect, functools
from typing import Callable, ParamSpec, TypeVar, Awaitable, Any, AsyncIterator, Union
import re
import json
from reson.services.inference_clients import ChatMessage, ChatRole

from reson.utils.parsers import OutputParser, get_default_parser
from reson.types import Deserializable

from reson.utils.inference import (
    create_google_gemini_api_client,
    create_openrouter_inference_client,
    create_anthropic_inference_client,
    create_bedrock_inference_client,
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
    return hasattr(type_annotation, "model_validate")

def _is_deserializable_type(type_annotation) -> bool:
    """Check if a type is Deserializable-based."""
    origin = get_origin(type_annotation)
    if origin is not None:
        args = get_args(type_annotation)
        return any(_is_deserializable_type(arg) for arg in args if arg != type(None))
    try:
        return hasattr(type_annotation, "__mro__") and Deserializable in type_annotation.__mro__
    except:
        return False

def _validate_callable_params(func: Callable, name: str) -> None:
    """Ensure all parameters are properly typed."""
    sig = inspect.signature(func)
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
        if param.annotation == inspect.Parameter.empty:
            raise ValueError(f"Parameter '{param_name}' in tool '{name}' must have a type annotation")

def _create_pydantic_tool_model(func: Callable, name: str) -> Type:
    """Create a Pydantic model from a callable's signature."""
    from typing import ClassVar
    
    sig = inspect.signature(func)
    
    # Build the class dynamically
    attrs = {
        '__annotations__': {},
        '__doc__': func.__doc__ or f"Tool for {name}",
        # Only store the tool name, not the function
        '_tool_name': name,
    }
    
    # Add annotations for ClassVar
    attrs['__annotations__']['_tool_name'] = ClassVar[str]
    
    # Add parameter fields
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
        
        attrs['__annotations__'][param_name] = param.annotation
        
        # Handle defaults
        if param.default != inspect.Parameter.empty:
            attrs[param_name] = param.default
    
    # Create the class
    tool_class = type(
        f"{name.capitalize().replace('_', '')}Tool",
        (BaseModel,),
        attrs
    )
    
    return tool_class

def _create_deserializable_tool_class(func: Callable, name: str) -> Type:
    """Create a Deserializable class from a callable's signature using type()."""
    sig = inspect.signature(func)
    
    # Build annotations
    annotations = {}
    
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
        annotations[param_name] = param.annotation
    
    # Define __init__ method for Deserializable
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        # Store tool name on instance
        self._tool_name = name
    
    # Build class attributes
    class_attrs = {
        '__annotations__': annotations,
        '__doc__': func.__doc__ or f"Tool for {name}",
        '__init__': __init__,
    }
    
    # Add defaults as class attributes
    for param_name, param in sig.parameters.items():
        if param_name != 'self' and param.default != inspect.Parameter.empty:
            class_attrs[param_name] = param.default
    
    # Create the class using type()
    tool_class = type(
        f"{name.capitalize()}Tool",
        (Deserializable,),  # Base class
        class_attrs
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
    model: str
    store: Store
    used: bool = Field(default=False)

    # ───── private runtime fields ─────
    _tools: dict[str, Callable] = PrivateAttr(default_factory=dict)
    _default_prompt: str = PrivateAttr(default="")
    _context: Optional[_Ctx] = PrivateAttr(default=None)
    _return_type: Optional[Type[Any]] = PrivateAttr(default=None)
    
    def model_post_init(self, __context):
        """Initialize private attributes after model fields are set."""
        self._context = _Ctx(self.store)

    # ───── public API ─────
    def tool(self, fn: Callable, *, name: str | None = None):
        """Register a callable so the LLM can invoke it as a tool."""
        self._tools[name or fn.__name__] = fn

    async def run(
        self,
        *,
        prompt: str | None = None,
        output_type: type | None = None,
    ):
        """Execute a single, non-streaming LLM call."""
        self.used = True
        prompt = prompt or self._default_prompt
        # Use _return_type if output_type is not provided
        effective_output_type = output_type if output_type is not None else self._return_type
        return await _call_llm(prompt, self.model, self._tools, effective_output_type, self.store)

    async def run_stream(
        self,
        *,
        prompt: str | None = None,
        output_type: type | None = None,
    ) -> AsyncIterator[Any]:
        """Execute a streaming LLM call yielding chunks as they arrive."""
        self.used = True
        prompt = prompt or self._default_prompt
        # Use _return_type if output_type is not provided
        effective_output_type = output_type if output_type is not None else self._return_type
        async for chunk in _call_llm_stream(
            prompt, self.model, self._tools, effective_output_type, self.store
        ):
            yield chunk
    
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
                # Extract the prompt from the BAML request
                prompt = parser.extract_prompt_from_baml_request(baml_request)
                
                # Use _return_type if output_type is not provided
                effective_output_type = output_type if output_type is not None else self._return_type
                
                # Call the LLM using the extracted prompt
                return await _call_llm(prompt, self.model, self._tools, effective_output_type, self.store)
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
                # Extract the prompt from the BAML request
                prompt = parser.extract_prompt_from_baml_request(baml_request)
                
                # Use _return_type if output_type is not provided
                effective_output_type = output_type if output_type is not None else self._return_type
                
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
        return hasattr(result, '_tool_name') and getattr(result, '_tool_name', None) in self._tools
    
    def get_tool_name(self, result: Any) -> Optional[str]:
        """Get the tool name from a tool call result."""
        return getattr(result, '_tool_name', None)
    
    async def execute_tool(self, tool_result: Any) -> Any:
        """Execute a tool call result."""
        if not self.is_tool_call(tool_result):
            raise ValueError("Not a tool call result")
        
        # Look up the function from the tools registry
        tool_name = self.get_tool_name(tool_result)
        func = self._tools.get(tool_name)
        
        if func is None:
            raise ValueError(f"Tool '{tool_name}' not found in runtime tools")
        
        # Convert the tool model instance to kwargs
        if hasattr(tool_result, "model_dump"):  # Pydantic
            # Get all fields excluding private attributes
            all_fields = tool_result.model_dump()
            # Remove any fields that start with underscore or are class-level attributes
            kwargs = {k: v for k, v in all_fields.items() 
                     if not k.startswith('_') and k not in ['_tool_func', '_tool_name']}
        else:  # Deserializable
            kwargs = {k: v for k, v in tool_result.__dict__.items() 
                     if not k.startswith('_')}
        
        # Debug print
        print(f"DEBUG execute_tool: tool_name={tool_name}, func={func}, kwargs={kwargs}")
        
        # Execute the function
        print(f"Executing tool: {tool_name} with args: {kwargs}")

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

async def _create_inference_client(model_str, store=None):
    """Create an appropriate inference client based on the model string."""
    # Parse model string to get provider and model
    parts = model_str.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid model string format: {model_str}")
    
    provider, model_name = parts
    
    if provider == "openrouter":
        client = create_openrouter_inference_client(model_name)
    elif provider == "anthropic":
        # Parse out thinking parameter if provided
        thinking_match = re.match(r"(.+)@thinking=(\d+)", model_name)
        if thinking_match:
            model_name, thinking = thinking_match.groups()
            client = create_anthropic_inference_client(model_name, thinking=int(thinking))
        else:
            client = create_anthropic_inference_client(model_name)
    elif provider == "bedrock":
        client = create_bedrock_inference_client(model_name)
    elif provider == "google-gemini":
        client = create_google_gemini_api_client(model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Wrap in TracingInferenceClient
    return TracingInferenceClient(client, store)

def _create_chat_message(prompt):
    """Convert a prompt string to a ChatMessage."""
    if isinstance(prompt, str):
        return [ChatMessage(role=ChatRole.USER, content=prompt)]
    return prompt

def _convert_tools_to_tuple_format(tools_dict):
    """Convert a tools dictionary to the format expected by tool_chain."""
    formatted_tools = []
    for tool_name, tool_fn in tools_dict.items():
        # Create tool spec
        spec = {
            "name": tool_name,
            "description": tool_fn.__doc__ or f"Tool {tool_name}",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
        
        # Extract function signature for parameters if possible
        sig = inspect.signature(tool_fn)
        for param_name, param in sig.parameters.items():
            if param_name != 'self':  # Skip self parameter for methods
                spec["parameters"]["properties"][param_name] = {"type": "string"}
        
        formatted_tools.append((tool_fn, spec))
    
    return formatted_tools

def _get_parser_for_type(output_type=None) -> OutputParser:
    """Get the appropriate parser for the given output type."""
    # For now, just use the default parser
    # In the future, we might select based on type or user configuration
    return get_default_parser()

import os  # Add import at the top if not already there

async def _call_llm(prompt, model, tools, output_type, store=None):
    """Execute a non-streaming LLM call, possibly with tool use."""
    print(f"Calling LLM with output type: {output_type}")
    client = await _create_inference_client(model, store)
    
    
    # Determine effective output type (with tools if applicable)
    effective_output_type = output_type
    
    if tools and output_type:
        # Validate all callables have typed parameters
        for name, func in tools.items():
            _validate_callable_params(func, name)
        
        # Determine which type system to use based on output_type
        tool_models = []
        if _is_pydantic_type(output_type):
            tool_models = [_create_pydantic_tool_model(func, name) 
                          for name, func in tools.items()]

        elif _is_deserializable_type(output_type):
            tool_models = [_create_deserializable_tool_class(func, name) 
                          for name, func in tools.items()]
        
        if tool_models:
            # Create Union type including tools and output
            effective_output_type = Union[*(tool_models + [output_type])]

    print(tool_models)

    # Get the appropriate parser
    parser = _get_parser_for_type(effective_output_type)
    
    # Enhance the prompt with type information
    # Parser expects a single type, not a Union, so use original output_type for prompt enhancement
    enhanced_prompt = parser.enhance_prompt(prompt, effective_output_type) if effective_output_type else prompt
    from pprint import pprint
    pprint(enhanced_prompt)  # Debugging: print the enhanced prompt
    
    messages = _create_chat_message(enhanced_prompt)
    
    # Standard LLM call (now includes tool types in effective_output_type)
    print(messages)

    result = await client.get_generation(
        messages=messages,
        max_tokens=8192,
        top_p=0.9,
        temperature=0.01,
    )

    print(result)

    # If output_type is provided, parse the result
    if effective_output_type and result is not None:
        parsed_result = parser.parse(result, effective_output_type)
        if parsed_result.success:
            print(parsed_result.value)  # Debugging: print the parsed value
            return parsed_result.value
        else:
            raise ValueError(f"Failed to parse result: {parsed_result.error}")

async def _call_llm_stream(prompt, model, tools, output_type, store=None):
    """Execute a streaming LLM call, possibly with tool use."""
    print(f"Streaming LLM call with output type: {output_type}")
    client = await _create_inference_client(model, store)
    
    # Get the appropriate parser
    parser = _get_parser_for_type(output_type)
    
    # Determine effective output type (with tools if applicable)
    effective_output_type = output_type
    
    if tools and output_type:
        # Validate all callables have typed parameters
        for name, func in tools.items():
            _validate_callable_params(func, name)
        
        # Determine which type system to use based on output_type
        tool_models = []
        if _is_pydantic_type(output_type):
            tool_models = [_create_pydantic_tool_model(func, name) 
                          for name, func in tools.items()]
        elif _is_deserializable_type(output_type):
            tool_models = [_create_deserializable_tool_class(func, name) 
                          for name, func in tools.items()]
        
        if tool_models:
            # Create Union type including tools and output
            effective_output_type = Union[tuple(tool_models + [output_type])]
    
    # Enhance the prompt with type information
    # Parser expects a single type, not a Union, so use original output_type for prompt enhancement
    enhanced_prompt = parser.enhance_prompt(prompt, output_type) if output_type else prompt
    
    messages = _create_chat_message(enhanced_prompt)
    
    # Create a streaming parser if output_type is provided
    stream_parser = None
    if effective_output_type:
        stream_parser = parser.create_stream_parser(effective_output_type)
    
    if tools and not _is_pydantic_type(output_type) and not _is_deserializable_type(output_type):
        # Fall back to existing tool_chain for non-typed outputs
        formatted_tools = _convert_tools_to_tuple_format(tools)
        
        async for chunk in client.connect_and_listen(
            messages=messages,
            max_tokens=8192,
            top_p=0.9,
            temperature=0.5,
        ):
            if effective_output_type and stream_parser:
                # Feed the chunk to the parser
                result = parser.feed_chunk(stream_parser, chunk)
                if result.value is not None:
                    yield result.value
            else:
                yield chunk

        
        # Validate the final result if using a parser
        if effective_output_type and stream_parser:
            final_result = parser.validate_final(stream_parser)
            if final_result.success and final_result.value is not None:
                yield final_result.value
    else:
        # Standard streaming (now includes tool types in effective_output_type)
        full_output = ""
        async for chunk in client.connect_and_listen(
            messages=messages,
            max_tokens=8192,
            top_p=0.9,
            temperature=0.5,
        ):
            full_output += chunk
            if effective_output_type and stream_parser:
                # Feed the chunk to the parser
                result = parser.feed_chunk(stream_parser, chunk)
                if result.value is not None:
                    yield result.value
            else:
                yield chunk
        
        # Validate the final result if using a parser
        if effective_output_type and stream_parser:
            final_result = parser.validate_final(stream_parser)
            if final_result.success and final_result.value is not None:
                yield final_result.value

def agentic(
    *,
    model: str,
    api_key: str | None = None,
    store_cfg: StoreConfigBase = MemoryStoreConfig(),
    autobind: bool = True,
) -> Callable[[Callable[..., R]],
              Callable[..., Awaitable[R]]]:
    """Decorator for *awaitable* agentic functions."""
    return cast(
        Callable[[Callable[..., R]],
                Callable[..., Awaitable[R]]],
        _agentic_core(
            model=model,
            api_key=api_key,
            store_cfg=store_cfg,
            autobind=autobind,
        ),
    )


def agentic_generator(
    *,
    model: str,
    api_key: str | None = None,
    store_cfg: StoreConfigBase = MemoryStoreConfig(),
    autobind: bool = True,
) -> Callable[[Callable[..., AsyncGenerator[Any, None]]],
              Callable[..., AsyncGenerator[Any, None]]]:
    """Decorator for *async-generator* agentic functions."""
    return cast(
        Callable[[Callable[..., AsyncGenerator[Any, None]]],
                Callable[..., AsyncGenerator[Any, None]]],
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
    model: str,
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
      • model (str) – model identifier (e.g., "openrouter:openai/gpt-4o")
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
                        origin is AsyncGenerator or 
                        origin is AsyncIterator or
                        (isinstance(origin, type) and issubclass(origin, (AsyncGenerator, AsyncIterator)))
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
                )
                rt._default_prompt = default_prompt
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

                if autobind:
                    for name, param in sig.parameters.items():
                        val = ba.arguments.get(name)
                        if callable(val):
                            rt.tool(val, name=name)
                
                # Iterate and yield values from the generator
                async for value in fn(*ba.args, **ba.kwargs):
                    yield value
                    
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
                )
                rt._default_prompt = default_prompt
                # Set the return type from the function's signature
                if ret_type != inspect.Signature.empty and ret_type is not None:
                    rt._return_type = ret_type

                ba = sig.bind_partial(*args, **kwargs, runtime=rt)

                if autobind:
                    for name, param in sig.parameters.items():
                        val = ba.arguments.get(name)
                        if callable(val):
                            rt.tool(val, name=name)

                result = await fn(*ba.args, **ba.kwargs)

                if not rt.used:
                    raise RuntimeError(
                        "agentic function completed without calling runtime.run() "
                        "or runtime.run_stream()"
                    )
                return result

        return functools.wraps(fn)(wrapper)
    return decorator


__all__ = ["agentic", "agentic_generator", "Runtime"]
