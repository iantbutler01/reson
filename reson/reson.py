# reson/agentic.py
from __future__ import annotations

from asimov.services.inference_clients import InferenceClient, ChatMessage, ChatRole
from typing import List, Dict, Any, AsyncGenerator, Optional, Type, TypeVar, get_origin, get_args, Callable, TypeVar, ParamSpec, Union, Awaitable
from reson.stores import StoreConfigBase, MemoryStore, MemoryStoreConfig, RedisStore, RedisStoreConfig, PostgresStore, PostgresStoreConfig, Store
from asimov.asimov_base import AsimovBase
from pydantic import PrivateAttr, Field

import inspect, functools
from typing import Callable, ParamSpec, TypeVar, Awaitable, Any, AsyncIterator, Union
import re
import json
from asimov.services.inference_clients import ChatMessage, ChatRole

from reson.utils.parsers import OutputParser, get_default_parser

from reson.utils.inference import (
    create_google_gemini_api_client,
    create_openrouter_inference_client,
    create_anthropic_inference_client,
    create_bedrock_inference_client,
)
from reson.tracing_inference_client import TracingInferenceClient

P = ParamSpec("P")
R = TypeVar("R")

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
class Runtime(AsimovBase):
    """
    Runtime object that wraps calls to the underlying LLM and exposes
    dynamically bound tools.

    Inherits from `AsimovBase` (a thin shim over `pydantic.BaseModel`)
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
    _context: _Ctx = PrivateAttr(default=None)
    _return_type: type | None = PrivateAttr(default=None)
    
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
    
    # Get the appropriate parser
    parser = _get_parser_for_type(output_type)
    
    # Enhance the prompt with type information if output_type is provided
    if output_type:
        enhanced_prompt = parser.enhance_prompt(prompt, output_type)
    else:
        enhanced_prompt = prompt
    
    messages = _create_chat_message(enhanced_prompt)
    
    if tools:
        # Convert tools to the format expected by tool_chain
        formatted_tools = _convert_tools_to_tuple_format(tools)
        
        serialized_messages = [m.model_dump() for m in messages]
        
        # Use unstructured method if available, otherwise fall back to tool_chain
        try:
            # Use the _unstructured method for unstructured tool calling
            result = await client._unstructured(
                serialized_messages=serialized_messages,
                system=None,
                max_tokens=8192,
                top_p=0.9,
                temperature=0.5,
            )
        except (AttributeError, NotImplementedError):
            # Fall back to regular tool_chain if _unstructured is not available
            result = await client.tool_chain(
                messages=messages,
                tools=formatted_tools,
                max_tokens=8192,
                top_p=0.9,
                temperature=0.5,
            )
        
        # If output_type is provided, parse the result
        if output_type:
            result_text = result if isinstance(result, str) else json.dumps(result)
            parsed_result = parser.parse(result_text, output_type)
            if parsed_result.success:
                return parsed_result.value
            else:
                # If parsing fails, create an appropriate empty value based on type
                return _create_empty_value(output_type)
        else:
            return result
    else:
        # Standard LLM call without tools
        result = await client.get_generation(
            messages=messages,
            max_tokens=8192,
            top_p=0.9,
            temperature=0.5,
        )
        
        # If output_type is provided, parse the result
        if output_type:
            print(f"LLM Response to parse:\n{result[:500]}...")  # Print first 500 chars
            
            # Log enhanced prompt for debugging
            print(f"Enhanced prompt was:\n{enhanced_prompt[:500]}...")
            
            parsed_result = parser.parse(result, output_type)
            if parsed_result.success:
                print(f"Successfully parsed result: {parsed_result.value}")
                return parsed_result.value
            else:
                print(f"Parsing failed with error: {parsed_result.error}")
                # Try a more aggressive approach to find any valid JSON in the response
                try:
                    import re
                    json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', result)
                    if json_match:
                        json_str = json_match.group(0)
                        print(f"Attempting to parse extracted JSON: {json_str[:100]}...")
                        json_data = json.loads(json_str)
                        
                        # For List[Person], try to convert the JSON to the right type
                        origin = get_origin(output_type)
                        args = get_args(output_type)
                        
                        if origin == list and args:
                            item_type = args[0]
                            # Try to convert each item to the expected type
                            if hasattr(item_type, "model_validate"):  # Pydantic v2
                                items = [item_type.model_validate(item) for item in json_data]
                                print(f"Manual JSON extraction succeeded with {len(items)} items")
                                return items
                except Exception as e:
                    print(f"Manual JSON extraction failed: {e}")
                
                # If all parsing attempts fail, create an appropriate empty value
                print(f"Returning empty value for type: {output_type}")
                return _create_empty_value(output_type)
        else:
            return result

async def _call_llm_stream(prompt, model, tools, output_type, store=None):
    """Execute a streaming LLM call, possibly with tool use."""
    print(f"Streaming LLM call with output type: {output_type}")
    client = await _create_inference_client(model, store)
    
    # Get the appropriate parser
    parser = _get_parser_for_type(output_type)
    
    # Enhance the prompt with type information if output_type is provided
    if output_type:
        enhanced_prompt = parser.enhance_prompt(prompt, output_type)
    else:
        enhanced_prompt = prompt
    
    messages = _create_chat_message(enhanced_prompt)
    
    # Create a streaming parser if output_type is provided
    stream_parser = None
    if output_type:
        stream_parser = parser.create_stream_parser(output_type)
    
    if tools:
        # Convert tools to the format expected by tool_chain
        formatted_tools = _convert_tools_to_tuple_format(tools)
        
        # Use tool_chain for tools in streaming mode
        serialized_messages = [m.model_dump() for m in messages]
        
        try:
            # First try to use _unstructured_stream
            async for chunk in client._unstructured_stream(
                serialized_messages=serialized_messages,
                system=None,
                max_tokens=8192,
                top_p=0.9,
                temperature=0.5,
            ):
                if output_type and stream_parser:
                    # Feed the chunk to the parser
                    result = parser.feed_chunk(stream_parser, chunk)
                    if result.value is not None:
                        yield result.value
                else:
                    yield chunk
        except (AttributeError, NotImplementedError):
            # Fall back to _tool_chain_stream if _unstructured_stream is not available
            async for chunk in client._tool_chain_stream(
                serialized_messages=serialized_messages,
                tools=formatted_tools,
                system=None,
                max_tokens=8192,
                top_p=0.9,
                temperature=0.5,
            ):
                if output_type and stream_parser:
                    # Feed the chunk to the parser
                    chunk_str = chunk if isinstance(chunk, str) else json.dumps(chunk)
                    result = parser.feed_chunk(stream_parser, chunk_str)
                    if result.value is not None:
                        yield result.value
                else:
                    yield chunk
        
        # Validate the final result if using a parser
        if output_type and stream_parser:
            final_result = parser.validate_final(stream_parser)
            if final_result.success and final_result.value is not None:
                yield final_result.value
    else:
        # Standard streaming without tools
        full_output = ""
        async for chunk in client.connect_and_listen(
            messages=messages,
            max_tokens=8192,
            top_p=0.9,
            temperature=0.5,
        ):
            full_output += chunk
            if output_type and stream_parser:
                # Feed the chunk to the parser
                result = parser.feed_chunk(stream_parser, chunk)
                if result.value is not None:
                    yield result.value
            else:
                yield chunk
        
        # Validate the final result if using a parser
        if output_type and stream_parser:
            final_result = parser.validate_final(stream_parser)
            if final_result.success and final_result.value is not None:
                yield final_result.value


# ───────── decorator ─────────
def agentic(
    *,
    model: str,
    api_key: str = None,
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
    def decorator(fn: Callable[P, R]) -> Callable[P, Union[Awaitable[R], AsyncGenerator]]:
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
            async def gen_wrapper(*args: P.args, **kwargs: P.kwargs):
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
            async def wrapper(*args: P.args, **kwargs: P.kwargs):   # type: ignore[override]
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


__all__ = ["agentic", "Runtime"]
