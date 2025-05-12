# reson/agentic.py
from __future__ import annotations

from asimov.services.inference_clients import InferenceClient, ChatMessage, ChatRole
from typing import Callable, TypeVar, ParamSpec
from reson.stores import StoreConfigBase, MemoryStore, MemoryStoreConfig, RedisStore, RedisStoreConfig, PostgresStore, PostgresStoreConfig, Store
from asimov.asimov_base import AsimovBase
from pydantic import PrivateAttr, Field

import inspect, functools
from typing import Callable, ParamSpec, TypeVar, Awaitable, Any, AsyncIterator

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
        return await _call_llm(prompt, self.model, self._tools, output_type, self.store)

    async def run_stream(
        self,
        *,
        prompt: str | None,
        output_type: type,
    ) -> AsyncIterator[Any]:
        """Execute a streaming LLM call yielding chunks as they arrive."""
        self.used = True
        prompt = prompt or self._default_prompt
        async for chunk in _call_llm_stream(
            prompt, self.model, self._tools, output_type, self.store
        ):
            yield chunk

    @property
    def context(self):
        """Legacy accessor for context."""
        return self._context


import re
from asimov.services.inference_clients import ChatMessage, ChatRole
from typing import List, Dict, Any, AsyncGenerator

from reson.utils.inference import (
    create_google_gemini_api_client,
    create_openrouter_inference_client,
    create_anthropic_inference_client,
    create_bedrock_inference_client,
    create_google_anthropic_inference_client,
)
from reson.tracing_inference_client import TracingInferenceClient

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
        client = create_anthropic_inference_client(model_name)
    elif provider == "bedrock":
        client = create_bedrock_inference_client(model_name)
    elif provider == "google-gemini":
        client = create_google_gemini_api_client(model_name)
    elif provider == "google-anthropic":
        client = create_google_anthropic_inference_client(model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    # Wrap in TracingInferenceClient
    return TracingInferenceClient(client, store)

def _create_chat_message(prompt):
    """Convert a prompt string to a ChatMessage."""
    if isinstance(prompt, str):
        return [ChatMessage(role=ChatRole.user, content=prompt)]
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

async def _call_llm(prompt, model, tools, output_type, store=None):
    """Execute a non-streaming LLM call, possibly with tool use."""
    client = await _create_inference_client(model, store)
    messages = _create_chat_message(prompt)
    
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
        
        return output_type() if output_type else result
    else:
        # Standard LLM call without tools
        result = await client.get_generation(
            messages=messages,
            max_tokens=8192,
            top_p=0.9,
            temperature=0.5,
        )
        return output_type() if output_type else result

async def _call_llm_stream(prompt, model, tools, output_type, store=None):
    """Execute a streaming LLM call, possibly with tool use."""
    client = await _create_inference_client(model, store)
    messages = _create_chat_message(prompt)
    
    if tools:
        # Convert tools to the format expected by tool_chain
        formatted_tools = _convert_tools_to_tuple_format(tools)
        
        # Use tool_chain for tools in streaming mode
        serialized_messages = [m.model_dump() for m in messages]
        
        try:
            # First try to use _unstructured_stream
            result = await client._unstructured_stream(
                serialized_messages=serialized_messages,
                system=None,
                max_tokens=8192,
                top_p=0.9,
                temperature=0.5,
            )
        except (AttributeError, NotImplementedError):
            # Fall back to _tool_chain_stream if _unstructured_stream is not available
            result = await client._tool_chain_stream(
                serialized_messages=serialized_messages,
                tools=formatted_tools,
                system=None,
                max_tokens=8192,
                top_p=0.9,
                temperature=0.5,
            )
        
        if output_type:
            yield output_type()
        else:
            yield result
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
            if output_type:
                # In a real implementation, you might need partial parsing logic here
                yield chunk
            else:
                yield chunk


# ───────── decorator ─────────
def agentic(
    *,
    model: str,
    api_key: str = None,
    store_cfg: StoreConfigBase = MemoryStoreConfig(),
    autobind: bool = True,
):
    """
    Decorator factory:
      • model   – model identifier ("openai:gpt-4o")
      • store   – "memory" | "redis" | "postgres"  (fill in adapters later)
      • autobind – automatically expose callable params as tools
    """

    store_obj = _build_store(store_cfg) 
    def decorator(fn: Callable[P, R]) -> Callable[P, Awaitable[R]]:
        sig = inspect.signature(fn)
        if "runtime" not in sig.parameters:
            raise TypeError("`runtime` parameter missing in agentic function")

        default_prompt = inspect.getdoc(fn) or ""
        ret_type       = sig.return_annotation

        async def wrapper(*args: P.args, **kwargs: P.kwargs):   # type: ignore[override]
            rt = Runtime(
                model=model,
                store=store_obj,
            )
            rt._default_prompt = default_prompt

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
