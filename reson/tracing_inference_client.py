import asyncio
from datetime import datetime, timedelta
import functools
import random
import logging
import math
from typing import AsyncGenerator, Dict, List, Callable, Awaitable, Any, Optional, Tuple
import opentelemetry.trace
from asimov.caches.cache import Cache
from asimov.services.inference_clients import (
    InferenceClient,
    ChatMessage,
    RetriesExceeded,
)

from reson.utils.tracing import trace_output

tracer = opentelemetry.trace.get_tracer(__name__)


_FALLBACK_TIME: Optional[datetime] = None

def with_fallback(func):
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def wrapped(self, *args, **kwargs):
            try:
                return await func(self, *args, **kwargs)
            except RetriesExceeded:
                logging.info(
                    "Switching to fallback client after retries exceeded on primary"
                )
                global _FALLBACK_TIME
                _FALLBACK_TIME = datetime.now()
                if self.fallback_client:
                    return await func(self, *args, **kwargs)
                raise

        return wrapped
    else:

        @functools.wraps(func)
        async def wrapped(self, *args, **kwargs):
            try:
                async for res in func(self, *args, **kwargs):
                    yield res
            except RetriesExceeded:
                logging.info(
                    "Switching to fallback client after retries exceeded on primary"
                )
                global _FALLBACK_TIME
                _FALLBACK_TIME = datetime.now()
                if not self.fallback_client:
                    raise
                async for res in func(self, *args, **kwargs):
                    yield res

        return wrapped


MILLION = 1_000_000


class TracingInferenceClient(InferenceClient):
    primary_client: InferenceClient
    fallback_client: Optional[InferenceClient]
    # When the fallback client was swapped to. Used to swap back after a timeout.
    _fallback_time: Optional[datetime]
    _cache: Optional[Cache]

    def __init__(
        self,
        inference_client: InferenceClient,
        cache: Optional[Cache],
        fallback_client: Optional[InferenceClient] = None,
    ):
        self.primary_client = inference_client
        self.primary_client.trace_cb = self._trace_cb
        self.fallback_client = fallback_client
        if self.fallback_client:
            self.fallback_client.trace_cb = self._trace_cb
        self._cache = cache

    @property
    def model(self) -> str:  # type: ignore
        return self.client.model

    @property
    def client(self) -> InferenceClient:
        global _FALLBACK_TIME
        if _FALLBACK_TIME is not None:
            if datetime.now() - _FALLBACK_TIME > timedelta(minutes=5):
                logging.info("Switching back to primary client")
                _FALLBACK_TIME = None
                return self.primary_client
            elif self.fallback_client is not None:
                return self.fallback_client
        return self.primary_client

    async def _trace_cb(self, id, req, resp, cost) -> None:
        if self._cache is None:
            return

        await trace_output(req, f"inference_req_{id}")
        await trace_output(resp, f"inference_resp_{id}")

        credit_usage = 0.0

        if "claude" in self.client.model:
            if "haiku" in self.client.model:
                # $0.8/MTok in
                credit_usage += cost.input_tokens * (80 / MILLION)
                # $4/MTok out
                credit_usage += cost.output_tokens * (400 / MILLION)
                # $0.08/MTok cache read
                credit_usage += cost.cache_read_input_tokens * (8 / MILLION)
                # $1/MTok cache write
                credit_usage += cost.cache_write_input_tokens * (100 / MILLION)
            elif "sonnet" in self.client.model:
                # $3/MTok in
                credit_usage += cost.input_tokens * (300 / MILLION)
                # $15/MTok out
                credit_usage += cost.output_tokens * (1500 / MILLION)
                # $0.3/MTok cache read
                credit_usage += cost.cache_read_input_tokens * (30 / MILLION)
                # $3.75/MTok cache write
                credit_usage += cost.cache_write_input_tokens * (375 / MILLION)
        elif "openai" in self.client.model:
            if self.client.model == "openai/o4-mini":
                # $1.1/MTok in
                credit_usage += cost.input_tokens * (110 / MILLION)
                # $4.4/MTok out
                credit_usage += cost.output_tokens * (440 / MILLION)
                # $0.275/MTok cache read
                credit_usage += cost.cache_read_input_tokens * (27.5 / MILLION)
            elif self.client.model == "openai/o3":
                # $10/MTok in
                credit_usage += cost.input_tokens * (1000 / MILLION)
                # $40/MTok out
                credit_usage += cost.output_tokens * (4000 / MILLION)
                # $2.5/MTok cache read
                credit_usage += cost.cache_read_input_tokens * (250 / MILLION)
        else:
            logging.warning(f"No cost information for model {self.client.model}.")

        credit_usage += cost.dollar_adjust * 100

        all_usage = await self._cache.get("credits_used", 0)
        all_usage += math.ceil(credit_usage)
        await self._cache.set("credits_used", all_usage)

    @with_fallback
    async def connect_and_listen(
        self, messages: List[ChatMessage], max_tokens=8192, top_p=0.9, temperature=0.5
    ):
        full_output = ""
        async for token in self.client.connect_and_listen(
            messages, max_tokens, top_p, temperature
        ):
            yield token
            full_output += token
        id = random.randrange(1000000)
        await trace_output([m.model_dump() for m in messages], f"streaming_input_{id}")
        await trace_output(full_output, f"streaming_output_{id}")

    @with_fallback
    async def get_generation(
        self, messages: List[ChatMessage], max_tokens=8192, top_p=0.9, temperature=0.5
    ):
        res = await self.client.get_generation(messages, max_tokens, top_p, temperature)
        id = random.randrange(1000000)
        await trace_output([m.model_dump() for m in messages], f"gen_input_{id}")
        await trace_output(res, f"gen_output_{id}")
        return res
    
    @with_fallback
    async def _unstructured(
        self,
        serialized_messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        max_tokens=8192,
        top_p=0.9,
        temperature=0.5,
        middlewares: List[Callable[[dict[str, Any]], Awaitable[None]]] = [],
    ):
        res = await self.client._unstructured(
            serialized_messages,
            system,
            max_tokens,
            top_p,
            temperature,
            middlewares,
        )
        id = random.randrange(1000000)
        await trace_output(serialized_messages, f"unstructured_input_{id}")
        await trace_output(res, f"unstructured_output_{id}")
        return res
        
    @with_fallback
    async def _unstructured_stream(
        self,
        serialized_messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        max_tokens=8192,
        top_p=0.9,
        temperature=0.5,
        middlewares: List[Callable[[dict[str, Any]], Awaitable[None]]] = [],
    ):
        res = await self.client._unstructured_stream(
            serialized_messages,
            system,
            max_tokens,
            top_p,
            temperature,
            middlewares,
        )
        
        id = random.randrange(1000000)
        await trace_output(serialized_messages, f"unstructured_stream_input_{id}")
        await trace_output(res, f"unstructured_stream_output_{id}")
        return res

    @with_fallback
    async def _tool_chain_stream(
        self,
        serialized_messages: List[Dict[str, Any]],
        tools: List[Tuple[Callable, Dict[str, Any]]],
        system: Optional[str] = None,
        max_tokens=8192,
        top_p=0.9,
        temperature=0.5,
        tool_choice="any",
        middlewares: List[Callable[[dict[str, Any]], Awaitable[None]]] = [],
    ) -> List[Dict[str, Any]]:
        if "openai" in self.client.model and tool_choice == "any":
            tool_choice = "required"

        res = await self.client._tool_chain_stream(
            serialized_messages,
            tools,
            system,
            max_tokens,
            top_p,
            temperature,
            tool_choice,
            middlewares,
        )
        id = random.randrange(1000000)
        await trace_output(serialized_messages, f"tool_chain_input_{id}")
        await trace_output(res, f"tool_chain_output_{id}")
        return res

    @with_fallback
    async def tool_chain(
        self,
        messages,
        tools,
        max_tokens=8192,
        top_p=0.9,
        temperature=0.5,
        max_iterations=10,
        tool_choice="any",
        middlewares: List[Callable[[dict[str, Any]], Awaitable[None]]] = [],
        mode_swap_callback: Optional[Callable] = None,
        tool_parser=None,
        tool_result_reducer=None
    ):
        if "openai" in self.client.model and tool_choice == "any":
            tool_choice = "required"

        hooked_tools = []
        for func, spec in tools:

            def capture(func, spec):
                async def hook(resp):
                    try:
                        with tracer.start_as_current_span(spec["name"]):
                            res = await func(resp)
                        await trace_output(res, spec["name"])
                        return res
                    except StopAsyncIteration:
                        await trace_output("StopAsyncIteration", spec["name"])
                        raise

                return hook

            hooked_tools.append((capture(func, spec), spec))
        return await self.client.tool_chain(
            messages,
            hooked_tools,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            max_iterations=max_iterations,
            tool_choice=tool_choice,
            middlewares=middlewares,
            mode_swap_callback=mode_swap_callback,
            tool_parser=tool_parser,
            tool_result_reducer=tool_result_reducer
        )
