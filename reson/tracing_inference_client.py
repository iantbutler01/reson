import asyncio
from datetime import datetime, timedelta
import functools
import random
import logging
import math
from typing import AsyncGenerator, Dict, List, Callable, Awaitable, Any, Optional, Tuple
import opentelemetry.trace
from reson.caches.cache import Cache
from reson.services.inference_clients import (
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
        async def wrapped(self, *args, **kwargs):  # type: ignore
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
            # Handle both tuple and string formats
            if isinstance(token, tuple):
                yield token  # Pass through the tuple
                # Only accumulate content tokens for tracing
                if len(token) == 2 and token[0] == "content":
                    full_output += token[1]
            else:
                yield token  # Legacy string format
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
