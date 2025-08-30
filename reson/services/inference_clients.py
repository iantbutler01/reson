import asyncio
from collections.abc import Hashable
import botocore.exceptions
from pydantic import Field
import json
from typing import (
    Awaitable,
    Callable,
    List,
    Dict,
    Any,
    Optional,
    Tuple,
    AsyncGenerator,
    Union,
)
from enum import Enum
from abc import ABC, abstractmethod
import logging

import aioboto3
import httpx
import backoff
import opentelemetry.instrumentation.httpx
import opentelemetry.trace
import pydantic_core
import google.auth
import google.auth.transport.requests
import google.genai.errors
from google import genai
from google.genai import types
import os

if os.environ.get("ART_ENABLED"):
    import art

from reson.reson_base import ResonBase

logger = logging.getLogger(__name__)
tracer = opentelemetry.trace.get_tracer(__name__)
opentelemetry.instrumentation.httpx.HTTPXClientInstrumentor().instrument()


class NonRetryableException(Exception):
    """
    An exception which, when raised within a module, will prevent the module from being retried
    regardless of the retry_on_failure configuration.
    """

    pass


class InferenceException(Exception):
    """
    A generic exception for inference errors.
    Should be safe to retry.
    ValueError is raised if the request is un-retryable (e.g. parameters are malformed).
    """

    pass


class ContextLengthExceeded(InferenceException, ValueError):
    pass


class RetriesExceeded(InferenceException):
    """
    Raised when the maximum number of retries is exceeded.
    """

    pass


class ChatRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"


class ChatMessage(ResonBase):
    role: ChatRole
    content: str
    cache_marker: bool = False
    model_families: List[str] = Field(default_factory=list)
    signature: Optional[str] = Field(default=None)


class ToolResult(ResonBase):
    """Encapsulates tool result content for conversation history."""

    tool_use_id: str
    content: str
    is_error: bool = False
    signature: Optional[str] = Field(
        default=None
    )  # For signature/thought_signature handling
    tool_obj: Optional[Any] = Field(
        default=None
    )  # Original tool call object (dict or marshalled Deserializable)

    def to_provider_format(self, provider: str) -> Dict[str, Any]:
        """Convert to provider-specific format for API calls."""
        if provider in ["openai", "openrouter", "custom-openai"]:
            return self._format_openai_content()
        elif provider in ["anthropic", "bedrock"]:
            return self._format_anthropic_content()
        elif provider and provider.startswith("google"):
            return self._format_google_content()
        else:
            # Fallback format
            return {"content": self.content}

    def to_chat_message(self) -> "ChatMessage":
        """Convert to ChatMessage for backward compatibility."""
        return ChatMessage(
            role=ChatRole.USER, content=self.content, signature=self.signature
        )

    @classmethod
    def create(
        cls,
        tool_call_obj_or_list: Union[Any, Tuple[Any, str], List[Tuple[Any, str]]],
        signature: Optional[str] = None,
    ) -> Union["ToolResult", List["ToolResult"]]:
        """Factory method to create ToolResult(s) from tool call objects."""

        if isinstance(tool_call_obj_or_list, tuple):
            tool_calls_and_results = [tool_call_obj_or_list]
        elif isinstance(tool_call_obj_or_list, list):
            tool_calls_and_results = tool_call_obj_or_list
        else:
            raise ValueError(
                "Invalid arguments - need either (tool_obj, result) or [(tool1, result1), ...]"
            )

        if not tool_calls_and_results:
            raise ValueError("No tool calls and results provided")

        # Create ToolResult objects
        results = []
        for tool_call_obj, result_content in tool_calls_and_results:
            # Extract tool_use_id from the tool call object
            tool_use_id = cls._extract_tool_use_id(tool_call_obj)

            tool_result = cls(
                tool_use_id=tool_use_id,
                content=result_content,
                signature=signature,
                tool_obj=tool_call_obj,
            )
            results.append(tool_result)

        # Return single result or list based on input
        if len(results) == 1:
            return results[0]
        else:
            return results

    @classmethod
    def _extract_tool_use_id(cls, tool_call_obj: Any) -> str:
        """Extract tool_use_id from tool call object."""
        if hasattr(tool_call_obj, "_tool_use_id"):
            return tool_call_obj._tool_use_id
        elif isinstance(tool_call_obj, dict):
            return tool_call_obj.get("id", "")
        else:
            return ""

    @classmethod
    def _extract_signature(cls, tool_call_obj: Any) -> Optional[str]:
        """Extract signature/thought_signature from tool call objects."""
        if hasattr(tool_call_obj, "signature"):
            return tool_call_obj.signature
        elif hasattr(tool_call_obj, "thought_signature"):
            return tool_call_obj.thought_signature
        elif isinstance(tool_call_obj, dict):
            return tool_call_obj.get("signature") or tool_call_obj.get(
                "thought_signature"
            )
        return None

    def _format_openai_content(self) -> Dict[str, Any]:
        """Format content for OpenAI/OpenRouter API calls."""
        return {
            "type": "function_call_output",
            "call_id": self.tool_use_id,
            "output": self.content,
        }

    def _format_anthropic_content(self) -> Dict[str, Any]:
        """Format content for Anthropic API calls."""
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content,
        }

    def _format_google_content(self) -> Dict[str, Any]:
        """Format content for Google API calls."""
        result = {
            "functionResponse": {
                "name": (
                    getattr(self.tool_obj, "_tool_name", "") if self.tool_obj else ""
                ),
                "response": {"result": self.content},
            }
        }

        # Add thought signature preservation for Google
        if self.signature:
            result["_google_thought_signature_required"] = True
            result["_original_response_required"] = True

        return result


class AnthropicRequest(ResonBase):
    anthropic_version: str
    system: str
    messages: List[Dict[str, Any]]
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    tool_choice: Dict[str, Any] = Field(default_factory=dict)
    max_tokens: int = 512
    temperature: float = 0.5
    top_p: float = 0.9
    top_k: int = 50


class InferenceCost(ResonBase):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    # Dollar amount that should be added on to the cost of the request
    dollar_adjust: float = 0.0


class InferenceClient(ABC):
    model: str
    _trace_id: int = 0
    trace_cb: Optional[
        Callable[
            [int, list[dict[str, Any]], list[dict[str, Any]], InferenceCost],
            Awaitable[None],
        ]
    ] = None
    _cost: InferenceCost
    _last_cost: InferenceCost

    def __init__(self):
        self._cost = InferenceCost()
        self._last_cost = InferenceCost()

    async def _trace(self, request, response):
        opentelemetry.trace.get_current_span().set_attribute(
            "inference.usage.input_tokens", self._cost.input_tokens
        )
        opentelemetry.trace.get_current_span().set_attribute(
            "inference.usage.cached_input_tokens",
            self._cost.cache_read_input_tokens,
        )
        opentelemetry.trace.get_current_span().set_attribute(
            "inference.usage.output_tokens", self._cost.output_tokens
        )

        if self.trace_cb:
            logger.debug(f"Request {self._trace_id} cost {self._cost}")
            await self.trace_cb(self._trace_id, request, response, self._cost)
            self._last_cost = self._cost
            self._cost = InferenceCost()
            self._trace_id += 1

    @abstractmethod
    def connect_and_listen(
        self,
        messages: List[Union[ChatMessage, "ToolResult"]],
        max_tokens=4096,
        top_p=0.9,
        temperature=0.5,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Tuple[str, Any], None]:
        pass

    @abstractmethod
    async def get_generation(
        self,
        messages: List[Union[ChatMessage, "ToolResult"]],
        max_tokens=4096,
        top_p=0.9,
        temperature=0.5,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        pass

    def _convert_messages_to_provider_format(
        self, messages: List[Union[ChatMessage, "ToolResult"]], provider: str
    ) -> List[Dict[str, Any]]:
        """Convert mixed ChatMessage/ToolResult list to provider-specific format."""
        converted_messages = []

        for msg in messages:
            if isinstance(msg, ChatMessage):
                # Handle ChatMessage
                converted_messages.append(
                    {"role": msg.role.value, "content": msg.content}
                )
            else:  # ToolResult
                # Convert ToolResult to provider format
                provider_content = msg.to_provider_format(provider)
                converted_messages.append(
                    {
                        "role": "user",
                        "content": (
                            provider_content
                            if isinstance(provider_content, str)
                            else json.dumps(provider_content)
                        ),
                    }
                )

        return converted_messages


class BedrockInferenceClient(InferenceClient):
    def __init__(self, model: str, region_name="us-east-1"):
        super().__init__()
        self.model = model
        self.region_name = region_name
        self.session = aioboto3.Session()
        self.anthropic_version = "bedrock-2023-05-31"

    @tracer.start_as_current_span(name="BedrockInferenceClient.get_generation")
    @backoff.on_exception(backoff.expo, InferenceException, max_time=60)
    async def get_generation(
        self,
        messages: List[Union[ChatMessage, "ToolResult"]],
        max_tokens=1024,
        top_p=0.9,
        temperature=0.5,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        # Convert messages to ChatMessage format
        processed_messages = []
        for msg in messages:
            if hasattr(msg, "to_chat_message"):  # ToolResult
                processed_messages.append(msg.to_chat_message())
            else:  # ChatMessage
                processed_messages.append(msg)

        system = ""
        if processed_messages[0].role == ChatRole.SYSTEM:
            system = processed_messages[0].content
            processed_messages = processed_messages[1:]

        request = AnthropicRequest(
            anthropic_version=self.anthropic_version,
            system=system,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": msg.role.value,
                    "content": [{"type": "text", "text": msg.content}],
                }
                for msg in processed_messages
            ],
        )

        async with self.session.client(
            service_name="bedrock-runtime",
            region_name=self.region_name,
        ) as client:
            try:
                response = await client.invoke_model(
                    body=request.model_dump_json(exclude={"tools", "tool_choice"}),
                    modelId=self.model,
                    contentType="application/json",
                    accept="application/json",
                )
            except client.exceptions.ValidationException as e:
                raise ValueError(str(e))
            except botocore.exceptions.ClientError as e:
                raise InferenceException(str(e))

            body: dict = json.loads(await response["body"].read())

            self._cost.input_tokens += body["usage"]["input_tokens"]
            self._cost.output_tokens += body["usage"]["output_tokens"]

            await self._trace(request.messages, body["content"])

            return body["content"][0]["text"]

    @tracer.start_as_current_span(name="BedrockInferenceClient.connect_and_listen")
    async def connect_and_listen(
        self,
        messages: List[ChatMessage],
        max_tokens=1024,
        top_p=0.9,
        temperature=0.5,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        system = ""
        if messages[0].role == ChatRole.SYSTEM:
            system = messages[0].content
            messages = messages[1:]

        request = AnthropicRequest(
            anthropic_version=self.anthropic_version,
            system=system,
            top_p=top_p,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": msg.role.value,
                    "content": [{"type": "text", "text": msg.content}],
                }
                for msg in messages
            ],
        )

        async with self.session.client(
            service_name="bedrock-runtime",
            region_name=self.region_name,
        ) as client:
            try:
                response = await client.invoke_model_with_response_stream(
                    body=request.model_dump_json(exclude={"tools", "tool_choice"}),
                    modelId=self.model,
                    contentType="application/json",
                    accept="application/json",
                )
            except client.exceptions.ValidationException as e:
                raise ValueError(str(e))
            except botocore.exceptions.ClientError as e:
                raise InferenceException(str(e))

            out = ""
            current_tool_blocks = {}  # Track tool building by index like OpenAI

            async for chunk in response["body"]:
                chunk_json = json.loads(chunk["chunk"]["bytes"].decode())
                chunk_type = chunk_json["type"]

                if chunk_type == "content_block_delta":
                    content_type = chunk_json["delta"]["type"]

                    if content_type == "text_delta":
                        text = chunk_json["delta"]["text"]
                        out += text
                        yield ("content", text)
                    elif content_type == "input_json_delta" and tools:
                        # Progressive tool input accumulation (like OpenAI)
                        index = chunk_json["index"]
                        if index in current_tool_blocks:
                            # Accumulate partial JSON
                            current_tool_blocks[index]["input"] += chunk_json["delta"][
                                "partial_json"
                            ]

                            # Convert to OpenAI format
                            openai_format = {
                                "id": current_tool_blocks[index]["id"],
                                "function": {
                                    "name": current_tool_blocks[index]["name"],
                                    "arguments": current_tool_blocks[index][
                                        "input"
                                    ],  # JSON string
                                },
                            }
                            yield ("tool_call_partial", openai_format)

                elif chunk_type == "content_block_start":
                    # Initialize tool tracking
                    content_block = chunk_json.get("content_block", {})
                    if content_block.get("type") == "tool_use":
                        index = chunk_json["index"]
                        current_tool_blocks[index] = {
                            "id": content_block["id"],
                            "name": content_block["name"],
                            "input": "",  # Start accumulating JSON
                        }

                elif chunk_type == "content_block_stop":
                    # Tool completion
                    index = chunk_json["index"]
                    if tools and index in current_tool_blocks:
                        # Convert to OpenAI format
                        openai_format = {
                            "id": current_tool_blocks[index]["id"],
                            "function": {
                                "name": current_tool_blocks[index]["name"],
                                "arguments": (
                                    json.dumps(
                                        json.loads(current_tool_blocks[index]["input"])
                                    )
                                    if current_tool_blocks[index]["input"]
                                    else "{}"
                                ),
                            },
                        }
                        yield ("tool_call_complete", openai_format)
                        # Clean up completed tool
                        del current_tool_blocks[index]
                elif chunk_type == "message_start":
                    self._cost.input_tokens += chunk_json["message"]["usage"][
                        "input_tokens"
                    ]
                elif chunk_type == "message_delta":
                    self._cost.output_tokens += chunk_json["usage"]["output_tokens"]

            await self._trace(request.messages, [{"text": out}])


class AnthropicInferenceClient(InferenceClient):
    def __init__(
        self,
        model: str,
        api_key: str,
        api_url: str = "https://api.anthropic.com/v1/messages",
        thinking: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.thinking = thinking

    async def _post(self, request: dict):
        async with httpx.AsyncClient() as client:
            return await client.post(
                self.api_url,
                timeout=300000,
                json=request,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "prompt-caching-2024-07-31,output-128k-2025-02-19",
                },
            )

    def _stream(self, client, request: dict):
        return client.stream(
            "POST",
            self.api_url,
            timeout=300000,
            json=request,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "prompt-caching-2024-07-31,output-128k-2025-02-19",
            },
        )

    @tracer.start_as_current_span(name="AnthropicInferenceClient.get_generation")
    @backoff.on_exception(backoff.expo, InferenceException, max_time=60)
    async def get_generation(
        self,
        messages: List[ChatMessage],
        max_tokens=1024,
        top_p=0.9,
        temperature=0.5,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        system = None
        if messages[0].role == ChatRole.SYSTEM:
            system = {
                "system": [
                    {"type": "text", "text": messages[0].content}
                    | (
                        {"cache_control": {"type": "ephemeral"}}
                        if messages[0].cache_marker
                        else {}
                    )
                ]
            }
            messages = messages[1:]

        request = {
            "model": self.model,
            "top_p": top_p,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": msg.role.value, "content": msg.content}
                | ({"cache_control": {"type": "ephemeral"}} if msg.cache_marker else {})
                for msg in messages
            ],
            "stream": False,
        }

        # Add tools if provided
        if tools:
            request["tools"] = tools
            request["tool_choice"] = {"type": "auto"}
            # Enable parallel tool calling (Anthropic default is disabled)
            request["disable_parallel_tool_use"] = False

        if system:
            request.update(system)

        if self.thinking:
            request["thinking"] = {"type": "enabled", "budget_tokens": self.thinking}
            request["max_tokens"] += self.thinking
            request["temperature"] = 1
            del request["top_p"]

        response = await self._post(
            request,
        )

        if response.status_code == 400:
            # TODO: ContextLengthExceeded
            raise ValueError(await response.aread())
        elif response.status_code != 200:
            raise InferenceException(await response.aread())

        body: dict = response.json()

        # https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#tracking-cache-performance
        self._cost.input_tokens += body["usage"]["input_tokens"]
        self._cost.cache_read_input_tokens += body["usage"].get(
            "cache_read_input_tokens", 0
        )
        self._cost.cache_write_input_tokens += body["usage"].get(
            "cache_creation_input_tokens", 0
        )
        self._cost.output_tokens += body["usage"]["output_tokens"]

        await self._trace(request["messages"], body["content"])

        # Return full response for tool extraction, or just content for traditional approach
        if tools:
            return body  # Return full response so tool calls can be extracted
        else:
            return next(block["text"] for block in body["content"] if "text" in block)

    @tracer.start_as_current_span(name="AnthropicInferenceClient.connect_and_listen")
    async def connect_and_listen(
        self,
        messages: List[ChatMessage],
        max_tokens=1024,
        top_p=0.9,
        temperature=0.5,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):

        system = None
        if messages[0].role == ChatRole.SYSTEM:
            system = {
                "system": [
                    {"type": "text", "text": messages[0].content}
                    | (
                        {"cache_control": {"type": "ephemeral"}}
                        if messages[0].cache_marker
                        else {}
                    )
                ]
            }
            messages = messages[1:]

        request = {
            "model": self.model,
            "top_p": top_p,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": msg.role.value,
                    "content": [
                        {"type": "text", "text": msg.content}
                        | (
                            {"cache_control": {"type": "ephemeral"}}
                            if msg.cache_marker
                            else {}
                        )
                    ],
                }
                for msg in messages
            ],
            "stream": True,
        }

        # Add tools if provided
        if tools:
            request["tools"] = tools
            request["tool_choice"] = {"type": "auto"}

        if system:
            request.update(system)

        if self.thinking:
            request["thinking"] = {"type": "enabled", "budget_tokens": self.thinking}
            request["max_tokens"] += self.thinking
            request["temperature"] = 1
            del request["top_p"]

        async with httpx.AsyncClient() as client:
            async with self._stream(
                client,
                request,
            ) as response:
                if response.status_code == 400:
                    raise ValueError(await response.aread())
                elif response.status_code != 200:
                    raise InferenceException(await response.aread())

                out = ""
                current_tool_blocks = {}  # Track tool building by index like OpenAI

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk_json = json.loads(line[6:])
                        chunk_type = chunk_json["type"]

                        if chunk_type == "content_block_delta":
                            content_type = chunk_json["delta"]["type"]

                            if content_type == "text_delta":
                                text = chunk_json["delta"]["text"]
                                out += text
                                yield ("content", text)
                            elif content_type == "input_json_delta" and tools:
                                # Progressive tool input accumulation (like OpenAI)
                                index = chunk_json["index"]
                                if index in current_tool_blocks:
                                    # Accumulate partial JSON
                                    current_tool_blocks[index]["input"] += chunk_json[
                                        "delta"
                                    ]["partial_json"]

                                    # Convert to OpenAI format
                                    openai_format = {
                                        "id": current_tool_blocks[index]["id"],
                                        "function": {
                                            "name": current_tool_blocks[index]["name"],
                                            "arguments": current_tool_blocks[index][
                                                "input"
                                            ],  # JSON string
                                        },
                                    }
                                    yield ("tool_call_partial", openai_format)

                        elif chunk_type == "content_block_start":
                            # Initialize tool tracking
                            content_block = chunk_json.get("content_block", {})
                            if content_block.get("type") == "tool_use":
                                index = chunk_json["index"]
                                current_tool_blocks[index] = {
                                    "id": content_block["id"],
                                    "name": content_block["name"],
                                    "input": "",  # Start accumulating JSON
                                }

                        elif chunk_type == "content_block_stop":
                            # Tool completion
                            index = chunk_json["index"]
                            if tools and index in current_tool_blocks:
                                # Convert to OpenAI format
                                openai_format = {
                                    "id": current_tool_blocks[index]["id"],
                                    "function": {
                                        "name": current_tool_blocks[index]["name"],
                                        "arguments": (
                                            json.dumps(
                                                json.loads(
                                                    current_tool_blocks[index]["input"]
                                                )
                                            )
                                            if current_tool_blocks[index]["input"]
                                            else "{}"
                                        ),
                                    },
                                }
                                yield ("tool_call_complete", openai_format)
                                # Clean up completed tool
                                del current_tool_blocks[index]
                        elif chunk_type == "message_start":
                            self._cost.input_tokens += chunk_json["message"]["usage"][
                                "input_tokens"
                            ]
                            self._cost.cache_read_input_tokens += chunk_json["message"][
                                "usage"
                            ].get("cache_read_input_tokens", 0)
                            self._cost.cache_write_input_tokens += chunk_json[
                                "message"
                            ]["usage"].get("cache_creation_input_tokens", 0)
                        elif chunk_type == "message_delta":
                            self._cost.output_tokens += chunk_json["usage"][
                                "output_tokens"
                            ]

                await self._trace(request["messages"], [{"text": out}])


def _proto_to_dict(obj):
    type_name = str(type(obj).__name__)
    if hasattr(obj, "DESCRIPTOR"):  # Is protobuf message
        return {
            field.name: _proto_to_dict(getattr(obj, field.name))
            for field in obj.DESCRIPTOR.fields
        }
    elif type_name in ("RepeatedComposite", "RepeatedScalarContainer", "list", "tuple"):
        return [_proto_to_dict(x) for x in obj]
    elif type_name in ("dict", "MapComposite", "MessageMap"):
        return {k: _proto_to_dict(v) for k, v in obj.items()}
    return obj


def smart_unescape_code(s: str) -> str:
    # Common patterns in double-escaped code
    replacements = {
        "\\n": "\n",  # Newlines
        '\\"': '"',  # Quotes
        "\\'": "'",  # Single quotes
        "\\\\": "\\",  # Actual backslashes
        "\\t": "\t",  # Tabs
    }

    result = s
    for escaped, unescaped in replacements.items():
        result = result.replace(escaped, unescaped)

    return result


class GoogleGenAIInferenceClient(InferenceClient):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        reasoning: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client(**kwargs)
        self.model = model
        self.reasoning = reasoning

    @tracer.start_as_current_span(name="GoogleGenAIInferenceClient.get_generation")
    @backoff.on_exception(backoff.expo, InferenceException, max_time=60)
    async def get_generation(
        self,
        messages: List[Union[ChatMessage, "ToolResult"]],
        max_tokens=4096,
        top_p=0.9,
        temperature=0.5,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        system_instruction = None

        if (
            len(messages) > 0
            and hasattr(messages[0], "role")
            and messages[0].role.value == "system"
        ):
            system_instruction = messages[0].content
            messages = messages[1:]

        # Convert messages to Google's format - handle both ChatMessage and ToolResult
        processed_messages = []
        for msg in messages:
            try:
                # ChatMessage
                processed_messages.append(
                    types.Content(
                        role=msg.role.value, parts=[types.Part(text=msg.content)]
                    )
                )
            except AttributeError:
                # ToolResult - convert to Google format
                google_content = msg.to_provider_format("google")
                processed_messages.append(
                    types.Content(
                        role="user", parts=[types.Part(text=json.dumps(google_content))]
                    )
                )

        # Prepare config with optional tools
        config_kwargs = {
            "temperature": temperature,
            "system_instruction": system_instruction,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
            "thinking_config": (
                types.GenerationConfigThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=self.reasoning,
                )
                if self.reasoning is not None
                else types.GenerationConfigThinkingConfig(
                    include_thoughts=True,
                )
            ),
        }

        # Add tools if provided (Google uses function_declarations)
        if tools and len(tools) > 0 and "function_declarations" in tools[0]:
            # Convert function declarations to Google Tool objects
            function_declarations = tools[0]["function_declarations"]
            google_tools = []
            for func_decl in function_declarations:
                # Create Google Tool object from function declaration
                tool = types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=func_decl["name"],
                            description=func_decl["description"],
                            parameters=func_decl["parameters"],
                        )
                    ]
                )
                google_tools.append(tool)
            config_kwargs["tools"] = google_tools

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=processed_messages,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except google.genai.errors.ClientError as e:
            if e.code == 400:
                raise NonRetryableException(str(e))
            raise InferenceException(str(e))

        if not response.candidates:
            return ""

        if not response.candidates[0].content.parts:
            # Sometimes seeing this where we only get a bunch of citation_metadata back and no actual content
            return ""

        reasoning = "\n".join(
            p.text for p in response.candidates[0].content.parts if p.text and p.thought
        )
        out_text = "\n".join(
            p.text
            for p in response.candidates[0].content.parts
            if p.text and not p.thought
        )

        # Convert messages for tracing (handle both ChatMessage and ToolResult)
        trace_messages = []
        for msg in messages:
            try:
                # ChatMessage
                trace_messages.append(
                    {
                        "role": msg.role.value,
                        "content": [{"type": "text", "text": msg.content}],
                    }
                )
            except AttributeError:
                # ToolResult - convert to simple format for tracing
                trace_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"[ToolResult: {msg.content[:100]}...]",
                            }
                        ],
                    }
                )

        await self._trace(trace_messages, [{"text": out_text}])

        # Return full response for tool extraction, or processed content for traditional approach
        if tools:
            return response  # Return full response so tool calls can be extracted
        else:
            return (out_text, reasoning)

    @tracer.start_as_current_span(name="GoogleGenAIInferenceClient.connect_and_listen")
    async def connect_and_listen(
        self,
        messages: List[Union[ChatMessage, "ToolResult"]],
        max_tokens=4096,
        top_p=0.9,
        temperature=0.5,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        system_instruction = None

        if (
            len(messages) > 0
            and hasattr(messages[0], "role")
            and messages[0].role.value == "system"
        ):
            system_instruction = messages[0].content
            messages = messages[1:]

        # Convert messages to Google's format - handle both ChatMessage and ToolResult
        processed_messages = []
        for msg in messages:
            try:
                # ChatMessage
                processed_messages.append(
                    types.Content(
                        role=msg.role.value, parts=[types.Part(text=msg.content)]
                    )
                )
            except AttributeError:
                # ToolResult - convert to Google format
                google_content = msg.to_provider_format("google")
                processed_messages.append(
                    types.Content(
                        role="user", parts=[types.Part(text=json.dumps(google_content))]
                    )
                )

        # Prepare config with optional tools
        config_kwargs = {
            "temperature": temperature,
            "system_instruction": system_instruction,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
            "thinking_config": (
                types.GenerationConfigThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=self.reasoning,
                )
                if self.reasoning is not None
                else types.GenerationConfigThinkingConfig(
                    include_thoughts=True,
                )
            ),
        }

        # Add tools if provided (Google uses function_declarations)
        if tools and len(tools) > 0 and "function_declarations" in tools[0]:
            # Convert function declarations to proper Google Tool objects for streaming
            function_declarations = tools[0]["function_declarations"]
            google_tools = []
            for func_decl in function_declarations:
                # Create Google Tool object from function declaration
                tool = types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=func_decl["name"],
                            description=func_decl["description"],
                            parameters=func_decl["parameters"],
                        )
                    ]
                )
                google_tools.append(tool)
            config_kwargs["tools"] = google_tools

        try:
            response = await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=processed_messages,
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except google.genai.errors.ClientError as e:
            if e.code == 400:
                raise NonRetryableException(str(e))
            raise InferenceException(str(e))

        out = ""

        async for chunk in response:
            if not chunk.candidates:
                continue
            if not chunk.candidates[0].content:
                # Sometimes seeing this where we only get a bunch of citation_metadata back and no actual content
                continue
            if not chunk.candidates[0].content.parts:
                # Sometimes seeing this where we only get a bunch of citation_metadata back and no actual content
                continue

            # Check for function calls in this chunk (Google buffers complete tool calls)
            tool_calls_detected = False
            for part in chunk.candidates[0].content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    # Convert Google function call to OpenAI format
                    func_call = part.function_call
                    openai_format = {
                        "id": f"google_{func_call.name}_{hash(str(func_call.args))}",  # Generate ID since Google doesn't provide one
                        "function": {
                            "name": func_call.name,
                            "arguments": json.dumps(dict(func_call.args)),
                        },
                    }
                    yield ("tool_call_complete", openai_format)
                    tool_calls_detected = True
                    # Don't break - process all function calls in the chunk

            # Only process text if no tool calls were detected
            if not tool_calls_detected:
                reasoning = "\n".join(
                    p.text
                    for p in chunk.candidates[0].content.parts
                    if p.text and p.thought
                )
                out_text = "\n".join(
                    p.text
                    for p in chunk.candidates[0].content.parts
                    if p.text and not p.thought
                )
                if reasoning:
                    yield ("reasoning", reasoning)
                elif out_text:  # Only yield content if there's actual text
                    yield ("content", out_text)
                out += reasoning + out_text

        # Convert messages for tracing (handle both ChatMessage and ToolResult)
        trace_messages = []
        for msg in messages:
            try:
                # ChatMessage
                trace_messages.append(
                    {
                        "role": msg.role.value,
                        "content": [{"type": "text", "text": msg.content}],
                    }
                )
            except AttributeError:
                # ToolResult - convert to simple format for tracing
                trace_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"[ToolResult: {msg.content[:100]}...]",
                            }
                        ],
                    }
                )

        await self._trace(trace_messages, [{"text": out}])


class OAIRequest(ResonBase):
    model: str
    messages: List[Dict[str, Any]]
    max_completion_tokens: int = 4096
    temperature: float = 0.5
    top_p: float = 0.9
    stream: bool = False
    stream_options: Dict[str, Any] = {}
    tools: list[dict] = [{}]
    tool_choice: str = "none"


class OAIInferenceClient(InferenceClient):
    def __init__(
        self,
        model: str,
        api_key: str,
        api_url: str = "https://api.openai.com/v1/chat/completions",
        reasoning: str = None,
    ):
        super().__init__()
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.reasoning = reasoning
        self.ranking_referer = None
        self.ranking_title = None
        self.provider = "openai"

    @tracer.start_as_current_span(name="OAIInferenceClient.get_generation")
    @backoff.on_exception(
        backoff.expo, (InferenceException, httpx.HTTPError), max_time=60
    )
    async def get_generation(
        self,
        messages: List[Union[ChatMessage, "ToolResult"]],
        max_tokens=4096,
        top_p=1.0,
        temperature=0.5,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        # Convert ToolResult objects to ChatMessage format
        formatted_messages = []

        for message in messages:
            try:
                msg = {
                    "role": message.role.value,
                    "content": message.content,
                }

                formatted_messages.append(msg)
            except AttributeError as e:
                msg = message.to_provider_format(self.provider)

                formatted_messages.append(msg)

        request = OAIRequest(
            model=self.model,
            messages=formatted_messages,
            max_completion_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            stream=False,
        ).model_dump(exclude={"stream_options"})

        # Add tools if provided
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        else:
            # Remove tools-related fields for non-tool calls
            request = {
                k: v for k, v in request.items() if k not in ["tools", "tool_choice"]
            }

        if self.model in (
            "o3",
            "gpt-5",
        ):
            request.pop("temperature", None)
            request.pop("top_p", None)

        if self.reasoning:
            # Determine if it's effort or max_tokens
            if self.reasoning.isdigit():
                request["reasoning"] = {"max_tokens": int(self.reasoning)}
            else:
                request["reasoning"] = {"effort": self.reasoning}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.api_url,
                json=request,
                timeout=180,
                headers=(
                    {
                        "Content-Type": "application/json",
                    }
                    | (
                        {"Authorization": f"Bearer {self.api_key}"}
                        if self.api_key
                        else {}
                    )
                    | (
                        {"HTTP-Referer": self.ranking_referer}
                        if self.ranking_referer
                        else ({})
                        | (
                            {"X-Title": self.ranking_title}
                            if self.ranking_title
                            else {}
                        )
                    )
                ),
            )

            if response.status_code == 400:
                raise ValueError(await response.aread())
            elif response.status_code != 200:
                raise InferenceException(await response.aread())

            body: dict = response.json()
            if body.get("error"):
                raise InferenceException(
                    body["error"]["message"] + f" ({body['error']})"
                )

            try:
                self._cost.input_tokens += body["usage"]["prompt_tokens"]
                self._cost.cache_read_input_tokens += (
                    body["usage"]
                    .get("prompt_tokens_details", {})
                    .get("cached_tokens", 0)
                )
                self._cost.output_tokens += body["usage"]["completion_tokens"]
            except KeyError:
                logger.warning(f"Malformed usage? {repr(body)}")

            await self._trace(
                request["messages"],
                [{"text": body["choices"][0]["message"]["content"]}],
            )

            # Return full response for tool extraction, or just content for traditional approach
            if tools:
                return body  # Return full response so tool calls can be extracted
            else:
                # Check for reasoning in the response
                content = body["choices"][0]["message"]["content"]
                reasoning = body["choices"][0]["message"].get("reasoning")

                if reasoning:
                    return (content, reasoning)
                return content

    @tracer.start_as_current_span(name="OAIInferenceClient.connect_and_listen")
    async def connect_and_listen(
        self,
        messages: List[ChatMessage | ToolResult],
        max_tokens=4096,
        top_p=1.0,
        temperature=0.5,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        formatted_messages = []

        for message in messages:
            try:
                msg = {
                    "role": message.role.value,
                    "content": message.content,
                }

                formatted_messages.append(msg)
            except AttributeError as e:
                msg = message.to_provider_format(self.provider)

                formatted_messages.append(msg)

        request = OAIRequest(
            model=self.model,
            messages=formatted_messages,
            max_completion_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        ).model_dump()

        # Add tools if provided
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        else:
            # Remove tools-related fields for non-tool calls
            request = {
                k: v for k, v in request.items() if k not in ["tools", "tool_choice"]
            }

        if self.reasoning:
            # Determine if it's effort or max_tokens
            if self.reasoning.isdigit():
                request["reasoning"] = {"max_tokens": int(self.reasoning)}
            else:
                request["reasoning"] = {"effort": self.reasoning}

        if self.model in (
            "o3",
            "gpt-5",
        ):
            request.pop("temperature", None)
            request.pop("top_p", None)

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                self.api_url,
                json=request,
                headers={
                    "Content-Type": "application/json",
                }
                | ({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})
                | (
                    {"HTTP-Referer": self.ranking_referer}
                    if self.ranking_referer
                    else ({})
                    | ({"X-Title": self.ranking_title} if self.ranking_title else {})
                ),
                timeout=180,
            ) as response:
                if response.status_code == 400:
                    raise ValueError(await response.aread())
                elif response.status_code != 200:
                    raise InferenceException(await response.aread())

                out = ""
                reasoning_out = ""
                current_tool_calls = {}  # Index-based accumulation for parallel tools

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        if line.strip() == "data: [DONE]":
                            if current_tool_calls:
                                print("HERE")
                                index = len(current_tool_calls) - 1
                                yield ("tool_call_complete", current_tool_calls[index])

                            break
                        data = json.loads(line[6:])
                        if "error" in data:
                            raise InferenceException(
                                data["error"]["message"] + f" ({data['error']})"
                            )
                        if data["choices"]:
                            delta = data["choices"][0]["delta"]
                            # Check for reasoning first (it can come alongside content)
                            if delta.get("reasoning"):
                                reasoning_out += delta["reasoning"]
                                yield ("reasoning", delta["reasoning"])
                            # Check for content
                            if delta.get("content"):
                                out += delta["content"]
                                yield ("content", delta["content"])

                            # Handle tool calls for native tools
                            if tools and delta.get("tool_calls"):
                                for tool_call_delta in delta["tool_calls"]:
                                    index = tool_call_delta.get("index", 0)

                                    if index not in current_tool_calls:
                                        if index - 1 >= 0 and current_tool_calls.get(
                                            index - 1, None
                                        ):
                                            print("HERE")
                                            yield (
                                                "tool_call_complete",
                                                current_tool_calls[index - 1],
                                            )

                                        current_tool_calls[index] = {
                                            "id": tool_call_delta.get("id", ""),
                                            "function": {
                                                "name": tool_call_delta.get(
                                                    "function", {}
                                                ).get("name", ""),
                                                "arguments": "",
                                            },
                                        }

                                    if tool_call_delta.get("function", {}).get(
                                        "arguments"
                                    ):
                                        current_tool_calls[index]["function"][
                                            "arguments"
                                        ] += tool_call_delta["function"]["arguments"]
                                        yield (
                                            "tool_call_partial",
                                            current_tool_calls[index],
                                        )
                        elif data.get("usage"):
                            self._cost.input_tokens += data["usage"]["prompt_tokens"]
                            # No reference to cached tokens in the docs for the streaming API response objects...
                            try:
                                self._cost.cache_read_input_tokens += (
                                    data["usage"]
                                    .get("prompt_tokens_details", {})
                                    .get("cached_tokens", 0)
                                )
                                self._cost.output_tokens += data["usage"][
                                    "completion_tokens"
                                ]
                            except AttributeError as e:
                                logger.warning(f"Malformed usage? {repr(data)}")

                await self._trace(
                    request["messages"],
                    [{"text": out}],
                )

    @property
    def include_cache_control(self):
        return "anthropic" in self.model


class OpenRouterInferenceClient(OAIInferenceClient):
    def __init__(
        self,
        model: str,
        api_key: str,
        reasoning: str = None,
        ranking_referer: Optional[str] = None,
        ranking_title: Optional[str] = None,
    ):
        super().__init__(
            model,
            api_key,
            api_url="https://openrouter.ai/api/v1/chat/completions",
            reasoning=reasoning,
        )
        self.ranking_referer = ranking_referer
        self.ranking_title = ranking_title
        self.provider = "openrouter"

    async def _populate_cost(self, id: str):
        await asyncio.sleep(0.5)
        async with httpx.AsyncClient() as client:
            for _ in range(3):
                response = await client.get(
                    f"https://openrouter.ai/api/v1/generation?id={id}",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )

                if response.status_code == 404:
                    await asyncio.sleep(0.5)
                    continue

                if response.status_code != 200:
                    await asyncio.sleep(0.5)
                    continue

                body: dict = response.json()

                self._cost.input_tokens += body["data"]["native_tokens_prompt"]
                self._cost.output_tokens += body["data"]["native_tokens_completion"]
                self._cost.dollar_adjust += -(body["data"]["cache_discount"] or 0)
                return


if os.environ.get("ART_ENABLED"):
    import art

    class ARTInferenceClient(InferenceClient):
        def __init__(
            self,
            name: str,
            model: str,
            project: str,
            backend,
            reasoning: Optional[str] = None,
        ):
            super().__init__()
            self.model = model
            self.art_model = art.TrainableModel(
                name=name,
                project=project,
                base_model=model,
            )
            self.backend = backend

            self.reasoning = reasoning
            self.openai_client: Any = None
            self._initialize_lock = asyncio.Lock()

        async def _initialize(self):
            async with self._initialize_lock:
                if self.openai_client:
                    return

                await self.art_model.register(self.backend)
                self.openai_client = self.art_model.openai_client()

        @tracer.start_as_current_span(name="ARTInferenceClient.get_generation")
        @backoff.on_exception(backoff.expo, InferenceException, max_time=60)
        async def get_generation(
            self,
            messages: List[ChatMessage],
            max_tokens=4096,
            top_p=1.0,
            temperature=0.5,
        ):
            if not self.openai_client:
                await self._initialize()

            processed_messages = [
                {"role": m.role.value, "content": m.content} for m in messages
            ]

            request = OAIRequest(
                model=self.model,
                messages=processed_messages,
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
                stream=False,
            ).model_dump(exclude={"stream_options", "tools", "tool_choice"})

            if self.reasoning:
                if self.reasoning.isdigit():
                    request["reasoning"] = {"max_tokens": int(self.reasoning)}
                else:
                    request["reasoning"] = {"effort": self.reasoning}

            try:
                response = await self.openai_client.chat.completions.create(**request)
            except Exception as e:
                raise InferenceException(str(e))

            content = response.choices[0].message.content
            reasoning = getattr(response.choices[0].message, "reasoning", None)

            if response.usage:
                self._cost.input_tokens += response.usage.prompt_tokens
                self._cost.output_tokens += response.usage.completion_tokens

            await self._trace(
                processed_messages,
                [{"text": content}],
            )

            if reasoning:
                return (content, reasoning)
            return content

        @tracer.start_as_current_span(name="ARTInferenceClient.connect_and_listen")
        async def connect_and_listen(
            self,
            messages: List[ChatMessage],
            max_tokens=4096,
            top_p=1.0,
            temperature=0.5,
        ):
            if not self.openai_client:
                await self._initialize()

            processed_messages = [
                {"role": msg.role.value, "content": msg.content} for msg in messages
            ]

            request = OAIRequest(
                model=self.model,
                messages=processed_messages,
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            ).model_dump(exclude={"tools", "tool_choice"})

            if self.reasoning:
                if self.reasoning.isdigit():
                    request["reasoning"] = {"max_tokens": int(self.reasoning)}
                else:
                    request["reasoning"] = {"effort": self.reasoning}

            try:
                stream = await self.openai_client.chat.completions.create(**request)
            except Exception as e:
                raise InferenceException(str(e))

            out = ""
            async for chunk in stream:
                if chunk.usage:
                    self._cost.input_tokens += chunk.usage.prompt_tokens
                    self._cost.output_tokens += chunk.usage.completion_tokens

                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if reasoning := getattr(delta, "reasoning", None):
                        yield "reasoning", reasoning
                    if content := getattr(delta, "content", None):
                        out += content
                        yield "content", content

            await self._trace(
                processed_messages,
                [{"text": out}],
            )


class GoogleAnthropicInferenceClient(AnthropicInferenceClient):
    def __init__(
        self,
        model: str,
        region: str = "us-east5",
        thinking: Optional[int] = None,
    ):
        InferenceClient.__init__(self)
        self.model = model
        self.region = region
        self.thinking = thinking
        self._get_token()

    def _get_token(self):
        if not hasattr(self, "creds"):
            import google.oauth2.id_token
            import google.auth.transport.requests

            self.creds, self.project_id = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

        if not self.creds.token or self.creds.expired:
            request = google.auth.transport.requests.Request()
            self.creds.refresh(request)

        return self.creds.token

    async def _post(self, request: dict):
        request.pop("model", None)
        request["anthropic_version"] = "vertex-2023-10-16"

        async with httpx.AsyncClient() as client:
            return await client.post(
                f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/anthropic/models/{self.model}:streamRawPredict",
                timeout=180,
                json=request,
                headers={
                    "Authorization": f"Bearer {self._get_token()}",
                },
            )

    def _stream(self, client, request: dict):
        request.pop("model", None)
        request["anthropic_version"] = "vertex-2023-10-16"

        return client.stream(
            "POST",
            f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/anthropic/models/{self.model}:streamRawPredict",
            timeout=180,
            json=request,
            headers={
                "Authorization": f"Bearer {self._get_token()}",
            },
        )
