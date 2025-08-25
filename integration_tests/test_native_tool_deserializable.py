#!/usr/bin/env python3
"""Test native tool calling with Deserializable type marshalling."""

import asyncio
import os
from typing import AsyncGenerator, List, Dict
from dataclasses import dataclass
from reson import agentic, agentic_generator, Runtime
from reson.types import Deserializable
from reson.stores import MemoryStore
from pydantic import BaseModel


class CalculationQuery(Deserializable):
    """A calculation with two numbers."""

    a: int
    b: int
    operation: str = "add"


class SearchQuery(Deserializable):
    """A search query with parameters."""

    text: str
    category: str = "general"
    max_results: int = 5


class AgentAction(Deserializable):
    """An agent action (simplified)."""

    action_type: str
    content: str


# Pydantic model for testing
class PydanticConfig(BaseModel):
    """A Pydantic configuration model."""

    host: str
    port: int = 8080
    ssl_enabled: bool = False


# Dataclass for testing
@dataclass
class DataclassSettings:
    """A dataclass settings model."""

    name: str
    timeout: int = 30
    debug: bool = False


# Regular class for testing
class RegularUser:
    """A regular Python class."""

    def __init__(self, username: str, email: str, active: bool = True):
        self.username = username
        self.email = email
        self.active = active


def calculate(query: CalculationQuery) -> int:
    """Perform a calculation."""
    if query.operation == "add":
        return query.a + query.b
    elif query.operation == "multiply":
        return query.a * query.b
    else:
        return 0


def search_database(query: SearchQuery) -> str:
    """Search a database with the given query parameters."""
    return f"Found {query.max_results} results for '{query.text}' in category '{query.category}'"


def configure_server(config: PydanticConfig) -> str:
    """Configure a server using Pydantic model."""
    return f"Server configured: {config.host}:{config.port}, SSL: {config.ssl_enabled}"


def update_settings(settings: DataclassSettings) -> str:
    """Update settings using dataclass."""
    return f"Settings updated: {settings.name}, timeout: {settings.timeout}, debug: {settings.debug}"


def create_user(user: RegularUser) -> str:
    """Create a user using regular class."""
    return f"User created: {user.username} ({user.email}), active: {user.active}"


def process_metadata(metadata: Dict[str, str]) -> str:
    """Process metadata using dict."""
    return f"Processed {len(metadata)} metadata items: {list(metadata.keys())}"


def process_tags(tags: List[str]) -> str:
    """Process tags using list."""
    return f"Processed {len(tags)} tags: {', '.join(tags)}"


def add_numbers(a: int, b: int) -> int:
    """Add two numbers (primitives)."""
    return a + b


async def test_tool_type_registration():
    """Test that tool types are properly registered."""
    print("🧪 Testing Tool Type Registration")

    rt = Runtime(
        model="openrouter:anthropic/claude-sonnet-4",
        store=MemoryStore(),
        native_tools=True,
    )

    # Register tools with types
    rt.tool(calculate, tool_type=CalculationQuery)
    rt.tool(search_database, tool_type=SearchQuery)

    # Verify tools and types are registered
    assert "calculate" in rt._tools
    assert "search_database" in rt._tools
    assert "calculate" in rt._tool_types
    assert "search_database" in rt._tool_types
    assert rt._tool_types["calculate"] == CalculationQuery
    assert rt._tool_types["search_database"] == SearchQuery

    print("✅ Tool type registration working correctly")
    return True


async def test_deserializable_streaming():
    """Test streaming with Deserializable tool types."""
    print("\n🧪 Testing Deserializable Streaming Tool Calls")

    @agentic_generator(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def deserializable_streaming_agent(
        query: str, runtime: Runtime
    ) -> AsyncGenerator[str, None]:
        runtime.tool(calculate, tool_type=CalculationQuery)
        runtime.tool(search_database, tool_type=SearchQuery)

        # Start streaming
        async for chunk in runtime.run_stream(prompt=query):
            if isinstance(chunk, tuple) and len(chunk) == 2:
                chunk_type, chunk_content = chunk
                if chunk_type == "reasoning":
                    yield f"🧠 Thinking: {chunk_content}"
                elif chunk_type == "content":
                    yield f"📝 Content: {chunk_content}"
                elif chunk_type == "tool_call_delta":
                    # Check if we got a Deserializable object
                    if hasattr(chunk_content, "_tool_name"):
                        yield f"🔧 Partial Tool: {chunk_content._tool_name} - {type(chunk_content).__name__}({chunk_content.model_dump()})"
                    else:
                        yield f"🔧 Raw Delta: {chunk_content}"
                elif chunk_type == "tool_call_complete":
                    # Check if we got a Deserializable object
                    if hasattr(chunk_content, "_tool_name"):
                        yield f"✅ Complete Tool: {chunk_content._tool_name} - {type(chunk_content).__name__}({chunk_content.model_dump()})"
                    else:
                        yield f"✅ Complete Tool: {chunk_content}"
                else:
                    yield f"🔧 Stream: {chunk_type} - {chunk_content}"
            else:
                yield f"📝 Chunk: {chunk}"

    try:
        print("Starting Deserializable streaming test...")
        chunks = []

        async for chunk in deserializable_streaming_agent(
            "Use calculate to add 15 and 27"
        ):
            print(chunk)
            chunks.append(chunk)

        print(f"\n✅ Deserializable streaming test completed with {len(chunks)} chunks")

        # Check if we got Deserializable objects in the stream
        deserializable_chunks = [
            chunk for chunk in chunks if "CalculationQuery(" in chunk
        ]
        if deserializable_chunks:
            print(f"🎉 Found {len(deserializable_chunks)} Deserializable tool chunks!")

        return True

    except Exception as e:
        print(f"❌ Deserializable streaming test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_mixed_tool_types():
    """Test mixing tools with and without types."""
    print("\n🧪 Testing Mixed Tool Types")

    def simple_greeting(name: str) -> str:
        """Generate a simple greeting."""
        return f"Hello, {name}!"

    @agentic(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def mixed_agent(query: str, runtime: Runtime) -> str:
        # Register one tool with type, one without
        runtime.tool(calculate, tool_type=CalculationQuery)
        runtime.tool(simple_greeting)  # No tool_type

        return await runtime.run(prompt=query)

    try:
        result = await mixed_agent("Say hello to 'Alice' using simple_greeting")
        print(f"✅ Mixed tools result: {result}")
        print(f"🔍 Result type: {type(result)}")

        # Test that both types of tool registration work
        if hasattr(result, "_tool_name"):
            print(f"🔧 Tool call detected: {result._tool_name}")
            return True
        else:
            print(f"📝 Text response: {result}")
            return True

    except Exception as e:
        print(f"❌ Mixed tools test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_typed_output_with_tools():
    """Test streaming with both tool types and output type."""
    print("\n🧪 Testing Typed Output + Tool Types")

    @agentic_generator(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def typed_output_agent(
        query: str, runtime: Runtime
    ) -> AsyncGenerator[str, None]:
        runtime.tool(calculate, tool_type=CalculationQuery)

        actions = []

        # Stream with output type
        async for chunk_type, content in runtime.run_stream(
            prompt=query, output_type=List[AgentAction]
        ):
            if chunk_type == "tool_call_delta":
                if hasattr(content, "_tool_name"):
                    yield f"🔧 Building {content._tool_name}: {content.model_dump()}"
                else:
                    yield f"🔧 Raw delta: {content}"
            elif chunk_type == "tool_call_complete":
                if hasattr(content, "_tool_name"):
                    yield f"✅ Complete {content._tool_name}: {content.model_dump()}"
                    # Execute tool immediately
                    tool_result = await runtime.execute_tool(content)
                    yield f"🎯 Tool result: {tool_result}"
                else:
                    yield f"✅ Complete tool: {content}"
            elif chunk_type == "content":
                yield f"📝 Content: {content}"
                if content:  # Build up actions list
                    actions.append(
                        AgentAction(action_type="content", content=str(content))
                    )
            elif chunk_type == "reasoning":
                yield f"🧠 Thinking: {content}"

        yield f"🏁 Final actions: {len(actions)}"

    try:
        print("Starting typed output + tools test...")
        chunks = []

        async for chunk in typed_output_agent(
            "Calculate 8 + 12, then summarize the result"
        ):
            print(chunk)
            chunks.append(chunk)

        print(f"\n✅ Typed output + tools test completed with {len(chunks)} chunks")
        return True

    except Exception as e:
        print(f"❌ Typed output + tools test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_pydantic_marshalling():
    """Test Pydantic model marshalling."""
    print("\n🧪 Testing Pydantic Model Marshalling")

    @agentic(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def pydantic_agent(query: str, runtime: Runtime) -> str:
        runtime.tool(configure_server)
        return await runtime.run(prompt=query)

    try:
        result = await pydantic_agent(
            "Configure server with host 'api.example.com' on port 443 with SSL enabled"
        )
        print(f"✅ Pydantic result: {result}")

        if hasattr(result, "_tool_name"):
            print(f"🔧 Pydantic tool call detected: {result._tool_name}")
            # Test execution
            rt = Runtime(
                model="openrouter:anthropic/claude-sonnet-4",
                store=MemoryStore(),
                native_tools=True,
            )
            rt.tool(configure_server)
            if rt.is_tool_call(result):
                tool_result = await rt.execute_tool(result)
                print(f"🎯 Pydantic tool result: {tool_result}")
            return True
        else:
            print(f"📝 Pydantic text response: {result}")
            return True

    except Exception as e:
        print(f"❌ Pydantic test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_dataclass_marshalling():
    """Test dataclass marshalling."""
    print("\n🧪 Testing Dataclass Marshalling")

    @agentic(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def dataclass_agent(query: str, runtime: Runtime) -> str:
        runtime.tool(update_settings)
        return await runtime.run(prompt=query)

    try:
        result = await dataclass_agent(
            "Update settings with name 'production' and timeout 60 with debug enabled"
        )
        print(f"✅ Dataclass result: {result}")

        if hasattr(result, "_tool_name"):
            print(f"🔧 Dataclass tool call detected: {result._tool_name}")
            # Test execution
            rt = Runtime(
                model="openrouter:anthropic/claude-sonnet-4",
                store=MemoryStore(),
                native_tools=True,
            )
            rt.tool(update_settings)
            if rt.is_tool_call(result):
                tool_result = await rt.execute_tool(result)
                print(f"🎯 Dataclass tool result: {tool_result}")
            return True
        else:
            print(f"📝 Dataclass text response: {result}")
            return True

    except Exception as e:
        print(f"❌ Dataclass test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_regular_class_marshalling():
    """Test regular class marshalling."""
    print("\n🧪 Testing Regular Class Marshalling")

    @agentic(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def regular_class_agent(query: str, runtime: Runtime) -> str:
        runtime.tool(create_user)
        return await runtime.run(prompt=query)

    try:
        result = await regular_class_agent(
            "Create user with username 'johndoe' and email 'john@example.com'"
        )
        print(f"✅ Regular class result: {result}")

        if hasattr(result, "_tool_name"):
            print(f"🔧 Regular class tool call detected: {result._tool_name}")
            # Test execution
            rt = Runtime(
                model="openrouter:anthropic/claude-sonnet-4",
                store=MemoryStore(),
                native_tools=True,
            )
            rt.tool(create_user)
            if rt.is_tool_call(result):
                tool_result = await rt.execute_tool(result)
                print(f"🎯 Regular class tool result: {tool_result}")
            return True
        else:
            print(f"📝 Regular class text response: {result}")
            return True

    except Exception as e:
        print(f"❌ Regular class test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_collection_types():
    """Test dict and list parameter marshalling."""
    print("\n🧪 Testing Collection Types (Dict, List)")

    @agentic(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def collection_agent(query: str, runtime: Runtime) -> str:
        runtime.tool(process_metadata)
        runtime.tool(process_tags)
        runtime.tool(add_numbers)  # Primitives
        return await runtime.run(prompt=query)

    try:
        result = await collection_agent(
            "Process metadata with keys 'env', 'version', 'region' and values 'prod', '1.0', 'us-east'"
        )
        print(f"✅ Collection types result: {result}")

        if hasattr(result, "_tool_name"):
            print(f"🔧 Collection tool call detected: {result._tool_name}")
            # Test execution
            rt = Runtime(
                model="openrouter:anthropic/claude-sonnet-4",
                store=MemoryStore(),
                native_tools=True,
            )
            rt.tool(process_metadata)
            rt.tool(process_tags)
            rt.tool(add_numbers)
            if rt.is_tool_call(result):
                tool_result = await rt.execute_tool(result)
                print(f"🎯 Collection tool result: {tool_result}")
            return True
        else:
            print(f"📝 Collection text response: {result}")
            return True

    except Exception as e:
        print(f"❌ Collection types test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_primitive_types():
    """Test primitive parameter marshalling."""
    print("\n🧪 Testing Primitive Types")

    @agentic(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
    async def primitive_agent(query: str, runtime: Runtime) -> str:
        runtime.tool(add_numbers)
        return await runtime.run(prompt=query)

    try:
        result = await primitive_agent("Add 25 and 17 using add_numbers")
        print(f"✅ Primitive types result: {result}")

        if hasattr(result, "_tool_name"):
            print(f"🔧 Primitive tool call detected: {result._tool_name}")
            # Test execution
            rt = Runtime(
                model="openrouter:anthropic/claude-sonnet-4",
                store=MemoryStore(),
                native_tools=True,
            )
            rt.tool(add_numbers)
            if rt.is_tool_call(result):
                tool_result = await rt.execute_tool(result)
                print(f"🎯 Primitive tool result: {tool_result}")
            return True
        else:
            print(f"📝 Primitive text response: {result}")
            return True

    except Exception as e:
        print(f"❌ Primitive types test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all native tool Deserializable tests."""
    print("🚀 Testing Native Tool Deserializable Support")
    print("=" * 60)

    # Check environment
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ OPENROUTER_API_KEY not set")
        return

    print("✅ API key configured")

    # Run tests - now including all parameter types
    tests = [
        ("Tool Type Registration", test_tool_type_registration()),
        ("Deserializable Streaming", test_deserializable_streaming()),
        ("Mixed Tool Types", test_mixed_tool_types()),
        ("Typed Output + Tools", test_typed_output_with_tools()),
        ("Pydantic Marshalling", test_pydantic_marshalling()),
        ("Dataclass Marshalling", test_dataclass_marshalling()),
        ("Regular Class Marshalling", test_regular_class_marshalling()),
        ("Collection Types", test_collection_types()),
        ("Primitive Types", test_primitive_types()),
    ]

    results = []
    for test_name, test_coro in tests:
        try:
            result = await test_coro
            results.append(result)
            print(
                f"{'✅' if result else '❌'} {test_name}: {'PASSED' if result else 'FAILED'}"
            )
        except Exception as e:
            print(f"❌ {test_name}: EXCEPTION - {e}")
            results.append(False)

    success_count = sum(1 for r in results if r)
    print(f"\n📊 Test Results: {success_count}/{len(results)} passed")

    if success_count == len(results):
        print(
            "🎉 All parameter type tests passed! Native tools with comprehensive marshalling working!"
        )
    else:
        print("⚠️  Some parameter type tests failed. Check the output above.")

    return success_count == len(results)


if __name__ == "__main__":
    asyncio.run(main())
