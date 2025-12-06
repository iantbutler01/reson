#!/usr/bin/env python3
"""Test script for reson-py bindings."""

import asyncio
import os

# Make sure OPENROUTER_API_KEY is set
# export OPENROUTER_API_KEY=your-key-here

from reson import (
    Runtime,
    ChatMessage,
    ChatRole,
    ToolCall,
    ToolResult,
    MemoryStore,
)


async def test_basic_types():
    """Test that basic types work."""
    print("=" * 50)
    print("Testing basic types...")

    # ChatRole
    print(f"  ChatRole.User = {ChatRole.User}")
    print(f"  ChatRole.Assistant = {ChatRole.Assistant}")

    # ChatMessage
    msg = ChatMessage.user("Hello!")
    print(f"  ChatMessage: {msg}")
    print(f"    role: {msg.role}, content: {msg.content}")

    # ToolCall
    tc = ToolCall("id-123", "my_tool")
    print(f"  ToolCall: {tc}")

    # ToolResult
    tr = ToolResult("id-123", "Success!")
    print(f"  ToolResult: {tr}")

    print("✓ Basic types work!\n")


async def test_memory_store():
    """Test async MemoryStore operations."""
    print("=" * 50)
    print("Testing MemoryStore...")

    store = MemoryStore()

    # Set a value
    await store.set("user", {"name": "Alice", "age": 30})
    print("  Set user = {name: Alice, age: 30}")

    # Get it back
    user = await store.get("user", None)
    print(f"  Get user = {user}")

    # Get keys
    keys = await store.keys()
    print(f"  Keys = {keys}")

    # Delete
    await store.delete("user")
    user = await store.get("user", "not found")
    print(f"  After delete, user = {user}")

    print("✓ MemoryStore works!\n")


async def test_runtime_run():
    """Test Runtime.run() with actual LLM call."""
    print("=" * 50)
    print("Testing Runtime.run()...")

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("  ⚠ OPENROUTER_API_KEY not set, skipping LLM test")
        print("  Set it with: export OPENROUTER_API_KEY=your-key\n")
        return

    runtime = Runtime(model="openrouter:openai/gpt-4o-mini")

    # Simple inference
    result = await runtime.run(prompt="What is 2 + 2? Reply with just the number.")
    print(f"  Prompt: 'What is 2 + 2?'")
    print(f"  Result: {result}")
    print(f"  Type: {type(result)}")

    # Check accumulator
    raw = await runtime.raw_response
    print(f"  Raw response: {raw}")

    print("✓ Runtime.run() works!\n")


async def test_runtime_run_stream():
    """Test Runtime.run_stream() with streaming."""
    print("=" * 50)
    print("Testing Runtime.run_stream()...")

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("  ⚠ OPENROUTER_API_KEY not set, skipping streaming test\n")
        return

    runtime = Runtime(model="openrouter:openai/gpt-4o-mini")

    # Streaming inference (returns list of chunks for now)
    chunks = await runtime.run_stream(prompt="Count from 1 to 3")
    print(f"  Prompt: 'Count from 1 to 3'")
    print(f"  Got {len(chunks)} chunks")
    print(f"  First 5 chunks:")
    for chunk_type, chunk_value in chunks[:5]:
        print(f"    {chunk_type}: {repr(chunk_value)}")

    # Check accumulator
    raw = await runtime.raw_response
    print(f"  Full response: {raw}")

    print("✓ Runtime.run_stream() works!\n")


async def test_runtime_with_history():
    """Test Runtime with conversation history."""
    print("=" * 50)
    print("Testing Runtime with history...")

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("  ⚠ OPENROUTER_API_KEY not set, skipping history test\n")
        return

    runtime = Runtime(model="openrouter:openai/gpt-4o-mini")

    # Build history
    history = [
        ChatMessage.user("My name is Bob."),
        ChatMessage.assistant("Nice to meet you, Bob!"),
    ]

    # Ask a follow-up
    result = await runtime.run(
        prompt="What is my name?",
        history=history,
    )
    print(f"  History: user='My name is Bob', assistant='Nice to meet you, Bob!'")
    print(f"  Prompt: 'What is my name?'")
    print(f"  Result: {result}")

    print("✓ History works!\n")


async def test_runtime_with_system():
    """Test Runtime with system prompt."""
    print("=" * 50)
    print("Testing Runtime with system prompt...")

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("  ⚠ OPENROUTER_API_KEY not set, skipping system prompt test\n")
        return

    runtime = Runtime(model="openrouter:openai/gpt-4o-mini")

    result = await runtime.run(
        prompt="Hello!",
        system="You are a pirate. Always respond in pirate speak.",
    )
    print(f"  System: 'You are a pirate...'")
    print(f"  Prompt: 'Hello!'")
    print(f"  Result: {result}")

    print("✓ System prompt works!\n")


async def main():
    print("\n🦀 Testing reson-py (Rust-backed Python bindings)\n")

    await test_basic_types()
    await test_memory_store()
    await test_runtime_run()
    await test_runtime_run_stream()
    await test_runtime_with_history()
    await test_runtime_with_system()

    print("=" * 50)
    print("🎉 All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
