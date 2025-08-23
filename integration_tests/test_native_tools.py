"""Integration test for native tool calling functionality."""

import pytest
from typing import Dict
from reson import agentic, Runtime
from reson.types import Deserializable
from reson.utils.schema_generators import supports_native_tools
from reson.stores import MemoryStore


class Calculator(Deserializable):
    """A calculator operation."""

    operation: str
    a: float
    b: float


def calculate(calc: Calculator) -> float:
    """Perform a calculation operation."""
    if calc.operation == "add":
        return calc.a + calc.b
    elif calc.operation == "multiply":
        return calc.a * calc.b
    elif calc.operation == "subtract":
        return calc.a - calc.b
    elif calc.operation == "divide":
        return calc.a / calc.b if calc.b != 0 else float("inf")
    else:
        raise ValueError(f"Unknown operation: {calc.operation}")


def simple_greeting(name: str) -> str:
    """Generate a simple greeting."""
    return f"Hello, {name}!"


class TestNativeToolsIntegration:
    """Test native tool calling integration."""

    def test_runtime_native_tools_validation(self):
        """Test that Runtime validates native tools support correctly."""
        # Test with supported provider
        rt = Runtime(model="openai", store=MemoryStore(), native_tools=True)
        assert rt.native_tools is True

        # Test with unsupported provider
        with pytest.raises(ValueError, match="Native tools not supported"):
            Runtime(
                model="unsupported-provider", store=MemoryStore(), native_tools=True
            )

        # Test without native_tools enabled (should work for any provider)
        rt2 = Runtime(
            model="unsupported-provider", store=MemoryStore(), native_tools=False
        )
        assert rt2.native_tools is False

    def test_decorator_native_tools_parameter(self):
        """Test that decorators accept and pass through native_tools parameter."""

        @agentic(model="openai", native_tools=True)
        async def test_function(runtime: Runtime) -> str:
            """Test function with native tools enabled."""
            return "test"

        # This should not raise an error if the parameter is accepted
        assert test_function is not None

        @agentic(model="openai", native_tools=False)
        async def test_function2(runtime: Runtime) -> str:
            """Test function with native tools disabled."""
            return "test"

        assert test_function2 is not None

    def test_tool_registration_works_with_native_tools(self):
        """Test that tool registration works when native_tools is enabled."""
        rt = Runtime(model="openai", store=MemoryStore(), native_tools=True)

        # Register tools
        rt.tool(calculate)
        rt.tool(simple_greeting)

        # Verify tools are registered
        assert "calculate" in rt._tools
        assert "simple_greeting" in rt._tools
        assert rt._tools["calculate"] == calculate
        assert rt._tools["simple_greeting"] == simple_greeting

    def test_schema_generation_integration(self):
        """Test that schema generation works with registered tools."""
        from reson.reson import _generate_native_tool_schemas

        tools = {"calculate": calculate, "simple_greeting": simple_greeting}

        # Test OpenAI schema generation
        schemas = _generate_native_tool_schemas(tools, "openai")
        assert len(schemas) == 2

        # Verify calculator tool schema
        calc_schema = next(s for s in schemas if s["function"]["name"] == "calculate")
        assert calc_schema["type"] == "function"
        assert "calc" in calc_schema["function"]["parameters"]["properties"]

        calc_param = calc_schema["function"]["parameters"]["properties"]["calc"]
        assert calc_param["type"] == "object"
        assert "operation" in calc_param["properties"]
        assert "a" in calc_param["properties"]
        assert "b" in calc_param["properties"]

        # Verify greeting tool schema
        greeting_schema = next(
            s for s in schemas if s["function"]["name"] == "simple_greeting"
        )
        assert (
            greeting_schema["function"]["parameters"]["properties"]["name"]["type"]
            == "string"
        )

    def test_tool_instance_creation(self):
        """Test creation of tool instances from arguments."""
        from reson.reson import _create_tool_instance

        # Test simple function
        arguments = {"name": "Alice"}
        tool_instance = _create_tool_instance(
            simple_greeting, arguments, "simple_greeting"
        )

        assert tool_instance._tool_name == "simple_greeting"
        assert tool_instance._tool_func == simple_greeting
        assert tool_instance.name == "Alice"

        # Test function with Deserializable parameter
        calc_args = {"calc": {"operation": "add", "a": 5.0, "b": 3.0}}
        calc_instance = _create_tool_instance(calculate, calc_args, "calculate")

        assert calc_instance._tool_name == "calculate"
        assert calc_instance._tool_func == calculate
        assert hasattr(calc_instance, "calc")

    @pytest.mark.parametrize(
        "provider", ["openai", "anthropic", "google-gemini", "openrouter"]
    )
    def test_provider_support(self, provider):
        """Test that all expected providers support native tools."""
        assert supports_native_tools(provider)

    def test_provider_prefix_parsing(self):
        """Test that provider prefixes are parsed correctly."""
        assert supports_native_tools("openai:gpt-4")
        assert supports_native_tools("anthropic:claude-3")
        assert supports_native_tools("google-gemini:gemini-pro")

    def test_native_tools_disabled_by_default(self):
        """Test that native_tools defaults to False."""
        rt = Runtime(model="openai", store=MemoryStore())
        assert rt.native_tools is False

        @agentic(model="openai")
        async def test_func(runtime: Runtime) -> str:
            return "test"

        # Should work without error - native_tools defaults to False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
