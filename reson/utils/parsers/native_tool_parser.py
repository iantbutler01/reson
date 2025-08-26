"""
Native tool parser for handling tool call deltas and complete calls with Deserializable marshalling.
"""

import json
from typing import Any, Dict, Type
from .type_parser import TypeParser
from .base import ParserResult
from reson.types import Deserializable
from json_repair import repair_json


class NativeToolParser(TypeParser):
    """Parser for native tool calls that builds Deserializable objects from JSON deltas."""

    def __init__(self, tools_registry: Dict[str, Type[Deserializable]]):
        """Initialize with tools registry for dynamic type lookup.

        Args:
            tools_registry: Mapping of tool names to their Deserializable types
        """
        super().__init__()
        self.tools_registry = tools_registry

    def parse_tool_delta(self, tool_name: str, delta_json: str) -> ParserResult:
        """Parse partial tool call JSON into partial Deserializable.

        Args:
            tool_name: Name of the tool being called
            delta_json: Partial JSON string for the tool arguments

        Returns:
            ParserResult with partial Deserializable object
        """
        try:
            # Get the tool type
            tool_type = self.tools_registry.get(tool_name)
            if not tool_type:
                return ParserResult(
                    error=ValueError(f"Tool '{tool_name}' not found in registry"),
                    is_partial=True,
                )

            partial_data = {}
            try:
                partial_data = json.loads(repair_json(delta_json))
                partial_tool = tool_type.__gasp_from_partial__(partial_data)

                # Add tool name for identification
                setattr(partial_tool, "_tool_name", tool_name)

                return ParserResult(
                    value=partial_tool, is_partial=True, raw_output=delta_json
                )
            except Exception as e:
                print(e)

                return ParserResult(
                    value=tool_type.__gasp_from_partial__({}),
                    is_partial=True,
                    raw_output=delta_json,
                )

        except json.JSONDecodeError as e:
            # Invalid JSON - likely still building up
            return ParserResult(error=e, is_partial=True, raw_output=delta_json)
        except Exception as e:
            return ParserResult(error=e, is_partial=True, raw_output=delta_json)

    def parse_tool_complete(self, tool_name: str, complete_json: str) -> ParserResult:
        """Parse complete tool call JSON into complete Deserializable.

        Args:
            tool_name: Name of the tool being called
            complete_json: Complete JSON string for the tool arguments

        Returns:
            ParserResult with complete Deserializable object
        """
        try:
            # Get the tool type
            tool_type = self.tools_registry.get(tool_name)
            if not tool_type:
                return ParserResult(
                    error=ValueError(f"Tool '{tool_name}' not found in registry"),
                    is_partial=False,
                )

            # Parse complete JSON data
            complete_data = json.loads(complete_json)

            # Use GASP's partial building capability (works for complete data too)
            complete_tool = tool_type.__gasp_from_partial__(complete_data)

            # Add tool name for identification
            setattr(complete_tool, "_tool_name", tool_name)

            return ParserResult(
                value=complete_tool, is_partial=False, raw_output=complete_json
            )

        except Exception as e:
            return ParserResult(error=e, is_partial=False, raw_output=complete_json)

    def extract_tool_name_from_delta(self, tool_call_data: Any) -> str:
        """Extract tool name from OpenAI tool call delta format.

        Args:
            tool_call_data: Tool call delta object from OpenAI/OpenRouter

        Returns:
            Tool name string
        """
        if isinstance(tool_call_data, dict):
            # OpenAI format: {"function": {"name": "tool_name"}}
            function_data = tool_call_data.get("function", {})
            return function_data.get("name", "")
        elif hasattr(tool_call_data, "function"):
            # Object format
            return getattr(tool_call_data.function, "name", "")

        return ""

    def extract_arguments_from_delta(self, tool_call_data: Any) -> str:
        """Extract arguments JSON from OpenAI tool call delta format.

        Args:
            tool_call_data: Tool call delta object from OpenAI/OpenRouter

        Returns:
            Arguments JSON string
        """
        if isinstance(tool_call_data, dict):
            # OpenAI format: {"function": {"arguments": "{\"a\": 5}"}}
            function_data = tool_call_data.get("function", {})
            return function_data.get("arguments", "{}")
        elif hasattr(tool_call_data, "function"):
            # Object format
            return getattr(tool_call_data.function, "arguments", "{}")

        return "{}"
