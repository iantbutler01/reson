"""Anthropic-format tool schema generator."""

from typing import Dict, Any, Callable, List

from .base import SchemaGenerator


class AnthropicSchemaGenerator(SchemaGenerator):
    """Generate Anthropic-format tool schemas."""

    def get_provider_name(self) -> str:
        return "anthropic"

    def generate_tool_schemas(self, tools: Dict[str, Callable]) -> List[Dict[str, Any]]:
        """Generate Anthropic tools format: [{"name": "...", "description": "...", "input_schema": {...}}]"""
        schemas = []
        for tool_name, tool_func in tools.items():
            func_info = self._extract_function_info(tool_func)

            schema = {
                "name": tool_name,
                "description": func_info["description"],
                "input_schema": {
                    "type": "object",
                    "properties": func_info["parameters"],
                    "required": func_info["required"],
                },
            }
            schemas.append(schema)
        return schemas
